#!/usr/bin/env python3
"""
Evaluation Service
==================

This script provides a complete evaluation service for AI agent trajectory data.
It integrates preprocessing and step-level evaluation with LLM APIs.

Features:
- Separate preprocessing and evaluation modes
- Support for multiple LLM providers (OpenAI, Google, Anthropic)
- Batch processing with rate limiting
- Comprehensive reporting and statistics

Usage:
    # Preprocessing only
    python src/evaluation_service.py --input data/demo01.jsonl --preprocess-only --output data/preprocessed.jsonl
    
    # LLM evaluation only (assumes input is already preprocessed)
    python src/evaluation_service.py --input data/preprocessed.jsonl --provider openai --api-key YOUR_KEY --output data/results.jsonl
    
    # Full pipeline (preprocess + evaluate)
    python src/evaluation_service.py --input data/demo01.jsonl --provider openai --api-key YOUR_KEY --full-pipeline --output data/results.jsonl
"""

import asyncio
import argparse
import logging
import json
import os
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import our modules
from preprocess_agent_data import AgentDataPreprocessor
from step_level_evaluator import StepLevelEvaluator, create_evaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EvaluationService:
    """Main evaluation service that coordinates preprocessing and evaluation."""
    
    def __init__(self, 
                 provider: str = "openai",
                 api_key: str = "",
                 model_name: Optional[str] = None,
                 batch_size: int = 3,
                 rate_limit_delay: float = 1.0,
                 preprocess_only: bool = False,
                 full_pipeline: bool = False):
        
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        self.preprocess_only = preprocess_only
        self.full_pipeline = full_pipeline
        
        # Initialize preprocessor
        self.preprocessor = AgentDataPreprocessor(beta_threshold=15)
        
        # Initialize evaluator only if not preprocessing-only mode
        if not preprocess_only:
            if not api_key:
                raise ValueError(f"API key is required for {provider} evaluation. Please provide --api-key or set environment variable.")
            
            try:
                self.evaluator = create_evaluator(provider, api_key, model_name)
                self.evaluator.llm_client.set_rate_limit_delay(rate_limit_delay)
                logger.info(f"Using {provider} for evaluation with model {model_name or 'default'}")
            except Exception as e:
                logger.error(f"Failed to initialize LLM client for {provider}: {e}")
                raise
        else:
            self.evaluator = None
            logger.info("Preprocessing-only mode enabled")
        
        # Statistics tracking
        self.stats = {
            'total_samples': 0,
            'preprocessed_samples': 0,
            'evaluated_samples': 0,
            'failed_evaluations': 0,
            'start_time': None,
            'end_time': None
        }
    
    def load_data(self, input_path: str) -> List[Dict]:
        """Load data from file or directory."""
        
        input_path = Path(input_path)
        
        if input_path.is_file():
            if input_path.suffix == '.jsonl':
                return self.preprocessor.load_jsonl(str(input_path))
            else:
                raise ValueError("Input file must be .jsonl format")
        
        elif input_path.is_dir():
            # Load all .jsonl files in directory
            all_data = []
            for file_path in input_path.glob("*.jsonl"):
                data = self.preprocessor.load_jsonl(str(file_path))
                all_data.extend(data)
            return all_data
        
        else:
            raise ValueError(f"Input path does not exist: {input_path}")
    
    async def evaluate_single_sample(self, sample: Dict) -> Dict:
        """Evaluate a single sample using the step-level evaluator."""
        if self.evaluator is None:
            raise RuntimeError("Evaluator not initialized. Cannot perform evaluation in preprocessing-only mode.")
        
        try:
            # Use the new method that generates full responses
            result = await self.evaluator.evaluate_trajectory_with_full_response(sample)
            self.stats['evaluated_samples'] += 1
            return result
        except Exception as e:
            logger.error(f"Failed to evaluate sample {sample.get('task_id', 'unknown')}: {e}")
            self.stats['failed_evaluations'] += 1
            raise  # Re-raise the exception instead of returning mock data
    
    async def evaluate_batch(self, samples: List[Dict]) -> List[Dict]:
        """Evaluate a batch of samples with rate limiting."""
        
        if self.evaluator is None:
            raise RuntimeError("Evaluator not initialized. Cannot perform evaluation in preprocessing-only mode.")
        
        logger.info(f"Evaluating batch of {len(samples)} samples")
        
        # Create semaphore to limit concurrent evaluations
        semaphore = asyncio.Semaphore(self.batch_size)
        
        async def evaluate_with_semaphore(sample):
            async with semaphore:
                return await self.evaluate_single_sample(sample)
        
        # Process all samples concurrently with rate limiting
        tasks = [evaluate_with_semaphore(sample) for sample in samples]
        
        try:
            evaluated_samples = await asyncio.gather(*tasks)
            return evaluated_samples
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            raise
    
    def preprocess_data(self, data: List[Dict]) -> List[Dict]:
        """Preprocess the data before evaluation."""
        
        logger.info(f"Preprocessing {len(data)} samples")
        self.stats['total_samples'] = len(data)
        
        preprocessed = self.preprocessor.preprocess_data(data)
        self.stats['preprocessed_samples'] = len(preprocessed)
        
        logger.info(f"Preprocessing completed: {len(preprocessed)} valid samples")
        return preprocessed
    
    async def run_preprocessing_only(self, data: List[Dict]) -> List[Dict]:
        """Run preprocessing only."""
        
        logger.info("Running preprocessing only...")
        self.stats['start_time'] = time.time()
        
        # Preprocess data
        preprocessed_data = self.preprocess_data(data)
        
        self.stats['end_time'] = time.time()
        return preprocessed_data
    
    async def run_evaluation_only(self, data: List[Dict]) -> List[Dict]:
        """Run evaluation only (assumes data is already preprocessed)."""
        
        logger.info("Running LLM evaluation only...")
        self.stats['start_time'] = time.time()
        
        # Set stats for evaluation-only mode
        self.stats['total_samples'] = len(data)
        self.stats['preprocessed_samples'] = len(data)
        
        if not data:
            logger.warning("No data provided for evaluation")
            return []
        
        # Evaluate in batches
        evaluated_samples = []
        batch_size = min(self.batch_size, 10)  # Limit batch size to avoid overwhelming APIs
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            try:
                batch_results = await self.evaluate_batch(batch)
                evaluated_samples.extend(batch_results)
                
                # Log progress
                logger.info(f"Completed batch {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}")
                
                # Add delay between batches
                if i + batch_size < len(data):
                    await asyncio.sleep(self.rate_limit_delay)
            except Exception as e:
                logger.error(f"Failed to evaluate batch {i//batch_size + 1}: {e}")
                raise
        
        self.stats['end_time'] = time.time()
        return evaluated_samples
    
    async def run_full_pipeline(self, data: List[Dict]) -> List[Dict]:
        """Run full pipeline: preprocess + evaluate."""
        
        logger.info("Running full pipeline (preprocess + evaluate)...")
        self.stats['start_time'] = time.time()
        
        # Preprocess data
        preprocessed_data = self.preprocess_data(data)
        
        if not preprocessed_data:
            logger.warning("No valid samples after preprocessing")
            return []
        
        # Evaluate in batches
        evaluated_samples = []
        batch_size = min(self.batch_size, 10)  # Limit batch size to avoid overwhelming APIs
        
        for i in range(0, len(preprocessed_data), batch_size):
            batch = preprocessed_data[i:i + batch_size]
            try:
                batch_results = await self.evaluate_batch(batch)
                evaluated_samples.extend(batch_results)
                
                # Log progress
                logger.info(f"Completed batch {i//batch_size + 1}/{(len(preprocessed_data)-1)//batch_size + 1}")
                
                # Add delay between batches
                if i + batch_size < len(preprocessed_data):
                    await asyncio.sleep(self.rate_limit_delay)
            except Exception as e:
                logger.error(f"Failed to evaluate batch {i//batch_size + 1}: {e}")
                raise
        
        self.stats['end_time'] = time.time()
        return evaluated_samples
    
    async def process_data(self, data: List[Dict]) -> List[Dict]:
        """Process data based on the configured mode."""
        
        if self.preprocess_only:
            return await self.run_preprocessing_only(data)
        elif self.full_pipeline:
            return await self.run_full_pipeline(data)
        else:
            # Evaluation only mode
            return await self.run_evaluation_only(data)
    
    def save_results(self, processed_data: List[Dict], output_path: str):
        """Save processing results to a single clean file."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate processing statistics for logging
        processing_time = None
        if self.stats['start_time'] and self.stats['end_time']:
            processing_time = self.stats['end_time'] - self.stats['start_time']
        
        # Log processing statistics
        logger.info(f"Processing completed:")
        logger.info(f"  - Total samples processed: {len(processed_data)}")
        logger.info(f"  - Processing time: {processing_time:.1f} seconds" if processing_time else "  - Processing time: N/A")
        
        if not self.preprocess_only:
            success_rate = (self.stats['evaluated_samples'] / 
                           self.stats['preprocessed_samples'] 
                           if self.stats['preprocessed_samples'] > 0 else 0)
            
            samples_per_second = (self.stats['evaluated_samples'] / processing_time 
                                 if processing_time and processing_time > 0 else 0)
            
            logger.info(f"  - Success rate: {success_rate:.1%}")
            logger.info(f"  - Samples per second: {samples_per_second:.2f}")
            logger.info(f"  - Provider: {self.provider}")
            logger.info(f"  - Model: {self.model_name or 'default'}")
            
            # Calculate evaluation overview for logging
            successful_evaluations = sum(1 for sample in processed_data 
                                       if sample.get('evaluation_metadata', {}).get('overall_trajectory_score', 0) > 0)
            average_clips = sum(len(sample.get('clip_evaluations', [])) for sample in processed_data) / len(processed_data) if processed_data else 0
            
            logger.info(f"Evaluation quality metrics:")
            logger.info(f"  - Successful evaluations: {successful_evaluations}/{len(processed_data)}")
            logger.info(f"  - Average clips per sample: {average_clips:.1f}")
        
        # Save clean JSONL file
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in processed_data:
                if self.preprocess_only:
                    # For preprocessing only, save the preprocessed sample
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                else:
                    # For evaluation, save clean sample with evaluation data
                    clean_sample = {
                        # Original trajectory data
                        'timestamp': sample.get('timestamp', ''),
                        'task_id': sample.get('task_id', ''),
                        'task_description': sample.get('task_description', ''),
                        'duration': sample.get('duration', 0),
                        'success': sample.get('success', False),
                        'final_result': sample.get('final_result', ''),
                        'raw_response': sample.get('raw_response', ''),
                        
                        # Full response with embedded evaluations
                        'full_response_with_evaluations': sample.get('full_response_with_evaluations', ''),
                        
                        # Core evaluation metadata only
                        'evaluation_metadata': sample.get('evaluation_metadata', {})
                    }
                    f.write(json.dumps(clean_sample, ensure_ascii=False) + '\n')
        
        mode_desc = "preprocessing" if self.preprocess_only else "evaluation"
        logger.info(f"Clean {mode_desc} results saved to {output_path}")
        
        if not self.preprocess_only:
            logger.info(f"File contains: original data + full_response_with_evaluations + evaluation_metadata")
    
    def print_summary(self):
        """Print processing summary."""
        
        processing_time = (self.stats['end_time'] - self.stats['start_time'] 
                          if self.stats['start_time'] and self.stats['end_time'] else 0)
        
        print("\n" + "="*60)
        if self.preprocess_only:
            print("PREPROCESSING SERVICE SUMMARY")
        else:
            print("EVALUATION SERVICE SUMMARY")
        print("="*60)
        
        if not self.preprocess_only:
            print(f"Provider: {self.provider}")
            print(f"Model: {self.model_name or 'default'}")
            print(f"Mode: {'Full Pipeline' if self.full_pipeline else 'Evaluation Only'}")
            print("-" * 30)
        
        print(f"Total samples: {self.stats['total_samples']}")
        print(f"Preprocessed: {self.stats['preprocessed_samples']}")
        
        if not self.preprocess_only:
            print(f"Evaluated: {self.stats['evaluated_samples']}")
            print(f"Failed: {self.stats['failed_evaluations']}")
            print(f"Success rate: {(self.stats['evaluated_samples']/self.stats['preprocessed_samples']*100 if self.stats['preprocessed_samples'] > 0 else 0):.1f}%")
        
        print("-" * 30)
        print(f"Processing time: {processing_time:.1f} seconds")
        
        if not self.preprocess_only:
            print(f"Samples per second: {(self.stats['evaluated_samples']/processing_time if processing_time > 0 else 0):.2f}")
        
        print("="*60)


async def main():
    """Main function for the evaluation service."""
    
    parser = argparse.ArgumentParser(description='AI Agent Trajectory Evaluation Service')
    
    # Input/Output arguments
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSONL file or directory path')
    parser.add_argument('--output', type=str, default='data/processed_results.jsonl',
                       help='Output file path for results')
    
    # Processing mode
    parser.add_argument('--preprocess-only', action='store_true',
                       help='Only run preprocessing, skip evaluation')
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run full pipeline: preprocess + evaluate')
    
    # API configuration (only needed for evaluation)
    parser.add_argument('--provider', type=str, default='openai',
                       choices=['openai', 'google', 'anthropic'],
                       help='LLM provider to use')
    parser.add_argument('--api-key', type=str, default='',
                       help='API key for the LLM provider')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model name to use')
    
    # Processing configuration
    parser.add_argument('--batch-size', type=int, default=3,
                       help='Number of concurrent evaluations')
    parser.add_argument('--rate-limit', type=float, default=1.0,
                       help='Delay between API requests (seconds)')
    
    # Logging configuration
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get API key from environment if not provided
    if not args.preprocess_only and not args.api_key:
        env_key_map = {
            'openai': 'OPENAI_API_KEY',
            'google': 'GOOGLE_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY'
        }
        env_key = env_key_map.get(args.provider)
        if env_key:
            args.api_key = os.getenv(env_key, '')
        
        if not args.api_key:
            logger.error(f"API key required for {args.provider} evaluation. Please provide --api-key or set environment variable {env_key}.")
            return
    
    try:
        # Initialize evaluation service
        service = EvaluationService(
            provider=args.provider,
            api_key=args.api_key,
            model_name=args.model,
            batch_size=args.batch_size,
            rate_limit_delay=args.rate_limit,
            preprocess_only=args.preprocess_only,
            full_pipeline=args.full_pipeline
        )
        
        # Load data
        logger.info(f"Loading data from {args.input}")
        data = service.load_data(args.input)
        
        if not data:
            logger.error("No data loaded. Exiting.")
            return
        
        # Process data
        logger.info("Starting processing...")
        processed_data = await service.process_data(data)
        
        # Save results
        service.save_results(processed_data, args.output)
        
        # Print summary
        service.print_summary()
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 