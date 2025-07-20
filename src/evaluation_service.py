#!/usr/bin/env python3
"""
Evaluation Service
==================

This script provides a complete evaluation service for AI agent trajectory data.
It integrates preprocessing and step-level evaluation with LLM APIs.

Features:
- Separate preprocessing and evaluation modes
- Support for multiple LLM providers (OpenAI, Google, Anthropic, DeepSeek, Kimi, Vertex AI)
- Multi-model evaluation with score averaging
- Batch processing with rate limiting
- Comprehensive reporting and statistics

Usage:
    # Preprocessing only
    python src/evaluation_service.py --input data/demo01.jsonl --preprocess-only --output data/preprocessed.jsonl
    
    # Single-model LLM evaluation
    python src/evaluation_service.py --input data/preprocessed.jsonl --provider openai --api-key YOUR_KEY --output data/results.jsonl
    
    # Multi-model evaluation
    python src/evaluation_service.py --input data/preprocessed.jsonl --multi-model --output data/results.jsonl
    
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
from step_level_evaluator import StepLevelEvaluator, MultiModelStepLevelEvaluator, create_evaluator, create_multi_model_evaluator
from llm_api_clients import ModelConfig, MultiModelEvaluationConfig

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
                 full_pipeline: bool = False,
                 beta_threshold: int = 15,
                 split_size: int = 0,
                 multi_model: bool = False,
                 model_configs: Optional[List[ModelConfig]] = None):
        
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        self.preprocess_only = preprocess_only
        self.full_pipeline = full_pipeline
        self.beta_threshold = beta_threshold
        self.split_size = split_size
        self.multi_model = multi_model
        self.model_configs = model_configs or []
        
        # Initialize preprocessor
        self.preprocessor = AgentDataPreprocessor(beta_threshold=beta_threshold, split_size=split_size)
        
        # Initialize evaluator only if not preprocessing-only mode
        if not preprocess_only:
            if multi_model and model_configs:
                # Multi-model evaluation
                try:
                    self.evaluator = create_multi_model_evaluator(model_configs, rate_limit_delay)
                    logger.info(f"Using multi-model evaluation with {len(model_configs)} models")
                except Exception as e:
                    logger.error(f"Failed to initialize multi-model evaluator: {e}")
                    raise
            else:
                # Single-model evaluation
                if not api_key:
                    raise ValueError(f"API key is required for {provider} evaluation. Please provide --api-key or set environment variable.")
                
                try:
                    self.evaluator = create_evaluator(provider, api_key, model_name)
                    self.evaluator.multi_evaluator.llm_clients[0].set_rate_limit_delay(rate_limit_delay)
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
        """Load data from JSONL file or directory."""
        try:
            path = Path(input_path)
            
            if path.is_file():
                # Single file
                data = self.preprocessor.load_jsonl(str(path))
                logger.info(f"Loaded {len(data)} samples from {input_path}")
                return data
            
            elif path.is_dir():
                # Directory with multiple files
                all_data = []
                jsonl_files = list(path.glob('*.jsonl'))
                
                if not jsonl_files:
                    logger.error(f"No JSONL files found in directory: {input_path}")
                    return []
                
                for file_path in sorted(jsonl_files):
                    file_data = self.preprocessor.load_jsonl(str(file_path))
                    all_data.extend(file_data)
                    logger.info(f"Loaded {len(file_data)} samples from {file_path.name}")
                
                logger.info(f"Total loaded: {len(all_data)} samples from {len(jsonl_files)} files")
                return all_data
            
            else:
                logger.error(f"Input path does not exist: {input_path}")
                return []
                
        except Exception as e:
            logger.error(f"Error loading data from {input_path}: {e}")
            return []
    
    async def process_data(self, data: List[Dict]) -> List[Dict]:
        """Process data based on configured mode."""
        self.stats['total_samples'] = len(data)
        self.stats['start_time'] = time.time()
        
        if self.preprocess_only:
            # Preprocessing only
            logger.info("Running preprocessing only...")
            processed_data = self.preprocessor.preprocess_data(data)
            self.stats['preprocessed_samples'] = len(processed_data)
            
        elif self.full_pipeline:
            # Full pipeline: preprocess + evaluate
            logger.info("Running full pipeline: preprocessing + evaluation...")
            
            # Preprocess first
            preprocessed_data = self.preprocessor.preprocess_data(data)
            self.stats['preprocessed_samples'] = len(preprocessed_data)
            
            if not preprocessed_data:
                logger.warning("No data remaining after preprocessing")
                processed_data = []
            else:
                # Then evaluate
                processed_data = await self._evaluate_data(preprocessed_data)
        
        else:
            # Evaluation only (assumes data is already preprocessed)
            logger.info("Running evaluation only...")
            processed_data = await self._evaluate_data(data)
        
        self.stats['end_time'] = time.time()
        return processed_data
    
    async def _evaluate_data(self, data: List[Dict]) -> List[Dict]:
        """Evaluate preprocessed data using configured evaluator."""
        if not self.evaluator:
            logger.error("No evaluator configured")
            return data
        
        logger.info(f"Starting evaluation of {len(data)} samples...")
        
        # Create batches for parallel processing
        batches = [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]
        
        evaluated_data = []
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} samples)")
            
            # Process batch in parallel
            tasks = []
            for sample in batch:
                task = asyncio.create_task(self._evaluate_single_sample(sample))
                tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Evaluation failed for sample {i * self.batch_size + j}: {result}")
                    self.stats['failed_evaluations'] += 1
                    # Add original sample with error info
                    error_sample = batch[j].copy()
                    error_sample['evaluation_error'] = str(result)
                    evaluated_data.append(error_sample)
                else:
                    self.stats['evaluated_samples'] += 1
                    evaluated_data.append(result)
            
            # Add delay between batches to respect rate limits
            if i < len(batches) - 1:  # Don't delay after last batch
                await asyncio.sleep(self.rate_limit_delay)
        
        logger.info(f"Evaluation completed: {self.stats['evaluated_samples']}/{len(data)} successful")
        return evaluated_data
    
    async def _evaluate_single_sample(self, sample: Dict) -> Dict:
        """Evaluate a single sample using the configured evaluator."""
        try:
            # Use the new output directory structure for multi-model files
            output_dir = "data/model_evaluations"
            
            if self.multi_model:
                # Multi-model evaluation
                result = await self.evaluator.evaluate_trajectory_with_full_response(sample, output_dir)
            else:
                # Single-model evaluation  
                result = await self.evaluator.evaluate_trajectory_with_full_response(sample)
            
            return result
            
        except Exception as e:
            logger.error(f"Sample evaluation failed for {sample.get('task_id', 'unknown')}: {e}")
            raise
    
    def save_results(self, processed_data: List[Dict], output_path: str):
        """Save processed results to file."""
        try:
            # Ensure output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save main results file
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in processed_data:
                    # Create clean output with only the required fields
                    if not self.preprocess_only:
                        # For evaluation modes, only keep essential fields
                        clean_sample = {
                            'timestamp': sample.get('timestamp'),
                            'task_id': sample.get('task_id'),
                            'task_description': sample.get('task_description'),
                            'duration': sample.get('duration'),
                            'success': sample.get('success'),
                            'final_result': sample.get('final_result'),
                            'full_response_with_evaluation': sample.get('full_response_with_evaluation'),
                            'evaluation_metadata': sample.get('evaluation_metadata')
                        }
                        
                        # Remove None values
                        clean_sample = {k: v for k, v in clean_sample.items() if v is not None}
                    else:
                        # For preprocessing mode, keep all fields
                        clean_sample = sample
                    
                    f.write(json.dumps(clean_sample, ensure_ascii=False) + '\n')
            
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results to {output_path}: {e}")
            raise
    
    def print_summary(self):
        """Print processing summary."""
        print("\n" + "="*60)
        print("EVALUATION SERVICE SUMMARY")
        print("="*60)
        
        # Processing mode
        if self.preprocess_only:
            mode = "Preprocessing Only"
        elif self.full_pipeline:
            mode = "Full Pipeline (Preprocess + Evaluate)"
        else:
            mode = "Evaluation Only"
        print(f"Mode: {mode}")
        
        # Multi-model info
        if not self.preprocess_only:
            if self.multi_model:
                print(f"Evaluation: Multi-model ({len(self.model_configs)} models)")
                for config in self.model_configs:
                    print(f"  • {config.provider}: {config.model_name}")
            else:
                print(f"Evaluation: Single model ({self.provider}: {self.model_name or 'default'})")
        
        # Statistics
        print(f"\nSample Statistics:")
        print(f"  Total samples: {self.stats['total_samples']}")
        
        if self.full_pipeline or self.preprocess_only:
            print(f"  Preprocessed: {self.stats['preprocessed_samples']}")
            
        if not self.preprocess_only:
            print(f"  Evaluated: {self.stats['evaluated_samples']}")
            print(f"  Failed: {self.stats['failed_evaluations']}")
            
            if self.stats['total_samples'] > 0:
                success_rate = (self.stats['evaluated_samples'] / self.stats['total_samples']) * 100
                print(f"  Success rate: {success_rate:.1f}%")
        
        # Timing
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            print(f"\nProcessing time: {duration:.2f} seconds")
            
            if self.stats['evaluated_samples'] > 0:
                avg_time = duration / self.stats['evaluated_samples']
                print(f"Average time per sample: {avg_time:.2f} seconds")
        
        print("="*60)


async def main():
    """Main function for running the evaluation service."""
    parser = argparse.ArgumentParser(
        description='AI Agent Trajectory Evaluation Service',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input JSONL file or directory path')
    parser.add_argument('--output', '-o', type=str, default='data/demo01_eva.jsonl',
                       help='Output file path for results')
    
    # Processing mode
    parser.add_argument('--preprocess-only', action='store_true',
                       help='Only run preprocessing, skip evaluation')
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run full pipeline: preprocess + evaluate')
    parser.add_argument('--multi-model', action='store_true',
                       help='Use multi-model evaluation (interactive configuration)')
    
    # Single model API configuration
    parser.add_argument('--provider', type=str, default='openai',
                       choices=['openai', 'google', 'anthropic', 'deepseek', 'kimi', 'vertex_ai'],
                       help='LLM provider for single-model evaluation')
    parser.add_argument('--api-key', type=str, default='',
                       help='API key for single-model evaluation')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name for single-model evaluation')
    
    # Processing configuration
    parser.add_argument('--batch-size', type=int, default=3,
                       help='Number of concurrent evaluations')
    parser.add_argument('--rate-limit', type=float, default=1.0,
                       help='Delay between API requests in seconds')
    parser.add_argument('--beta-threshold', type=int, default=15,
                       help='Maximum tool calls per sample in preprocessing')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Multi-model configuration
    model_configs = []
    if args.multi_model and not args.preprocess_only:
        model_configs = configure_multi_model()
        if not model_configs:
            logger.error("No models configured for multi-model evaluation")
            return
    
    try:
        # Initialize service
        service = EvaluationService(
            provider=args.provider,
            api_key=args.api_key,
            model_name=args.model,
            batch_size=args.batch_size,
            rate_limit_delay=args.rate_limit,
            preprocess_only=args.preprocess_only,
            full_pipeline=args.full_pipeline,
            beta_threshold=args.beta_threshold,
            multi_model=args.multi_model,
            model_configs=model_configs
        )
        
        # Load and process data
        data = service.load_data(args.input)
        if not data:
            logger.error("No data loaded")
            return
        
        processed_data = await service.process_data(data)
        
        # Save results
        service.save_results(processed_data, args.output)
        
        # Print summary
        if not args.quiet:
            service.print_summary()
            
    except Exception as e:
        logger.error(f"Service failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


def configure_multi_model() -> List[ModelConfig]:
    """Interactive configuration for multi-model evaluation."""
    print("\n" + "="*60)
    print("MULTI-MODEL EVALUATION CONFIGURATION")
    print("="*60)
    print("Configure multiple LLM models for evaluation.")
    print("Supported providers: openai, google, anthropic, deepseek, kimi, vertex_ai")
    print("Press Enter with empty input to finish configuration.")
    print()
    
    model_configs = []
    
    while True:
        print(f"\nConfiguring Model #{len(model_configs) + 1}")
        print("-" * 30)
        
        # Get provider
        provider = input("Provider (openai/google/anthropic/deepseek/kimi/vertex_ai): ").strip().lower()
        if not provider:
            break
        
        if provider not in ['openai', 'google', 'anthropic', 'deepseek', 'kimi', 'vertex_ai']:
            print("Invalid provider. Please choose from: openai, google, anthropic, deepseek, kimi, vertex_ai")
            continue
        
        # Get API key
        api_key = input(f"API key for {provider}: ").strip()
        if not api_key:
            print("API key is required")
            continue
        
        # Get model name (with defaults)
        defaults = {
            'openai': 'gpt-4o',
            'google': 'gemini-1.5-pro',
            'anthropic': 'claude-3-5-sonnet-20241022',
            'deepseek': 'deepseek-chat',
            'kimi': 'moonshot-v1-8k',
            'vertex_ai': 'gemini-1.5-pro'
        }
        
        default_model = defaults.get(provider, '')
        model_name = input(f"Model name (default: {default_model}): ").strip() or default_model
        
        # Additional configuration for specific providers
        endpoint_url = None
        project_id = None
        
        if provider in ['deepseek', 'kimi']:
            endpoint_url = input(f"Endpoint URL (press Enter for default): ").strip() or None
        
        if provider == 'vertex_ai':
            project_id = input("Google Cloud Project ID: ").strip()
            if not project_id:
                print("Project ID is required for Vertex AI")
                continue
        
        # Create model config
        config = ModelConfig(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            endpoint_url=endpoint_url,
            project_id=project_id
        )
        
        model_configs.append(config)
        print(f"✓ Added {provider}:{model_name}")
        
        # Ask if user wants to add more models
        if len(model_configs) >= 5:  # Limit to prevent too many models
            print("\nMaximum of 5 models reached.")
            break
        
        add_more = input("\nAdd another model? (y/N): ").strip().lower()
        if add_more not in ['y', 'yes']:
            break
    
    if model_configs:
        print(f"\n✓ Multi-model configuration completed with {len(model_configs)} models:")
        for i, config in enumerate(model_configs, 1):
            print(f"  {i}. {config.provider}: {config.model_name}")
    else:
        print("No models configured.")
    
    print("="*60)
    return model_configs


if __name__ == "__main__":
    asyncio.run(main()) 