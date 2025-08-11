#!/usr/bin/env python3
"""
AI Agent Step-Level Evaluation System - Main Entry Point
=======================================================

This script provides a command-line interface for the AI agent trajectory evaluation system.
It supports preprocessing, evaluation with multiple LLM providers, and various output formats.

Usage Examples:
    # Preprocessing only
    python run_evaluation.py --input data/demo01.jsonl --preprocess-only --output data/preprocessed.jsonl

    # Single-model LLM evaluation (assumes input is already preprocessed)
    python run_evaluation.py --input data/preprocessed.jsonl --provider openai --api-key YOUR_KEY --output data/results.jsonl

    # Full pipeline with single model (preprocess + evaluate)
    python run_evaluation.py --input data/demo01.jsonl --provider openai --api-key YOUR_KEY --full-pipeline --output data/results.jsonl

    # Multi-model evaluation with interactive configuration (preprocess + evaluate)
    python run_evaluation.py --input data/demo01.jsonl --multi-model --full-pipeline --output data/results.jsonl

    # Multi-model evaluation only (assumes preprocessed data)
    python run_evaluation.py --input data/preprocessed.jsonl --multi-model --output data/results.jsonl

    # Test multi-model configuration before running evaluation
    python src/test_multi_model_interactive.py

    # Batch evaluation with custom settings
    python run_evaluation.py --input data/ --provider google --model gemini-1.5-pro --batch-size 5 --rate-limit 2.0 --full-pipeline

    # Evaluate all JSONL files in a folder
    python run_evaluation.py --input data/predemo02/ --provider openai --api-key YOUR_KEY --batch-folder

    # Use environment variables for API keys (single-model only)
    export OPENAI_API_KEY=your_key
    python run_evaluation.py --input data/demo01.jsonl --provider openai --full-pipeline
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional, List
import glob

# Import our modules
from evaluation_service import EvaluationService
from llm_api_clients import ModelConfig
from preprocess_agent_data import AgentDataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up and return the argument parser."""
    
    parser = argparse.ArgumentParser(
        description='AI Agent Step-Level Evaluation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocessing only
  python run_evaluation.py --input data/demo01.jsonl --preprocess-only

  # Single-model LLM evaluation (assumes input is already preprocessed)
  python run_evaluation.py --input data/preprocessed.jsonl --provider openai --api-key YOUR_KEY

  # Full pipeline with single model (preprocess + evaluate)
  python run_evaluation.py --input data/demo01.jsonl --provider openai --api-key YOUR_KEY --full-pipeline

  # Multi-model evaluation (interactive configuration)
  python run_evaluation.py --input data/demo01.jsonl --multi-model --full-pipeline

  # Multi-model evaluation only (assumes preprocessed data)
  python run_evaluation.py --input data/preprocessed.jsonl --multi-model

  # Test multi-model configuration before running evaluation
  python src/test_multi_model_interactive.py

  # Batch evaluation with custom settings
  python run_evaluation.py --input data/ --provider google --model gemini-1.5-pro --batch-size 5 --full-pipeline

  # Evaluate all JSONL files in a folder
  python run_evaluation.py --input data/predemo02/ --provider openai --api-key YOUR_KEY --batch-folder

  # Use environment variables for API keys (single-model only)
  export OPENAI_API_KEY=your_key
  python run_evaluation.py --input data/demo01.jsonl --provider openai --full-pipeline
        """
    )
    
    # Input/Output arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input JSONL file or directory path')
    parser.add_argument('--output', '-o', type=str, default='data/demo01_eva.jsonl',
                       help='Output file path for results (default: data/demo01_eva.jsonl)')
    
    # Processing mode
    parser.add_argument('--preprocess-only', action='store_true',
                       help='Only run preprocessing, skip evaluation')
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run full pipeline: preprocess + evaluate')
    parser.add_argument('--batch-folder', action='store_true',
                       help='Process all JSONL files in the input folder')
    parser.add_argument('--multi-model', action='store_true',
                       help='Use multi-model evaluation (interactive configuration)')
    
    # API configuration (required for evaluation modes)
    parser.add_argument('--provider', type=str, default='openai',
                       choices=['openai', 'google', 'anthropic', 'deepseek', 'kimi', 'vertex_ai',
                               'openai_compatible', 'anthropic_compatible', 'openrouter', 'together', 'groq', 'fireworks'],
                       help='LLM provider to use (default: openai)')
    parser.add_argument('--api-key', type=str, default='',
                       help='API key for the LLM provider (or set environment variable)')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model name to use (provider default if not specified)')
    
    # Processing configuration
    parser.add_argument('--batch-size', type=int, default=3,
                       help='Number of concurrent evaluations (default: 3)')
    parser.add_argument('--rate-limit', type=float, default=1.0,
                       help='Delay between API requests in seconds (default: 1.0)')
    parser.add_argument('--beta-threshold', type=int, default=15,
                       help='Maximum tool calls per sample in preprocessing (default: 15)')
    
    # Logging and output
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress output')
    
    return parser


def get_api_key(provider: str, provided_key: str) -> Optional[str]:
    """Get API key from command line argument or environment variable."""
    
    if provided_key:
        return provided_key
    
    # Try to get from environment
    env_key_map = {
        'openai': 'OPENAI_API_KEY',
        'google': 'GOOGLE_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY'
    }
    
    env_key = env_key_map.get(provider.lower())
    if env_key:
        return os.getenv(env_key)
    
    return None


def find_jsonl_files(input_path: str) -> List[str]:
    """Find all JSONL files in the input path."""
    
    input_path_obj = Path(input_path)
    
    if input_path_obj.is_file():
        # Single file
        if input_path_obj.suffix == '.jsonl':
            return [str(input_path_obj)]
        else:
            logger.error(f"Input file must have .jsonl extension: {input_path}")
            return []
    
    elif input_path_obj.is_dir():
        # Directory - find all JSONL files
        jsonl_files = list(input_path_obj.glob('*.jsonl'))
        if not jsonl_files:
            logger.error(f"No JSONL files found in directory: {input_path}")
            return []
        
        # Sort files for consistent processing order
        jsonl_files.sort()
        return [str(f) for f in jsonl_files]
    
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return []


def generate_output_path(input_file: str, base_output: str, file_index: int = None) -> str:
    """Generate output path for a given input file."""
    
    input_path = Path(input_file)
    base_output_path = Path(base_output)
    
    if file_index is not None:
        # Multiple files - create unique output names
        stem = input_path.stem
        suffix = base_output_path.suffix or '.jsonl'
        output_dir = base_output_path.parent
        output_name = f"{stem}_eva{suffix}"
        return str(output_dir / output_name)
    else:
        # Single file - use provided output path
        return base_output


def validate_args(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    
    # Check if input file/directory exists
    if not Path(args.input).exists():
        logger.error(f"Input path does not exist: {args.input}")
        return False
    
    # Check API key requirements for evaluation modes (skip for multi-model mode)
    if not args.preprocess_only and not args.multi_model:
        api_key = get_api_key(args.provider, args.api_key)
        if not api_key:
            env_key_map = {
                'openai': 'OPENAI_API_KEY',
                'google': 'GOOGLE_API_KEY',
                'anthropic': 'ANTHROPIC_API_KEY'
            }
            env_key = env_key_map.get(args.provider)
            logger.error(f"API key required for {args.provider} evaluation. Please:")
            logger.error(f"  1. Use --api-key YOUR_KEY")
            logger.error(f"  2. Set environment variable: export {env_key}=your_key")
            logger.error(f"  3. Use --multi-model for interactive model configuration")
            logger.error(f"  4. Or use --preprocess-only to skip evaluation")
            return False
    
    # Validate batch size
    if args.batch_size < 1:
        logger.error("Batch size must be at least 1")
        return False
    
    # Validate rate limit
    if args.rate_limit < 0:
        logger.error("Rate limit must be non-negative")
        return False
    
    # Check for conflicting modes
    if args.preprocess_only and args.full_pipeline:
        logger.error("Cannot use --preprocess-only and --full-pipeline together")
        return False
    
    # Check batch-folder mode compatibility
    if args.batch_folder and not Path(args.input).is_dir():
        logger.error("--batch-folder requires input to be a directory")
        return False
    
    return True


def print_banner():
    """Print system banner."""
    print("=" * 60)
    print("AI Agent Step-Level Evaluation System")
    print("=" * 60)
    print("Features:")
    print("  • Step-level evaluation with tool-specific metrics")
    print("  • Support for 4 MCP servers: microsandbox, deepsearch, browser_use, search_tool")
    print("  • Multiple LLM providers: OpenAI, Google, Anthropic")
    print("  • Comprehensive preprocessing and validation")
    print("  • Separate preprocessing and evaluation modes")
    print("  • Batch processing of multiple JSONL files")
    print("=" * 60)


def print_mode_info(args: argparse.Namespace, jsonl_files: List[str]):
    """Print information about the selected processing mode."""
    
    if args.batch_folder:
        print(f"\n MODE: Batch Folder Processing")
        print(f"  • Processing {len(jsonl_files)} JSONL files")
        print(f"  • Files: {[Path(f).name for f in jsonl_files]}")
    
    if args.preprocess_only:
        print("\n MODE: Preprocessing Only")
        print("  • Will clean and validate input data")
        print("  • Remove samples with excessive tool calls")
        print("  • Fix XML format issues")
        print("  • No LLM evaluation will be performed")
    elif args.full_pipeline:
        print(f"\n MODE: Full Pipeline")
        print("  • Will preprocess data first")
        print("  • Then evaluate with LLM")
        if args.multi_model:
            print("  • Evaluation: Multi-model (interactive configuration)")
        else:
            print(f"  • Provider: {args.provider}")
            print(f"  • Model: {args.model or 'default'}")
        print(f"  • Batch size: {args.batch_size}")
    else:
        print(f"\n MODE: Evaluation Only")
        print("  • Assumes input is already preprocessed")
        print("  • Will evaluate directly with LLM")
        if args.multi_model:
            print("  • Evaluation: Multi-model (interactive configuration)")
        else:
            print(f"  • Provider: {args.provider}")
            print(f"  • Model: {args.model or 'default'}")
        print(f"  • Batch size: {args.batch_size}")


def print_completion_summary(args: argparse.Namespace, processed_files: List[str]):
    """Print completion summary."""
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETED")
    print("=" * 60)
    
    if args.batch_folder:
        print(f"✓ Batch processing completed successfully")
        print(f"  Files processed: {len(processed_files)}")
        for file_path in processed_files:
            print(f"    • {file_path}")
    else:
        if args.preprocess_only:
            print("✓ Preprocessing completed successfully")
            print(f"  Output: {args.output}")
            print("  Data is now ready for LLM evaluation")
        elif args.full_pipeline:
            print("✓ Full pipeline completed successfully")
            print(f"  Results file: {args.output}")
            print("  File contains:")
            print("    • Original trajectory data")
            print("    • Full response with embedded evaluations")
            print("    • Core evaluation metadata")
        else:
            print("✓ LLM evaluation completed successfully")
            print(f"  Results file: {args.output}")
            print("  File contains:")
            print("    • Original trajectory data")
            print("    • Full response with embedded evaluations")
            print("    • Core evaluation metadata")
    
    print("=" * 60)


def configure_multi_model() -> List[ModelConfig]:
    """Interactive configuration for multi-model evaluation."""
    print("\n" + "="*60)
    print("MULTI-MODEL EVALUATION CONFIGURATION")
    print("="*60)
    print("Configure multiple LLM models for evaluation.")
    print()
    print("Available providers:")
    print("  • Native: openai, google, anthropic, deepseek, kimi, vertex_ai")
    print("  • Third-party: openai_compatible, anthropic_compatible, openrouter, together, groq, fireworks")
    print()
    print("Instructions:")
    print("  • You will be prompted to enter information for each model")
    print("  • Type 'enough' when you're done adding models")
    print("  • At least one model is required for evaluation")
    print("="*60)
    
    model_configs = []
    model_number = 1
    
    while True:
        print(f"\n🔧 CONFIGURING MODEL #{model_number}")
        print("-" * 40)
        
        # Get provider
        while True:
            print("\nStep 1: Choose Provider")
            print("Native providers: openai, google, anthropic, deepseek, kimi, vertex_ai")
            print("Third-party providers: openai_compatible, anthropic_compatible, openrouter, together, groq, fireworks")
            provider = input(f"Provider for Model #{model_number} (or 'enough' to finish): ").strip().lower()
            
            if provider == 'enough':
                if len(model_configs) == 0:
                    print("You need to configure at least one model. Please continue.")
                    continue
                else:
                    print(f"Configuration complete with {len(model_configs)} models.")
                    break
            
            supported_providers = [
                'openai', 'google', 'anthropic', 'deepseek', 'kimi', 'vertex_ai',
                'openai_compatible', 'anthropic_compatible', 'openrouter', 'together', 'groq', 'fireworks'
            ]
            
            if provider in supported_providers:
                break
            else:
                print(f"Invalid provider '{provider}'. Please choose from: {', '.join(supported_providers)}")
        
        # Break if user typed 'enough'
        if provider == 'enough':
            break
        
        # Get API key
        while True:
            print(f"\nStep 2: API Key")
            api_key = input(f"API key for {provider}: ").strip()
            if api_key:
                break
            else:
                print("API key is required. Please enter a valid API key.")
        
        # Get model name (with defaults)
        print(f"\nStep 3: Model Name")
        defaults = {
            'openai': 'gpt-4o',
            'google': 'gemini-1.5-pro',
            'anthropic': 'claude-3-5-sonnet-20241022',
            'deepseek': 'deepseek-chat',
            'kimi': 'moonshot-v1-8k',
            'vertex_ai': 'gemini-1.5-pro',
            'openai_compatible': 'gpt-4',
            'anthropic_compatible': 'claude-3-sonnet',
            'openrouter': 'gpt-4',
            'together': 'meta-llama/Llama-2-70b-chat-hf',
            'groq': 'llama2-70b-4096',
            'fireworks': 'accounts/fireworks/models/llama-v2-70b-chat'
        }
        
        default_model = defaults.get(provider, 'default-model')
        model_name = input(f"Model name (default: {default_model}): ").strip() or default_model
        
        # Additional configuration for specific providers
        endpoint_url = None
        base_url = None
        project_id = None
        protocol = None
        custom_headers = None
        
        # Third-party providers need base_url
        if provider in ['openai_compatible', 'anthropic_compatible', 'openrouter', 'together', 'groq', 'fireworks']:
            print(f"\nStep 4: Base URL (Required for {provider})")
            while True:
                base_url = input(f"Base URL for {provider}: ").strip()
                if base_url:
                    break
                else:
                    print("Base URL is required for third-party providers.")
            
            # Ask for custom headers (optional)
            print(f"\nStep 5: Custom Headers (Optional)")
            print("Some third-party services require custom headers (e.g., x-foo: true)")
            headers_input = input("Custom headers in format 'key1:value1,key2:value2' (press Enter to skip): ").strip()
            if headers_input:
                try:
                    custom_headers = {}
                    for header_pair in headers_input.split(','):
                        key, value = header_pair.split(':')
                        custom_headers[key.strip()] = value.strip()
                    print(f"Custom headers set: {custom_headers}")
                except Exception as e:
                    print(f" Invalid header format, skipping: {e}")
            
            # Set protocol based on provider
            if provider in ['openai_compatible', 'openrouter', 'together', 'groq', 'fireworks']:
                protocol = 'openai'
            elif provider == 'anthropic_compatible':
                protocol = 'anthropic'
        
        # Native providers that might need endpoint URL
        elif provider in ['deepseek', 'kimi']:
            print(f"\nStep 4: Endpoint URL (Optional)")
            endpoint_url = input(f"Endpoint URL for {provider} (press Enter for default): ").strip() or None
        
        # Vertex AI specific
        elif provider == 'vertex_ai':
            print(f"\nStep 4: Google Cloud Configuration")
            while True:
                project_id = input("Google Cloud Project ID: ").strip()
                if project_id:
                    break
                else:
                    print("Project ID is required for Vertex AI.")
        
        # Create model config
        try:
            config = ModelConfig(
                provider=provider,
                model_name=model_name,
                api_key=api_key,
                endpoint_url=endpoint_url,
                project_id=project_id,
                base_url=base_url,
                protocol=protocol,
                custom_headers=custom_headers
            )
            
            model_configs.append(config)
            print(f"\nSuccessfully added Model #{model_number}:")
            print(f"   Provider: {provider}")
            print(f"   Model: {model_name}")
            if base_url:
                print(f"   Base URL: {base_url}")
            if project_id:
                print(f"   Project ID: {project_id}")
            
            model_number += 1
            
            # Limit to prevent too many models
            if len(model_configs) >= 5:
                print(f"\n Maximum of 5 models reached. Configuration complete.")
                break
                
        except Exception as e:
            print(f"Error creating model configuration: {e}")
            print("Please try again with different settings.")
            continue
    
    # Summary
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    if model_configs:
        print(f"Successfully configured {len(model_configs)} models:")
        for i, config in enumerate(model_configs, 1):
            display_name = f"{config.provider}: {config.model_name}"
            if config.base_url:
                display_name += f" (URL: {config.base_url})"
            print(f"   {i}. {display_name}")
        print(f"\nThese models will be used for step-level evaluation.")
    else:
        print("No models configured.")
    
    print("="*60)
    return model_configs


async def process_single_file(input_file: str, output_file: str, args: argparse.Namespace) -> bool:
    """Process a single JSONL file."""
    
    try:
        # Multi-model configuration
        model_configs = []
        if args.multi_model and not args.preprocess_only:
            try:
                print("\n Starting multi-model configuration...")
                model_configs = configure_multi_model()
                if not model_configs:
                    logger.error(" No models configured for multi-model evaluation")
                    return False
                else:
                    print(f" Multi-model configuration completed with {len(model_configs)} models")
            except KeyboardInterrupt:
                print("\n  Configuration interrupted by user")
                return False
            except Exception as e:
                logger.error(f" Error during multi-model configuration: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                return False
        
        # Get API key for evaluation modes (single model)
        api_key = get_api_key(args.provider, args.api_key) if not args.preprocess_only and not args.multi_model else ""
        
        # Initialize evaluation service
        try:
            if args.multi_model and model_configs:
                print(f"🔧 Initializing multi-model evaluation service with {len(model_configs)} models...")
            elif not args.preprocess_only:
                print(f"🔧 Initializing single-model evaluation service ({args.provider})...")
            else:
                print("🔧 Initializing preprocessing-only service...")
                
            service = EvaluationService(
                provider=args.provider,
                api_key=api_key,
                model_name=args.model,
                batch_size=args.batch_size,
                rate_limit_delay=args.rate_limit,
                preprocess_only=args.preprocess_only,
                full_pipeline=args.full_pipeline,
                beta_threshold=args.beta_threshold,
                multi_model=args.multi_model,
                model_configs=model_configs
            )
            print(" Evaluation service initialized successfully")
        except Exception as e:
            logger.error(f" Failed to initialize evaluation service: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return False
        
        # Load data
        logger.info(f"Processing file: {input_file}")
        data = service.load_data(input_file)
        
        if not data:
            logger.error(f"No data loaded from {input_file}")
            return False
        
        # Process data based on mode
        processed_data = await service.process_data(data)
        
        # Save results
        service.save_results(processed_data, output_file)
        
        if not args.quiet:
            service.print_summary()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {input_file}: {e}")
        return False


async def main():
    """Main function."""
    
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Print banner
    if not args.quiet:
        print_banner()
    
    # Validate arguments
    if not validate_args(args):
        sys.exit(1)
    
    # Find JSONL files to process
    jsonl_files = find_jsonl_files(args.input)
    if not jsonl_files:
        sys.exit(1)
    
    # Print mode information
    if not args.quiet:
        print_mode_info(args, jsonl_files)
    
    # Create output directory if needed
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        processed_files = []
        
        if args.batch_folder and len(jsonl_files) > 1:
            # Process multiple files
            logger.info(f"Processing {len(jsonl_files)} JSONL files in batch mode")
            
            for i, input_file in enumerate(jsonl_files):
                output_file = generate_output_path(input_file, args.output, i)
                
                logger.info(f"Processing file {i+1}/{len(jsonl_files)}: {Path(input_file).name}")
                success = await process_single_file(input_file, output_file, args)
                
                if success:
                    processed_files.append(output_file)
                    logger.info(f"✓ Completed: {Path(input_file).name} -> {Path(output_file).name}")
                else:
                    logger.error(f"✗ Failed: {Path(input_file).name}")
        
        else:
            # Process single file
            input_file = jsonl_files[0]
            output_file = generate_output_path(input_file, args.output)
            
            success = await process_single_file(input_file, output_file, args)
            if success:
                processed_files.append(output_file)
        
        # Print completion summary
        if not args.quiet:
            print_completion_summary(args, processed_files)
        
        if not processed_files:
            logger.error("No files were processed successfully")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 