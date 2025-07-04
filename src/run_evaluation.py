#!/usr/bin/env python3
"""
AI Agent Step-Level Evaluation System - Main Entry Point
=======================================================

This script provides a command-line interface for the AI agent trajectory evaluation system.
It supports preprocessing, evaluation with multiple LLM providers, and various output formats.

Usage Examples:
    # Preprocessing only
    python run_evaluation.py --input data/demo01.jsonl --preprocess-only --output data/preprocessed.jsonl

    # LLM evaluation only (assumes input is already preprocessed)
    python run_evaluation.py --input data/preprocessed.jsonl --provider openai --api-key YOUR_KEY --output data/results.jsonl

    # Full pipeline (preprocess + evaluate)
    python run_evaluation.py --input data/demo01.jsonl --provider openai --api-key YOUR_KEY --full-pipeline --output data/results.jsonl

    # Batch evaluation with custom settings
    python run_evaluation.py --input data/ --provider google --model gemini-1.5-pro --batch-size 5 --rate-limit 2.0 --full-pipeline

    # Use environment variables for API keys
    export OPENAI_API_KEY=your_key
    python run_evaluation.py --input data/demo01.jsonl --provider openai --full-pipeline
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional

# Import our modules
from evaluation_service import EvaluationService
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

  # LLM evaluation only (assumes input is already preprocessed)
  python run_evaluation.py --input data/preprocessed.jsonl --provider openai --api-key YOUR_KEY

  # Full pipeline (preprocess + evaluate)
  python run_evaluation.py --input data/demo01.jsonl --provider openai --api-key YOUR_KEY --full-pipeline

  # Batch evaluation with custom settings
  python run_evaluation.py --input data/ --provider google --model gemini-1.5-pro --batch-size 5 --full-pipeline

  # Use environment variables for API keys
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
    
    # API configuration (required for evaluation modes)
    parser.add_argument('--provider', type=str, default='openai',
                       choices=['openai', 'google', 'anthropic'],
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


def validate_args(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    
    # Check if input file/directory exists
    if not Path(args.input).exists():
        logger.error(f"Input path does not exist: {args.input}")
        return False
    
    # Check API key requirements for evaluation modes
    if not args.preprocess_only:
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
            logger.error(f"  3. Or use --preprocess-only to skip evaluation")
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
    print("=" * 60)


def print_mode_info(args: argparse.Namespace):
    """Print information about the selected processing mode."""
    
    if args.preprocess_only:
        print("\n🔧 MODE: Preprocessing Only")
        print("  • Will clean and validate input data")
        print("  • Remove samples with excessive tool calls")
        print("  • Fix XML format issues")
        print("  • No LLM evaluation will be performed")
    elif args.full_pipeline:
        print(f"\n🚀 MODE: Full Pipeline")
        print("  • Will preprocess data first")
        print("  • Then evaluate with LLM")
        print(f"  • Provider: {args.provider}")
        print(f"  • Model: {args.model or 'default'}")
        print(f"  • Batch size: {args.batch_size}")
    else:
        print(f"\n⚡ MODE: Evaluation Only")
        print("  • Assumes input is already preprocessed")
        print("  • Will evaluate directly with LLM")
        print(f"  • Provider: {args.provider}")
        print(f"  • Model: {args.model or 'default'}")
        print(f"  • Batch size: {args.batch_size}")


def print_completion_summary(args: argparse.Namespace):
    """Print completion summary."""
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETED")
    print("=" * 60)
    
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
        print_mode_info(args)
    
    # Validate arguments
    if not validate_args(args):
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get API key for evaluation modes
        api_key = get_api_key(args.provider, args.api_key) if not args.preprocess_only else ""
        
        # Initialize evaluation service
        service = EvaluationService(
            provider=args.provider,
            api_key=api_key,
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
            sys.exit(1)
        
        # Process data based on mode
        logger.info("Starting processing...")
        processed_data = await service.process_data(data)
        
        # Save results
        service.save_results(processed_data, args.output)
        
        # Print summary
        if not args.quiet:
            service.print_summary()
            print_completion_summary(args)
            
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