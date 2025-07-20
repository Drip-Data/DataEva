#!/usr/bin/env python3
"""
API Connectivity Test
=====================

This script tests the connectivity to different LLM providers.
It allows users to test their API keys and endpoints before running the full evaluation.

Usage:
    python src/test_api_connectivity.py
"""

import asyncio
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
from llm_api_clients import LLMClientFactory


async def test_provider_connectivity(provider: str, api_key: str, model_name: Optional[str] = None, **kwargs):
    """Test connectivity to a specific provider."""
    
    print(f"\nTesting {provider.upper()} connectivity...")
    print("-" * 40)
    
    try:
        # Create client
        client = LLMClientFactory.create_client(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            **kwargs
        )
        
        print(f"✓ Client created successfully")
        print(f"  Provider: {client.provider_name}")
        print(f"  Model: {client.model_name}")
        
        # Test a simple evaluation
        test_prompt = """
You are evaluating an AI agent's response. Please provide your evaluation in JSON format.

Task: Calculate 2 + 2
Agent Response: The answer is 4.

Please evaluate this response with the following JSON format:
{
    "scores": {
        "accuracy": 1.0,
        "clarity": 1.0
    },
    "summary": "Correct calculation",
    "reasoning": "The agent correctly calculated 2 + 2 = 4"
}
"""
        
        print("Testing API call...")
        response = await client.evaluate_clip(test_prompt, max_tokens=500)
        
        if response.success:
            print("✓ API call successful")
            print(f"  Response received: {len(response.raw_response or '')} characters")
            print(f"  Scores found: {len(response.scores)} metrics")
            print(f"  Summary: {response.summary[:50]}...")
            return True
        else:
            print("✗ API call failed")
            print(f"  Error: {response.error_message}")
            return False
            
    except Exception as e:
        print("✗ Connection failed")
        print(f"  Error: {e}")
        return False


def get_provider_config(provider: str):
    """Get configuration for a specific provider."""
    
    print(f"\nConfiguring {provider.upper()} connection...")
    
    # Get API key
    api_key = input(f"Enter API key for {provider}: ").strip()
    if not api_key:
        print("API key is required")
        return None
    
    config = {"api_key": api_key}
    
    # Get model name with default
    defaults = {
        'openai': 'gpt-4o-mini',  # Use mini for testing to save costs
        'google': 'gemini-1.5-flash',  # Use flash for testing
        'anthropic': 'claude-3-haiku-20240307',  # Use haiku for testing
        'deepseek': 'deepseek-chat',
        'kimi': 'moonshot-v1-8k',
        'vertex_ai': 'gemini-1.5-flash'
    }
    
    default_model = defaults.get(provider, '')
    model_name = input(f"Model name (default: {default_model}): ").strip()
    config["model_name"] = model_name or default_model
    
    # Additional configuration for specific providers
    if provider in ['deepseek', 'kimi']:
        endpoint = input("Endpoint URL (press Enter for default): ").strip()
        if endpoint:
            config["endpoint_url"] = endpoint
    
    if provider == 'vertex_ai':
        project_id = input("Google Cloud Project ID: ").strip()
        if not project_id:
            print("Project ID is required for Vertex AI")
            return None
        config["project_id"] = project_id
        
        location = input("Location (default: us-central1): ").strip()
        config["location"] = location or "us-central1"
    
    return config


async def main():
    """Main test function."""
    
    print("LLM API Connectivity Test")
    print("=" * 50)
    print("This tool helps you test API connectivity for all supported LLM providers.")
    print("Supported providers: openai, google, anthropic, deepseek, kimi, vertex_ai")
    print()
    
    results = {}
    
    while True:
        print("\nAvailable providers:")
        providers = ['openai', 'google', 'anthropic', 'deepseek', 'kimi', 'vertex_ai']
        for i, provider in enumerate(providers, 1):
            status = "✓" if results.get(provider) else "○"
            print(f"  {i}. {provider} {status}")
        
        print("  0. Exit")
        print()
        
        try:
            choice = input("Select provider to test (1-6) or 0 to exit: ").strip()
            
            if choice == '0':
                break
            
            provider_index = int(choice) - 1
            if 0 <= provider_index < len(providers):
                provider = providers[provider_index]
                
                # Get configuration
                config = get_provider_config(provider)
                if config:
                    # Test connectivity
                    success = await test_provider_connectivity(provider, **config)
                    results[provider] = success
                    
                    if success:
                        print(f"\n🎉 {provider.upper()} connection successful!")
                    else:
                        print(f"\n❌ {provider.upper()} connection failed.")
            else:
                print("Invalid selection. Please choose 1-6 or 0.")
                
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nTest interrupted by user.")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
    
    # Final summary
    if results:
        print("\n" + "=" * 50)
        print("CONNECTIVITY TEST SUMMARY")
        print("=" * 50)
        
        successful = [provider for provider, success in results.items() if success]
        failed = [provider for provider, success in results.items() if not success]
        
        if successful:
            print("✓ Successful connections:")
            for provider in successful:
                print(f"  - {provider}")
        
        if failed:
            print("✗ Failed connections:")
            for provider in failed:
                print(f"  - {provider}")
        
        print(f"\nTotal tested: {len(results)}")
        print(f"Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
        
        if successful:
            print(f"\n✓ You can use multi-model evaluation with {len(successful)} provider(s)!")
        else:
            print("\n❌ No successful connections. Please check your API keys and network.")
    
    print("\nTest completed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted.")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc() 