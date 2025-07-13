#!/usr/bin/env python3
"""
LLM API Connection Test Tool

This tool allows you to test connections to different LLM providers (OpenAI, Google Gemini, Anthropic Claude)
by sending a simple prompt and verifying the response.
"""

import asyncio
import sys
import os
import json
from typing import Optional

# Import the required libraries directly
import openai
import google.generativeai as genai
import anthropic

def print_header():
    """Print the header information."""
    print("="*60)
    print("           LLM API Connection Test Tool")
    print("="*60)
    print()
    print("This tool helps you test connections to different LLM providers.")
    print("Supported providers: OpenAI, Google Gemini, Anthropic Claude")
    print()

def get_user_choice() -> str:
    """Get the user's choice of LLM provider."""
    print("Available LLM providers:")
    print("1. OpenAI (ChatGPT)")
    print("2. Google Gemini")
    print("3. Anthropic Claude")
    print()
    
    while True:
        choice = input("Select LLM provider (1-3): ").strip()
        if choice == "1":
            return "openai"
        elif choice == "2":
            return "google"
        elif choice == "3":
            return "anthropic"
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def get_api_key(provider: str) -> str:
    """Get API key from user."""
    provider_names = {
        "openai": "OpenAI",
        "google": "Google Gemini",
        "anthropic": "Anthropic Claude"
    }
    
    print(f"\nEnter your {provider_names[provider]} API key:")
    api_key = input("API Key: ").strip()
    
    if not api_key:
        print("ERROR: API key cannot be empty.")
        sys.exit(1)
    
    return api_key

def get_model_name(provider: str) -> str:
    """Get model name from user."""
    default_models = {
        "openai": "gpt-4o",
        "google": "gemini-1.5-pro",
        "anthropic": "claude-3-5-sonnet-20241022"
    }
    
    print(f"\nModel name (press Enter for default: {default_models[provider]}):")
    model_name = input("Model: ").strip()
    
    return model_name if model_name else default_models[provider]

def get_custom_prompt() -> str:
    """Get custom prompt from user."""
    print("\nCustom prompt (press Enter for default: 'Hello world, how are you'):")
    custom_prompt = input("Prompt: ").strip()
    
    return custom_prompt if custom_prompt else "Hello world, how are you"

async def test_openai_connection(api_key: str, model_name: str, prompt: str):
    """Test OpenAI API connection."""
    try:
        client = openai.AsyncOpenAI(api_key=api_key)
        
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return True, response.choices[0].message.content
        
    except Exception as e:
        return False, str(e)

async def test_google_connection(api_key: str, model_name: str, prompt: str):
    """Test Google Gemini API connection."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=500,
            temperature=0.7
        )
        
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=generation_config
        )
        
        return True, response.text
        
    except Exception as e:
        return False, str(e)

async def test_anthropic_connection(api_key: str, model_name: str, prompt: str):
    """Test Anthropic Claude API connection."""
    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        
        response = await client.messages.create(
            model=model_name,
            max_tokens=500,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return True, response.content[0].text
        
    except Exception as e:
        return False, str(e)

async def test_llm_connection(provider: str, api_key: str, model_name: str, prompt: str):
    """Test the LLM connection."""
    print("\n" + "="*60)
    print("           CONNECTION TEST RESULTS")
    print("="*60)
    
    print(f"[1/4] Testing {provider.upper()} API connection...")
    print(f"    Provider: {provider}")
    print(f"    Model: {model_name}")
    print(f"    Prompt: '{prompt}'")
    
    print(f"\n[2/4] Sending request to {provider.upper()} API...")
    
    try:
        if provider == "openai":
            success, response = await test_openai_connection(api_key, model_name, prompt)
        elif provider == "google":
            success, response = await test_google_connection(api_key, model_name, prompt)
        elif provider == "anthropic":
            success, response = await test_anthropic_connection(api_key, model_name, prompt)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        print(f"\n[3/4] Processing response...")
        
        if success:
            print(f"✓ API connection successful!")
            print(f"    Response received: {len(response)} characters")
        else:
            print(f"✗ API connection failed!")
            print(f"    Error: {response}")
            return False
        
        # Display response
        print(f"\n[4/4] LLM Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        print("✓ Test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Connection test failed!")
        print(f"    Error: {str(e)}")
        print(f"    Error type: {type(e).__name__}")
        return False

def main():
    """Main function to run the test."""
    print_header()
    
    try:
        # Get user inputs
        provider = get_user_choice()
        api_key = get_api_key(provider)
        model_name = get_model_name(provider)
        prompt = get_custom_prompt()
        
        # Run the test
        print(f"\nStarting connection test...")
        success = asyncio.run(test_llm_connection(provider, api_key, model_name, prompt))
        
        if success:
            print(f"\n🎉 SUCCESS: {provider.upper()} API is working correctly!")
        else:
            print(f"\n❌ FAILURE: {provider.upper()} API connection failed!")
            print("\nTroubleshooting tips:")
            print("1. Check your API key is correct")
            print("2. Verify your account has sufficient credits/quota")
            print("3. Check your internet connection")
            print("4. Try a different model if available")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 