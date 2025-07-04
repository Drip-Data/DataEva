import openai
import google.generativeai as genai
import anthropic
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
import os
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResponse:
    """Structure for evaluation response from LLM APIs."""
    scores: Dict[str, float]
    summary: str
    reasoning: str
    success: bool
    error_message: Optional[str] = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM API clients."""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.rate_limit_delay = 1.0  # Default 1 second between requests
    
    @abstractmethod
    async def evaluate_clip(self, prompt: str, max_tokens: int = 2000) -> EvaluationResponse:
        """Evaluate a clip using the LLM API."""
        pass
    
    def set_rate_limit_delay(self, delay: float):
        """Set the delay between API requests for rate limiting."""
        self.rate_limit_delay = delay


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT API client for evaluation."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        super().__init__(api_key, model_name)
        openai.api_key = api_key
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    async def evaluate_clip(self, prompt: str, max_tokens: int = 2000) -> EvaluationResponse:
        """Evaluate a clip using OpenAI GPT API."""
        try:
            # Add rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of AI agent trajectories. Provide detailed, objective evaluations with precise scores."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for consistent evaluation
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            
            # Parse the JSON response
            import json
            eval_data = json.loads(content)
            
            return EvaluationResponse(
                scores=eval_data.get("scores", {}),
                summary=eval_data.get("summary", ""),
                reasoning=eval_data.get("reasoning", ""),
                success=True
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return EvaluationResponse(
                scores={},
                summary="",
                reasoning="",
                success=False,
                error_message=str(e)
            )


class GoogleGeminiClient(BaseLLMClient):
    """Google Gemini API client for evaluation."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        super().__init__(api_key, model_name)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    async def evaluate_clip(self, prompt: str, max_tokens: int = 2000) -> EvaluationResponse:
        """Evaluate a clip using Google Gemini API."""
        try:
            # Add rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.1,
                response_mime_type="application/json"
            )
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=generation_config
            )
            
            content = response.text
            
            # Parse the JSON response
            import json
            eval_data = json.loads(content)
            
            return EvaluationResponse(
                scores=eval_data.get("scores", {}),
                summary=eval_data.get("summary", ""),
                reasoning=eval_data.get("reasoning", ""),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Google Gemini API error: {e}")
            return EvaluationResponse(
                scores={},
                summary="",
                reasoning="",
                success=False,
                error_message=str(e)
            )


class AnthropicClaudeClient(BaseLLMClient):
    """Anthropic Claude API client for evaluation."""
    
    def __init__(self, api_key: str, model_name: str = "claude-3-5-sonnet-20241022"):
        super().__init__(api_key, model_name)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    async def evaluate_clip(self, prompt: str, max_tokens: int = 2000) -> EvaluationResponse:
        """Evaluate a clip using Anthropic Claude API."""
        try:
            # Add rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=0.1,
                system="You are an expert evaluator of AI agent trajectories. Provide detailed, objective evaluations with precise scores. Return your response as valid JSON.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            
            # Parse the JSON response
            import json
            eval_data = json.loads(content)
            
            return EvaluationResponse(
                scores=eval_data.get("scores", {}),
                summary=eval_data.get("summary", ""),
                reasoning=eval_data.get("reasoning", ""),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Anthropic Claude API error: {e}")
            return EvaluationResponse(
                scores={},
                summary="",
                reasoning="",
                success=False,
                error_message=str(e)
            )


class LLMClientFactory:
    """Factory class to create LLM clients."""
    
    @staticmethod
    def create_client(provider: str, api_key: str, model_name: Optional[str] = None) -> BaseLLMClient:
        """Create an LLM client based on the provider."""
        
        if provider.lower() == "openai":
            model = model_name or "gpt-4o"
            return OpenAIClient(api_key, model)
        
        elif provider.lower() == "google" or provider.lower() == "gemini":
            model = model_name or "gemini-1.5-pro"
            return GoogleGeminiClient(api_key, model)
        
        elif provider.lower() == "anthropic" or provider.lower() == "claude":
            model = model_name or "claude-3-5-sonnet-20241022"
            return AnthropicClaudeClient(api_key, model)
        
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: openai, google, anthropic")


class EvaluationConfig:
    """Configuration class for evaluation settings."""
    
    def __init__(self, 
                 provider: str = "openai",
                 model_name: Optional[str] = None,
                 rate_limit_delay: float = 1.0,
                 max_tokens: int = 2000,
                 concurrent_requests: int = 5):
        self.provider = provider
        self.model_name = model_name
        self.rate_limit_delay = rate_limit_delay
        self.max_tokens = max_tokens
        self.concurrent_requests = concurrent_requests
    
    @classmethod
    def from_env(cls) -> 'EvaluationConfig':
        """Create configuration from environment variables."""
        return cls(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            model_name=os.getenv("LLM_MODEL_NAME"),
            rate_limit_delay=float(os.getenv("RATE_LIMIT_DELAY", "1.0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "2000")),
            concurrent_requests=int(os.getenv("CONCURRENT_REQUESTS", "5"))
        ) 