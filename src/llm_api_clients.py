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
import json

# Optional imports for additional providers
try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform import gapic
    HAS_VERTEX_AI = True
except ImportError:
    HAS_VERTEX_AI = False
    logger = logging.getLogger(__name__)
    logger.warning("Vertex AI not available. Install google-cloud-aiplatform to use Vertex AI.")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger = logging.getLogger(__name__)
    logger.warning("Requests not available. Some providers may not work.")

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
    model_name: str = ""
    provider: str = ""
    raw_response: Optional[str] = None  # For debugging purposes


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    provider: str
    model_name: str
    api_key: str
    endpoint_url: Optional[str] = None  # For custom endpoints
    project_id: Optional[str] = None    # For Vertex AI
    protocol: Optional[str] = None      # Protocol type: 'openai', 'anthropic', or None for native
    base_url: Optional[str] = None      # Base URL for OpenAI/Anthropic compatible APIs
    custom_headers: Optional[Dict[str, str]] = None  # Custom headers for third-party services
    
    def __post_init__(self):
        """Validate configuration and set defaults."""
        # Set protocol based on provider if not specified
        if self.protocol is None:
            if self.provider.lower() in ['openai', 'openai_compatible']:
                self.protocol = 'openai'
            elif self.provider.lower() in ['anthropic', 'anthropic_compatible']:
                self.protocol = 'anthropic'
            elif self.provider.lower() in ['google', 'gemini']:
                self.protocol = 'google'
            elif self.provider.lower() in ['vertex_ai', 'vertex']:
                self.protocol = 'vertex'
            else:
                # For custom providers, try to infer from endpoint_url or base_url
                url = self.endpoint_url or self.base_url or ""
                if 'openai' in url.lower() or '/chat/completions' in url:
                    self.protocol = 'openai'
                elif 'anthropic' in url.lower() or 'claude' in url.lower():
                    self.protocol = 'anthropic'
                else:
                    self.protocol = 'openai'  # Default to OpenAI protocol
        
        # Set base_url from endpoint_url if base_url is not provided
        if self.base_url is None and self.endpoint_url is not None:
            # Extract base URL from full endpoint URL
            if '/chat/completions' in self.endpoint_url:
                self.base_url = self.endpoint_url.replace('/chat/completions', '')
            elif '/v1/messages' in self.endpoint_url:
                self.base_url = self.endpoint_url.replace('/v1/messages', '')
            else:
                self.base_url = self.endpoint_url


class BaseLLMClient(ABC):
    """Abstract base class for LLM API clients."""
    
    def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.rate_limit_delay = 1.0  # Default 1 second between requests
        self.provider_name = self.__class__.__name__.replace("Client", "").lower()
    
    @abstractmethod
    async def evaluate_clip(self, prompt: str, max_tokens: int = 2000) -> EvaluationResponse:
        """Evaluate a clip using the LLM API."""
        pass
    
    def set_rate_limit_delay(self, delay: float):
        """Set the delay between API requests for rate limiting."""
        self.rate_limit_delay = delay


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT API client for evaluation."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o", base_url: Optional[str] = None):
        super().__init__(api_key, model_name, base_url)
        openai.api_key = api_key
        # Use custom base_url if provided, otherwise use default
        if base_url:
            self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
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
            eval_data = json.loads(content)
            
            return EvaluationResponse(
                scores=eval_data.get("scores", {}),
                summary=eval_data.get("summary", ""),
                reasoning=eval_data.get("reasoning", ""),
                success=True,
                model_name=self.model_name,
                provider="openai",
                raw_response=content
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return EvaluationResponse(
                scores={},
                summary="",
                reasoning="",
                success=False,
                error_message=str(e),
                model_name=self.model_name,
                provider="openai"
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
            eval_data = json.loads(content)
            
            return EvaluationResponse(
                scores=eval_data.get("scores", {}),
                summary=eval_data.get("summary", ""),
                reasoning=eval_data.get("reasoning", ""),
                success=True,
                model_name=self.model_name,
                provider="google",
                raw_response=content
            )
            
        except Exception as e:
            logger.error(f"Google Gemini API error: {e}")
            return EvaluationResponse(
                scores={},
                summary="",
                reasoning="",
                success=False,
                error_message=str(e),
                model_name=self.model_name,
                provider="google"
            )


class AnthropicClaudeClient(BaseLLMClient):
    """Anthropic Claude API client for evaluation."""
    
    def __init__(self, api_key: str, model_name: str = "claude-3-5-sonnet-20241022", base_url: Optional[str] = None):
        super().__init__(api_key, model_name, base_url)
        # Use custom base_url if provided, otherwise use default
        if base_url:
            self.client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url)
        else:
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
            eval_data = json.loads(content)
            
            return EvaluationResponse(
                scores=eval_data.get("scores", {}),
                summary=eval_data.get("summary", ""),
                reasoning=eval_data.get("reasoning", ""),
                success=True,
                model_name=self.model_name,
                provider="anthropic",
                raw_response=content
            )
            
        except Exception as e:
            logger.error(f"Anthropic Claude API error: {e}")
            return EvaluationResponse(
                scores={},
                summary="",
                reasoning="",
                success=False,
                error_message=str(e),
                model_name=self.model_name,
                provider="anthropic"
            )


class DeepSeekClient(BaseLLMClient):
    """DeepSeek API client for evaluation."""
    
    def __init__(self, api_key: str, model_name: str = "deepseek-chat", endpoint_url: str = "https://api.deepseek.com/v1/chat/completions"):
        super().__init__(api_key, model_name)
        self.endpoint_url = endpoint_url
    
    async def evaluate_clip(self, prompt: str, max_tokens: int = 2000) -> EvaluationResponse:
        """Evaluate a clip using DeepSeek API."""
        try:
            if not HAS_REQUESTS:
                raise ImportError("requests library required for DeepSeek client")
            
            # Add rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are an expert evaluator of AI agent trajectories. Provide detailed, objective evaluations with precise scores. Return your response as valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
            
            # Use asyncio.to_thread for blocking requests call
            response = await asyncio.to_thread(
                requests.post, 
                self.endpoint_url, 
                headers=headers, 
                json=data, 
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse the JSON response
            eval_data = json.loads(content)
            
            return EvaluationResponse(
                scores=eval_data.get("scores", {}),
                summary=eval_data.get("summary", ""),
                reasoning=eval_data.get("reasoning", ""),
                success=True,
                model_name=self.model_name,
                provider="deepseek",
                raw_response=content
            )
            
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            return EvaluationResponse(
                scores={},
                summary="",
                reasoning="",
                success=False,
                error_message=str(e),
                model_name=self.model_name,
                provider="deepseek"
            )


class KimiClient(BaseLLMClient):
    """Kimi (Moonshot) API client for evaluation."""
    
    def __init__(self, api_key: str, model_name: str = "moonshot-v1-8k", endpoint_url: str = "https://api.moonshot.cn/v1/chat/completions"):
        super().__init__(api_key, model_name)
        self.endpoint_url = endpoint_url
    
    async def evaluate_clip(self, prompt: str, max_tokens: int = 2000) -> EvaluationResponse:
        """Evaluate a clip using Kimi API."""
        try:
            if not HAS_REQUESTS:
                raise ImportError("requests library required for Kimi client")
            
            # Add rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are an expert evaluator of AI agent trajectories. Provide detailed, objective evaluations with precise scores. Return your response as valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1
            }
            
            # Use asyncio.to_thread for blocking requests call
            response = await asyncio.to_thread(
                requests.post, 
                self.endpoint_url, 
                headers=headers, 
                json=data, 
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Try to extract JSON from content if it's not pure JSON
            try:
                eval_data = json.loads(content)
            except json.JSONDecodeError:
                # Try to find JSON block in response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    eval_data = json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in response")
            
            return EvaluationResponse(
                scores=eval_data.get("scores", {}),
                summary=eval_data.get("summary", ""),
                reasoning=eval_data.get("reasoning", ""),
                success=True,
                model_name=self.model_name,
                provider="kimi",
                raw_response=content
            )
            
        except Exception as e:
            logger.error(f"Kimi API error: {e}")
            return EvaluationResponse(
                scores={},
                summary="",
                reasoning="",
                success=False,
                error_message=str(e),
                model_name=self.model_name,
                provider="kimi"
            )


class VertexAIClient(BaseLLMClient):
    """Google Vertex AI API client for evaluation.
    
    Vertex AI supports multiple models including:
    - Gemini models (gemini-1.5-pro, gemini-1.5-flash)
    - Claude models (claude-3-5-sonnet@20241022, claude-3-haiku@20240307)
    - And other models available through Vertex AI Model Garden
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro", project_id: str = "", location: str = "us-central1"):
        super().__init__(api_key, model_name)
        self.project_id = project_id
        self.location = location
        
        if not HAS_VERTEX_AI:
            raise ImportError("google-cloud-aiplatform required for Vertex AI client")
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        
        # Set up authentication if needed
        if api_key and api_key != "":
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = api_key  # For service account JSON file path
        
    async def evaluate_clip(self, prompt: str, max_tokens: int = 2000) -> EvaluationResponse:
        """Evaluate a clip using Vertex AI API."""
        try:
            # Add rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            # Handle different model types in Vertex AI
            if self.model_name.startswith("gemini"):
                # Use Vertex AI Gemini models
                response = await self._call_gemini_vertex(prompt, max_tokens)
            elif self.model_name.startswith("claude"):
                # Use Vertex AI Claude models (Anthropic on Vertex AI)
                response = await self._call_claude_vertex(prompt, max_tokens)
            else:
                # Try generic Vertex AI prediction API
                response = await self._call_generic_vertex(prompt, max_tokens)
            
            return response
            
        except Exception as e:
            logger.error(f"Vertex AI API error: {e}")
            return EvaluationResponse(
                scores={},
                summary="",
                reasoning="",
                success=False,
                error_message=str(e),
                model_name=self.model_name,
                provider="vertex_ai"
            )
    
    async def _call_gemini_vertex(self, prompt: str, max_tokens: int) -> EvaluationResponse:
        """Call Gemini models through Vertex AI."""
        try:
            from vertexai.generative_models import GenerativeModel, GenerationConfig
            
            model = GenerativeModel(self.model_name)
            
            generation_config = GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.1,
                response_mime_type="application/json"
            )
            
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=generation_config
            )
            
            content = response.text
            
            # Parse the JSON response
            eval_data = json.loads(content)
            
            return EvaluationResponse(
                scores=eval_data.get("scores", {}),
                summary=eval_data.get("summary", ""),
                reasoning=eval_data.get("reasoning", ""),
                success=True,
                model_name=self.model_name,
                provider="vertex_ai",
                raw_response=content
            )
            
        except Exception as e:
            raise Exception(f"Gemini Vertex AI call failed: {e}")
    
    async def _call_claude_vertex(self, prompt: str, max_tokens: int) -> EvaluationResponse:
        """Call Claude models through Vertex AI."""
        try:
            # Use the Vertex AI REST API for Claude models
            from google.auth import default
            from google.auth.transport.requests import Request
            
            # Get credentials
            credentials, _ = default()
            credentials.refresh(Request())
            
            # Prepare the request
            endpoint = f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/publishers/anthropic/models/{self.model_name}:streamRawPredict"
            
            headers = {
                "Authorization": f"Bearer {credentials.token}",
                "Content-Type": "application/json"
            }
            
            # Format for Claude on Vertex AI
            data = {
                "anthropic_version": "vertex-2023-10-16",
                "messages": [
                    {
                        "role": "user",
                        "content": f"You are an expert evaluator of AI agent trajectories. Provide detailed, objective evaluations with precise scores. Return your response as valid JSON.\n\n{prompt}"
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1
            }
            
            # Make the API call
            response = await asyncio.to_thread(
                requests.post,
                endpoint,
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            content = result.get("content", [{}])[0].get("text", "")
            
            # Parse the JSON response
            eval_data = json.loads(content)
            
            return EvaluationResponse(
                scores=eval_data.get("scores", {}),
                summary=eval_data.get("summary", ""),
                reasoning=eval_data.get("reasoning", ""),
                success=True,
                model_name=self.model_name,
                provider="vertex_ai",
                raw_response=content
            )
            
        except Exception as e:
            raise Exception(f"Claude Vertex AI call failed: {e}")
    
    async def _call_generic_vertex(self, prompt: str, max_tokens: int) -> EvaluationResponse:
        """Call other models through Vertex AI Prediction API."""
        try:
            # Use the generic prediction API for other models
            from google.cloud import aiplatform
            
            # Prepare the instance
            instance = {
                "prompt": f"You are an expert evaluator of AI agent trajectories. Provide detailed, objective evaluations with precise scores. Return your response as valid JSON.\n\n{prompt}",
                "max_output_tokens": max_tokens,
                "temperature": 0.1
            }
            
            # Get the endpoint
            endpoint = aiplatform.Endpoint.list(
                filter=f'display_name="{self.model_name}"',
                project=self.project_id,
                location=self.location
            )
            
            if not endpoint:
                raise ValueError(f"No endpoint found for model {self.model_name}")
            
            # Make prediction
            response = await asyncio.to_thread(
                endpoint[0].predict,
                instances=[instance]
            )
            
            # Extract content from response
            predictions = response.predictions
            if predictions:
                content = predictions[0].get("content", "")
                
                # Try to parse as JSON
                eval_data = json.loads(content)
                
                return EvaluationResponse(
                    scores=eval_data.get("scores", {}),
                    summary=eval_data.get("summary", ""),
                    reasoning=eval_data.get("reasoning", ""),
                    success=True,
                    model_name=self.model_name,
                    provider="vertex_ai",
                    raw_response=content
                )
            else:
                raise ValueError("No predictions in response")
                
        except Exception as e:
            raise Exception(f"Generic Vertex AI call failed: {e}")


class GenericOpenAIClient(BaseLLMClient):
    """Generic OpenAI-compatible API client for third-party providers.
    
    This client can be used with any provider that follows the OpenAI API protocol,
    such as OpenRouter, Together AI, Groq, etc.
    """
    
    def __init__(self, api_key: str, model_name: str, base_url: str, custom_headers: Optional[Dict[str, str]] = None):
        super().__init__(api_key, model_name, base_url)
        
        # Set up custom headers for third-party services
        default_headers = custom_headers or {}
        
        # Common headers that many third-party services expect
        if 'x-foo' not in default_headers:
            default_headers['x-foo'] = 'true'
        
        self.client = openai.AsyncOpenAI(
            api_key=api_key, 
            base_url=base_url,
            default_headers=default_headers
        )
    
    async def evaluate_clip(self, prompt: str, max_tokens: int = 2000) -> EvaluationResponse:
        """Evaluate a clip using OpenAI-compatible API."""
        try:
            # Add rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            # Prepare the request parameters
            request_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are an expert evaluator of AI agent trajectories. Provide detailed, objective evaluations with precise scores. Return your response as valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1,  # Low temperature for consistent evaluation
            }
            
            # Try with response_format first (for OpenAI-native services)
            try:
                request_params["response_format"] = {"type": "json_object"}
                response = await self.client.chat.completions.create(**request_params)
            except Exception as e:
                # If response_format fails, try without it (for third-party services)
                logger.debug(f"Failed with response_format, retrying without: {e}")
                request_params.pop("response_format", None)
                response = await self.client.chat.completions.create(**request_params)
            
            content = response.choices[0].message.content
            
            # Flexible JSON parsing for third-party services
            eval_data = self._parse_response_content(content)
            
            return EvaluationResponse(
                scores=eval_data.get("scores", {}),
                summary=eval_data.get("summary", ""),
                reasoning=eval_data.get("reasoning", ""),
                success=True,
                model_name=self.model_name,
                provider="openai_compatible",
                raw_response=content
            )
            
        except Exception as e:
            logger.error(f"OpenAI-compatible API error: {e}")
            return EvaluationResponse(
                scores={},
                summary="",
                reasoning="",
                success=False,
                error_message=str(e),
                model_name=self.model_name,
                provider="openai_compatible"
            )
    
    def _parse_response_content(self, content: str) -> Dict[str, Any]:
        """Parse response content with flexible JSON extraction."""
        try:
            # Try direct JSON parsing first
            return json.loads(content)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from the response
            import re
            
            # Look for JSON blocks in various formats
            json_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple nested JSON
                r'```json\s*(\{.*?\})\s*```',         # JSON in code blocks
                r'```\s*(\{.*?\})\s*```',             # JSON in generic code blocks
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, content, re.DOTALL)
                for match in matches:
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
            
            # If no JSON found, create a basic response structure
            logger.warning(f"Could not parse JSON from response: {content[:200]}...")
            return {
                "scores": {},
                "summary": content[:100] + "..." if len(content) > 100 else content,
                "reasoning": "Response could not be parsed as JSON"
            }


class GenericAnthropicClient(BaseLLMClient):
    """Generic Anthropic-compatible API client for third-party providers.
    
    This client can be used with any provider that follows the Anthropic API protocol.
    """
    
    def __init__(self, api_key: str, model_name: str, base_url: str):
        super().__init__(api_key, model_name, base_url)
        self.client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url)
    
    async def evaluate_clip(self, prompt: str, max_tokens: int = 2000) -> EvaluationResponse:
        """Evaluate a clip using Anthropic-compatible API."""
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
            eval_data = json.loads(content)
            
            return EvaluationResponse(
                scores=eval_data.get("scores", {}),
                summary=eval_data.get("summary", ""),
                reasoning=eval_data.get("reasoning", ""),
                success=True,
                model_name=self.model_name,
                provider="anthropic_compatible",
                raw_response=content
            )
            
        except Exception as e:
            logger.error(f"Anthropic-compatible API error: {e}")
            return EvaluationResponse(
                scores={},
                summary="",
                reasoning="",
                success=False,
                error_message=str(e),
                model_name=self.model_name,
                provider="anthropic_compatible"
            )


class LLMClientFactory:
    """Factory class to create LLM clients."""
    
    @staticmethod
    def create_client(provider: str, api_key: str, model_name: Optional[str] = None, **kwargs) -> BaseLLMClient:
        """Create an LLM client based on the provider."""
        
        # Extract common parameters
        base_url = kwargs.get("base_url")
        endpoint_url = kwargs.get("endpoint_url")
        protocol = kwargs.get("protocol")
        custom_headers = kwargs.get("custom_headers")
        
        # Determine if this is a third-party provider using OpenAI or Anthropic protocol
        if protocol == "openai" or (base_url and provider.lower() not in ["openai", "google", "anthropic", "deepseek", "kimi", "vertex_ai", "vertex"]):
            model = model_name or "gpt-3.5-turbo"  # Default for third-party OpenAI-compatible
            if not base_url:
                raise ValueError("base_url is required for OpenAI-compatible third-party providers")
            return GenericOpenAIClient(api_key, model, base_url, custom_headers)
        
        elif protocol == "anthropic":
            model = model_name or "claude-3-sonnet"  # Default for third-party Anthropic-compatible
            if not base_url:
                raise ValueError("base_url is required for Anthropic-compatible third-party providers")
            return GenericAnthropicClient(api_key, model, base_url)
        
        # Handle native providers
        elif provider.lower() == "openai":
            model = model_name or "gpt-4o"
            return OpenAIClient(api_key, model, base_url)
        
        elif provider.lower() == "google" or provider.lower() == "gemini":
            model = model_name or "gemini-1.5-pro"
            return GoogleGeminiClient(api_key, model)
        
        elif provider.lower() == "anthropic" or provider.lower() == "claude":
            model = model_name or "claude-3-5-sonnet-20241022"
            return AnthropicClaudeClient(api_key, model, base_url)
        
        elif provider.lower() == "deepseek":
            model = model_name or "deepseek-chat"
            endpoint_url = endpoint_url or "https://api.deepseek.com/v1/chat/completions"
            return DeepSeekClient(api_key, model, endpoint_url)
        
        elif provider.lower() == "kimi" or provider.lower() == "moonshot":
            model = model_name or "moonshot-v1-8k"
            endpoint_url = endpoint_url or "https://api.moonshot.cn/v1/chat/completions"
            return KimiClient(api_key, model, endpoint_url)
        
        elif provider.lower() == "vertex_ai" or provider.lower() == "vertex":
            model = model_name or "gemini-1.5-pro"
            project_id = kwargs.get("project_id", "")
            location = kwargs.get("location", "us-central1")
            return VertexAIClient(api_key, model, project_id, location)
        
        # Handle special provider names for third-party services
        elif provider.lower() in ["openai_compatible", "openrouter", "together", "groq", "fireworks"]:
            model = model_name or "gpt-3.5-turbo"
            if not base_url:
                raise ValueError(f"base_url is required for {provider}")
            return GenericOpenAIClient(api_key, model, base_url, custom_headers)
        
        elif provider.lower() in ["anthropic_compatible"]:
            model = model_name or "claude-3-sonnet"
            if not base_url:
                raise ValueError(f"base_url is required for {provider}")
            return GenericAnthropicClient(api_key, model, base_url)
        
        else:
            supported_providers = ["openai", "google", "anthropic", "deepseek", "kimi", "vertex_ai", 
                                 "openai_compatible", "anthropic_compatible", "openrouter", "together", "groq", "fireworks"]
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: {', '.join(supported_providers)}")
    
    @staticmethod
    def get_supported_providers() -> List[str]:
        """Get list of supported providers."""
        return ["openai", "google", "anthropic", "deepseek", "kimi", "vertex_ai", 
                "openai_compatible", "anthropic_compatible", "openrouter", "together", "groq", "fireworks"]


class MultiModelEvaluationConfig:
    """Configuration for multi-model evaluation."""
    
    def __init__(self, model_configs: List[ModelConfig], rate_limit_delay: float = 1.0, max_tokens: int = 2000):
        self.model_configs = model_configs
        self.rate_limit_delay = rate_limit_delay
        self.max_tokens = max_tokens
    
    def get_client_count(self) -> int:
        """Get number of configured models."""
        return len(self.model_configs)
    
    def create_clients(self) -> List[BaseLLMClient]:
        """Create all LLM clients from configurations."""
        clients = []
        for config in self.model_configs:
            try:
                client = LLMClientFactory.create_client(
                    provider=config.provider,
                    api_key=config.api_key,
                    model_name=config.model_name,
                    endpoint_url=config.endpoint_url,
                    project_id=config.project_id,
                    base_url=config.base_url,
                    protocol=config.protocol,
                    custom_headers=config.custom_headers
                )
                client.set_rate_limit_delay(self.rate_limit_delay)
                clients.append(client)
            except Exception as e:
                logger.error(f"Failed to create client for {config.provider}:{config.model_name}: {e}")
                continue
        return clients


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