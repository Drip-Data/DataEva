#!/usr/bin/env python3
"""
Multi-Model Evaluation Test
===========================

This script tests the multi-model evaluation functionality without requiring real API keys.
It creates mock LLM clients to simulate the multi-model evaluation process.

Usage:
    python src/test_multi_model_evaluation.py
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
from llm_api_clients import EvaluationResponse, ModelConfig, MultiModelEvaluationConfig
from step_level_evaluator import MultiModelStepLevelEvaluator, Clip


class MockLLMClient:
    """Mock LLM client for testing purposes."""
    
    def __init__(self, provider: str, model_name: str, base_scores: Dict[str, float]):
        self.provider_name = provider
        self.model_name = model_name
        self.base_scores = base_scores
        self.api_key = "mock_key"
        self.rate_limit_delay = 0.1  # Short delay for testing
    
    def set_rate_limit_delay(self, delay: float):
        """Set rate limit delay."""
        self.rate_limit_delay = delay
    
    async def evaluate_clip(self, prompt: str, max_tokens: int = 2000) -> EvaluationResponse:
        """Mock evaluation with simulated variation."""
        # Add some random variation to base scores (±0.1)
        import random
        varied_scores = {}
        for key, base_score in self.base_scores.items():
            variation = random.uniform(-0.1, 0.1)
            varied_scores[key] = max(0.0, min(1.0, base_score + variation))
        
        # Simulate API delay
        await asyncio.sleep(self.rate_limit_delay)
        
        return EvaluationResponse(
            scores=varied_scores,
            summary=f"Mock evaluation from {self.provider_name}:{self.model_name}",
            reasoning=f"This is a mock reasoning from {self.provider_name} model {self.model_name}",
            success=True,
            model_name=self.model_name,
            provider=self.provider_name,
            raw_response=json.dumps({
                "scores": varied_scores,
                "summary": f"Mock evaluation from {self.provider_name}:{self.model_name}",
                "reasoning": f"Mock reasoning from {self.provider_name}"
            })
        )


def create_mock_multi_model_evaluator() -> MultiModelStepLevelEvaluator:
    """Create a multi-model evaluator with mock clients."""
    
    # Create mock model configurations
    model_configs = [
        ModelConfig(provider="openai", model_name="gpt-4o", api_key="mock_openai_key"),
        ModelConfig(provider="google", model_name="gemini-1.5-pro", api_key="mock_google_key"),
        ModelConfig(provider="anthropic", model_name="claude-3-5-sonnet-20241022", api_key="mock_anthropic_key")
    ]
    
    # Create the multi-model configuration
    multi_config = MultiModelEvaluationConfig(model_configs, rate_limit_delay=0.1)
    
    # Create the evaluator
    evaluator = MultiModelStepLevelEvaluator(multi_config)
    
    # Replace real clients with mock clients
    mock_clients = [
        MockLLMClient("openai", "gpt-4o", {
            "code_correctness": 0.85,
            "computational_efficiency": 0.75,
            "error_handling": 0.65,
            "result_interpretation": 0.90
        }),
        MockLLMClient("google", "gemini-1.5-pro", {
            "code_correctness": 0.80,
            "computational_efficiency": 0.80,
            "error_handling": 0.70,
            "result_interpretation": 0.85
        }),
        MockLLMClient("anthropic", "claude-3-5-sonnet-20241022", {
            "code_correctness": 0.90,
            "computational_efficiency": 0.70,
            "error_handling": 0.75,
            "result_interpretation": 0.88
        })
    ]
    
    evaluator.llm_clients = mock_clients
    
    return evaluator


def create_test_trajectory() -> Dict[str, Any]:
    """Create a test trajectory with multiple tool calls."""
    
    return {
        "timestamp": "2025-01-07T12:00:00.000000",
        "task_id": "test_multi_model_001",
        "task_description": "Test task for multi-model evaluation",
        "duration": 5.0,
        "success": True,
        "final_result": "Task completed successfully",
        "raw_response": """
Let me solve this step by step.

<microsandbox>
# Calculate the fibonacci sequence
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)
print(f"Fibonacci(10) = {result}")
</microsandbox>

The fibonacci sequence calculation is complete. Now let me search for more information.

<deepsearch>
Search for "fibonacci sequence mathematical properties"
</deepsearch>

Based on the search results, I can provide additional context about fibonacci numbers.

Final answer: The 10th fibonacci number is 55, and fibonacci numbers have many interesting mathematical properties including the golden ratio relationship.
"""
    }


async def test_multi_model_clip_evaluation():
    """Test multi-model evaluation of a single clip."""
    
    print("Testing Multi-Model Clip Evaluation...")
    print("=" * 50)
    
    # Create mock evaluator
    evaluator = create_mock_multi_model_evaluator()
    
    # Create a test clip
    test_clip = Clip(
        content="<microsandbox>\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nresult = fibonacci(10)\nprint(f\"Fibonacci(10) = {result}\")\n</microsandbox>",
        tool_type="microsandbox",
        start_index=0,
        end_index=100,
        has_tool_call=True,
        tool_call_content="def fibonacci(n): ...",
        result_content="Fibonacci(10) = 55"
    )
    
    # Evaluate the clip with all models
    evaluation = await evaluator.evaluate_clip_with_all_models(
        clip=test_clip,
        task_description="Test fibonacci calculation",
        previous_context="This is the first step"
    )
    
    # Print results
    print(f"Clip Tool Type: {evaluation.clip.tool_type}")
    print(f"Number of Models: {len(evaluation.model_evaluations)}")
    print(f"Successful Evaluations: {evaluation.num_successful_models}")
    print(f"Overall Success: {evaluation.success}")
    
    print("\nIndividual Model Results:")
    for model_key, model_eval in evaluation.model_evaluations.items():
        print(f"  {model_key}:")
        print(f"    Success: {model_eval.success}")
        if model_eval.success:
            print(f"    Scores: {model_eval.scores}")
            print(f"    Summary: {model_eval.summary}")
    
    print(f"\nAveraged Results:")
    print(f"  Averaged Scores: {evaluation.averaged_scores}")
    print(f"  Averaged Summary: {evaluation.averaged_summary}")
    
    # Verify averaging works correctly
    if evaluation.success and len(evaluation.model_evaluations) > 1:
        # Check if scores are properly averaged
        for score_key in evaluation.averaged_scores:
            individual_scores = []
            for model_eval in evaluation.model_evaluations.values():
                if model_eval.success and score_key in model_eval.scores:
                    individual_scores.append(model_eval.scores[score_key])
            
            if individual_scores:
                expected_avg = statistics.mean(individual_scores)
                actual_avg = evaluation.averaged_scores[score_key]
                
                if abs(expected_avg - actual_avg) < 0.001:  # Allow small floating point differences
                    print(f"    ✓ {score_key}: {actual_avg:.3f} (correctly averaged)")
                else:
                    print(f"    ✗ {score_key}: {actual_avg:.3f} (expected {expected_avg:.3f})")
    
    print("=" * 50)
    return evaluation.success


async def test_full_trajectory_evaluation():
    """Test multi-model evaluation of a full trajectory."""
    
    print("\nTesting Full Trajectory Multi-Model Evaluation...")
    print("=" * 50)
    
    # Create mock evaluator
    evaluator = create_mock_multi_model_evaluator()
    
    # Create test trajectory
    trajectory = create_test_trajectory()
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        # Evaluate the trajectory
        result = await evaluator.evaluate_trajectory_with_full_response(trajectory, str(output_dir))
        
        # Check results
        print(f"Task ID: {result.get('task_id')}")
        print(f"Has full_response_with_evaluation: {'full_response_with_evaluation' in result}")
        print(f"Has evaluation_metadata: {'evaluation_metadata' in result}")
        
        if 'evaluation_metadata' in result:
            metadata = result['evaluation_metadata']
            print(f"Number of clips: {metadata.get('num_clips', 0)}")
            print(f"Number of models: {metadata.get('num_models', 0)}")
            print(f"Model names: {metadata.get('model_names', [])}")
            print(f"Successful evaluations: {metadata.get('successful_evaluations', 0)}")
            print(f"Overall trajectory score: {metadata.get('overall_trajectory_score', 0):.3f}")
        
        # Check if individual model files were created
        model_files = list(output_dir.glob("*_eva.json"))
        print(f"\nIndividual model files created: {len(model_files)}")
        for file_path in model_files:
            print(f"  - {file_path.name}")
            
            # Verify file content
            try:
                with open(file_path, 'r') as f:
                    file_data = json.load(f)
                    print(f"    Task ID: {file_data.get('task_id')}")
                    print(f"    Model: {file_data.get('model_name')}")
                    print(f"    Total clips: {file_data.get('total_clips', 0)}")
            except Exception as e:
                print(f"    Error reading file: {e}")
        
        # Check if full response contains embedded evaluations
        if 'full_response_with_evaluation' in result:
            full_response = result['full_response_with_evaluation']
            evaluation_count = full_response.count('<clip_evaluation>')
            print(f"\nEmbedded evaluations in full response: {evaluation_count}")
            
            if evaluation_count > 0:
                print("✓ Evaluations successfully embedded in response")
            else:
                print("✗ No evaluations found in response")
    
    print("=" * 50)
    return 'full_response_with_evaluation' in result and 'evaluation_metadata' in result


def test_model_config_creation():
    """Test model configuration creation."""
    
    print("\nTesting Model Configuration Creation...")
    print("=" * 50)
    
    # Test individual model configs
    configs = [
        ModelConfig(provider="openai", model_name="gpt-4o", api_key="test_key_1"),
        ModelConfig(provider="google", model_name="gemini-1.5-pro", api_key="test_key_2"),
        ModelConfig(provider="anthropic", model_name="claude-3-5-sonnet-20241022", api_key="test_key_3"),
        ModelConfig(provider="deepseek", model_name="deepseek-chat", api_key="test_key_4"),
        ModelConfig(provider="kimi", model_name="moonshot-v1-8k", api_key="test_key_5"),
        ModelConfig(provider="vertex_ai", model_name="gemini-1.5-pro", api_key="test_key_6", project_id="test_project")
    ]
    
    print(f"Created {len(configs)} model configurations:")
    for config in configs:
        print(f"  - {config.provider}: {config.model_name}")
        if config.project_id:
            print(f"    Project ID: {config.project_id}")
    
    # Test multi-model configuration
    multi_config = MultiModelEvaluationConfig(configs, rate_limit_delay=1.0)
    print(f"\nMulti-model config created with {multi_config.get_client_count()} models")
    print(f"Rate limit delay: {multi_config.rate_limit_delay}s")
    print(f"Max tokens: {multi_config.max_tokens}")
    
    print("=" * 50)
    return True


async def main():
    """Run all tests."""
    
    print("Multi-Model Evaluation System Test Suite")
    print("=" * 70)
    
    try:
        # Test 1: Model configuration
        test1_success = test_model_config_creation()
        
        # Test 2: Single clip evaluation
        test2_success = await test_multi_model_clip_evaluation()
        
        # Test 3: Full trajectory evaluation
        test3_success = await test_full_trajectory_evaluation()
        
        # Summary
        print("\nTest Results Summary:")
        print("=" * 30)
        print(f"Model Configuration: {'✓ PASS' if test1_success else '✗ FAIL'}")
        print(f"Clip Evaluation: {'✓ PASS' if test2_success else '✗ FAIL'}")
        print(f"Trajectory Evaluation: {'✓ PASS' if test3_success else '✗ FAIL'}")
        
        all_passed = test1_success and test2_success and test3_success
        print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        
        if all_passed:
            print("\n🎉 Multi-model evaluation system is working correctly!")
        else:
            print("\n❌ Some issues found in multi-model evaluation system.")
            
        return all_passed
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 