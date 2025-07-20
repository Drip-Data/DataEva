import re
import json
import logging
import asyncio
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import statistics

from llm_api_clients import BaseLLMClient, LLMClientFactory, EvaluationConfig, EvaluationResponse, MultiModelEvaluationConfig, ModelConfig

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class Clip:
    """Represents a clip in the trajectory for evaluation."""
    content: str
    tool_type: str
    start_index: int
    end_index: int
    has_tool_call: bool = True
    tool_call_content: str = ""
    result_content: str = ""


@dataclass
class ClipEvaluation:
    """Evaluation result for a single clip."""
    clip: Clip
    scores: Dict[str, float]
    summary: str
    reasoning: str
    tool_metrics: List[str]
    success: bool


@dataclass
class MultiModelClipEvaluation:
    """Evaluation result from multiple models for a single clip."""
    clip: Clip
    model_evaluations: Dict[str, EvaluationResponse]  # model_name -> evaluation
    averaged_scores: Dict[str, float]
    averaged_summary: str
    averaged_reasoning: str
    tool_metrics: List[str]
    success: bool
    num_successful_models: int


class PromptTemplates:
    """Prompt templates for different tool types."""
    
    MICROSANDBOX_TEMPLATE = """
You are evaluating an AI agent's use of the MicroSandbox tool (code execution and computational tools).

Task Description: {task_description}

Previous Context: {previous_context}

Current Clip Content:
{clip_content}

Evaluation Criteria for MicroSandbox:
1. Code Correctness (0.0-1.0): Is the code syntactically correct and logically sound?
2. Computational Efficiency (0.0-1.0): Does the code solve the problem efficiently?
3. Error Handling (0.0-1.0): Are potential errors and edge cases handled properly?
4. Result Interpretation (0.0-1.0): Are the computational results correctly interpreted?

Please provide your evaluation in the following JSON format:
{{
    "scores": {{
        "code_correctness": 0.85,
        "computational_efficiency": 0.70,
        "error_handling": 0.60,
        "result_interpretation": 0.90
    }},
    "summary": "Brief summary of the clip evaluation",
    "reasoning": "Detailed reasoning for the scores given"
}}
"""

    DEEPSEARCH_TEMPLATE = """
You are evaluating an AI agent's use of the DeepSearch tool (research and information gathering).

Task Description: {task_description}

Previous Context: {previous_context}

Current Clip Content:
{clip_content}

Evaluation Criteria for DeepSearch:
1. Query Formulation (0.0-1.0): How well-formulated and specific are the search queries?
2. Information Relevance (0.0-1.0): How relevant is the retrieved information to the task?
3. Source Quality (0.0-1.0): Are high-quality, credible sources being used?
4. Information Synthesis (0.0-1.0): How well is information from multiple sources synthesized?

Please provide your evaluation in the following JSON format:
{{
    "scores": {{
        "query_formulation": 0.85,
        "information_relevance": 0.90,
        "source_quality": 0.75,
        "information_synthesis": 0.80
    }},
    "summary": "Brief summary of the clip evaluation",
    "reasoning": "Detailed reasoning for the scores given"
}}
"""

    BROWSER_USE_TEMPLATE = """
You are evaluating an AI agent's use of the Browser Use tool (web browsing and content extraction).

Task Description: {task_description}

Previous Context: {previous_context}

Current Clip Content:
{clip_content}

Evaluation Criteria for Browser Use:
1. Navigation Efficiency (0.0-1.0): How efficiently does the agent navigate web pages?
2. Content Extraction (0.0-1.0): How accurately is relevant content extracted?
3. Interaction Quality (0.0-1.0): How appropriate are the web interactions (clicks, inputs)?
4. Goal Achievement (0.0-1.0): How well does the browsing contribute to task completion?

Please provide your evaluation in the following JSON format:
{{
    "scores": {{
        "navigation_efficiency": 0.85,
        "content_extraction": 0.90,
        "interaction_quality": 0.75,
        "goal_achievement": 0.80
    }},
    "summary": "Brief summary of the clip evaluation",
    "reasoning": "Detailed reasoning for the scores given"
}}
"""

    SEARCH_TOOL_TEMPLATE = """
You are evaluating an AI agent's use of the Search Tool (file and code search capabilities).

Task Description: {task_description}

Previous Context: {previous_context}

Current Clip Content:
{clip_content}

Evaluation Criteria for Search Tool:
1. Search Strategy (0.0-1.0): How effective are the search strategies employed?
2. Pattern Recognition (0.0-1.0): How well does the agent identify relevant patterns?
3. Result Filtering (0.0-1.0): How appropriately are search results filtered and selected?
4. Tool Integration (0.0-1.0): How well is the search tool integrated with other tools?

Please provide your evaluation in the following JSON format:
{{
    "scores": {{
        "search_strategy": 0.85,
        "pattern_recognition": 0.90,
        "result_filtering": 0.75,
        "tool_integration": 0.80
    }},
    "summary": "Brief summary of the clip evaluation",
    "reasoning": "Detailed reasoning for the scores given"
}}
"""

    FINAL_TEMPLATE = """
You are evaluating the final part of an AI agent's response (task completion and overall quality).

Task Description: {task_description}

Previous Context: {previous_context}

Current Clip Content:
{clip_content}

Evaluation Criteria for Final Response:
1. Task Completion (0.0-1.0): How completely is the original task addressed?
2. Response Quality (0.0-1.0): How clear, accurate, and well-structured is the response?
3. Reasoning Coherence (0.0-1.0): How logical and coherent is the overall reasoning chain?
4. Problem Resolution (0.0-1.0): How effectively are any encountered problems resolved?

Please provide your evaluation in the following JSON format:
{{
    "scores": {{
        "task_completion": 0.85,
        "response_quality": 0.90,
        "reasoning_coherence": 0.75,
        "problem_resolution": 0.80
    }},
    "summary": "Brief summary of the clip evaluation",
    "reasoning": "Detailed reasoning for the scores given"
}}
"""


class MultiModelStepLevelEvaluator:
    """Multi-model step-level evaluator for AI agent trajectories."""
    
    def __init__(self, multi_model_config: MultiModelEvaluationConfig):
        """Initialize with multiple LLM clients."""
        self.multi_model_config = multi_model_config
        self.llm_clients = multi_model_config.create_clients()
        
        if not self.llm_clients:
            raise ValueError("No valid LLM clients could be created from the configuration")
        
        # Tool-specific prompt templates
        self.prompt_templates = {
            'microsandbox': PromptTemplates.MICROSANDBOX_TEMPLATE,
            'deepsearch': PromptTemplates.DEEPSEARCH_TEMPLATE,
            'browser_use': PromptTemplates.BROWSER_USE_TEMPLATE,
            'search_tool': PromptTemplates.SEARCH_TOOL_TEMPLATE,
            'final': PromptTemplates.FINAL_TEMPLATE
        }
        
        # Tool-specific metrics
        self.tool_metrics = {
            'microsandbox': ['code_correctness', 'computational_efficiency', 'error_handling', 'result_interpretation'],
            'deepsearch': ['query_formulation', 'information_relevance', 'source_quality', 'information_synthesis'],
            'browser_use': ['navigation_efficiency', 'content_extraction', 'interaction_quality', 'goal_achievement'],
            'search_tool': ['search_strategy', 'pattern_recognition', 'result_filtering', 'tool_integration'],
            'final': ['task_completion', 'response_quality', 'reasoning_coherence', 'problem_resolution']
        }
        
        logger.info(f"Initialized multi-model evaluator with {len(self.llm_clients)} models")
    
    def extract_clips(self, raw_response: str) -> List[Clip]:
        """Extract clips from the trajectory based on tool usage patterns."""
        clips = []
        
        # Find all tool call patterns
        tool_patterns = {
            'microsandbox': r'<microsandbox[^>]*>(.*?)</microsandbox>',
            'deepsearch': r'<deepsearch[^>]*>(.*?)</deepsearch>',
            'browser_use': r'<browser_use[^>]*>(.*?)</browser_use>',
            'search_tool': r'<search_tool[^>]*>(.*?)</search_tool>'
        }
        
        # Find all tool calls and their positions
        tool_matches = []
        for tool_type, pattern in tool_patterns.items():
            matches = list(re.finditer(pattern, raw_response, re.DOTALL | re.IGNORECASE))
            for match in matches:
                tool_matches.append({
                    'tool_type': tool_type,
                    'start': match.start(),
                    'end': match.end(),
                    'content': match.group(1).strip(),
                    'full_match': match.group(0)
                })
        
        # Sort matches by position
        tool_matches.sort(key=lambda x: x['start'])
        
        # Create clips
        last_end = 0
        
        for i, match in enumerate(tool_matches):
            # Get context before the tool call
            context_start = max(last_end, match['start'] - 500)  # Include some context
            context_end = match['end']
            
            # Find any result content after the tool call
            next_start = tool_matches[i + 1]['start'] if i + 1 < len(tool_matches) else len(raw_response)
            result_end = min(next_start, match['end'] + 200)  # Look ahead for results
            
            clip_content = raw_response[context_start:result_end].strip()
            
            # Extract tool call content and result
            tool_call_content = match['content']
            result_content = raw_response[match['end']:result_end].strip()
            
            clip = Clip(
                content=clip_content,
                tool_type=match['tool_type'],
                start_index=context_start,
                end_index=result_end,
                has_tool_call=True,
                tool_call_content=tool_call_content,
                result_content=result_content
            )
            clips.append(clip)
            last_end = result_end
        
        # Add final clip for task completion if there's remaining content
        if last_end < len(raw_response) - 50:  # Only if there's substantial content
            final_content = raw_response[last_end:].strip()
            if final_content:
                final_clip = Clip(
                    content=final_content,
                    tool_type='final',
                    start_index=last_end,
                    end_index=len(raw_response),
                    has_tool_call=False,
                    tool_call_content="",
                    result_content=final_content
                )
                clips.append(final_clip)
        
        logger.info(f"Extracted {len(clips)} clips: {[clip.tool_type for clip in clips]}")
        return clips
    
    def _build_context_summary(self, clips: List[Clip], current_index: int) -> str:
        """Build a summary of previous clips for context."""
        if current_index == 0:
            return "This is the first step in the trajectory."
        
        summaries = []
        for i in range(min(3, current_index)):  # Include up to 3 previous clips
            clip = clips[current_index - 1 - i]
            tool_info = f"Used {clip.tool_type}"
            if clip.tool_call_content:
                call_preview = clip.tool_call_content[:100] + "..." if len(clip.tool_call_content) > 100 else clip.tool_call_content
                tool_info += f": {call_preview}"
            summaries.insert(0, tool_info)  # Insert at beginning to maintain order
        
        return " → ".join(summaries)
    
    async def evaluate_clip_with_all_models(self, clip: Clip, task_description: str, previous_context: str) -> MultiModelClipEvaluation:
        """Evaluate a single clip using all configured models."""
        # Get the appropriate prompt template
        template = self.prompt_templates.get(clip.tool_type, PromptTemplates.FINAL_TEMPLATE)
        prompt = template.format(
            task_description=task_description,
            previous_context=previous_context,
            clip_content=clip.content
        )
        
        # Evaluate with all models in parallel
        tasks = []
        for client in self.llm_clients:
            task = asyncio.create_task(client.evaluate_clip(prompt, self.multi_model_config.max_tokens))
            tasks.append(task)
        
        # Wait for all evaluations to complete
        evaluations = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        model_evaluations = {}
        successful_evaluations = []
        
        for i, evaluation in enumerate(evaluations):
            client = self.llm_clients[i]
            model_key = f"{client.provider_name}_{client.model_name}"
            
            if isinstance(evaluation, Exception):
                logger.error(f"Evaluation failed for {model_key}: {evaluation}")
                model_evaluations[model_key] = EvaluationResponse(
                    scores={}, summary="", reasoning="", success=False, 
                    error_message=str(evaluation), model_name=client.model_name, provider=client.provider_name
                )
            else:
                model_evaluations[model_key] = evaluation
                if evaluation.success:
                    successful_evaluations.append(evaluation)
        
        # Calculate averaged scores
        if successful_evaluations:
            averaged_scores = self._average_scores([eval.scores for eval in successful_evaluations])
            averaged_summary = self._average_text([eval.summary for eval in successful_evaluations])
            averaged_reasoning = self._average_text([eval.reasoning for eval in successful_evaluations])
            success = True
        else:
            averaged_scores = {}
            averaged_summary = "All model evaluations failed"
            averaged_reasoning = "No successful evaluations available"
            success = False
        
        return MultiModelClipEvaluation(
            clip=clip,
            model_evaluations=model_evaluations,
            averaged_scores=averaged_scores,
            averaged_summary=averaged_summary,
            averaged_reasoning=averaged_reasoning,
            tool_metrics=self.tool_metrics.get(clip.tool_type, []),
            success=success,
            num_successful_models=len(successful_evaluations)
        )
    
    def _average_scores(self, score_lists: List[Dict[str, float]]) -> Dict[str, float]:
        """Average scores across multiple evaluations."""
        if not score_lists:
            return {}
        
        # Collect all unique score keys
        all_keys = set()
        for scores in score_lists:
            all_keys.update(scores.keys())
        
        averaged = {}
        for key in all_keys:
            values = [scores.get(key, 0.0) for scores in score_lists if key in scores]
            if values:
                averaged[key] = statistics.mean(values)
        
        return averaged
    
    def _average_text(self, texts: List[str]) -> str:
        """Create an averaged/combined text from multiple texts."""
        if not texts:
            return ""
        
        # For now, we'll just combine the texts
        # In the future, this could use LLM to create a better summary
        filtered_texts = [text.strip() for text in texts if text.strip()]
        if len(filtered_texts) == 1:
            return filtered_texts[0]
        elif len(filtered_texts) > 1:
            return f"Combined evaluation: {' | '.join(filtered_texts[:3])}"  # Limit to first 3
        else:
            return ""
    
    async def evaluate_trajectory_with_full_response(self, trajectory_data: Dict[str, Any], output_dir: str = "data") -> Dict[str, Any]:
        """Evaluate a trajectory with multi-model approach and return full response with embedded evaluations."""
        
        task_description = trajectory_data.get('task_description', 'No task description provided')
        raw_response = trajectory_data.get('raw_response', '')
        
        if not raw_response:
            logger.error("No raw_response found in trajectory data")
            return trajectory_data
        
        logger.info(f"Starting multi-model evaluation for task: {trajectory_data.get('task_id', 'unknown')}")
        
        # Extract clips
        clips = self.extract_clips(raw_response)
        if not clips:
            logger.warning("No clips extracted from trajectory")
            return trajectory_data
        
        # Evaluate each clip with all models
        multi_model_evaluations = []
        for i, clip in enumerate(clips):
            previous_context = self._build_context_summary(clips, i)
            
            logger.info(f"Evaluating clip {i+1}/{len(clips)} ({clip.tool_type}) with {len(self.llm_clients)} models")
            evaluation = await self.evaluate_clip_with_all_models(clip, task_description, previous_context)
            multi_model_evaluations.append(evaluation)
        
        # Save individual model results
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        task_id = trajectory_data.get('task_id', 'unknown')
        
        # Save individual model files
        self._save_individual_model_results(multi_model_evaluations, task_id, output_path)
        
        # Create full response with embedded averaged evaluations
        full_response_with_evaluations = self._insert_averaged_evaluations(raw_response, multi_model_evaluations)
        
        # Calculate overall averaged metrics
        overall_metrics = self._calculate_overall_averaged_metrics(multi_model_evaluations)
        
        # Update trajectory data with averaged results
        result = trajectory_data.copy()
        result['full_response_with_evaluation'] = full_response_with_evaluations
        result['evaluation_metadata'] = {
            'num_clips': len(clips),
            'num_models': len(self.llm_clients),
            'model_names': [f"{client.provider_name}_{client.model_name}" for client in self.llm_clients],
            **overall_metrics
        }
        
        logger.info(f"Multi-model evaluation completed: {overall_metrics.get('successful_evaluations', 0)}/{len(clips)} clips successful")
        
        return result
    
    def _save_individual_model_results(self, evaluations: List[MultiModelClipEvaluation], task_id: str, output_dir: Path):
        """Save detailed results for each model to separate JSON files."""
        
        # Group results by model
        model_results = defaultdict(list)
        
        for evaluation in evaluations:
            for model_key, model_eval in evaluation.model_evaluations.items():
                model_results[model_key].append({
                    'clip_index': len(model_results[model_key]),
                    'tool_type': evaluation.clip.tool_type,
                    'clip_content': evaluation.clip.content,
                    'evaluation_input': {
                        'prompt_length': len(str(evaluation.clip.content)),
                        'tool_type': evaluation.clip.tool_type,
                        'has_tool_call': evaluation.clip.has_tool_call
                    },
                    'evaluation_output': {
                        'success': model_eval.success,
                        'scores': model_eval.scores,
                        'summary': model_eval.summary,
                        'reasoning': model_eval.reasoning,
                        'model_name': model_eval.model_name,
                        'provider': model_eval.provider,
                        'raw_response': model_eval.raw_response,
                        'error_message': model_eval.error_message
                    }
                })
        
        # Save each model's results to a separate file
        for model_key, results in model_results.items():
            filename = f"{model_key}_{task_id}_eva.json"
            filepath = output_dir / filename
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump({
                        'task_id': task_id,
                        'model_name': model_key,
                        'total_clips': len(results),
                        'evaluations': results
                    }, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved individual model results: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save model results for {model_key}: {e}")
    
    def _insert_averaged_evaluations(self, raw_response: str, evaluations: List[MultiModelClipEvaluation]) -> str:
        """Insert averaged evaluation results into the raw response."""
        response_with_eval = raw_response
        offset = 0
        
        for evaluation in evaluations:
            if not evaluation.success:
                continue
                
            clip = evaluation.clip
            
            # Create evaluation XML
            eval_xml = f"""
<clip_evaluation>
<scores>
{json.dumps(evaluation.averaged_scores, indent=2)}
</scores>
<summary>{evaluation.averaged_summary}</summary>
<reasoning>{evaluation.averaged_reasoning}</reasoning>
<model_info>Averaged from {evaluation.num_successful_models} models</model_info>
</clip_evaluation>
"""
            
            # Insert after the clip's end position
            insert_position = clip.end_index + offset
            response_with_eval = (
                response_with_eval[:insert_position] + 
                eval_xml + 
                response_with_eval[insert_position:]
            )
            offset += len(eval_xml)
        
        return response_with_eval
    
    def _calculate_overall_averaged_metrics(self, evaluations: List[MultiModelClipEvaluation]) -> Dict[str, Any]:
        """Calculate overall averaged metrics from multi-model evaluations."""
        total_clips = len(evaluations)
        successful_clips = sum(1 for eval in evaluations if eval.success)
        
        # Group scores by tool type
        tool_scores = defaultdict(list)
        for evaluation in evaluations:
            if evaluation.success:
                tool_scores[evaluation.clip.tool_type].append(evaluation.averaged_scores)
        
        # Calculate tool averages
        tool_averages = {}
        for tool_type, scores_list in tool_scores.items():
            if scores_list:
                averaged_tool_scores = self._average_scores(scores_list)
                overall_avg = statistics.mean(averaged_tool_scores.values()) if averaged_tool_scores else 0.0
                
                tool_averages[tool_type] = {
                    'average_scores': averaged_tool_scores,
                    'clip_count': len(scores_list),
                    'overall_average': overall_avg
                }
        
        # Calculate weighted overall score
        total_weight = 0
        weighted_sum = 0
        
        for tool_type, metrics in tool_averages.items():
            clip_count = metrics['clip_count']
            avg_score = metrics['overall_average']
            
            # Weight by number of clips
            weight = clip_count
            weighted_sum += avg_score * weight
            total_weight += weight
        
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return {
            'total_clips': total_clips,
            'successful_evaluations': successful_clips,
            'success_rate': successful_clips / total_clips if total_clips > 0 else 0.0,
            'overall_trajectory_score': overall_score,
            'tool_averages': tool_averages
        }


# Backward compatibility - single model evaluator
class StepLevelEvaluator:
    """Single-model step-level evaluator (for backward compatibility)."""
    
    def __init__(self, llm_client: BaseLLMClient):
        """Initialize with a single LLM client."""
        # Create a multi-model config with just one model
        model_config = ModelConfig(
            provider=llm_client.provider_name,
            model_name=llm_client.model_name,
            api_key=llm_client.api_key
        )
        
        multi_config = MultiModelEvaluationConfig([model_config])
        self.multi_evaluator = MultiModelStepLevelEvaluator(multi_config)
        self.multi_evaluator.llm_clients = [llm_client]  # Use the provided client directly
    
    async def evaluate_trajectory_with_full_response(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trajectory using single model (backward compatibility)."""
        result = await self.multi_evaluator.evaluate_trajectory_with_full_response(trajectory_data)
        return result


def create_evaluator(provider: str, api_key: str, model_name: Optional[str] = None) -> StepLevelEvaluator:
    """Create a single-model evaluator (backward compatibility)."""
    client = LLMClientFactory.create_client(provider, api_key, model_name)
    return StepLevelEvaluator(client)


def create_multi_model_evaluator(model_configs: List[ModelConfig], rate_limit_delay: float = 1.0) -> MultiModelStepLevelEvaluator:
    """Create a multi-model evaluator."""
    multi_config = MultiModelEvaluationConfig(model_configs, rate_limit_delay)
    return MultiModelStepLevelEvaluator(multi_config) 