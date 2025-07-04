import re
import json
import logging
import asyncio
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from llm_api_clients import BaseLLMClient, LLMClientFactory, EvaluationConfig, EvaluationResponse

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


class PromptTemplates:
    """Prompt templates for different tool types."""
    
    MICROSANDBOX_TEMPLATE = """
You are evaluating an AI agent's use of the MicroSandbox tool (code execution and computational tools).

Task Description: {task_description}

Previous Context: {previous_context}

Current Clip Content:
{clip_content}

Evaluation Criteria for MicroSandbox:
1. Code Correctness (0-1): Syntactic and logical correctness of generated code
2. Computational Efficiency (0-1): Appropriateness of algorithmic approach  
3. Error Handling (0-1): Proper handling of edge cases and errors
4. Result Interpretation (0-1): Accurate interpretation and integration of execution results

Please evaluate this clip and provide:
1. Scores for each criterion (0.0 to 1.0)
2. A one-sentence summary of what this clip accomplished
3. Detailed reasoning for your evaluation

Return your response as JSON:
{{
    "scores": {{
        "code_correctness": 0.0,
        "computational_efficiency": 0.0,
        "error_handling": 0.0,
        "result_interpretation": 0.0
    }},
    "summary": "One sentence summary of what this clip did and the result",
    "reasoning": "Detailed explanation of your evaluation and scoring rationale"
}}
"""

    DEEPSEARCH_TEMPLATE = """
You are evaluating an AI agent's use of the DeepSearch tool (research and information gathering).

Task Description: {task_description}

Previous Context: {previous_context}

Current Clip Content:
{clip_content}

Evaluation Criteria for DeepSearch:
1. Search Depth Appropriateness (0-1): Whether the depth of search matches task complexity
2. Query Refinement Quality (0-1): Effectiveness of iterative query improvements
3. Source Diversity (0-1): Breadth of information sources consulted
4. Synthesis Quality (0-1): Ability to synthesize information from multiple sources

Please evaluate this clip and provide:
1. Scores for each criterion (0.0 to 1.0)
2. A one-sentence summary of what this clip accomplished
3. Detailed reasoning for your evaluation

Return your response as JSON:
{{
    "scores": {{
        "search_depth_appropriateness": 0.0,
        "query_refinement_quality": 0.0,
        "source_diversity": 0.0,
        "synthesis_quality": 0.0
    }},
    "summary": "One sentence summary of what this clip did and the result",
    "reasoning": "Detailed explanation of your evaluation and scoring rationale"
}}
"""

    BROWSER_USE_TEMPLATE = """
You are evaluating an AI agent's use of the Browser Use tool (web browsing and information extraction).

Task Description: {task_description}

Previous Context: {previous_context}

Current Clip Content:
{clip_content}

Evaluation Criteria for Browser Use:
1. Query Relevance (0-1): How well the search query relates to the reasoning context
2. Information Extraction Quality (0-1): Effectiveness of extracting relevant information from results
3. Navigation Efficiency (0-1): Appropriateness of website selection and browsing strategy
4. Content Integration (0-1): How well retrieved information is integrated into reasoning

Please evaluate this clip and provide:
1. Scores for each criterion (0.0 to 1.0)
2. A one-sentence summary of what this clip accomplished
3. Detailed reasoning for your evaluation

Return your response as JSON:
{{
    "scores": {{
        "query_relevance": 0.0,
        "information_extraction_quality": 0.0,
        "navigation_efficiency": 0.0,
        "content_integration": 0.0
    }},
    "summary": "One sentence summary of what this clip did and the result",
    "reasoning": "Detailed explanation of your evaluation and scoring rationale"
}}
"""

    SEARCH_TOOL_TEMPLATE = """
You are evaluating an AI agent's use of the Search Tool (file and code search capabilities).

Task Description: {task_description}

Previous Context: {previous_context}

Current Clip Content:
{clip_content}

Evaluation Criteria for Search Tool:
1. Tool Selection Accuracy (0-1): Appropriateness of selected tool for the task
2. Parameter Optimization (0-1): Quality of parameters passed to the selected tool
3. Fallback Strategy (0-1): Effectiveness of alternative approaches when primary tool fails
4. Meta-Reasoning Quality (0-1): Quality of reasoning about tool selection process

Please evaluate this clip and provide:
1. Scores for each criterion (0.0 to 1.0)
2. A one-sentence summary of what this clip accomplished
3. Detailed reasoning for your evaluation

Return your response as JSON:
{{
    "scores": {{
        "tool_selection_accuracy": 0.0,
        "parameter_optimization": 0.0,
        "fallback_strategy": 0.0,
        "meta_reasoning_quality": 0.0
    }},
    "summary": "One sentence summary of what this clip did and the result",
    "reasoning": "Detailed explanation of your evaluation and scoring rationale"
}}
"""

    FINAL_CLIP_TEMPLATE = """
You are evaluating the final clip of an AI agent trajectory that may not contain any tool calls.

Task Description: {task_description}

Previous Context: {previous_context}

Final Clip Content:
{clip_content}

This is the final portion of the agent's response. Please evaluate based on:
1. Task Completion (0-1): How well the agent completed the overall task
2. Response Quality (0-1): Clarity and usefulness of the final response
3. Reasoning Coherence (0-1): Logical flow and consistency of reasoning
4. Problem Resolution (0-1): Effectiveness in addressing the original problem

Please evaluate this final clip and provide:
1. Scores for each criterion (0.0 to 1.0)
2. A one-sentence summary of what this clip accomplished
3. Detailed reasoning for your evaluation

Return your response as JSON:
{{
    "scores": {{
        "task_completion": 0.0,
        "response_quality": 0.0,
        "reasoning_coherence": 0.0,
        "problem_resolution": 0.0
    }},
    "summary": "One sentence summary of what this final clip accomplished",
    "reasoning": "Detailed explanation of your evaluation and scoring rationale"
}}
"""


class TrajectoryParser:
    """Parses AI agent trajectories to extract clips for evaluation."""
    
    def __init__(self):
        # Define the four MCP servers
        self.tool_servers = ['microsandbox', 'deepsearch', 'browser_use', 'search_tool']
        
    def identify_tool_calls(self, raw_response: str) -> List[Tuple[int, int, str]]:
        """
        Identify tool call positions and types in the raw response.
        
        Returns:
            List of (start_pos, end_pos, tool_type) tuples
        """
        tool_calls = []
        
        for tool in self.tool_servers:
            # Pattern to match tool calls with their content
            pattern = f'<{tool}(?:\\s[^>]*)?>.*?</{tool}>'
            matches = re.finditer(pattern, raw_response, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                tool_calls.append((match.start(), match.end(), tool))
        
        # Sort by start position
        tool_calls.sort(key=lambda x: x[0])
        return tool_calls
    
    def extract_clips(self, raw_response: str) -> List[Clip]:
        """
        Extract clips from the raw response based on tool call positions.
        
        Each clip includes:
        - All content from previous clip end to current tool call end
        - Or final content if no more tool calls
        """
        tool_calls = self.identify_tool_calls(raw_response)
        clips = []
        
        if not tool_calls:
            # No tool calls, treat entire response as final clip
            clips.append(Clip(
                content=raw_response,
                tool_type="final",
                start_index=0,
                end_index=len(raw_response),
                has_tool_call=False
            ))
            return clips
        
        start_pos = 0
        
        for i, (tool_start, tool_end, tool_type) in enumerate(tool_calls):
            # Extract content from start_pos to tool_end (inclusive)
            clip_content = raw_response[start_pos:tool_end]
            
            # Extract tool call content
            tool_call_content = raw_response[tool_start:tool_end]
            
            # Look for result content after tool call
            result_content = ""
            next_tool_start = tool_calls[i + 1][0] if i + 1 < len(tool_calls) else len(raw_response)
            
            # Search for <result> tags after the tool call
            result_pattern = r'<result>(.*?)</result>'
            result_search_area = raw_response[tool_end:next_tool_start]
            result_match = re.search(result_pattern, result_search_area, re.DOTALL)
            
            if result_match:
                result_content = result_match.group(1).strip()
                # Include result in clip content
                result_end = tool_end + result_match.end()
                clip_content = raw_response[start_pos:result_end]
                start_pos = result_end
            else:
                start_pos = tool_end
            
            clips.append(Clip(
                content=clip_content,
                tool_type=tool_type,
                start_index=start_pos if i == 0 else clips[-1].end_index,
                end_index=tool_end,
                has_tool_call=True,
                tool_call_content=tool_call_content,
                result_content=result_content
            ))
        
        # Handle final clip if there's remaining content
        if start_pos < len(raw_response):
            final_content = raw_response[start_pos:]
            if final_content.strip():  # Only add if there's meaningful content
                clips.append(Clip(
                    content=final_content,
                    tool_type="final",
                    start_index=start_pos,
                    end_index=len(raw_response),
                    has_tool_call=False
                ))
        
        return clips


class StepLevelEvaluator:
    """Main class for step-level evaluation of AI agent trajectories."""
    
    def __init__(self, llm_client: BaseLLMClient, config: EvaluationConfig):
        self.llm_client = llm_client
        self.config = config
        self.parser = TrajectoryParser()
        self.templates = PromptTemplates()
        
        # Template mapping
        self.template_map = {
            'microsandbox': self.templates.MICROSANDBOX_TEMPLATE,
            'deepsearch': self.templates.DEEPSEARCH_TEMPLATE,
            'browser_use': self.templates.BROWSER_USE_TEMPLATE,
            'search_tool': self.templates.SEARCH_TOOL_TEMPLATE,
            'final': self.templates.FINAL_CLIP_TEMPLATE
        }
    
    async def evaluate_clip(self, clip: Clip, task_description: str, previous_context: str = "") -> ClipEvaluation:
        """Evaluate a single clip using the appropriate template."""
        
        # Get the appropriate template
        template = self.template_map.get(clip.tool_type, self.templates.FINAL_CLIP_TEMPLATE)
        
        # Format the prompt
        prompt = template.format(
            task_description=task_description,
            previous_context=previous_context,
            clip_content=clip.content
        )
        
        # Call LLM API
        response = await self.llm_client.evaluate_clip(prompt, self.config.max_tokens)
        
        if response.success:
            # Get metric names for this tool type
            tool_metrics = self._get_tool_metrics(clip.tool_type)
            
            return ClipEvaluation(
                clip=clip,
                scores=response.scores,
                summary=response.summary,
                reasoning=response.reasoning,
                tool_metrics=tool_metrics,
                success=True
            )
        else:
            logger.error(f"Failed to evaluate clip: {response.error_message}")
            return ClipEvaluation(
                clip=clip,
                scores={},
                summary=f"Evaluation failed: {response.error_message}",
                reasoning="",
                tool_metrics=[],
                success=False
            )
    
    def _get_tool_metrics(self, tool_type: str) -> List[str]:
        """Get the metric names for a specific tool type."""
        metric_map = {
            'microsandbox': ['code_correctness', 'computational_efficiency', 'error_handling', 'result_interpretation'],
            'deepsearch': ['search_depth_appropriateness', 'query_refinement_quality', 'source_diversity', 'synthesis_quality'],
            'browser_use': ['query_relevance', 'information_extraction_quality', 'navigation_efficiency', 'content_integration'],
            'search_tool': ['tool_selection_accuracy', 'parameter_optimization', 'fallback_strategy', 'meta_reasoning_quality'],
            'final': ['task_completion', 'response_quality', 'reasoning_coherence', 'problem_resolution']
        }
        return metric_map.get(tool_type, [])
    
    async def evaluate_trajectory(self, sample: Dict) -> Dict:
        """
        Evaluate a complete trajectory using step-level evaluation.
        
        Args:
            sample: Dictionary containing task data with 'task_description' and 'raw_response'
            
        Returns:
            Dictionary with evaluation results inserted into the trajectory
        """
        task_description = sample.get('task_description', '')
        raw_response = sample.get('raw_response', '')
        
        # Parse trajectory into clips
        clips = self.parser.extract_clips(raw_response)
        
        logger.info(f"Evaluating trajectory {sample.get('task_id', 'unknown')} with {len(clips)} clips")
        
        # Evaluate each clip
        clip_evaluations = []
        previous_context = ""
        
        for i, clip in enumerate(clips):
            logger.debug(f"Evaluating clip {i+1}/{len(clips)} (tool: {clip.tool_type})")
            
            # Evaluate the clip
            evaluation = await self.evaluate_clip(clip, task_description, previous_context)
            clip_evaluations.append(evaluation)
            
            # Update previous context for next clip
            if evaluation.success:
                previous_context += f" [Previous: {evaluation.summary}]"
            
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(0.1)
        
        # Insert evaluation results into the trajectory
        evaluated_sample = self._insert_evaluations(sample, clip_evaluations)
        
        # Add overall statistics
        evaluated_sample['evaluation_metadata'] = self._calculate_overall_metrics(clip_evaluations)
        
        return evaluated_sample
    
    def _insert_evaluations(self, sample: Dict, clip_evaluations: List[ClipEvaluation]) -> Dict:
        """Insert evaluation results after each clip in the raw response."""
        
        raw_response = sample['raw_response']
        evaluated_response = ""
        last_end = 0
        
        for evaluation in clip_evaluations:
            clip = evaluation.clip
            
            # Add content up to clip end
            evaluated_response += raw_response[last_end:clip.end_index]
            
            # Add evaluation XML in the requested format
            if evaluation.success:
                eval_xml = f"""
<clip_evaluation>
<scores>
{self._format_scores_xml(evaluation.scores)}
</scores>
<summary>{evaluation.summary}</summary>
<reasoning>{evaluation.reasoning}</reasoning>
</clip_evaluation>
"""
                evaluated_response += eval_xml
            
            last_end = clip.end_index
        
        # Add any remaining content
        if last_end < len(raw_response):
            evaluated_response += raw_response[last_end:]
        
        # Create new sample with evaluated response
        evaluated_sample = sample.copy()
        evaluated_sample['raw_response'] = evaluated_response
        
        return evaluated_sample
    
    def _format_scores_xml(self, scores: Dict[str, float]) -> str:
        """Format scores as XML elements."""
        xml_elements = []
        for metric, score in scores.items():
            xml_elements.append(f"<{metric}>{score:.3f}</{metric}>")
        return "\n".join(xml_elements)
    
    def _calculate_overall_metrics(self, clip_evaluations: List[ClipEvaluation]) -> Dict:
        """Calculate overall trajectory evaluation metrics."""
        
        # Group scores by tool type
        tool_scores = defaultdict(list)
        total_clips = len(clip_evaluations)
        successful_clips = sum(1 for eval in clip_evaluations if eval.success)
        
        for evaluation in clip_evaluations:
            if evaluation.success:
                tool_scores[evaluation.clip.tool_type].append(evaluation.scores)
        
        # Calculate averages for each tool type
        tool_averages = {}
        for tool_type, score_list in tool_scores.items():
            if score_list:
                avg_scores = {}
                # Get all metric names from first score dict
                if score_list:
                    metrics = score_list[0].keys()
                    for metric in metrics:
                        scores = [scores_dict.get(metric, 0) for scores_dict in score_list]
                        avg_scores[metric] = sum(scores) / len(scores)
                
                tool_averages[tool_type] = {
                    'average_scores': avg_scores,
                    'clip_count': len(score_list),
                    'overall_average': sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0
                }
        
        return {
            'total_clips': total_clips,
            'successful_evaluations': successful_clips,
            'success_rate': successful_clips / total_clips if total_clips > 0 else 0,
            'tool_averages': tool_averages,
            'overall_trajectory_score': self._calculate_weighted_score(tool_averages)
        }
    
    def _calculate_weighted_score(self, tool_averages: Dict) -> float:
        """Calculate a weighted overall score for the trajectory."""
        if not tool_averages:
            return 0.0
        
        total_weight = 0
        weighted_sum = 0
        
        for tool_type, metrics in tool_averages.items():
            clip_count = metrics['clip_count']
            avg_score = metrics['overall_average']
            
            # Weight by number of clips (more clips = more influence)
            weight = clip_count
            weighted_sum += avg_score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def generate_full_response_with_evaluations(self, sample: Dict, clip_evaluations: List[ClipEvaluation]) -> str:
        """
        Generate a full response with embedded clip evaluations.
        
        This method creates a complete response where evaluation information is inserted
        after each clip in the specified XML format.
        
        Args:
            sample: The original sample data
            clip_evaluations: List of clip evaluations
            
        Returns:
            Complete response string with embedded evaluations
        """
        raw_response = sample['raw_response']
        full_response = ""
        last_end = 0
        
        for i, evaluation in enumerate(clip_evaluations):
            clip = evaluation.clip
            
            # Add content up to clip end
            full_response += raw_response[last_end:clip.end_index]
            
            # Add evaluation XML in the requested format
            if evaluation.success:
                eval_xml = f"""
<clip_evaluation>
<scores>
{self._format_scores_xml(evaluation.scores)}
</scores>
<summary>{evaluation.summary}</summary>
<reasoning>{evaluation.reasoning}</reasoning>
</clip_evaluation>"""
                full_response += eval_xml
            
            last_end = clip.end_index
        
        # Add any remaining content
        if last_end < len(raw_response):
            full_response += raw_response[last_end:]
        
        return full_response

    async def evaluate_trajectory_with_full_response(self, sample: Dict) -> Dict:
        """
        Evaluate a trajectory and return both traditional and full response formats.
        
        Args:
            sample: Dictionary containing task data
            
        Returns:
            Dictionary with both evaluation results and full response
        """
        task_description = sample.get('task_description', '')
        raw_response = sample.get('raw_response', '')
        
        # Parse trajectory into clips
        clips = self.parser.extract_clips(raw_response)
        
        logger.info(f"Evaluating trajectory {sample.get('task_id', 'unknown')} with {len(clips)} clips")
        
        # Evaluate each clip
        clip_evaluations = []
        previous_context = ""
        
        for i, clip in enumerate(clips):
            logger.debug(f"Evaluating clip {i+1}/{len(clips)} (tool: {clip.tool_type})")
            
            # Evaluate the clip
            evaluation = await self.evaluate_clip(clip, task_description, previous_context)
            clip_evaluations.append(evaluation)
            
            # Update previous context for next clip
            if evaluation.success:
                previous_context += f" [Previous: {evaluation.summary}]"
            
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(0.1)
        
        # Generate full response with evaluations
        full_response = self.generate_full_response_with_evaluations(sample, clip_evaluations)
        
        # Create evaluation result
        evaluated_sample = sample.copy()
        evaluated_sample['full_response_with_evaluations'] = full_response
        evaluated_sample['evaluation_metadata'] = self._calculate_overall_metrics(clip_evaluations)
        evaluated_sample['clip_evaluations'] = [
            {
                'clip_index': i,
                'tool_type': eval.clip.tool_type,
                'scores': eval.scores,
                'summary': eval.summary,
                'reasoning': eval.reasoning,
                'success': eval.success
            }
            for i, eval in enumerate(clip_evaluations)
        ]
        
        return evaluated_sample


# Factory function for easy initialization
def create_evaluator(provider: str = "openai", api_key: str = "", model_name: Optional[str] = None) -> StepLevelEvaluator:
    """Create a step-level evaluator with the specified provider."""
    
    config = EvaluationConfig(provider=provider, model_name=model_name)
    llm_client = LLMClientFactory.create_client(provider, api_key, model_name)
    
    return StepLevelEvaluator(llm_client, config) 