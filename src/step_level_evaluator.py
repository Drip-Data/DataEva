import re
import json
import logging
import asyncio
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import statistics
import time

from llm_api_clients import BaseLLMClient, LLMClientFactory, EvaluationConfig, EvaluationResponse, MultiModelEvaluationConfig, ModelConfig

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class FineTuningQAPair:
    """Represents a query-answer pair for fine-tuning."""
    query: str  # The prompt sent to the LLM
    answer: str  # The response from the LLM
    model_name: str  # Which model generated this response
    provider: str  # Which provider was used
    task_id: str  # Which task this came from
    evaluation_type: str  # Type of evaluation: "clip", "category", or "final"
    clip_number: Optional[int] = None  # For clip evaluations
    category: Optional[str] = None  # For category evaluations
    timestamp: str = field(default_factory=lambda: time.strftime('%Y-%m-%d %H:%M:%S'))

@dataclass 
class FineTuningDataCollector:
    """Collects fine-tuning data during evaluation."""
    qa_pairs: List[FineTuningQAPair] = field(default_factory=list)
    enabled: bool = False
    
    def add_qa_pair(self, query: str, answer: str, model_name: str, provider: str, 
                   task_id: str, evaluation_type: str, clip_number: Optional[int] = None, 
                   category: Optional[str] = None):
        """Add a new QA pair to the collection."""
        if self.enabled:
            qa_pair = FineTuningQAPair(
                query=query,
                answer=answer,
                model_name=model_name,
                provider=provider,
                task_id=task_id,
                evaluation_type=evaluation_type,
                clip_number=clip_number,
                category=category
            )
            self.qa_pairs.append(qa_pair)
    
    def save_to_llamafactory_format(self, output_path: str):
        """Save collected QA pairs in LLaMa Factory format."""
        if not self.qa_pairs:
            logger.info("No fine-tuning QA pairs to save")
            return
            
        llamafactory_data = []
        for qa_pair in self.qa_pairs:
            # Use conversation format for LLaMa Factory
            llamafactory_entry = {
                "conversations": [
                    {"from": "human", "value": qa_pair.query},
                    {"from": "gpt", "value": qa_pair.answer}
                ],
                "metadata": {
                    "model_name": qa_pair.model_name,
                    "provider": qa_pair.provider,
                    "task_id": qa_pair.task_id,
                    "evaluation_type": qa_pair.evaluation_type,
                    "clip_number": qa_pair.clip_number,
                    "category": qa_pair.category,
                    "timestamp": qa_pair.timestamp
                }
            }
            llamafactory_data.append(llamafactory_entry)
        
        # Save to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in llamafactory_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            logger.info(f"Saved {len(llamafactory_data)} fine-tuning QA pairs to {output_path}")
            logger.info(f"  Models involved: {len(set(qa.model_name for qa in self.qa_pairs))}")
            logger.info(f"  Tasks covered: {len(set(qa.task_id for qa in self.qa_pairs))}")
            
        except Exception as e:
            logger.error(f"Failed to save fine-tuning data: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected fine-tuning data."""
        if not self.qa_pairs:
            return {}
            
        return {
            "total_qa_pairs": len(self.qa_pairs),
            "unique_models": len(set(qa.model_name for qa in self.qa_pairs)),
            "unique_tasks": len(set(qa.task_id for qa in self.qa_pairs)),
            "evaluation_types": {
                eval_type: len([qa for qa in self.qa_pairs if qa.evaluation_type == eval_type])
                for eval_type in set(qa.evaluation_type for qa in self.qa_pairs)
            },
            "models": {
                model: len([qa for qa in self.qa_pairs if qa.model_name == model])
                for model in set(qa.model_name for qa in self.qa_pairs)
            }
        }

@dataclass
class Clip:
    """Represents a clip in the trajectory for evaluation."""
    content: str
    tool_type: str  # Now represents the categorized MCP type (deepsearch, microsandbox, tavily, perform_web_task)
    start_index: int
    end_index: int
    clip_number: int  # clip1, clip2, etc.
    is_final_clip: bool = False  # True if this is the final <think></think> <answer></answer> clip
    think_content: str = ""
    result_content: str = ""
    sub_task_think_content: str = ""  # Content from <sub_task_think> if present
    tool_calls: List[str] = field(default_factory=list)  # List of meta tool calls found in this clip


@dataclass
class ClipEvaluation:
    """Evaluation result for a single clip."""
    clip: Clip
    score: float  # 0.0-1.0 score
    summary: str  # One-sentence summary of what was done in the clip
    success: bool


@dataclass
class MultiModelClipEvaluation:
    """Evaluation result from multiple models for a single clip."""
    clip: Clip
    model_evaluations: Dict[str, EvaluationResponse]  # model_name -> evaluation
    averaged_score: float  # Average score (0.0-1.0) - reasonableness_score or correctness_score
    score_type: str  # Type of score: "reasonableness_score" or "correctness_score"
    averaged_summary: str  # Combined summary
    success: bool
    num_successful_models: int


@dataclass
class CategoryEvaluation:
    """Evaluation result for a group of clips in the same MCP category."""
    category: str
    clips: List[Clip]
    clip_evaluations: List[MultiModelClipEvaluation]
    scores: Dict[str, float]  # Category-specific detailed scores
    summary: str
    reasoning: str
    success: bool


@dataclass
class MultiModelCategoryEvaluation:
    """Multi-model evaluation result for a category."""
    category: str
    clips: List[Clip]
    clip_evaluations: List[MultiModelClipEvaluation]
    model_evaluations: Dict[str, EvaluationResponse]  # model_name -> category evaluation
    averaged_scores: Dict[str, float]  # Averaged detailed scores
    averaged_summary: str
    individual_reasoning: Dict[str, str]  # model_name -> reasoning (not concatenated)
    success: bool
    num_successful_models: int


class MetaToolCategorizer:
    """Categorizes meta tool calls into MCP server categories."""
    
    # Based on explaination.md
    TOOL_CATEGORIES = {
        'deepsearch': ['deepsearch'],
        'microsandbox': ['sandbox_start', 'sandbox_stop', 'sandbox_run_code', 'sandbox_run_command', 'sandbox_get_metrice'],
        'tavily': ['tavily-search', 'tavily-extract', 'tavily-crawl', 'tavily-map'],
        'perform_web_task': [
            'search_google', 'go_to_url', 'go_back', 'click_element_by_index', 
            'input_text', 'switch_tab', 'close_tab', 'extract_structured_data', 
            'scroll', 'done', 'write_file', 'replace_file_str', 'wait'
        ]
    }
    
    @classmethod
    def categorize_tool(cls, tool_name: str) -> str:
        """Categorize a meta tool call into its MCP server category."""
        for category, tools in cls.TOOL_CATEGORIES.items():
            if tool_name in tools:
                return category
        return 'unknown'  # Fallback for uncategorized tools
    
    @classmethod
    def extract_tool_calls_from_content(cls, content: str) -> List[str]:
        """Extract tool call names from clip content."""
        import re
        # Look for "name": "tool_name" patterns in tool_call blocks
        tool_pattern = r'"name":\s*"([^"]+)"'
        matches = re.findall(tool_pattern, content)
        return matches


class PromptTemplates:
    """Prompt templates for clip and category evaluation."""
    
    BASE_TEMPLATE = """
You are evaluating an AI agent's step-by-step task execution. Each clip represents a thinking-action-result sequence.

Task Description: {task_description}

Previous Clips Summary: {previous_context}

Current Clip (Clip {clip_number}):
Tool Category: {tool_category}
Tools Used: {tools_used}
Content:
{clip_content}

Please evaluate this clip and provide your assessment in the following JSON format:
{{
    "reasonableness_score": 0.85,  // Score from 0.0 to 1.0 indicating how reasonable this clip is
    "summary": "Detailed summary of what was done in this clip: the thinking process, which specific tools were called with what parameters, the results obtained, and how they contribute to solving the task. This summary will be used for subsequent evaluations so include important context and details."
}}

Evaluation Guidelines for Reasonableness Score (0.0 - 1.0):
- 0.8-1.0: Excellent reasoning, optimal tool selection, clear logic, effective results
- 0.6-0.8: Good reasoning with minor issues, appropriate tool usage, mostly effective
- 0.4-0.6: Adequate reasoning but with notable issues, suboptimal but acceptable approach
- 0.2-0.4: Poor reasoning, questionable tool selection, limited effectiveness
- 0.0-0.2: Very poor reasoning, inappropriate tools, ineffective or counterproductive

Focus on:
1. Quality of reasoning in the <think> section - is the logic clear and sound?
2. Appropriateness of tool selection for the specific task requirements
3. Effectiveness of tool invocation - are parameters reasonable and well-chosen?
4. Quality and usefulness of results obtained from the <result> section
5. How well this clip builds upon and integrates with previous clips' work
6. Overall contribution to task progress and problem-solving
"""

    FINAL_CLIP_TEMPLATE = """
You are evaluating the final part of an AI agent's task execution where it provides the final answer.

Task Description: {task_description}

Previous Clips Summary: {previous_context}

Final Clip:
Content:
{clip_content}

Ground Truth Answer: {ground_truth_answer}
Model Output Answer: {model_output_answer}

Please evaluate this final clip and provide your assessment in the following JSON format:
{{
    "correctness_score": 0.95,  // Score from 0.0 to 1.0 indicating answer correctness
    "summary": "Detailed summary of the final answer: what conclusion was reached, how it addresses the original task, comparison with ground truth, and overall task completion status."
}}

Evaluation Guidelines for Correctness Score (0.0 - 1.0):
- 1.0: Perfect match with ground truth, completely correct answer
- 0.8-0.9: Mostly correct with minor formatting differences or additional context
- 0.6-0.8: Correct core answer but with some inaccuracies or incomplete information
- 0.4-0.6: Partially correct but missing important details or has notable errors
- 0.2-0.4: Incorrect answer but shows some understanding of the task
- 0.0-0.2: Completely wrong or nonsensical answer

Focus on:
1. Exact correctness compared to ground truth answer
2. Completeness and accuracy of the final answer
3. How well the answer addresses the original task requirements
4. Quality of reasoning that led to the conclusion
5. Whether the answer format is appropriate and clear
"""

    # Category-specific comprehensive evaluation templates
    DEEPSEARCH_CATEGORY_TEMPLATE = """
You are conducting a comprehensive evaluation of an AI agent's use of deepsearch tools across multiple clips.

Task Description: {task_description}

Clips in this deepsearch category (with individual evaluations): {clips_summary}

Please evaluate this deepsearch category comprehensively using these metrics:
1. Information Relevance (0.0-1.0): How relevant is the retrieved information to the task?
2. Tool use quality (0.0-1.0): Is the sequence of tool use logical and efficient?
3. Source Quality (0.0-1.0): Are high-quality, credible and rich sources being used?
4. Information Synthesis (0.0-1.0): How well is information from multiple sources synthesized and summarized?

Please provide your evaluation in the following JSON format:
{{
    "scores": {{
        "information_relevance": 0.85,
        "tool_use_quality": 0.90,
        "source_quality": 0.75,
        "information_synthesis": 0.80
    }},
    "summary": "One sentence overall assessment of the deepsearch category performance",
    "reasoning": "Detailed reasoning for each score: explain why you gave each metric its score, how the clips work together as a sequence, what was done well, what could be improved, and how this category contributes to the overall task completion. Provide specific examples from the clips to justify your scoring."
}}
"""

    MICROSANDBOX_CATEGORY_TEMPLATE = """
You are conducting a comprehensive evaluation of an AI agent's use of microsandbox tools across multiple clips.

Task Description: {task_description}

Clips in this microsandbox category: {clips_summary}

Please evaluate this microsandbox category comprehensively using these metrics:
1. Code Correctness (0.0-1.0): Is the code syntactically correct and logically sound?
2. Tool use quality (0.0-1.0): Is the sequence of tool use logical and efficient?
3. Computational Efficiency (0.0-1.0): Does the code solve the problem efficiently?
4. Result Interpretation (0.0-1.0): Are the computational results correctly interpreted?

Please provide your evaluation in the following JSON format:
{{
    "scores": {{
        "code_correctness": 0.85,
        "tool_use_quality": 0.90,
        "computational_efficiency": 0.75,
        "result_interpretation": 0.80
    }},
    "summary": "One sentence overall assessment of the microsandbox category performance",
    "reasoning": "Detailed reasoning for each score: explain why you gave each metric its score, how the clips work together as a sequence, what was done well, what could be improved, and how this category contributes to the overall task completion. Provide specific examples from the clips to justify your scoring."
}}
"""

    PERFORM_WEB_TASK_CATEGORY_TEMPLATE = """
You are conducting a comprehensive evaluation of an AI agent's use of web task tools across multiple clips.

Task Description: {task_description}

Clips in this perform_web_task category: {clips_summary}

Please evaluate this perform_web_task category comprehensively using these metrics:
1. Tool use quality (0.0-1.0): Is the sequence of tool use logical and efficient?
2. Content Extraction (0.0-1.0): How accurately is relevant content extracted?
3. Interaction Quality (0.0-1.0): How appropriate are the web interactions (clicks, inputs)?
4. Goal Achievement (0.0-1.0): How well does the browsing contribute to task completion?

Please provide your evaluation in the following JSON format:
{{
    "scores": {{
        "tool_use_quality": 0.85,
        "content_extraction": 0.90,
        "interaction_quality": 0.75,
        "goal_achievement": 0.80
    }},
    "summary": "One sentence overall assessment of the perform_web_task category performance",
    "reasoning": "Detailed reasoning for each score: explain why you gave each metric its score, how the clips work together as a sequence, what was done well, what could be improved, and how this category contributes to the overall task completion. Provide specific examples from the clips to justify your scoring."
}}
"""

    TAVILY_CATEGORY_TEMPLATE = """
You are conducting a comprehensive evaluation of an AI agent's use of tavily tools across multiple clips.

Task Description: {task_description}

Clips in this tavily category: {clips_summary}

Please evaluate this tavily category comprehensively using these metrics:
1. Tool use quality (0.0-1.0): Is the sequence of tool use logical and efficient?
2. Information Relevance (0.0-1.0): How relevant is the retrieved information to the task?
3. Content Extraction (0.0-1.0): How accurately is relevant content extracted?
4. Goal Achievement (0.0-1.0): How well does the browsing contribute to task completion?

Please provide your evaluation in the following JSON format:
{{
    "scores": {{
        "tool_use_quality": 0.85,
        "information_relevance": 0.90,
        "content_extraction": 0.75,
        "goal_achievement": 0.80
    }},
    "summary": "One sentence overall assessment of the tavily category performance",
    "reasoning": "Detailed reasoning for each score: explain why you gave each metric its score, how the clips work together as a sequence, what was done well, what could be improved, and how this category contributes to the overall task completion. Provide specific examples from the clips to justify your scoring."
}}
"""

    FINAL_CATEGORY_TEMPLATE = """
You are conducting a comprehensive evaluation of the entire trajectory completion across all clips.

Task Description: {task_description}

Complete trajectory with all clip summaries and evaluations:
{clips_summary}

Ground Truth Answer: {ground_truth_answer}
Model Final Result: {model_final_result}

Please evaluate the overall trajectory comprehensively using these metrics:
1. Task Completion (0.0-1.0): How completely is the task addressed?
2. Tool use quality (0.0-1.0): Is the sequence of tool use logical and efficient?
3. Reasoning Coherence (0.0-1.0): How logical and coherent is the overall reasoning chain?
4. Problem Resolution (0.0-1.0): How effectively are any encountered problems resolved?
5. Answer Correctness (0.0-1.0): How accurate is the final answer compared to ground truth?

Please provide your evaluation in the following JSON format:
{{
    "scores": {{
        "task_completion": 0.85,
        "tool_use_quality": 0.90,
        "reasoning_coherence": 0.75,
        "problem_resolution": 0.80,
        "answer_correctness": 0.95
    }},
    "summary": "One sentence overall assessment of the complete trajectory performance including answer accuracy",
    "reasoning": "Detailed reasoning for each score: explain why you gave each metric its score, how all the clips and categories work together to complete the task, what was done well throughout the trajectory, what could be improved, the accuracy of the final answer compared to ground truth, and the overall quality of task completion. Provide specific examples from different parts of the trajectory to justify your scoring."
}}
"""


class MultiModelStepLevelEvaluator:
    """Multi-model step-level evaluator for AI agent trajectories."""
    
    def __init__(self, multi_model_config: MultiModelEvaluationConfig, 
                 collect_finetune_data: bool = False):
        """Initialize with multiple LLM clients."""
        self.multi_model_config = multi_model_config
        self.llm_clients = multi_model_config.create_clients()
        
        # Initialize fine-tuning data collector
        self.finetune_collector = FineTuningDataCollector()
        self.finetune_collector.enabled = collect_finetune_data
        
        if not self.llm_clients:
            raise ValueError("No valid LLM clients could be created from the configuration")
        
        logger.info(f"Initialized multi-model evaluator with {len(self.llm_clients)} models")
        if collect_finetune_data:
            logger.info("Fine-tuning data collection enabled")
    
    def extract_clips(self, raw_response: str) -> List[Clip]:
        """Extract clips from the trajectory based on <think></think> <tool_call></tool_call> <result></result> patterns."""
        clips = []
        clip_number = 1
        
        # Find pattern: <think>...</think><tool_call>...</tool_call><result>...</result>
        # Or final pattern: <think>...</think><answer>...</answer>
        
        # First, find all <think> blocks with optional <sub_task_think>
        think_pattern = r'<think>(.*?)</think>(?:\s*<sub_task_think>(.*?)</sub_task_think>)?'
        think_matches = list(re.finditer(think_pattern, raw_response, re.DOTALL | re.IGNORECASE))
        
        # Find all <tool_call> blocks  
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        tool_call_matches = list(re.finditer(tool_call_pattern, raw_response, re.DOTALL | re.IGNORECASE))
        
        # Find all <result> and <answer> blocks
        result_pattern = r'<result>(.*?)</result>'
        answer_pattern = r'<answer>(.*?)</answer>'
        
        result_matches = list(re.finditer(result_pattern, raw_response, re.DOTALL | re.IGNORECASE))
        answer_matches = list(re.finditer(answer_pattern, raw_response, re.DOTALL | re.IGNORECASE))
        
        # For each <think>, find the corresponding <tool_call> and <result>/<answer>
        for think_match in think_matches:
            think_content = think_match.group(1).strip()
            sub_task_think_content = think_match.group(2).strip() if think_match.group(2) else ""
            think_end = think_match.end()
            
            # Find next tool_call after this think
            corresponding_tool_call = None
            for tool_call in tool_call_matches:
                if tool_call.start() > think_end:
                    corresponding_tool_call = tool_call
                    break
            
            # Find next result/answer after this think (or tool_call)
            search_start = corresponding_tool_call.end() if corresponding_tool_call else think_end
            corresponding_result = None
            is_final_clip = False
            
            # Check for result first
            for result in result_matches:
                if result.start() > search_start:
                    corresponding_result = result
                    break
            
            # If no result found, check for answer (final clip)
            if not corresponding_result:
                for answer in answer_matches:
                    if answer.start() > search_start:
                        corresponding_result = answer
                        is_final_clip = True
                        break
            
            if corresponding_result:
                # Extract the full clip content
                clip_start = think_match.start()
                clip_end = corresponding_result.end()
                clip_content = raw_response[clip_start:clip_end].strip()
                
                # Extract tool calls from the tool_call block if it exists
                tool_calls = []
                if corresponding_tool_call:
                    tool_calls = MetaToolCategorizer.extract_tool_calls_from_content(corresponding_tool_call.group(1))
                
                # Determine tool category based on tool calls
                tool_category = 'final' if is_final_clip else 'unknown'
                if tool_calls and not is_final_clip:
                    categories = [MetaToolCategorizer.categorize_tool(tool) for tool in tool_calls]
                    if categories:
                        from collections import Counter
                        tool_category = Counter(categories).most_common(1)[0][0]
                
                clip = Clip(
                    content=clip_content,
                    tool_type=tool_category,
                    start_index=clip_start,
                    end_index=clip_end,
                    clip_number=clip_number,
                    is_final_clip=is_final_clip,
                    think_content=think_content,
                    result_content=corresponding_result.group(1).strip(),
                    sub_task_think_content=sub_task_think_content,
                    tool_calls=tool_calls
                )
                clips.append(clip)
                clip_number += 1
        
        logger.info(f"Extracted {len(clips)} clips: {[f'clip{clip.clip_number}({clip.tool_type})' for clip in clips]}")
        return clips
    
    def _build_context_summary(self, clips: List[Clip], current_index: int, clip_evaluations: List[MultiModelClipEvaluation] = None) -> str:
        """Build a summary of previous clips with their evaluation results for context."""
        if current_index == 0:
            return "This is the first step in the trajectory."
        
        summaries = []
        # Include ALL previous clips, not just the last 3
        for i in range(current_index):
            clip = clips[i]
            tools_used = ", ".join(clip.tool_calls) if clip.tool_calls else "no tools"
            
            # Include evaluation result if available
            if clip_evaluations and i < len(clip_evaluations):
                evaluation_score = clip_evaluations[i].averaged_score
                summary_text = clip_evaluations[i].averaged_summary
                summary = f"Clip {clip.clip_number} ({clip.tool_type}): {summary_text} (Score: {evaluation_score:.2f})"
            else:
                summary = f"Clip {clip.clip_number} ({clip.tool_type}): Used {tools_used} (Score: not yet available)"
            
            summaries.append(summary)
        
        return "\n".join(summaries)
    
    def _group_clips_by_category(self, clips: List[Clip]) -> List[List[Clip]]:
        """Group consecutive clips by their MCP category."""
        if not clips:
            return []
        
        groups = []
        current_group = [clips[0]]
        current_category = clips[0].tool_type
        
        for clip in clips[1:]:
            # Final clips are always in their own group
            if clip.is_final_clip:
                if current_group:
                    groups.append(current_group)
                groups.append([clip])  # Final clip gets its own group
                current_group = []
                current_category = None
            elif clip.tool_type == current_category:
                current_group.append(clip)
            else:
                # Category changed, start new group
                if current_group:
                    groups.append(current_group)
                current_group = [clip]
                current_category = clip.tool_type
        
        # Add remaining group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    async def evaluate_clip_with_all_models(self, clip: Clip, task_description: str, previous_context: str, ground_truth_answer: str = "", model_final_result: str = "", task_id: str = "unknown") -> MultiModelClipEvaluation:
        """Evaluate a single clip using all configured models."""
        # Choose appropriate template based on whether it's the final clip
        if clip.is_final_clip:
            template = PromptTemplates.FINAL_CLIP_TEMPLATE
            prompt = template.format(
                task_description=task_description,
                previous_context=previous_context,
                clip_content=clip.content,
                ground_truth_answer=ground_truth_answer,
                model_output_answer=model_final_result
            )
        else:
            template = PromptTemplates.BASE_TEMPLATE
            tools_used = ", ".join(clip.tool_calls) if clip.tool_calls else "no tools"
            prompt = template.format(
                task_description=task_description,
                previous_context=previous_context,
                clip_number=clip.clip_number,
                tool_category=clip.tool_type,
                tools_used=tools_used,
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
                
                # Collect fine-tuning data if enabled
                if evaluation.success and hasattr(evaluation, 'raw_response') and evaluation.raw_response:
                    self.finetune_collector.add_qa_pair(
                        query=prompt,
                        answer=evaluation.raw_response,
                        model_name=client.model_name,
                        provider=client.provider_name,
                        task_id=task_id,
                        evaluation_type="clip",
                        clip_number=clip.clip_number
                    )
        
        # Calculate averaged evaluation
        if successful_evaluations:
            # Average numerical scores (reasonableness_score or correctness_score)
            if clip.is_final_clip:
                # For final clips, use correctness_score
                scores = [eval.scores.get('correctness_score', 0.0) for eval in successful_evaluations]
                averaged_score = statistics.mean(scores) if scores else 0.0
                score_key = 'correctness_score'
            else:
                # For regular clips, use reasonableness_score
                scores = [eval.scores.get('reasonableness_score', 0.0) for eval in successful_evaluations]
                averaged_score = statistics.mean(scores) if scores else 0.0
                score_key = 'reasonableness_score'
            
            # Concatenate summaries with model names
            summaries_by_model = {}
            for i, eval in enumerate(successful_evaluations):
                client = self.llm_clients[i] if i < len(self.llm_clients) else None
                model_key = f"{client.provider_name}_{client.model_name}" if client else f"model_{i}"
                summaries_by_model[model_key] = eval.summary
            averaged_summary = self._concatenate_summaries(summaries_by_model)
            success = True
        else:
            averaged_score = 0.0  # Default to 0 if all failed
            score_key = 'correctness_score' if clip.is_final_clip else 'reasonableness_score'
            averaged_summary = "All model evaluations failed"
            success = False
        
        return MultiModelClipEvaluation(
            clip=clip,
            model_evaluations=model_evaluations,
            averaged_score=averaged_score,
            score_type=score_key,
            averaged_summary=averaged_summary,
            success=success,
            num_successful_models=len(successful_evaluations)
        )
    

    
    def _concatenate_summaries(self, summaries_by_model: Dict[str, str]) -> str:
        """Concatenate summaries from different models."""
        if not summaries_by_model:
            return ""
        
        concatenated = []
        for model_name, summary in summaries_by_model.items():
            if summary.strip():
                concatenated.append(f"{model_name}: {summary.strip()}")
        
        return " | ".join(concatenated)
    
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
    
    async def evaluate_category_with_all_models(self, category: str, clips: List[Clip], clip_evaluations: List[MultiModelClipEvaluation], task_description: str, task_id: str = "unknown") -> MultiModelCategoryEvaluation:
        """Evaluate a category (group of clips) with all models."""
        
        # Build clips summary for category evaluation as required: summary + evaluation result for each clip
        clips_summary_parts = []
        for i, (clip, evaluation) in enumerate(zip(clips, clip_evaluations)):
            clip_summary = f"Clip {clip.clip_number}: {evaluation.averaged_summary} (Score: {evaluation.averaged_score:.2f})"
            clips_summary_parts.append(clip_summary)
        
        clips_summary = "\n".join(clips_summary_parts)
        
        # Choose appropriate template based on category
        category_templates = {
            'deepsearch': PromptTemplates.DEEPSEARCH_CATEGORY_TEMPLATE,
            'microsandbox': PromptTemplates.MICROSANDBOX_CATEGORY_TEMPLATE,
            'perform_web_task': PromptTemplates.PERFORM_WEB_TASK_CATEGORY_TEMPLATE,
            'tavily': PromptTemplates.TAVILY_CATEGORY_TEMPLATE,
            'final': PromptTemplates.FINAL_CATEGORY_TEMPLATE
        }
        
        template = category_templates.get(category, PromptTemplates.FINAL_CATEGORY_TEMPLATE)
        prompt = template.format(
            task_description=task_description,
            clips_summary=clips_summary
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
                logger.error(f"Category evaluation failed for {model_key}: {evaluation}")
                model_evaluations[model_key] = EvaluationResponse(
                    scores={}, summary="", reasoning="", success=False, 
                    error_message=str(evaluation), model_name=client.model_name, provider=client.provider_name
                )
            else:
                model_evaluations[model_key] = evaluation
                if evaluation.success:
                    successful_evaluations.append(evaluation)
                
                # Collect fine-tuning data if enabled
                if evaluation.success and hasattr(evaluation, 'raw_response') and evaluation.raw_response:
                    self.finetune_collector.add_qa_pair(
                        query=prompt,
                        answer=evaluation.raw_response,
                        model_name=client.model_name,
                        provider=client.provider_name,
                        task_id=task_id,
                        evaluation_type="category",
                        category=category
                    )
        
        # Calculate averaged results
        if successful_evaluations:
            # Average the detailed scores
            averaged_scores = self._average_scores([eval.scores for eval in successful_evaluations])
            # Concatenate summaries with model names
            summaries_by_model = {}
            individual_reasoning = {}
            for i, eval in enumerate(successful_evaluations):
                client = self.llm_clients[i] if i < len(self.llm_clients) else None
                model_key = f"{client.provider_name}_{client.model_name}" if client else f"model_{i}"
                summaries_by_model[model_key] = eval.summary
                individual_reasoning[model_key] = eval.reasoning  # Keep reasoning separate
            averaged_summary = self._concatenate_summaries(summaries_by_model)
            success = True
        else:
            averaged_scores = {}
            averaged_summary = "All category evaluations failed"
            individual_reasoning = {}
            success = False
        
        return MultiModelCategoryEvaluation(
            category=category,
            clips=clips,
            clip_evaluations=clip_evaluations,
            model_evaluations=model_evaluations,
            averaged_scores=averaged_scores,
            averaged_summary=averaged_summary,
            individual_reasoning=individual_reasoning,
            success=success,
            num_successful_models=len(successful_evaluations)
        )
    
    async def evaluate_final_trajectory_with_all_models(self, all_clips: List[Clip], all_clip_evaluations: List[MultiModelClipEvaluation], task_description: str, ground_truth_answer: str = "", model_final_result: str = "", task_id: str = "unknown") -> MultiModelCategoryEvaluation:
        """Evaluate the entire trajectory with all models for final assessment."""
        
        # Build complete trajectory summary with ALL clips and their evaluations
        clips_summary_parts = []
        for i, (clip, evaluation) in enumerate(zip(all_clips, all_clip_evaluations)):
            clip_summary = f"Clip {clip.clip_number} ({clip.tool_type}): {evaluation.averaged_summary} (Score: {evaluation.averaged_score:.2f})"
            clips_summary_parts.append(clip_summary)
        
        clips_summary = "\n".join(clips_summary_parts)
        
        # Use the final category template for complete trajectory evaluation
        template = PromptTemplates.FINAL_CATEGORY_TEMPLATE
        prompt = template.format(
            task_description=task_description,
            clips_summary=clips_summary,
            ground_truth_answer=ground_truth_answer,
            model_final_result=model_final_result
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
                logger.error(f"Final trajectory evaluation failed for {model_key}: {evaluation}")
                model_evaluations[model_key] = EvaluationResponse(
                    scores={}, summary="", reasoning="", success=False, 
                    error_message=str(evaluation), model_name=client.model_name, provider=client.provider_name
                )
            else:
                model_evaluations[model_key] = evaluation
                if evaluation.success:
                    successful_evaluations.append(evaluation)
                
                # Collect fine-tuning data if enabled
                if evaluation.success and hasattr(evaluation, 'raw_response') and evaluation.raw_response:
                    self.finetune_collector.add_qa_pair(
                        query=prompt,
                        answer=evaluation.raw_response,
                        model_name=client.model_name,
                        provider=client.provider_name,
                        task_id=task_id,
                        evaluation_type="final",
                        category="final"
                    )
        
        # Calculate averaged results for the entire trajectory
        if successful_evaluations:
            # Average the detailed scores across all models
            averaged_scores = self._average_scores([eval.scores for eval in successful_evaluations])
            # Concatenate summaries with model names
            summaries_by_model = {}
            individual_reasoning = {}
            for i, eval in enumerate(successful_evaluations):
                client = self.llm_clients[i] if i < len(self.llm_clients) else None
                model_key = f"{client.provider_name}_{client.model_name}" if client else f"model_{i}"
                summaries_by_model[model_key] = eval.summary
                individual_reasoning[model_key] = eval.reasoning  # Keep reasoning separate
            averaged_summary = self._concatenate_summaries(summaries_by_model)
            success = True
        else:
            averaged_scores = {}
            averaged_summary = "All final trajectory evaluations failed"
            individual_reasoning = {}
            success = False
        
        return MultiModelCategoryEvaluation(
            category="final",
            clips=all_clips,
            clip_evaluations=all_clip_evaluations,
            model_evaluations=model_evaluations,
            averaged_scores=averaged_scores,
            averaged_summary=averaged_summary,
            individual_reasoning=individual_reasoning,
            success=success,
            num_successful_models=len(successful_evaluations)
        )
    
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
        """Evaluate a trajectory with sequential clip and category evaluation."""
        
        # Extract task instruction, output, and answer from the data format
        if 'sft_data' in trajectory_data:
            task_description = trajectory_data['sft_data'].get('instruction', 'No task description provided')
            raw_response = trajectory_data['sft_data'].get('output', '')
            model_final_result = trajectory_data['sft_data'].get('final_result', '')
        else:
            # Fallback to old format
            task_description = trajectory_data.get('task_description', 'No task description provided')
            raw_response = trajectory_data.get('raw_response', '')
            model_final_result = trajectory_data.get('final_result', '')
        
        # Extract ground truth answer and task_id
        ground_truth_answer = trajectory_data.get('answer', '')
        task_id = trajectory_data.get('task_id', 'unknown')
        
        if not raw_response:
            logger.error("No raw_response found in trajectory data")
            return trajectory_data
        
        logger.info(f"Starting sequential multi-model evaluation for task: {task_id}")
        
        # Extract clips and group by category
        clips = self.extract_clips(raw_response)
        if not clips:
            logger.warning("No clips extracted from trajectory")
            return trajectory_data
        
        clip_groups = self._group_clips_by_category(clips)
        logger.info(f"Grouped {len(clips)} clips into {len(clip_groups)} categories")
        
        # Sequential evaluation: clip->clip->category->clip->clip->category->final
        all_clip_evaluations = []
        all_category_evaluations = []
        
        for group_idx, clip_group in enumerate(clip_groups):
            category = clip_group[0].tool_type
            logger.info(f"Evaluating category {group_idx+1}/{len(clip_groups)}: {category} ({len(clip_group)} clips)")
            
            # Evaluate each clip in the group
            group_clip_evaluations = []
            for clip in clip_group:
                # Build context from ALL previous clips with their evaluation results
                previous_context = self._build_context_summary(clips, clip.clip_number - 1, all_clip_evaluations)
                
                logger.info(f"  Evaluating clip {clip.clip_number} ({category})")
                evaluation = await self.evaluate_clip_with_all_models(
                    clip, task_description, previous_context, ground_truth_answer, model_final_result, task_id
                )
                group_clip_evaluations.append(evaluation)
                all_clip_evaluations.append(evaluation)
            
            # After evaluating all clips in the group, evaluate the category comprehensively
            if len(clip_group) > 0:  # Should always be true, but safety check
                logger.info(f"  Evaluating {category} category comprehensively")
                
                # Special case: if this is the final clip (last group), evaluate the entire trajectory
                is_final_group = (group_idx == len(clip_groups) - 1)
                if is_final_group and clip_group[0].is_final_clip:
                    # For final trajectory evaluation, include ALL clips, not just the final one
                    category_evaluation = await self.evaluate_final_trajectory_with_all_models(
                        clips, all_clip_evaluations, task_description, ground_truth_answer, model_final_result, task_id
                    )
                else:
                    category_evaluation = await self.evaluate_category_with_all_models(
                        category, clip_group, group_clip_evaluations, task_description, task_id
                    )
                all_category_evaluations.append(category_evaluation)
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        task_id = trajectory_data.get('task_id', 'unknown')
        
        # Save detailed results
        self._save_comprehensive_results(all_clip_evaluations, all_category_evaluations, task_id, output_path)
        
        # Create tool_call_statistics with both clip and category evaluations
        result = trajectory_data.copy()
        if 'tool_call_statistics' not in result:
            result['tool_call_statistics'] = {}
        
        # Add clip evaluations (streamlined for eva.jsonl - detailed summaries saved separately in model_evaluations)
        clip_evaluations = []
        for evaluation in all_clip_evaluations:
            clip_eval = {
                'clip_number': evaluation.clip.clip_number,
                'tool_category': evaluation.clip.tool_type,
                'tools_used': evaluation.clip.tool_calls,
                'score': evaluation.averaged_score,
                'score_type': evaluation.score_type,
                'is_final_clip': evaluation.clip.is_final_clip
            }
            clip_evaluations.append(clip_eval)
        
        # Add category evaluations (streamlined for eva.jsonl - detailed summaries saved separately in model_evaluations)
        category_evaluations = []
        for evaluation in all_category_evaluations:
            category_eval = {
                'category': evaluation.category,
                'clips_in_category': [c.clip_number for c in evaluation.clips],
                'averaged_scores': evaluation.averaged_scores
            }
            category_evaluations.append(category_eval)
        
        # Calculate overall statistics
        successful_scores = [eval.averaged_score for eval in all_clip_evaluations if eval.success]
        score_statistics = {}
        if successful_scores:
            score_statistics = {
                'average_score': statistics.mean(successful_scores),
                'min_score': min(successful_scores),
                'max_score': max(successful_scores),
                'median_score': statistics.median(successful_scores)
            }
        
        result['tool_call_statistics'].update({
            'clip_evaluations': clip_evaluations,
            'category_evaluations': category_evaluations,
            'evaluation_summary': {
                'total_clips': len(clips),
                'total_categories': len(all_category_evaluations),
                'successful_clip_evaluations': len(successful_scores),
                'successful_category_evaluations': len([e for e in all_category_evaluations if e.success]),
                'score_statistics': score_statistics,
                'num_models_used': len(self.llm_clients),
                'model_names': [f"{client.provider_name}_{client.model_name}" for client in self.llm_clients]
            }
        })
        
        logger.info(f"Sequential evaluation completed: {len(successful_scores)}/{len(clips)} clips, {len([e for e in all_category_evaluations if e.success])}/{len(all_category_evaluations)} categories")
        
        # Save fine-tuning data if enabled
        if self.finetune_collector.enabled and self.finetune_collector.qa_pairs:
            finetune_output_path = output_path / "finetune.jsonl"
            self.finetune_collector.save_to_llamafactory_format(str(finetune_output_path))
            
            # Log fine-tuning statistics
            stats = self.finetune_collector.get_statistics()
            if stats:
                logger.info(f"Fine-tuning data collection summary:")
                logger.info(f"  Total QA pairs: {stats.get('total_qa_pairs', 0)}")
                logger.info(f"  Models involved: {stats.get('unique_models', 0)}")
                logger.info(f"  Evaluation types: {list(stats.get('evaluation_types', {}).keys())}")
        
        return result
    
    def _save_comprehensive_results(self, clip_evaluations: List[MultiModelClipEvaluation], category_evaluations: List[MultiModelCategoryEvaluation], task_id: str, output_dir: Path):
        """Save comprehensive evaluation results with both clips and categories."""
        
        # Create comprehensive structure
        comprehensive_data = {
            'task_id': task_id,
            'evaluation_metadata': {
                'total_clips': len(clip_evaluations),
                'total_categories': len(category_evaluations),
                'total_models': len(self.llm_clients),
                'model_names': [f"{client.provider_name}_{client.model_name}" for client in self.llm_clients],
                'successful_clips': sum(1 for eval in clip_evaluations if eval.success),
                'successful_categories': sum(1 for eval in category_evaluations if eval.success),
                'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'clip_evaluations': [],
            'category_evaluations': []
        }
        
        # Process clip evaluations
        for clip_eval in clip_evaluations:
            clip_data = {
                'clip_number': clip_eval.clip.clip_number,
                'tool_category': clip_eval.clip.tool_type,
                'is_final_clip': clip_eval.clip.is_final_clip,
                'tools_used': clip_eval.clip.tool_calls,
                'averaged_results': {
                    'score': clip_eval.averaged_score,
                    'score_type': clip_eval.score_type,
                    'summary': clip_eval.averaged_summary,
                    'success': clip_eval.success
                },
                'individual_model_results': {}
            }
            
            # Add individual model results for clips
            for model_key, model_eval in clip_eval.model_evaluations.items():
                # Extract the appropriate score based on clip type
                if clip_eval.clip.is_final_clip:
                    score = model_eval.scores.get('correctness_score', 0.0) if model_eval.scores else 0.0
                    score_type = 'correctness_score'
                else:
                    score = model_eval.scores.get('reasonableness_score', 0.0) if model_eval.scores else 0.0
                    score_type = 'reasonableness_score'
                
                clip_data['individual_model_results'][model_key] = {
                    'score': score,
                    'score_type': score_type,
                    'summary': model_eval.summary,
                    'success': model_eval.success,
                    'error_message': model_eval.error_message if not model_eval.success else None
                }
            
            comprehensive_data['clip_evaluations'].append(clip_data)
        
        # Process category evaluations
        for cat_eval in category_evaluations:
            category_data = {
                'category': cat_eval.category,
                'clips_in_category': [c.clip_number for c in cat_eval.clips],
                'averaged_results': {
                    'scores': cat_eval.averaged_scores,
                    'summary': cat_eval.averaged_summary,
                    'success': cat_eval.success
                },
                'individual_model_results': {}
            }
            
            # Add individual model results for categories
            for model_key, model_eval in cat_eval.model_evaluations.items():
                category_data['individual_model_results'][model_key] = {
                    'scores': model_eval.scores,
                    'summary': model_eval.summary,
                    'reasoning': model_eval.reasoning,
                    'success': model_eval.success,
                    'error_message': model_eval.error_message if not model_eval.success else None
                }
            
            comprehensive_data['category_evaluations'].append(category_data)
        
        # Save comprehensive file
        filename = f"comprehensive_{task_id}_evaluation.json"
        filepath = output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f" Saved comprehensive evaluation: {filepath}")
            logger.info(f"    {len(clip_evaluations)} clips, {len(category_evaluations)} categories, {len(self.llm_clients)} models")
            
        except Exception as e:
            logger.error(f" Failed to save comprehensive results: {e}")
    
    def _save_individual_model_results(self, evaluations: List[MultiModelClipEvaluation], task_id: str, output_dir: Path):
        """Save consolidated multi-model evaluation results to a single JSON file per task."""
        
        # Create consolidated structure
        consolidated_data = {
            'task_id': task_id,
            'evaluation_metadata': {
                'total_clips': len(evaluations),
                'total_models': len(self.llm_clients),
                'model_names': [f"{client.provider_name}_{client.model_name}" for client in self.llm_clients],
                'successful_clips': sum(1 for eval in evaluations if eval.success),
                'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'clip_evaluations': []
        }
        
        # Process each clip evaluation
        for clip_index, evaluation in enumerate(evaluations):
            clip_data = {
                'clip_index': clip_index,
                'input': {
                    'tool_type': evaluation.clip.tool_type,
                    'clip_content': evaluation.clip.content,
                    'clip_number': evaluation.clip.clip_number,
                    'is_final_clip': evaluation.clip.is_final_clip,
                    'think_content': evaluation.clip.think_content,
                    'result_content': evaluation.clip.result_content,
                    'sub_task_think_content': evaluation.clip.sub_task_think_content,
                    'tool_calls': evaluation.clip.tool_calls,
                    'start_index': evaluation.clip.start_index,
                    'end_index': evaluation.clip.end_index,
                    'prompt_length': len(str(evaluation.clip.content))
                },
                'model_outputs': {},
                'averaged_results': {
                    'success': evaluation.success,
                    'score': evaluation.averaged_score,
                    'score_type': evaluation.score_type,
                    'summary': evaluation.averaged_summary,
                    'successful_models': evaluation.num_successful_models,
                    'total_models': len(evaluation.model_evaluations)
                }
            }
            
            # Add individual model outputs
            for model_key, model_eval in evaluation.model_evaluations.items():
                # Extract the appropriate score based on clip type
                if evaluation.clip.is_final_clip:
                    score = model_eval.scores.get('correctness_score', 0.0) if model_eval.scores else 0.0
                    score_type = 'correctness_score'
                else:
                    score = model_eval.scores.get('reasonableness_score', 0.0) if model_eval.scores else 0.0
                    score_type = 'reasonableness_score'
                
                clip_data['model_outputs'][model_key] = {
                    'success': model_eval.success,
                    'score': score,
                    'score_type': score_type,
                    'summary': model_eval.summary,
                    'model_name': model_eval.model_name,
                    'provider': model_eval.provider,
                    'raw_response': model_eval.raw_response,
                    'error_message': model_eval.error_message if not model_eval.success else None
                }
            
            consolidated_data['clip_evaluations'].append(clip_data)
        
        # Calculate overall task-level evaluation statistics
        all_successful_scores = []
        for evaluation in evaluations:
            if evaluation.success:
                all_successful_scores.append(evaluation.averaged_score)
        
        if all_successful_scores:
            # Calculate score statistics
            import statistics
            
            consolidated_data['task_level_summary'] = {
                'score_statistics': {
                    'average_score': statistics.mean(all_successful_scores),
                    'min_score': min(all_successful_scores),
                    'max_score': max(all_successful_scores),
                    'median_score': statistics.median(all_successful_scores)
                },
                'total_successful_clips': len(all_successful_scores),
                'total_clips': len(evaluations),
                'success_ratio': len(all_successful_scores) / len(evaluations) if evaluations else 0.0,
                'overall_quality_score': statistics.mean(all_successful_scores)
            }
        else:
            consolidated_data['task_level_summary'] = {
                'score_statistics': {
                    'average_score': 0.0,
                    'min_score': 0.0,
                    'max_score': 0.0,
                    'median_score': 0.0
                },
                'total_successful_clips': 0,
                'total_clips': len(evaluations),
                'success_ratio': 0.0,
                'overall_quality_score': 0.0,
                'note': 'No successful evaluations'
            }
        
        # Save consolidated file
        filename = f"multi_model_{task_id}_evaluation.json"
        filepath = output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f" Saved consolidated multi-model evaluation: {filepath}")
            logger.info(f"    {len(evaluations)} clips, {len(self.llm_clients)} models, {len(all_successful_scores)} successful clip evaluations")
            
            # Log task-level summary
            if consolidated_data['task_level_summary']['score_statistics']:
                score_stats = consolidated_data['task_level_summary']['score_statistics']
                avg_score = score_stats['average_score']
                overall_quality_score = consolidated_data['task_level_summary']['overall_quality_score']
                logger.info(f"    Task score statistics: avg={avg_score:.3f}, overall_quality={overall_quality_score:.3f}")
            
        except Exception as e:
            logger.error(f" Failed to save consolidated model results: {e}")
            # Fallback to save basic structure
            try:
                fallback_data = {
                    'task_id': task_id,
                    'error': str(e),
                    'basic_info': {
                        'total_clips': len(evaluations),
                        'total_models': len(self.llm_clients)
                    }
                }
                with open(output_dir / f"error_{task_id}_evaluation.json", 'w') as f:
                    json.dump(fallback_data, f, indent=2)
                logger.info(f" Saved error fallback file for task {task_id}")
            except:
                logger.error(f" Complete failure to save any results for task {task_id}")
    

    



# Backward compatibility - single model evaluator
class StepLevelEvaluator:
    """Single-model step-level evaluator (for backward compatibility)."""
    
    def __init__(self, llm_client: BaseLLMClient, collect_finetune_data: bool = False):
        """Initialize with a single LLM client."""
        # Create a multi-model config with just one model
        model_config = ModelConfig(
            provider=llm_client.provider_name,
            model_name=llm_client.model_name,
            api_key=llm_client.api_key
        )
        
        multi_config = MultiModelEvaluationConfig([model_config])
        self.multi_evaluator = MultiModelStepLevelEvaluator(multi_config, collect_finetune_data)
        self.multi_evaluator.llm_clients = [llm_client]  # Use the provided client directly
    
    async def evaluate_trajectory_with_full_response(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trajectory using single model (backward compatibility)."""
        result = await self.multi_evaluator.evaluate_trajectory_with_full_response(trajectory_data)
        return result


def create_evaluator(provider: str, api_key: str, model_name: Optional[str] = None) -> StepLevelEvaluator:
    """Create a single-model evaluator (backward compatibility)."""
    client = LLMClientFactory.create_client(provider, api_key, model_name)
    return StepLevelEvaluator(client)


def create_multi_model_evaluator(model_configs: List[ModelConfig], rate_limit_delay: float = 1.0, collect_finetune_data: bool = False) -> MultiModelStepLevelEvaluator:
    """Create a multi-model evaluator."""
    multi_config = MultiModelEvaluationConfig(model_configs, rate_limit_delay)
    return MultiModelStepLevelEvaluator(multi_config, collect_finetune_data) 