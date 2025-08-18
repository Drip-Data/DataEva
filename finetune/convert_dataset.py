#!/usr/bin/env python3
"""
Dataset Conversion Script for Process Reward Model Training
Converts finetune.jsonl to LLaMA Factory compatible format with metadata filtering
"""

import json
import argparse
import os
from typing import List, Dict, Any, Optional
from pathlib import Path


class DatasetConverter:
    def __init__(self, input_file: str, output_dir: str = "data"):
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_data(self) -> List[Dict[Any, Any]]:
        """Load data from JSONL file"""
        data = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def filter_by_metadata(self, data: List[Dict], 
                          evaluation_types: Optional[List[str]] = None,
                          model_names: Optional[List[str]] = None,
                          task_ids: Optional[List[str]] = None) -> List[Dict]:
        """Filter data based on metadata criteria"""
        filtered_data = []
        
        for entry in data:
            metadata = entry.get('metadata', {})
            
            # Filter by evaluation_type
            if evaluation_types:
                eval_type = metadata.get('evaluation_type', '')
                if eval_type not in evaluation_types:
                    continue
            
            # Filter by model_name
            if model_names:
                model_name = metadata.get('model_name', '')
                if model_name not in model_names:
                    continue
            
            # Filter by task_id
            if task_ids:
                task_id = metadata.get('task_id', '')
                if task_id not in task_ids:
                    continue
            
            filtered_data.append(entry)
        
        return filtered_data
    
    def convert_to_llamafactory_format(self, data: List[Dict], 
                                     include_metadata_in_prompt: bool = True) -> List[Dict]:
        """Convert data to LLaMA Factory format"""
        converted_data = []
        
        for entry in data:
            conversations = entry.get('conversations', [])
            metadata = entry.get('metadata', {})
            
            if len(conversations) < 2:
                continue
            
            human_content = conversations[0]['value']
            gpt_content = conversations[1]['value']
            
            # Enhance input with important metadata
            if include_metadata_in_prompt:
                metadata_prefix = ""
                
                # Add evaluation type
                eval_type = metadata.get('evaluation_type', '')
                if eval_type:
                    metadata_prefix += f"[EVAL_TYPE: {eval_type}] "
                
                # Add model name
                model_name = metadata.get('model_name', '')
                if model_name:
                    metadata_prefix += f"[MODEL: {model_name}] "
                
                # Add task ID
                task_id = metadata.get('task_id', '')
                if task_id:
                    metadata_prefix += f"[TASK: {task_id}] "
                
                # Add category if available
                category = metadata.get('category')
                if category:
                    metadata_prefix += f"[CATEGORY: {category}] "
                
                human_content = metadata_prefix + human_content
            
            converted_entry = {
                "instruction": human_content,
                "output": gpt_content,
                "metadata": metadata  # Keep original metadata for reference
            }
            converted_data.append(converted_entry)
        
        return converted_data
    
    def save_dataset(self, data: List[Dict], filename: str):
        """Save converted dataset"""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} samples to {output_path}")
    
    def generate_dataset_configs(self, datasets: Dict[str, str]):
        """Generate dataset_info.json for LLaMA Factory"""
        dataset_info = {}
        
        for name, filename in datasets.items():
            dataset_info[name] = {
                "file_name": filename,
                "columns": {
                    "prompt": "instruction",
                    "response": "output"
                }
            }
        
        config_path = self.output_dir / "dataset_info.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        print(f"Generated dataset configuration: {config_path}")
    
    def convert_all_variants(self):
        """Convert data into different training variants"""
        print("Loading original data...")
        data = self.load_data()
        print(f"Loaded {len(data)} total samples")
        
        datasets = {}
        
        # 1. Full dataset (all evaluation types, all models)
        full_data = self.convert_to_llamafactory_format(data)
        if full_data:
            filename = "reward_model_full.json"
            self.save_dataset(full_data, filename)
            datasets["reward_model_full"] = filename
        
        # 2. Clip evaluation only
        clip_data = self.filter_by_metadata(data, evaluation_types=["clip"])
        if clip_data:
            clip_converted = self.convert_to_llamafactory_format(clip_data)
            filename = "reward_model_clip.json"
            self.save_dataset(clip_converted, filename)
            datasets["reward_model_clip"] = filename
        
        # 3. Category evaluation only
        category_data = self.filter_by_metadata(data, evaluation_types=["category"])
        if category_data:
            category_converted = self.convert_to_llamafactory_format(category_data)
            filename = "reward_model_category.json"
            self.save_dataset(category_converted, filename)
            datasets["reward_model_category"] = filename
        
        # 4. Final evaluation only
        final_data = self.filter_by_metadata(data, evaluation_types=["final"])
        if final_data:
            final_converted = self.convert_to_llamafactory_format(final_data)
            filename = "reward_model_final.json"
            self.save_dataset(final_converted, filename)
            datasets["reward_model_final"] = filename
        
        # 5. Single model variants (example for each unique model)
        unique_models = set()
        for entry in data:
            model_name = entry.get('metadata', {}).get('model_name', '')
            if model_name:
                unique_models.add(model_name)
        
        for model in unique_models:
            model_data = self.filter_by_metadata(data, model_names=[model])
            if model_data:
                model_converted = self.convert_to_llamafactory_format(model_data)
                safe_model_name = model.replace('-', '_').replace('.', '_')
                filename = f"reward_model_{safe_model_name}.json"
                self.save_dataset(model_converted, filename)
                datasets[f"reward_model_{safe_model_name}"] = filename
        
        # 6. Task-specific variants (example for each unique task)
        unique_tasks = set()
        for entry in data:
            task_id = entry.get('metadata', {}).get('task_id', '')
            if task_id:
                unique_tasks.add(task_id)
        
        for task in unique_tasks:
            task_data = self.filter_by_metadata(data, task_ids=[task])
            if task_data:
                task_converted = self.convert_to_llamafactory_format(task_data)
                safe_task_name = task.replace('-', '_').replace('.', '_')
                filename = f"reward_model_{safe_task_name}.json"
                self.save_dataset(task_converted, filename)
                datasets[f"reward_model_{safe_task_name}"] = filename
        
        # Generate dataset configuration
        self.generate_dataset_configs(datasets)
        
        print(f"\nGenerated {len(datasets)} dataset variants:")
        for name, filename in datasets.items():
            print(f"  - {name}: {filename}")
        
        return datasets


def main():
    parser = argparse.ArgumentParser(description="Convert finetune.jsonl to LLaMA Factory format")
    parser.add_argument("--input", "-i", default="finetune.jsonl", 
                       help="Input JSONL file path")
    parser.add_argument("--output-dir", "-o", default="data",
                       help="Output directory for converted datasets")
    parser.add_argument("--evaluation-types", nargs="+", 
                       help="Filter by evaluation types (clip, category, final)")
    parser.add_argument("--model-names", nargs="+",
                       help="Filter by model names")
    parser.add_argument("--task-ids", nargs="+",
                       help="Filter by task IDs")
    parser.add_argument("--no-metadata", action="store_true",
                       help="Don't include metadata in prompts")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found!")
        return
    
    converter = DatasetConverter(args.input, args.output_dir)
    
    if args.evaluation_types or args.model_names or args.task_ids:
        # Custom filtering
        print("Loading and filtering data...")
        data = converter.load_data()
        filtered_data = converter.filter_by_metadata(
            data, args.evaluation_types, args.model_names, args.task_ids
        )
        converted_data = converter.convert_to_llamafactory_format(
            filtered_data, not args.no_metadata
        )
        converter.save_dataset(converted_data, "reward_model_custom.json")
        converter.generate_dataset_configs({"reward_model_custom": "reward_model_custom.json"})
    else:
        # Generate all variants
        converter.convert_all_variants()


if __name__ == "__main__":
    main()