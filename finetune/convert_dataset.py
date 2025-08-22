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
                                     include_metadata_in_prompt: bool = True,
                                     split_eval: bool = False,
                                     eval_ratio: float = 0.1,
                                     min_samples_for_split: int = 5) -> Dict[str, List[Dict]]:
        """Convert data to LLaMA Factory format with optional train/eval split"""
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
        
        if split_eval and len(converted_data) >= min_samples_for_split:  # Only split if we have enough data
            # Shuffle data for random split
            import random
            random.seed(42)  # For reproducible splits
            random.shuffle(converted_data)
            
            # Calculate split point
            eval_size = max(1, int(len(converted_data) * eval_ratio))
            train_size = len(converted_data) - eval_size
            
            return {
                'train': converted_data[:train_size],
                'eval': converted_data[train_size:]
            }
        else:
            return {'train': converted_data}
    
    def save_dataset(self, data, filename: str):
        """Save converted dataset(s)"""
        if isinstance(data, dict) and 'train' in data:
            # Save train/eval split
            base_name = filename.replace('.json', '')
            
            # Save training data
            train_path = self.output_dir / f"{base_name}_train.json"
            with open(train_path, 'w', encoding='utf-8') as f:
                json.dump(data['train'], f, ensure_ascii=False, indent=2)
            print(f"Saved {len(data['train'])} training samples to {train_path}")
            
            # Save eval data if exists
            if 'eval' in data:
                eval_path = self.output_dir / f"{base_name}_eval.json"
                with open(eval_path, 'w', encoding='utf-8') as f:
                    json.dump(data['eval'], f, ensure_ascii=False, indent=2)
                print(f"Saved {len(data['eval'])} evaluation samples to {eval_path}")
                
                return {
                    'train': f"{base_name}_train.json",
                    'eval': f"{base_name}_eval.json"
                }
            else:
                return {'train': f"{base_name}_train.json"}
        else:
            # Save single dataset
            output_path = self.output_dir / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(data)} samples to {output_path}")
            return {'train': filename}
    
    def generate_dataset_configs(self, datasets: Dict[str, Dict[str, str]]):
        """Generate dataset_info.json for LLaMA Factory"""
        dataset_info = {}
        
        for name, files in datasets.items():
            if isinstance(files, dict) and 'train' in files:
                # Train/eval split dataset
                config = {
                    "file_name": files['train'],
                    "columns": {
                        "prompt": "instruction",
                        "response": "output"
                    }
                }
                if 'eval' in files:
                    config["file_name_eval"] = files['eval']
                dataset_info[name] = config
            else:
                # Single file dataset (backward compatibility)
                dataset_info[name] = {
                    "file_name": files if isinstance(files, str) else files.get('train', files),
                    "columns": {
                        "prompt": "instruction",
                        "response": "output"
                    }
                }
        
        config_path = self.output_dir / "dataset_info.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        print(f"Generated dataset configuration: {config_path}")
    
    def convert_all_variants(self, split_eval: bool = False, eval_ratio: float = 0.15, min_samples: int = 5):
        """Convert data into different training variants with user-controlled splitting"""
        print("Loading original data...")
        data = self.load_data()
        print(f"Loaded {len(data)} total samples")
        
        if split_eval:
            print(f"Will create train/eval splits with {eval_ratio:.1%} for evaluation")
            print(f"Minimum samples required for splitting: {min_samples}")
        else:
            print("Will create training-only datasets (no evaluation split)")
        
        datasets = {}
        
        # 1. Full dataset (all evaluation types, all models)
        full_data = self.convert_to_llamafactory_format(
            data, 
            split_eval=split_eval, 
            eval_ratio=eval_ratio,
            min_samples_for_split=min_samples
        )
        if full_data and full_data.get('train'):
            files = self.save_dataset(full_data, "reward_model_full.json")
            datasets["reward_model_full"] = files
        
        # 2. Clip evaluation only
        clip_data = self.filter_by_metadata(data, evaluation_types=["clip"])
        if clip_data:
            clip_converted = self.convert_to_llamafactory_format(
                clip_data, 
                split_eval=split_eval, 
                eval_ratio=eval_ratio,
                min_samples_for_split=min_samples
            )
            files = self.save_dataset(clip_converted, "reward_model_clip.json")
            datasets["reward_model_clip"] = files
        
        # 3. Category evaluation only
        category_data = self.filter_by_metadata(data, evaluation_types=["category"])
        if category_data:
            category_converted = self.convert_to_llamafactory_format(
                category_data, 
                split_eval=split_eval, 
                eval_ratio=eval_ratio,
                min_samples_for_split=min_samples
            )
            files = self.save_dataset(category_converted, "reward_model_category.json")
            datasets["reward_model_category"] = files
        
        # 4. Final evaluation only
        final_data = self.filter_by_metadata(data, evaluation_types=["final"])
        if final_data:
            final_converted = self.convert_to_llamafactory_format(
                final_data, 
                split_eval=split_eval, 
                eval_ratio=eval_ratio,
                min_samples_for_split=min_samples
            )
            files = self.save_dataset(final_converted, "reward_model_final.json")
            datasets["reward_model_final"] = files
        
        # 5. Single model variants (example for each unique model)
        unique_models = set()
        for entry in data:
            model_name = entry.get('metadata', {}).get('model_name', '')
            if model_name:
                unique_models.add(model_name)
        
        for model in unique_models:
            model_data = self.filter_by_metadata(data, model_names=[model])
            if model_data:
                # Only split if user requested AND we have enough samples
                should_split = split_eval and len(model_data) >= min_samples
                model_converted = self.convert_to_llamafactory_format(
                    model_data, 
                    split_eval=should_split, 
                    eval_ratio=eval_ratio,
                    min_samples_for_split=min_samples
                )
                safe_model_name = model.replace('-', '_').replace('.', '_')
                filename = f"reward_model_{safe_model_name}.json"
                files = self.save_dataset(model_converted, filename)
                datasets[f"reward_model_{safe_model_name}"] = files
        
        # 6. Task-specific variants - REMOVED
        # Individual task datasets are not needed and create too many files
        # Users can create custom task-specific datasets using:
        # python convert_dataset.py --task-ids specific_task_name
        
        # Generate dataset configuration
        self.generate_dataset_configs(datasets)
        
        print(f"\nGenerated {len(datasets)} dataset variants:")
        for name, files in datasets.items():
            if isinstance(files, dict) and 'eval' in files:
                print(f"  - {name}: {files['train']} + {files['eval']} (with eval split)")
            else:
                train_file = files.get('train', files) if isinstance(files, dict) else files
                print(f"  - {name}: {train_file} (training only)")
        
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
    
    # NEW: Train/Eval split parameters
    parser.add_argument("--split-eval", action="store_true",
                       help="Create train/eval splits (default: training only)")
    parser.add_argument("--eval-ratio", type=float, default=0.15,
                       help="Ratio of data for evaluation (default: 0.15 = 15%%)")
    parser.add_argument("--min-samples", type=int, default=5,
                       help="Minimum samples required to create eval split (default: 5)")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found!")
        return
    
    converter = DatasetConverter(args.input, args.output_dir)
    
    # Validate arguments
    if args.eval_ratio <= 0 or args.eval_ratio >= 1:
        print("Error: --eval-ratio must be between 0 and 1")
        return
    
    if args.min_samples < 2:
        print("Error: --min-samples must be at least 2")
        return
    
    # Print configuration
    print(f"Configuration:")
    print(f"  Input file: {args.input}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Create eval splits: {'Yes' if args.split_eval else 'No'}")
    if args.split_eval:
        print(f"  Evaluation ratio: {args.eval_ratio:.1%}")
        print(f"  Min samples for split: {args.min_samples}")
    print()
    
    if args.evaluation_types or args.model_names or args.task_ids:
        # Custom filtering
        print("Loading and filtering data...")
        data = converter.load_data()
        filtered_data = converter.filter_by_metadata(
            data, args.evaluation_types, args.model_names, args.task_ids
        )
        converted_data = converter.convert_to_llamafactory_format(
            filtered_data, 
            not args.no_metadata,
            split_eval=args.split_eval,
            eval_ratio=args.eval_ratio,
            min_samples_for_split=args.min_samples
        )
        files = converter.save_dataset(converted_data, "reward_model_custom.json")
        converter.generate_dataset_configs({"reward_model_custom": files})
    else:
        # Generate all variants with user-specified splitting
        converter.convert_all_variants(
            split_eval=args.split_eval,
            eval_ratio=args.eval_ratio,
            min_samples=args.min_samples
        )


if __name__ == "__main__":
    main()