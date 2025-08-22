#!/usr/bin/env python3
"""
Process Reward Model Training Script
Main training script for fine-tuning process reward models using LLaMA Factory
"""

import os
import sys
import yaml
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Add LLaMA Factory to Python path if needed
# Uncomment and modify the path if LLaMA Factory is not in your Python path
# sys.path.append('/path/to/LLaMA-Factory')

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("Warning: SwanLab not available. Training will proceed without monitoring.")


class RewardModelTrainer:
    def __init__(self, config_path: str = "training_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_paths()
        
    def load_config(self) -> Dict[str, Any]:
        """Load training configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_paths(self):
        """Setup and validate all required paths"""
        # Model path
        self.model_path = Path(self.config['model_name_or_path'])
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        
        # Data path
        self.data_path = Path(self.config['dataset_dir'])
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        
        # Dataset info file
        self.dataset_info_path = self.data_path / "dataset_info.json"
        if not self.dataset_info_path.exists():
            raise FileNotFoundError(f"Dataset info file not found: {self.dataset_info_path}")
        
        # Output directory
        self.output_path = Path(self.config['output_dir'])
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Logs directory
        self.logs_path = Path(self.config['logging_dir'])
        self.logs_path.mkdir(parents=True, exist_ok=True)
    
    def setup_monitoring(self):
        """Setup SwanLab monitoring"""
        if SWANLAB_AVAILABLE and self.config.get('experiment_name'):
            try:
                # Disable GPU monitoring to avoid cleanup issues
                import os
                os.environ['SWANLAB_DISABLE_GPU_MONITOR'] = '1'
                
                swanlab.init(
                    project=self.config.get('experiment_name', 'reward_model_training'),
                    experiment_name=self.config.get('run_name', 'process_reward_model_v1'),
                    config=self.config,
                    # Additional parameters to prevent GPU monitoring issues
                    settings={
                        'hardware_monitor': False,  # Disable hardware monitoring
                        'gpu_monitor': False,       # Explicitly disable GPU monitoring
                    } if hasattr(swanlab, 'init') else {}
                )
                print("✓ SwanLab monitoring initialized (GPU monitoring disabled)")
                return True
            except Exception as e:
                print(f"Warning: Failed to initialize SwanLab: {e}")
                return False
        return False
    
    def prepare_llamafactory_args(self) -> list:
        """Convert config to LLaMA Factory command line arguments"""
        args = []
        
        # Essential arguments
        args.extend(['--stage', self.config.get('stage', 'sft')])
        args.extend(['--do_train', 'True'])
        args.extend(['--model_name_or_path', str(self.model_path)])
        args.extend(['--dataset_dir', str(self.data_path)])
        args.extend(['--dataset', self.config.get('dataset', 'reward_model_full')])
        args.extend(['--template', self.config.get('template', 'qwen')])
        args.extend(['--finetuning_type', self.config.get('finetuning_type', 'lora')])
        args.extend(['--output_dir', str(self.output_path)])
        args.extend(['--overwrite_output_dir', 'True'])
        args.extend(['--overwrite_cache', 'True'])
        
        # LoRA settings
        if self.config.get('finetuning_type') == 'lora':
            args.extend(['--lora_target', self.config.get('lora_target', 'all')])
            args.extend(['--lora_rank', str(self.config.get('lora_rank', 64))])
            args.extend(['--lora_alpha', str(self.config.get('lora_alpha', 128))])
            args.extend(['--lora_dropout', str(self.config.get('lora_dropout', 0.1))])
        
        # Training hyperparameters
        args.extend(['--learning_rate', str(self.config.get('learning_rate', 1e-4))])
        args.extend(['--num_train_epochs', str(self.config.get('num_train_epochs', 3.0))])
        args.extend(['--per_device_train_batch_size', str(self.config.get('per_device_train_batch_size', 2))])
        args.extend(['--gradient_accumulation_steps', str(self.config.get('gradient_accumulation_steps', 8))])
        args.extend(['--lr_scheduler_type', self.config.get('lr_scheduler_type', 'cosine')])
        args.extend(['--warmup_ratio', str(self.config.get('warmup_ratio', 0.1))])
        args.extend(['--weight_decay', str(self.config.get('weight_decay', 0.01))])
        args.extend(['--max_grad_norm', str(self.config.get('max_grad_norm', 1.0))])
        
        # Memory and performance
        if self.config.get('bf16', True):
            args.append('--bf16')
        if self.config.get('fp16', False):
            args.append('--fp16')
        if self.config.get('tf32', True):
            args.extend(['--tf32', 'True'])
        else:
            args.extend(['--tf32', 'False'])
        
        # Logging
        args.extend(['--logging_steps', str(self.config.get('logging_steps', 5))])
        args.extend(['--save_steps', str(self.config.get('save_steps', 500))])
        args.extend(['--save_total_limit', str(self.config.get('save_total_limit', 3))])
        
        # Evaluation
        if self.config.get('do_eval', False):
            args.extend(['--do_eval', 'True'])
            args.extend(['--eval_strategy', self.config.get('eval_strategy', 'steps')])  # FIXED: Use eval_strategy from config
            args.extend(['--eval_steps', str(self.config.get('eval_steps', 500))])
            args.extend(['--per_device_eval_batch_size', str(self.config.get('per_device_eval_batch_size', 2))])
        
        # Data settings
        args.extend(['--cutoff_len', str(self.config.get('cutoff_len', 4096))])
        args.extend(['--preprocessing_num_workers', str(self.config.get('preprocessing_num_workers', 8))])
        
        if self.config.get('max_samples', -1) > 0:
            args.extend(['--max_samples', str(self.config.get('max_samples'))])
        
        # Additional LLaMA Factory specific arguments
        if self.config.get('group_by_length', False):
            args.append('--group_by_length')
        if self.config.get('dataloader_pin_memory', True):
            args.append('--dataloader_pin_memory')
        if self.config.get('remove_unused_columns', True):
            args.append('--remove_unused_columns')
        
        # Reproducibility
        args.extend(['--seed', str(self.config.get('seed', 42))])
        
        # Resume from checkpoint
        resume_checkpoint = self.config.get('resume_from_checkpoint', '')
        if resume_checkpoint:
            args.extend(['--resume_from_checkpoint', resume_checkpoint])
        
        # DeepSpeed config
        deepspeed_config = self.config.get('deepspeed', '')
        if deepspeed_config:
            args.extend(['--deepspeed', deepspeed_config])
        
        # Plot loss
        args.append('--plot_loss')
        
        return args
    
    def validate_dataset(self) -> bool:
        """Validate that the selected dataset exists"""
        with open(self.dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
        
        dataset_name = self.config.get('dataset', 'reward_model_full')
        if dataset_name not in dataset_info:
            print(f"Error: Dataset '{dataset_name}' not found in dataset_info.json")
            print(f"Available datasets: {list(dataset_info.keys())}")
            return False
        
        # Check if dataset file exists
        dataset_file = dataset_info[dataset_name]['file_name']
        dataset_file_path = self.data_path / dataset_file
        if not dataset_file_path.exists():
            print(f"Error: Dataset file not found: {dataset_file_path}")
            return False
        
        print(f"✓ Using dataset: {dataset_name} ({dataset_file})")
        return True
    
    def print_training_info(self):
        """Print training configuration summary"""
        print("=" * 80)
        print("🚀 PROCESS REWARD MODEL TRAINING")
        print("=" * 80)
        print(f"Model: {self.config['model_name_or_path']}")
        print(f"Dataset: {self.config.get('dataset', 'reward_model_full')}")
        print(f"Template: {self.config.get('template', 'qwen')}")
        print(f"Fine-tuning: {self.config.get('finetuning_type', 'lora')}")
        print(f"Output: {self.output_path}")
        print(f"Epochs: {self.config.get('num_train_epochs', 3.0)}")
        print(f"Learning Rate: {self.config.get('learning_rate', 1e-4)}")
        print(f"Batch Size: {self.config.get('per_device_train_batch_size', 2)} × {self.config.get('gradient_accumulation_steps', 8)} = {self.config.get('per_device_train_batch_size', 2) * self.config.get('gradient_accumulation_steps', 8)}")
        print(f"Max Length: {self.config.get('cutoff_len', 4096)}")
        print(f"Precision: {'BF16' if self.config.get('bf16') else 'FP32'}")
        print("=" * 80)
    
    def run_training(self):
        """Execute the training process"""
        print("Preparing training...")
        
        # Setup monitoring
        self.setup_monitoring()
        
        # Validate dataset
        if not self.validate_dataset():
            return False
        
        # Print training info
        self.print_training_info()
        
        # Prepare arguments
        args = self.prepare_llamafactory_args()
        
        # Build command
        cmd = ['llamafactory-cli', 'train'] + args
        
        print("Executing command:")
        print(" ".join(cmd))
        print("=" * 80)
        
        try:
            # Execute training
            result = subprocess.run(cmd, check=True, capture_output=False)
            print("✓ Training completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Training failed with error code {e.returncode}")
            return False
        except KeyboardInterrupt:
            print("\n✗ Training interrupted by user")
            return False
        except Exception as e:
            print(f"✗ Training failed with error: {e}")
            return False
    
    def export_model(self):
        """Export the trained model"""
        export_dir = self.config.get('export_dir', './exported_model')
        export_path = Path(export_dir)
        
        print(f"\nExporting model to {export_path}...")
        
        cmd = [
            'llamafactory-cli', 'export',
            '--model_name_or_path', str(self.model_path),
            '--adapter_name_or_path', str(self.output_path),
            '--template', self.config.get('template', 'qwen'),
            '--finetuning_type', self.config.get('finetuning_type', 'lora'),
            '--export_dir', str(export_path),
            '--export_size', str(self.config.get('export_size', 2)),
            '--export_device', self.config.get('export_device', 'cpu')
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Model exported to {export_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Model export failed with error code {e.returncode}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Train Process Reward Model")
    parser.add_argument("--config", "-c", default="training_config.yaml",
                       help="Training configuration file")
    parser.add_argument("--dataset", "-d", type=str,
                       help="Override dataset name from config")
    parser.add_argument("--epochs", "-e", type=float,
                       help="Override number of epochs")
    parser.add_argument("--learning-rate", "-lr", type=float,
                       help="Override learning rate")
    parser.add_argument("--batch-size", "-bs", type=int,
                       help="Override batch size")
    parser.add_argument("--export", action="store_true",
                       help="Export model after training")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate configuration and exit")
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found!")
        return
    
    try:
        # Initialize trainer
        trainer = RewardModelTrainer(args.config)
        
        # Override config parameters if provided
        if args.dataset:
            trainer.config['dataset'] = args.dataset
        if args.epochs:
            trainer.config['num_train_epochs'] = args.epochs
        if args.learning_rate:
            trainer.config['learning_rate'] = args.learning_rate
        if args.batch_size:
            trainer.config['per_device_train_batch_size'] = args.batch_size
        
        # Validate configuration
        if not trainer.validate_dataset():
            return
        
        if args.validate_only:
            print("✓ Configuration validation passed!")
            trainer.print_training_info()
            return
        
        # Run training
        success = trainer.run_training()
        
        # Export model if requested and training succeeded
        if success and args.export:
            trainer.export_model()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()