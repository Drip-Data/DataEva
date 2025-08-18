# Process Reward Model Training Framework

This framework provides a complete solution for fine-tuning process reward models using LLaMA Factory. It's specifically designed for training models that can evaluate AI agent performance step-by-step.

##  Framework Structure

```
finetune/
├── finetune.jsonl              # Your training data
├── model/                      # Pre-downloaded model directory
├── data/                       # Converted training datasets (auto-generated)
├── saves/                      # Training outputs and checkpoints (auto-generated)
├── logs/                       # Training logs (auto-generated)
├── exported_model/             # Exported model for deployment (auto-generated)
├── convert_dataset.py          # Data conversion script
├── training_config.yaml        # Training configuration
├── train.py                    # Main training script
├── run_training.sh             # Convenient shell script
└── explanation.md              # This guide
```

##  Quick Start

### 1. Prepare Your Environment

Make sure you have:
- LLaMA Factory installed and working
- Your model downloaded to `./model/` directory
- Your training data in `finetune.jsonl`

### 2. Install Dependencies (if needed)

```bash
# Install SwanLab for monitoring
./run_training.sh --install-deps
```

### 3. Run Training

**Basic training with default settings:**
```bash
./run_training.sh
```

**Training with specific parameters:**
```bash
./run_training.sh -d reward_model_full -e 5 --export
```

**Validate configuration only:**
```bash
./run_training.sh --validate-only
```

##  Dataset Variants

The framework automatically generates multiple dataset variants from your `finetune.jsonl`:

### By Evaluation Type
- `reward_model_clip` - Only clip-level evaluations
- `reward_model_category` - Only category-level evaluations  
- `reward_model_final` - Only final trajectory evaluations
- `reward_model_full` - All evaluation types (default)

### By Model Name
- `reward_model_deepseek_chat` - Only DeepSeek Chat evaluations
- `reward_model_kimi_k2_0711_preview` - Only Kimi evaluations
- *(Additional variants based on your data)*

### By Task ID
- `reward_model_webshaper_qa_0` - Task-specific training
- *(Additional variants based on your data)*

##  Configuration

### Training Configuration (`training_config.yaml`)

Key parameters you can modify:

```yaml
# Dataset selection
dataset: reward_model_full  # Change to use different variants

# Model settings
model_name_or_path: ./model
template: qwen  # Change based on your model (qwen, llama3, etc.)

# Training hyperparameters
learning_rate: 1.0e-4
num_train_epochs: 3.0
per_device_train_batch_size: 2
gradient_accumulation_steps: 8

# LoRA settings
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.1

# Memory optimization
bf16: true  # Use bfloat16 for A100
cutoff_len: 4096
```

### Template Selection

Choose the correct template based on your model:
- `qwen` - For Qwen models
- `llama3` - For Llama 3 models
- `mistral` - For Mistral models
- `chatglm3` - For ChatGLM models

##  Training Process

### Step 1: Data Conversion
```bash
python convert_dataset.py --input finetune.jsonl --output-dir data
```

This creates:
- Multiple dataset variants in `./data/`
- `dataset_info.json` configuration file

### Step 2: Training Execution
```bash
python train.py --config training_config.yaml
```

Or use the convenience script:
```bash
./run_training.sh -d reward_model_full -e 3 --export
```

### Step 3: Model Export (Optional)
```bash
python train.py --config training_config.yaml --export
```

##  Monitoring

### SwanLab Integration

The framework uses SwanLab for training monitoring:

1. **Automatic Initialization**: SwanLab is automatically initialized when available
2. **Project Tracking**: All runs are tracked under the `reward_model_training` project
3. **Configuration Logging**: All hyperparameters are logged for reproducibility

### GPU Monitoring

Monitor GPU usage during training:
```bash
nvidia-smi -l 1
```

### Training Logs

Check training progress:
```bash
tail -f ./logs/trainer_log.jsonl
```

##  Advanced Usage

### Custom Dataset Filtering

Create custom datasets with specific criteria:

```bash
# Train only on clip evaluations from DeepSeek
python convert_dataset.py \
    --input finetune.jsonl \
    --output-dir data \
    --evaluation-types clip \
    --model-names deepseek-chat

# Train on specific tasks
python convert_dataset.py \
    --input finetune.jsonl \
    --output-dir data \
    --task-ids webshaper_qa_0
```

### Training Parameter Override

Override configuration parameters at runtime:

```bash
python train.py \
    --config training_config.yaml \
    --dataset reward_model_clip \
    --epochs 5 \
    --learning-rate 2e-4 \
    --batch-size 4
```

### Resume Training

Resume from a checkpoint:

```yaml
# In training_config.yaml
resume_from_checkpoint: "./saves/reward_model_training/checkpoint-500"
```

##  Troubleshooting

### Common Issues

1. **"llamafactory-cli not found"**
   ```bash
   pip install llamafactory[torch,metrics]
   ```

2. **"Model directory not found"**
   - Download your model to `./model/` directory
   - Ensure the model files are directly in `./model/`, not in a subdirectory

3. **"Dataset not found"**
   - Run data conversion first: `python convert_dataset.py`
   - Check that `./data/dataset_info.json` exists

4. **Out of Memory Error**
   - Reduce `per_device_train_batch_size` in config
   - Increase `gradient_accumulation_steps` to maintain effective batch size
   - Reduce `cutoff_len` if sequences are too long

5. **SwanLab Issues**
   - Install: `pip install swanlab`
   - The framework will work without SwanLab, just without monitoring

### Memory Optimization

For A100 40GB GPU:
- **Recommended**: `batch_size=2`, `gradient_accumulation_steps=8`
- **If OOM**: `batch_size=1`, `gradient_accumulation_steps=16`
- **For larger models**: Enable DeepSpeed ZeRO Stage 2

### Performance Tuning

1. **Faster Training**:
   - Use `bf16=true` for A100
   - Set `tf32=true`
   - Increase `preprocessing_num_workers`

2. **Better Results**:
   - Increase `num_train_epochs`
   - Lower `learning_rate` for stability
   - Adjust `lora_rank` based on model complexity

##  Training Checklist

Before starting training:

- [ ] Model downloaded to `./model/`
- [ ] Training data in `finetune.jsonl`
- [ ] LLaMA Factory installed
- [ ] SwanLab installed (optional)
- [ ] GPU available and working
- [ ] Configuration file reviewed
- [ ] Dataset converted successfully

## 🎓 Model Usage

After training, use your model for evaluation:

### With LLaMA Factory CLI
```bash
llamafactory-cli chat \
    --model_name_or_path ./model \
    --adapter_name_or_path ./saves/reward_model_training \
    --template qwen \
    --finetuning_type lora
```

### With Exported Model
```bash
llamafactory-cli chat \
    --model_name_or_path ./exported_model \
    --template qwen
```

### Programmatic Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your trained model
tokenizer = AutoTokenizer.from_pretrained("./exported_model")
model = AutoModelForCausalLM.from_pretrained("./exported_model")

# Use for evaluation
inputs = tokenizer("Your evaluation prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review LLaMA Factory documentation
3. Verify your environment setup
4. Check GPU memory and availability

## 🔄 Continuous Training

For production use:

1. **Regular Updates**: Re-train with new evaluation data
2. **Version Control**: Keep track of different model versions
3. **Performance Monitoring**: Monitor model performance on validation sets
4. **A/B Testing**: Compare different model versions in production

---

**Happy Training! **