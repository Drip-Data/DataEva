#!/bin/bash

# Process Reward Model Training Script
# This script provides easy commands to set up and run training

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Default values
CONFIG_FILE="training_config.yaml"
DATASET_NAME=""
EPOCHS=""
LEARNING_RATE=""
BATCH_SIZE=""
EXPORT_MODEL=false
VALIDATE_ONLY=false
CONVERT_DATA=true
INSTALL_DEPS=false

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --config FILE        Training configuration file (default: training_config.yaml)"
    echo "  -d, --dataset NAME       Dataset name to use for training"
    echo "  -e, --epochs NUM         Number of training epochs"
    echo "  -lr, --learning-rate NUM Learning rate"
    echo "  -bs, --batch-size NUM    Batch size per device"
    echo "  --export                 Export model after training"
    echo "  --validate-only          Only validate configuration"
    echo "  --no-convert             Skip data conversion step"
    echo "  --install-deps           Install required dependencies"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Basic training with default settings"
    echo "  $0 -d reward_model_clip -e 5         # Train on clip data for 5 epochs"
    echo "  $0 --validate-only                   # Just validate configuration"
    echo "  $0 --export -e 3                     # Train for 3 epochs and export model"
    echo "  $0 --install-deps                    # Install dependencies only"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -lr|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -bs|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --export)
            EXPORT_MODEL=true
            shift
            ;;
        --validate-only)
            VALIDATE_ONLY=true
            shift
            ;;
        --no-convert)
            CONVERT_DATA=false
            shift
            ;;
        --install-deps)
            INSTALL_DEPS=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install dependencies
install_dependencies() {
    print_step "Installing required dependencies..."
    
    # Check if pip is available
    if ! command_exists pip; then
        print_error "pip is not available. Please install Python and pip first."
        exit 1
    fi
    
    # Install SwanLab for monitoring
    print_status "Installing SwanLab..."
    pip install swanlab
    
    # Install other potential dependencies
    print_status "Installing other dependencies..."
    pip install pyyaml
    
    print_status "Dependencies installed successfully!"
}

# Function to check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    # Check if Python is available
    if ! command_exists python; then
        print_error "Python is not available. Please install Python first."
        exit 1
    fi
    
    # Check if llamafactory-cli is available
    if ! command_exists llamafactory-cli; then
        print_error "llamafactory-cli is not available. Please install LLaMA Factory first."
        print_error "You can install it with: pip install llamafactory[torch,metrics]"
        exit 1
    fi
    
    # Check if config file exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        print_error "Configuration file '$CONFIG_FILE' not found!"
        exit 1
    fi
    
    # Check if model directory exists
    if [[ ! -d "model" ]]; then
        print_error "Model directory './model' not found!"
        print_error "Please download your model to the './model' directory."
        exit 1
    fi
    
    # Check if finetune.jsonl exists
    if [[ ! -f "finetune.jsonl" ]]; then
        print_error "Training data file 'finetune.jsonl' not found!"
        exit 1
    fi
    
    print_status "Prerequisites check passed!"
}

# Function to convert data
convert_data() {
    print_step "Converting training data..."
    
    if [[ ! -f "convert_dataset.py" ]]; then
        print_error "convert_dataset.py not found!"
        exit 1
    fi
    
    python convert_dataset.py --input finetune.jsonl --output-dir data
    
    if [[ $? -eq 0 ]]; then
        print_status "Data conversion completed successfully!"
    else
        print_error "Data conversion failed!"
        exit 1
    fi
}

# Function to run training
run_training() {
    print_step "Starting training..."
    
    # Build training command
    TRAIN_CMD="python train.py --config $CONFIG_FILE"
    
    # Add optional parameters
    if [[ -n "$DATASET_NAME" ]]; then
        TRAIN_CMD="$TRAIN_CMD --dataset $DATASET_NAME"
    fi
    
    if [[ -n "$EPOCHS" ]]; then
        TRAIN_CMD="$TRAIN_CMD --epochs $EPOCHS"
    fi
    
    if [[ -n "$LEARNING_RATE" ]]; then
        TRAIN_CMD="$TRAIN_CMD --learning-rate $LEARNING_RATE"
    fi
    
    if [[ -n "$BATCH_SIZE" ]]; then
        TRAIN_CMD="$TRAIN_CMD --batch-size $BATCH_SIZE"
    fi
    
    if [[ "$EXPORT_MODEL" == true ]]; then
        TRAIN_CMD="$TRAIN_CMD --export"
    fi
    
    if [[ "$VALIDATE_ONLY" == true ]]; then
        TRAIN_CMD="$TRAIN_CMD --validate-only"
    fi
    
    print_status "Executing: $TRAIN_CMD"
    eval $TRAIN_CMD
}

# Function to show training info
show_training_info() {
    echo ""
    echo "========================================"
    echo "🚀 Process Reward Model Training"
    echo "========================================"
    echo "Configuration: $CONFIG_FILE"
    if [[ -n "$DATASET_NAME" ]]; then
        echo "Dataset: $DATASET_NAME"
    fi
    if [[ -n "$EPOCHS" ]]; then
        echo "Epochs: $EPOCHS"
    fi
    if [[ -n "$LEARNING_RATE" ]]; then
        echo "Learning Rate: $LEARNING_RATE"
    fi
    if [[ -n "$BATCH_SIZE" ]]; then
        echo "Batch Size: $BATCH_SIZE"
    fi
    echo "Export Model: $EXPORT_MODEL"
    echo "Validate Only: $VALIDATE_ONLY"
    echo "========================================"
    echo ""
}

# Function to monitor GPU usage
monitor_gpu() {
    if command_exists nvidia-smi; then
        print_status "GPU monitoring available. You can run 'nvidia-smi -l 1' in another terminal to monitor GPU usage."
    else
        print_warning "nvidia-smi not available. GPU monitoring not possible."
    fi
}

# Main execution
main() {
    # Show banner
    echo ""
    echo "🤖 Process Reward Model Training Framework"
    echo "==========================================="
    echo ""
    
    # Install dependencies if requested
    if [[ "$INSTALL_DEPS" == true ]]; then
        install_dependencies
        exit 0
    fi
    
    # Show training info
    show_training_info
    
    # Check prerequisites
    check_prerequisites
    
    # Convert data if needed
    if [[ "$CONVERT_DATA" == true ]]; then
        convert_data
    fi
    
    # Monitor GPU
    monitor_gpu
    
    # Run training
    run_training
    
    print_status "Training process completed!"
    
    # Show next steps
    echo ""
    echo "📋 Next steps:"
    echo "1. Check training logs in ./logs/"
    echo "2. Find trained model in ./saves/reward_model_training/"
    if [[ "$EXPORT_MODEL" == true ]]; then
        echo "3. Exported model available in ./exported_model/"
    else
        echo "3. Run with --export to export the trained model"
    fi
    echo "4. Use the model for inference or further training"
    echo ""
}

# Error handling
trap 'print_error "Script interrupted"; exit 1' INT

# Run main function
main