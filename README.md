# AI Agent Step-Level Evaluation System

A comprehensive evaluation system for AI agent trajectory data with advanced sequential clip evaluation and multi-model scoring. This system provides detailed, step-by-step evaluation of AI agent responses containing XML-formatted tool calls from multiple MCP (Model Context Protocol) servers, with context-aware evaluation and comprehensive category-level assessment.

##  Latest Updates (Version 2.0)

### New Advanced Evaluation Features
- **Sequential Context-Aware Evaluation**: Each clip is evaluated with complete context from all previous clips and their evaluation results
- **Enhanced Category Assessment**: Comprehensive MCP category evaluation after individual clip assessments  
- **Final Trajectory Scoring**: Overall trajectory assessment using all clips and category evaluations
- **Multi-Model Robustness**: Support for 6+ LLM providers with advanced score averaging and bias reduction
- **Integrated Testing**: Comprehensive test suite with `test_integration.py` for system verification

### Architecture Improvements
- **Simplified Service Layer**: Streamlined `evaluation_service.py` focused on core functionality
- **Unified Entry Point**: `run_evaluation.py` now handles all execution logic and multi-model configuration
- **Enhanced Data Structures**: Rich evaluation metadata with individual model tracking and category assessments


## 1. Project Overview

### Purpose and Goals

This system is designed to evaluate AI agent trajectories that utilize multiple tools to accomplish complex tasks. Instead of providing a single holistic score, the system breaks down agent responses into sequential **clips** and evaluates each clip independently using large language models (LLMs). This approach provides granular insights into:

- Tool usage effectiveness
- Reasoning quality at each step
- Task progression and completion
- Context utilization between steps

### Key Features

- **Advanced Sequential Evaluation**: Context-aware evaluation with comprehensive trajectory assessment
- **Multi-Tool Support**: Handles 4+ MCP servers (microsandbox, deepsearch, tavily, perform_web_task) 
- **Comprehensive Context Chaining**: Each evaluation includes summaries and results from ALL previous clips
- **Category-Level Assessment**: Detailed evaluation for each MCP tool category with specialized metrics
- **Final Trajectory Scoring**: Complete trajectory evaluation using all clips and category assessments
- **Multiple LLM Providers**: Supports 6+ providers (OpenAI, Google, Anthropic, DeepSeek, Kimi, Vertex AI)
- **Multi-Model Evaluation**: Parallel evaluation with multiple LLMs for reduced bias
- **Advanced Score Averaging**: Sophisticated aggregation with mode selection and reasoning preservation
- **Individual Model Tracking**: Comprehensive per-model result storage and analysis
- **Embedded Results**: Rich evaluations embedded directly into agent responses
- **Robust Error Handling**: Graceful failure handling without mock data generation
- **Integrated Testing**: Built-in test suite for system verification and API connectivity
- **Batch Processing**: Efficient evaluation of multiple files with progress tracking

### Supported Tool Types

| Tool Type | Description | Use Cases | Evaluation Metrics |
|-----------|-------------|-----------|-------------------|
| **MicroSandbox** | Code execution environment | Python scripting, data analysis, algorithm implementation | Code Correctness, Computational Efficiency, Result Interpretation |
| **DeepSearch** | Research and information gathering | Literature review, fact-finding, comprehensive research | Information Relevance, Source Quality, Information Synthesis |
| **Tavily** | Advanced web search and extraction | Real-time search, content extraction, web crawling | Information Relevance, Content Extraction, Goal Achievement |
| **Perform Web Task** | Comprehensive web automation | Google search, web navigation, structured data extraction | Tool Use Quality, Content Extraction, Interaction Quality |

## 2. System Architecture

The system follows a clear separation of concerns with two main phases:

```
Raw Agent Data → Preprocessing → Clean Data → Evaluation → Evaluated Results
     ↓                ↓              ↓           ↓            ↓
 demo01.jsonl → preprocess_agent → preprocessed → step_level → final_results
                    _data.py        _data.jsonl   _evaluator     .jsonl
                                                    .py
```

### 2.1 Enhanced Workflow with File Splitting and Batch Processing

```
Large Dataset → Preprocessing → Split Files → Batch Evaluation → Aggregated Results
     ↓               ↓              ↓              ↓                ↓
demo02.jsonl → preprocess_agent → predemo02/ → run_evaluation → multiple_eva.jsonl
  (97 tasks)      _data.py        ├─demo0201.jsonl  --batch-folder   ├─demo0201_eva.jsonl
                 --split-size 10  ├─demo0202.jsonl                   ├─demo0202_eva.jsonl
                                  ├─...                               ├─...
                                  └─demo0210.jsonl                   └─demo0210_eva.jsonl
                                    (7 tasks)                         (evaluated)
```

### Core Components

1. **`run_evaluation.py`**: Main CLI entry point and workflow orchestration
2. **`evaluation_service.py`**: Service layer managing the entire evaluation pipeline
3. **`preprocess_agent_data.py`**: Data cleaning, validation, and normalization
4. **`step_level_evaluator.py`**: Core evaluation logic and clip processing
5. **`llm_api_clients.py`**: LLM provider integrations and API management

## 3. Data Preprocessing Process

### Preprocessing Indicators and Rules

The preprocessing phase applies three critical filters to ensure data quality:

#### Tool-Call Frequency Control (β-Threshold)
- **Purpose**: Remove samples with excessive tool calls that may indicate stuck loops or inefficient behavior
- **Default Threshold**: 15 tool calls per sample
- **Implementation**: Counts all XML tags matching the four MCP server patterns
- **Action**: Samples exceeding the threshold are discarded with logging

#### Duplicate Tool-Call Detection
- **Purpose**: Eliminate redundant or repeated tool calls that don't contribute to task progression
- **Detection Method**: 
  - Identifies consecutive identical tool calls within the same server type
  - Compares tool call content for exact matches
  - Checks for identical call patterns with same parameters
- **Action**: Samples with detected duplicates are removed

#### XML Format Normalization
- **Purpose**: Ensure proper XML structure for reliable parsing
- **Validation Checks**:
  - Matching opening and closing tags
  - Proper tag nesting
  - Valid XML syntax
- **Auto-Correction**: Attempts to fix minor issues like missing closing tags
- **Metadata**: Records correction actions in sample metadata

### Preprocessing Operation Procedure

#### Step 1: Data Loading
```python
# Load JSONL files containing agent trajectories
data = preprocessor.load_jsonl("data/demo01.jsonl")
```

#### Step 2: Sample Processing
For each sample in the dataset:

1. **Extract XML Tags**: Parse `raw_response` field to identify all XML-like structures
2. **Count Tool Calls**: Identify and count calls to the four MCP servers:
   - `<microsandbox>...</microsandbox>`
   - `<deepsearch>...</deepsearch>`
   - `<browser_use>...</browser_use>`
   - `<search_tool>...</search_tool>`

3. **Apply Frequency Filter**: 
   ```python
   tool_call_count = count_tool_calls(tag_content)
   if tool_call_count > beta_threshold:
       # Discard sample
       stats['removed_frequency'] += 1
       continue
   ```

4. **Detect Duplicates**:
   ```python
   if detect_duplicate_tool_calls(tag_content):
       # Discard sample
       stats['removed_duplicates'] += 1
       continue
   ```

5. **Validate and Normalize XML**:
   ```python
   is_valid, corrected_response = validate_xml_format(raw_response)
   if not is_valid:
       sample['raw_response'] = corrected_response
       sample['preprocessing_notes'] = 'Format issues detected and corrected'
   ```

6. **Add Metadata**:
   ```python
   sample['preprocessing_metadata'] = {
       'tool_call_count': tool_call_count,
       'has_duplicates': False,
       'format_corrected': not is_valid,
       'tag_analysis': {tag: len(content) for tag, content in tag_content.items()}
   }
   ```

#### Step 3: Output Generation
- Save cleaned samples to new JSONL file
- Generate preprocessing statistics
- Log processing summary

### File Splitting Feature

#### Purpose and Usage
The preprocessing system now supports splitting large JSONL files into smaller chunks for easier processing and evaluation. This is particularly useful when:
- Processing large datasets that exceed memory limits
- Distributing evaluation tasks across multiple sessions
- Managing computational resources more efficiently

#### Splitting Logic
```python
# Example: demo02.jsonl with 97 tasks, split_size=10
# Results in: predemo02/demo0201.jsonl, predemo02/demo0202.jsonl, ..., predemo02/demo0210.jsonl
# Last file (demo0210.jsonl) contains 7 tasks (97 % 10 = 7)

num_files = math.ceil(len(processed_data) / split_size)
for i in range(num_files):
    start_idx = i * split_size
    end_idx = min((i + 1) * split_size, len(processed_data))
    chunk = processed_data[start_idx:end_idx]
    output_filename = f"{base_name}{i+1:02d}.jsonl"
```

#### Directory Structure
```
data/
├── demo02.jsonl                 # Original file (97 tasks)
├── demo02_preprocessed.jsonl    # Main preprocessed file
└── predemo02/                   # Split files directory
    ├── demo0201.jsonl          # Tasks 1-10
    ├── demo0202.jsonl          # Tasks 11-20
    ├── ...
    └── demo0210.jsonl          # Tasks 91-97 (7 tasks)
```

### Preprocessing Operation Instructions

#### Command Syntax
```bash
python src/run_evaluation.py \
    --input <input_file_or_directory> \
    --preprocess-only \
    --output <output_file> \
    [--beta-threshold <number>] \
    [--verbose]

# For standalone preprocessing with splitting
python src/preprocess_agent_data.py \
    --input <input_file> \
    --output <output_file> \
    --split-size <number> \
    [--beta-threshold <number>]
```

#### Examples
```bash
# Basic preprocessing
python src/run_evaluation.py \
    --input data/demo01.jsonl \
    --preprocess-only \
    --output data/preprocessed.jsonl

# Custom threshold and verbose logging
python src/run_evaluation.py \
    --input data/raw_data/ \
    --preprocess-only \
    --output data/clean_data.jsonl \
    --beta-threshold 20 \
    --verbose

# Preprocessing with file splitting
python src/preprocess_agent_data.py \
    --input data/demo02.jsonl \
    --output data/demo02_preprocessed.jsonl \
    --split-size 10 \
    --beta-threshold 15

# Batch evaluation of split files
python src/run_evaluation.py \
    --input data/predemo02/ \
    --provider openai \
    --api-key YOUR_KEY \
    --batch-folder
```


### Supported LLM Providers

The system now supports **6 major LLM providers** with their respective models:

| Provider | Default Model | Alternative Models | API Type |
|----------|---------------|-------------------|----------|
| **OpenAI** | `gpt-4o` | `gpt-4o-mini`, `gpt-4-turbo` | OpenAI API |
| **Google** | `gemini-1.5-pro` | `gemini-1.5-flash`, `gemini-1.0-pro` | Google AI API |
| **Anthropic** | `claude-3-5-sonnet-20241022` | `claude-3-haiku-20240307`, `claude-3-opus-20240229` | Anthropic API |
| **DeepSeek** | `deepseek-chat` | `deepseek-coder` | Custom REST API |
| **Kimi** | `moonshot-v1-8k` | `moonshot-v1-32k` | Custom REST API |
| **Vertex AI** | `gemini-1.5-pro` | `claude-3-5-sonnet@20241022`, Various Model Garden models | Google Cloud API |

#### 5.2.1 Vertex AI Platform Support

Vertex AI provides a unified platform to access multiple model types:
- **Gemini models**: Direct access to Google's latest models
- **Claude models**: Anthropic models through Vertex AI
- **Open-source models**: Access to models like Qwen, DeepSeek through Model Garden
- **Custom endpoints**: Support for deployed custom models

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Required Python packages (see requirements.txt)
- API keys for LLM providers (optional for preprocessing-only mode)

### Installation Steps
```bash
# Clone the repository
git clone <repository-url>
cd dataeva

# Install dependencies
pip install -r requirements.txt

# Verify installation
python src/run_evaluation.py --help
```

### 6.3 API Key Configuration
```bash
# Option 1: Environment variables (recommended)
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Option 2: Command line arguments
python src/run_evaluation.py --api-key YOUR_KEY --provider openai ...
```

## Usage Examples

### Preprocessing Only
```bash
# Basic preprocessing
python src/run_evaluation.py \
    --input data/demo01.jsonl \
    --preprocess-only \
    --output data/preprocessed.jsonl

# Custom threshold and verbose logging
python src/run_evaluation.py \
    --input data/raw_data/ \
    --preprocess-only \
    --output data/clean_data.jsonl \
    --beta-threshold 20 \
    --verbose
```

###  Evaluation Only (Preprocessed Data)
```bash
# Evaluate with OpenAI
python src/run_evaluation.py \
    --input data/preprocessed.jsonl \
    --provider openai \
    --api-key YOUR_OPENAI_KEY \
    --output data/evaluated.jsonl

# Evaluate with custom batch size and rate limiting
python src/run_evaluation.py \
    --input data/preprocessed.jsonl \
    --provider anthropic \
    --model claude-3-sonnet-20240229 \
    --batch-size 2 \
    --rate-limit 2.0 \
    --output data/evaluated.jsonl
```

###  Full Pipeline
```bash
# Complete preprocessing and evaluation
python src/run_evaluation.py \
    --input data/demo01.jsonl \
    --provider openai \
    --full-pipeline \
    --output data/final_results.jsonl

# Full pipeline with Google Gemini
python src/run_evaluation.py \
    --input data/raw_trajectories/ \
    --provider google \
    --model gemini-1.5-pro \
    --full-pipeline \
    --batch-size 3 \
    --output data/gemini_results.jsonl
```

### Batch Folder Evaluation
```bash
# Evaluate all JSONL files in a folder
python src/run_evaluation.py \
    --input data/predemo02/ \
    --provider openai \
    --api-key YOUR_KEY \
    --batch-folder \
    --output data/batch_results.jsonl

# Batch evaluation with custom settings
python src/run_evaluation.py \
    --input data/split_files/ \
    --provider anthropic \
    --model claude-3-5-sonnet-20241022 \
    --batch-folder \
    --batch-size 2 \
    --rate-limit 1.5 \
    --full-pipeline

# Batch preprocessing with file splitting
python src/preprocess_agent_data.py \
    --input data/large_dataset.jsonl \
    --split-size 20 \
    --beta-threshold 12 \
    --output data/large_dataset_preprocessed.jsonl
```

###  Multi-Model Evaluation
```bash
# Interactive multi-model configuration (recommended)
python run_evaluation.py \
    --input data/demo01.jsonl \
    --multi-model \
    --full-pipeline \
    --output data/multi_model_results.jsonl

# Multi-model with batch processing
python run_evaluation.py \
    --input data/predemo02/ \
    --multi-model \
    --batch-folder \
    --batch-size 2 \
    --rate-limit 2.0

# Multi-model evaluation only (assumes preprocessed data)
python run_evaluation.py \
    --input data/preprocessed.jsonl \
    --multi-model \
    --output data/multi_results.jsonl
```
