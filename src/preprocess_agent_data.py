import json
import re
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
from pathlib import Path
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AgentDataPreprocessor:
    """
    A comprehensive preprocessor for AI agent trajectory data with tool-call analysis.
    
    This class handles three main preprocessing steps:
    1. Tool-call Frequency Control: Remove samples with excessive tool calls
    2. Duplicate Tool-call Removal: Eliminate redundant tool calls
    3. Format Normalization: Standardize XML tag formats and ensure proper pairing
    4. File Splitting: Split large JSONL files into smaller chunks (optional)
    """
    
    def __init__(self, beta_threshold: int = 15, split_size: int = 0):
        """
        Initialize the preprocessor.
        
        Args:
            beta_threshold: Maximum allowed tool calls before sample removal (default: 15)
            split_size: Number of samples per split file (0 = no splitting, default: 0)
        """
        self.beta_threshold = beta_threshold
        self.split_size = split_size
        
        # Define the four MCP servers and their XML tags
        self.mcp_servers = {
            'microsandbox': ['microsandbox_execute', 'microsandbox_install_package', 
                           'microsandbox_list_sessions', 'microsandbox_close_session',
                           'microsandbox_cleanup_expired', 'microsandbox_get_performance_stats',
                           'microsandbox_get_health_status'],
            'deepsearch': ['research', 'quick_research', 'comprehensive_research'],
            'browser_use': ['browser_use_execute_task', 'browser_navigate', 'browser_search_google',
                          'browser_click_element', 'browser_input_text', 'browser_screenshot'],
            'search_tool': ['search_file_content', 'list_code_definitions', 'analyze_tool_needs',
                          'search_and_install_tools']
        }
        
        # Define all XML tags that may appear in raw_response
        self.xml_tags = {
            'reasoning': ['think'],
            'tool_servers': ['microsandbox', 'deepsearch', 'browser_use', 'search_tool'],
            'results': ['result', 'answer'],
            'execution': ['execute_tools'],
            'logic': ['parallel', 'sequential']
        }
        
        # Statistics tracking
        self.stats = {
            'total_samples': 0,
            'removed_frequency': 0,
            'removed_duplicates': 0,
            'format_issues_fixed': 0,
            'valid_samples': 0,
            'split_files_created': 0
        }
    
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """Load JSONL file and return list of dictionaries."""
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error at line {line_num}: {e}")
                        continue
            logger.info(f"Successfully loaded {len(data)} samples from {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading JSONL file: {e}")
            return []
    
    def extract_xml_tags(self, raw_response: str) -> Dict[str, List[str]]:
        """
        Extract all XML tags and their content from raw_response.
        
        Args:
            raw_response: The raw response string containing XML tags
            
        Returns:
            Dictionary mapping tag names to their content
        """
        tag_content = defaultdict(list)
        
        # Extract all XML-like tags using regex
        # Pattern to match opening and closing tags with content
        pattern = r'<(\w+)(?:\s[^>]*)?>(.*?)</\1>'
        matches = re.findall(pattern, raw_response, re.DOTALL | re.IGNORECASE)
        
        for tag_name, content in matches:
            tag_content[tag_name.lower()].append(content.strip())
        
        # Also extract self-closing tags
        self_closing_pattern = r'<(\w+)(?:\s[^>]*)?/>'
        self_closing_matches = re.findall(self_closing_pattern, raw_response, re.IGNORECASE)
        
        for tag_name in self_closing_matches:
            tag_content[tag_name.lower()].append('')
        
        return dict(tag_content)
    
    def count_tool_calls(self, tag_content: Dict[str, List[str]]) -> int:
        """
        Count the total number of tool calls in the response.
        
        Args:
            tag_content: Dictionary mapping tag names to their content
            
        Returns:
            Total number of tool calls
        """
        tool_call_count = 0
        
        # Count MCP server tool calls
        for server in self.xml_tags['tool_servers']:
            tool_call_count += len(tag_content.get(server, []))
        
        # Count execute_tools as additional tool calls
        tool_call_count += len(tag_content.get('execute_tools', []))
        
        return tool_call_count
    
    def detect_duplicate_tool_calls(self, tag_content: Dict[str, List[str]]) -> bool:
        """
        Detect if there are duplicate tool calls in the response.
        
        Args:
            tag_content: Dictionary mapping tag names to their content
            
        Returns:
            True if duplicates are found, False otherwise
        """
        # Check for duplicate tool calls within each server type
        for server in self.xml_tags['tool_servers']:
            server_calls = tag_content.get(server, [])
            if len(server_calls) > 1:
                # Check for identical consecutive calls
                for i in range(len(server_calls) - 1):
                    if server_calls[i] == server_calls[i + 1]:
                        return True
                
                # Check for identical calls with same content pattern
                unique_calls = set(server_calls)
                if len(unique_calls) < len(server_calls):
                    return True
        
        return False
    
    def validate_xml_format(self, raw_response: str) -> Tuple[bool, str]:
        """
        Validate and fix XML format issues in the response.
        
        Args:
            raw_response: The raw response string
            
        Returns:
            Tuple of (is_valid, corrected_response)
        """
        corrected_response = raw_response
        format_issues = []
        
        # Check for proper tag pairing
        all_tags = set()
        for category in self.xml_tags.values():
            if isinstance(category, list):
                all_tags.update(category)
        
        for tag in all_tags:
            # Find opening tags
            opening_pattern = f'<{tag}(?:\\s[^>]*)?>'
            opening_matches = re.findall(opening_pattern, corrected_response, re.IGNORECASE)
            
            # Find closing tags
            closing_pattern = f'</{tag}>'
            closing_matches = re.findall(closing_pattern, corrected_response, re.IGNORECASE)
            
            # Check for mismatched tags
            if len(opening_matches) != len(closing_matches):
                format_issues.append(f"Mismatched {tag} tags: {len(opening_matches)} opening, {len(closing_matches)} closing")
                
                # Try to fix by adding missing closing tags
                if len(opening_matches) > len(closing_matches):
                    missing_closing = len(opening_matches) - len(closing_matches)
                    for _ in range(missing_closing):
                        corrected_response += f'</{tag}>'
        
        # Check for malformed XML structure
        try:
            # Try to parse as XML (this is a basic check)
            # We'll wrap the content to make it valid XML for testing
            test_xml = f"<root>{corrected_response}</root>"
            # Replace any remaining unclosed tags
            test_xml = re.sub(r'<(\w+)(?:\s[^>]*)?(?<!/)>', r'<\1>', test_xml)
            
        except ParseError as e:
            format_issues.append(f"XML parsing error: {e}")
        
        # Fix common formatting issues
        # 1. Fix malformed self-closing tags
        corrected_response = re.sub(r'<(\w+)(?:\s[^>]*)?(?<!/)>', r'<\1>', corrected_response)
        
        # 2. Ensure proper spacing around tags
        corrected_response = re.sub(r'<(\w+)>', r'<\1>', corrected_response)
        corrected_response = re.sub(r'</(\w+)>', r'</\1>', corrected_response)
        
        is_valid = len(format_issues) == 0
        
        if format_issues:
            logger.debug(f"Format issues found: {format_issues}")
        
        return is_valid, corrected_response
    
    def preprocess_sample(self, sample: Dict) -> Tuple[bool, Dict]:
        """
        Preprocess a single sample applying all three preprocessing steps.
        
        Args:
            sample: Dictionary containing sample data
            
        Returns:
            Tuple of (is_valid, processed_sample)
        """
        if 'raw_response' not in sample:
            logger.warning(f"Sample {sample.get('task_id', 'unknown')} missing raw_response field")
            return False, sample
        
        raw_response = sample['raw_response']
        
        # Step 1: Extract XML tags and content
        tag_content = self.extract_xml_tags(raw_response)
        
        # Step 2: Tool-call Frequency Control
        tool_call_count = self.count_tool_calls(tag_content)
        if tool_call_count > self.beta_threshold:
            logger.info(f"Sample {sample.get('task_id', 'unknown')} removed: {tool_call_count} tool calls > {self.beta_threshold}")
            self.stats['removed_frequency'] += 1
            return False, sample
        
        # Step 3: Duplicate Tool-call Removal
        if self.detect_duplicate_tool_calls(tag_content):
            logger.info(f"Sample {sample.get('task_id', 'unknown')} removed: duplicate tool calls detected")
            self.stats['removed_duplicates'] += 1
            return False, sample
        
        # Step 4: Format Normalization
        is_valid_format, corrected_response = self.validate_xml_format(raw_response)
        if not is_valid_format:
            sample['raw_response'] = corrected_response
            sample['preprocessing_notes'] = 'Format issues detected and corrected'
            self.stats['format_issues_fixed'] += 1
            logger.debug(f"Sample {sample.get('task_id', 'unknown')} format issues fixed")
        
        # Add preprocessing metadata
        sample['preprocessing_metadata'] = {
            'tool_call_count': tool_call_count,
            'has_duplicates': False,  # Since we would have removed it otherwise
            'format_corrected': not is_valid_format,
            'tag_analysis': {tag: len(content) for tag, content in tag_content.items()}
        }
        
        return True, sample
    
    def preprocess_data(self, data: List[Dict]) -> List[Dict]:
        """
        Preprocess all samples in the dataset.
        
        Args:
            data: List of sample dictionaries
            
        Returns:
            List of valid preprocessed samples
        """
        self.stats['total_samples'] = len(data)
        valid_samples = []
        
        logger.info(f"Starting preprocessing of {len(data)} samples...")
        
        for sample in data:
            is_valid, processed_sample = self.preprocess_sample(sample)
            if is_valid:
                valid_samples.append(processed_sample)
                self.stats['valid_samples'] += 1
        
        logger.info(f"Preprocessing completed. {len(valid_samples)} valid samples remaining.")
        return valid_samples
    
    def split_data_into_files(self, processed_data: List[Dict], input_file_path: str) -> List[str]:
        """
        Split processed data into multiple JSONL files.
        
        Args:
            processed_data: List of preprocessed samples
            input_file_path: Original input file path for naming convention
            
        Returns:
            List of output file paths created
        """
        if self.split_size <= 0 or len(processed_data) <= self.split_size:
            logger.info("No splitting required (split_size=0 or data size <= split_size)")
            return []
        
        # Create output directory
        input_path = Path(input_file_path)
        base_name = input_path.stem  # e.g., "demo02" from "demo02.jsonl"
        output_dir = input_path.parent / f"pre{base_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate number of files needed
        num_files = math.ceil(len(processed_data) / self.split_size)
        output_files = []
        
        logger.info(f"Splitting {len(processed_data)} samples into {num_files} files of {self.split_size} samples each")
        
        for i in range(num_files):
            start_idx = i * self.split_size
            end_idx = min((i + 1) * self.split_size, len(processed_data))
            chunk = processed_data[start_idx:end_idx]
            
            # Create output filename: xxxx01.jsonl, xxxx02.jsonl, etc.
            output_filename = f"{base_name}{i+1:02d}.jsonl"
            output_path = output_dir / output_filename
            
            # Save chunk to file
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    for sample in chunk:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                
                output_files.append(str(output_path))
                self.stats['split_files_created'] += 1
                logger.info(f"Created {output_filename} with {len(chunk)} samples")
                
            except Exception as e:
                logger.error(f"Error saving split file {output_path}: {e}")
                continue
        
        logger.info(f"Successfully created {len(output_files)} split files in {output_dir}")
        return output_files
    
    def save_preprocessed_data(self, processed_data: List[Dict], output_path: str):
        """
        Save preprocessed data to JSONL file.
        
        Args:
            processed_data: List of preprocessed samples
            output_path: Path to save the preprocessed data
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in processed_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logger.info(f"Preprocessed data saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving preprocessed data: {e}")
    
    def save_statistics(self, stats_path: str):
        """
        Save preprocessing statistics to JSON file.
        
        Args:
            stats_path: Path to save the statistics
        """
        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
            logger.info(f"Statistics saved to {stats_path}")
        except Exception as e:
            logger.error(f"Error saving statistics: {e}")
    
    def print_statistics(self):
        """Print preprocessing statistics."""
        print("\n" + "="*50)
        print("PREPROCESSING STATISTICS")
        print("="*50)
        print(f"Total samples: {self.stats['total_samples']}")
        print(f"Valid samples: {self.stats['valid_samples']}")
        print(f"Removed (frequency): {self.stats['removed_frequency']}")
        print(f"Removed (duplicates): {self.stats['removed_duplicates']}")
        print(f"Format issues fixed: {self.stats['format_issues_fixed']}")
        print(f"Split files created: {self.stats['split_files_created']}")
        print(f"Success rate: {self.stats['valid_samples']/self.stats['total_samples']*100:.2f}%")
        print("="*50)


def main():
    """Main function to run the preprocessing pipeline."""
    
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Preprocess AI agent trajectory data')
    parser.add_argument('--input', '-i', type=str, default="data/demo01.jsonl",
                       help='Input JSONL file path')
    parser.add_argument('--output', '-o', type=str, default="data/demo01_preprocessed.jsonl",
                       help='Output JSONL file path')
    parser.add_argument('--beta-threshold', type=int, default=15,
                       help='Maximum tool calls per sample (default: 15)')
    parser.add_argument('--split-size', type=int, default=0,
                       help='Number of samples per split file (0 = no splitting)')
    parser.add_argument('--stats', type=str, default="data/preprocessing_stats.json",
                       help='Statistics output file path')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = AgentDataPreprocessor(
        beta_threshold=args.beta_threshold,
        split_size=args.split_size
    )
    
    # Load data
    data = preprocessor.load_jsonl(args.input)
    
    if not data:
        logger.error("No data loaded. Exiting.")
        return
    
    # Preprocess data
    processed_data = preprocessor.preprocess_data(data)
    
    # Save main preprocessed file
    preprocessor.save_preprocessed_data(processed_data, args.output)
    
    # Split data if requested
    if args.split_size > 0:
        split_files = preprocessor.split_data_into_files(processed_data, args.input)
        if split_files:
            logger.info(f"Created {len(split_files)} split files:")
            for file_path in split_files:
                logger.info(f"  - {file_path}")
    
    # Save statistics
    preprocessor.save_statistics(args.stats)
    
    # Print statistics
    preprocessor.print_statistics()


if __name__ == "__main__":
    main() 