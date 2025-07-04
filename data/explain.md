## MCPserver(four)
1. MicroSandbox
  - xml tag <microsandbox></microsandbox>
  - The specific content of the tools
    - microsandbox_execute - 
    - microsandbox_install_package - 
    - microsandbox_list_sessions - 
    - microsandbox_close_session - 
    - microsandbox_cleanup_expired - 
    - microsandbox_get_performance_stats - 
    - microsandbox_get_health_status 
  - core evaluation and score metric
    - Code Correctness (0-1): Syntactic and logical correctness of generated code
    - Computational Efficiency (0-1): Appropriateness of algorithmic approach
    - Error Handling (0-1): Proper handling of edge cases and errors
    - Result Interpretation (0-1): Accurate interpretation and integration of execution results
2. DeepSearch 
  - xml tag <deepsearch></deepsearch>
  - The specific content of the tools
    - research
    - quick_research
    - comprehensive_research
  - core evaluation and score metric
    - Search Depth Appropriateness (0-1): Whether the depth of search matches task complexity
    - Query Refinement Quality (0-1): Effectiveness of iterative query improvements
    - Source Diversity (0-1): Breadth of information sources consulted
    - Synthesis Quality (0-1): Ability to synthesize information from multiple sources
3. Browser Use 
  - xml tag <browser_use></browser_use>
  - The specific content of the tools
    - browser_use_execute_task - AI browser task execution
    - browser_navigate - Navigate to the website
    - browser_search_google - Google Search
    - browser_click_element - Click on the page element
    - browser_input_text - input text
    - browser_screenshot - screen shot
    - 26 browser automation tools, etc
  - core evaluation and score metric
    - Query Relevance (0-1): How well the search query relates to the reasoning context
    - Information Extraction Quality (0-1): Effectiveness of extracting relevant information from results
    - Navigation Efficiency (0-1): Appropriateness of website selection and browsing strategy
    - Content Integration (0-1): How well retrieved information is integrated into reasoning
4. Search Tool 
  - xml tag <search_tool></search_tool>
  - The specific content of the tools
    - search_file_content - 搜索文件内容
    - list_code_definitions - 列出代码定义
    - analyze_tool_needs - 分析工具需求
    - search_and_install_tools - 搜索安装工具
  - core evaluation and score metric
    - Tool Selection Accuracy (0-1): Appropriateness of selected tool for the task
    - Parameter Optimization (0-1): Quality of parameters passed to the selected tool
    - Fallback Strategy (0-1): Effectiveness of alternative approaches when primary tool fails
    - Meta-Reasoning Quality (0-1): Quality of reasoning about tool selection process


## All xml tags
  1. Content of reasoning and thinking <think></think>
  2. Mcp server <microsandbox></microsandbox>, <deepsearch></deepsearch>, <browser_use></browser_use>, <search_tool></search_tool>
  3. Content of result and anseer <reslut></result>, <answer></answer>
  4. other tags <execute_tools/> etc.
  5. tool use logic <parallel></parallel>, <sequential></sequential>

