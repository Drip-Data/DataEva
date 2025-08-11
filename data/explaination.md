## Tool Category

1. deepsearch
2. microsandbox
    - sandbox_start
    - sandbox_stop
    - sandbox_run_code
    - sandbox_run_command
    - snadbox_get_metrice
3. tavily
    - tavily-search: avily's powerful web search tool supports real-time, categorized, in-depth, country-specific, image and other parameter configurations.
    - tavily-extract: Extract web page content from the given URL. You can choose the extraction depth, format (Markdown or plain text), etc.
    - tavily-crawl: Structured web crawler, recursively crawls web pages starting from the specified URL, and can set depth, breadth, restrictions, extraction categories, etc.
    - wavily-map: Website structure mapping tool, used for analyzing site structure and navigation paths. Suitable for content discovery, website audits, etc.
4. perform_web_task
    - search_google: 'Search the query in Google, the query should be a search query like humans search in Google, concrete and not vague or super long.’
    - go_to_url: 'Navigate to URL, set new_tab=True to open in new tab, False to navigate in current tab'
    - go_back: 'Go back’
    - click_element_by_index: 'Click element by index’
    - input_text: 'Click and input text into a input interactive element’
    - switch_tab: 'Switch tab’
    - close_tab: 'Close an existing tab'
    - extract_structured_data: "Extract structured, semantic data (e.g. product description, price, all information about XYZ) from the current webpage based on a textual query.This tool takes the entire markdown of the page and extracts the query from it.Set extract_links=True ONLY if your query requires extracting links/URLs from the page.Only use this for specific queries for information retrieval from the page. Don't use this to get interactive elements - the tool does not see HTML elements, only the markdown."
    - scroll: 'Scroll the page by specified number of pages (set down=True to scroll down, down=False to scroll up, num_pages=number of pages to scroll like 0.5 for half page, 1.0 for one page, etc.). Optional index parameter to scroll within a specific element or its scroll container (works well for dropdowns and custom UI components).’
    - done: 'Complete task - with return text and if the task is finished (success=True) or not yet completely finished (success=False), because last step is reached’
    - write_file: 'Write or append content to file_name in file system. Allowed extensions are .md, .txt, .json, .csv, .pdf. For .pdf files, write the content in markdown format and it will automatically be converted to a properly formatted PDF document.’
    - replace_file_str: 'Replace old_str with new_str in file_name. old_str must exactly match the string to replace in original text. Recommended tool to mark completed items in todo.md or change specific contents in a file.'
    - wait :  'Wait for x seconds default 3 (max 10 seconds). This can be used to wait until the page is fully loaded.’
  
## Evaluation metrics

1. deepsearch
    1. Information Relevance (0.0-1.0): How relevant is the retrieved information to the task?
    2. Tool use quality (0.0-1.0):  Is the sequence of tool use logical and efficient?
    3. Source Quality (0.0-1.0): Are high-quality, credible and rich sources being used?
    4. Information Synthesis (0.0-1.0): How well is information from multiple sources synthesized and summarized ?
2. microsandbox
    1. Code Correctness (0.0-1.0): Is the code syntactically correct and logically sound?
    2. Tool use quality (0.0-1.0):  Is the sequence of tool use logical and efficient?
    3. Computational Efficiency (0.0-1.0): Does the code solve the problem efficiently?
    4. Result Interpretation (0.0-1.0): Are the computational results correctly interpreted?
3. perform_web_task
    1. Tool use quality (0.0-1.0):  Is the sequence of tool use logical and efficient?
    2.  Content Extraction (0.0-1.0): How accurately is relevant content extracted?
    3. Interaction Quality (0.0-1.0): How appropriate are the web interactions (clicks, inputs)?
    4. Goal Achievement (0.0-1.0): How well does the browsing contribute to task completion?
4. tavily
    1. Tool use quality (0.0-1.0):  Is the sequence of tool use logical and efficient?
    2. Information Relevance (0.0-1.0): How relevant is the retrieved information to the task?
    3. Content Extraction (0.0-1.0): How accurately is relevant content extracted?
    4. Goal Achievement (0.0-1.0): How well does the browsing contribute to task completion?
5. final
    1. Task Completion (0.0-1.0): How completely is the  task addressed?
    2. Tool use quality (0.0-1.0):  Is the sequence of tool use logical and efficient?
    3. Reasoning Coherence (0.0-1.0): How logical and coherent is the overall reasoning chain？
    4. Problem Resolution (0.0-1.0): How effectively are any encountered problems resolved?