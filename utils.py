from datetime import datetime
from rich.console import Console
from rich.panel import Panel
import json
from typing_extensions import List, Literal

from langchain.chat_models import init_chat_model 
from langchain_core.messages import HumanMessage
from tavily import TavilyClient

from models import Summary, MedicalSummary
from prompts import summarize_medical_webpage_prompt
import os
from dotenv import load_dotenv

load_dotenv()

console = Console()

def create_llm():
    provider = os.getenv("LLM_PROVIDER", "google_genai")
    model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    return init_chat_model(
        model=model,
        model_provider=provider,
        temperature=temperature
    )


summarization_model = create_llm() 
tavily_client = TavilyClient()

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

def format_message_content(message):
    """Convert message content to displayable string"""
    parts = []
    tool_calls_processed = False
    
    # Handle main content
    if isinstance(message.content, str):
        parts.append(message.content)
    elif isinstance(message.content, list):
        # Handle complex content like tool calls (Anthropic format)
        for item in message.content:
            if item.get('type') == 'text':
                parts.append(item['text'])
            elif item.get('type') == 'tool_use':
                parts.append(f"\n🔧 Tool Call: {item['name']}")
                parts.append(f"   Args: {json.dumps(item['input'], indent=2)}")
                parts.append(f"   ID: {item.get('id', 'N/A')}")
                tool_calls_processed = True
    else:
        parts.append(str(message.content))
    
    # Handle tool calls attached to the message (OpenAI format) - only if not already processed
    if not tool_calls_processed and hasattr(message, 'tool_calls') and message.tool_calls:
        for tool_call in message.tool_calls:
            parts.append(f"\n🔧 Tool Call: {tool_call['name']}")
            parts.append(f"   Args: {json.dumps(tool_call['args'], indent=2)}")
            parts.append(f"   ID: {tool_call['id']}")
    
    return "\n".join(parts)

def format_messages(messages):
    """Format and display a list of messages with Rich formatting"""
    for m in messages:
        msg_type = m.__class__.__name__.replace('Message', '')
        content = format_message_content(m)

        if msg_type == 'Human':
            console.print(Panel(content, title="🧑 Human", border_style="blue"))
        elif msg_type == 'Ai':
            console.print(Panel(content, title="🤖 Assistant", border_style="green"))
        elif msg_type == 'Tool':
            console.print(Panel(content, title="🔧 Tool Output", border_style="yellow"))
        else:
            console.print(Panel(content, title=f"📝 {msg_type}", border_style="white"))


# Search functions

def tavily_medical_search(
    search_queries: List[str],
    symptoms: str = "",
    conditions: list = None,
    max_results: int = 3,
    include_raw_content: bool = True,
) -> str:
    """Perform medical-focused search with clinical context.
    
    Args:
        search_queries: List of medical search queries
        symptoms: Current patient symptoms for context
        conditions: List of possible conditions being considered
        max_results: Maximum number of results per query
        include_raw_content: Whether to include raw webpage content
        
    Returns:
        Formatted string of medical search results with clinical summaries
    """
    # Execute medical searches
    search_results = tavily_search_multiple(
        search_queries,
        max_results=max_results,
        topic="general",  # Use general for medical topics
        include_raw_content=include_raw_content,
    )
    
    # Deduplicate results by URL
    unique_results = deduplicate_search_results(search_results)
    
    # Process results with medical context
    medical_results = process_medical_search_results(
        unique_results,
        symptoms=symptoms,
        conditions=conditions,
        focus_area="medical"
    )
    
    # Format output for clinical use
    return format_medical_search_output(medical_results)

def tavily_search_multiple(
    search_queries: List[str], 
    max_results: int = 3, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
) -> List[dict]:
    """Perform search using Tavily API for multiple queries.

    Args:
        search_queries: List of search queries to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content

    Returns:
        List of search result dictionaries
    """
    
    # Execute searches sequentially. Note: yon can use AsyncTavilyClient to parallelize this step.
    search_docs = []
    for query in search_queries:
        result = tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        search_docs.append(result)

    return search_docs

def summarize_webpage_content(webpage_content: str) -> str:
    """Summarize webpage content using the configured summarization model.
    
    Args:
        webpage_content: Raw webpage content to summarize
        
    Returns:
        Formatted summary with key excerpts
    """
    try:
        # Set up structured output model for summarization
        structured_model = summarization_model.with_structured_output(Summary)
        
        # Generate summary using old prompt for backward compatibility
        summary = structured_model.invoke([
            HumanMessage(content="Summarize this content:\n\n" + webpage_content + "\n\nToday's date: " + get_today_str())
        ])
        
        # Format summary with clear structure
        formatted_summary = (
            "<summary>\n" + summary.summary + "\n</summary>\n\n" +
            "<key_excerpts>\n" + summary.key_excerpts + "\n</key_excerpts>"
        )
        
        return formatted_summary
        
    except Exception as e:
        print("Failed to summarize webpage: " + str(e))
        return webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content

def summarize_medical_content(webpage_content: str, symptoms: str = "", conditions: list = None, focus_area: str = "") -> str:
    """Summarize medical webpage content with clinical context.
    
    Args:
        webpage_content: Raw medical webpage content to summarize
        symptoms: Current patient symptoms for context
        conditions: List of possible conditions being considered
        focus_area: Medical focus area (e.g., cardiology, respiratory)
        
    Returns:
        Formatted medical summary with clinical excerpts
    """
    try:
        # Set up structured output model for medical summarization
        medical_model = summarization_model.with_structured_output(MedicalSummary)
        
        # Prepare conditions string
        conditions_str = str(conditions) if conditions else "No specific conditions"
        
        # Generate medical summary
        summary = medical_model.invoke([
            HumanMessage(content=summarize_medical_webpage_prompt.format(
                webpage_content=webpage_content,
                symptoms=symptoms,
                conditions=conditions_str,
                focus_area=focus_area,
                date=get_today_str()
            ))
        ])
        
        # Format medical summary with clinical structure
        formatted_summary = (
            "<medical_summary>\n" + summary.medical_summary + "\n</medical_summary>\n\n" +
            "<clinical_excerpts>\n" + summary.key_clinical_excerpts + "\n</clinical_excerpts>\n\n" +
            "<relevance>\n" + summary.relevance_to_symptoms + "\n</relevance>\n\n" +
            "<reliability>\n" + summary.reliability_assessment + "\n</reliability>"
        )
        
        return formatted_summary
        
    except Exception as e:
        print("Failed to summarize medical content: " + str(e))
        return webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content

def deduplicate_search_results(search_results: List[dict]) -> dict:
    """Deduplicate search results by URL to avoid processing duplicate content.
    
    Args:
        search_results: List of search result dictionaries
        
    Returns:
        Dictionary mapping URLs to unique results
    """
    unique_results = {}
    
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result
    
    return unique_results

def process_search_results(unique_results: dict) -> dict:
    """Process search results by summarizing content where available.
    
    Args:
        unique_results: Dictionary of unique search results
        
    Returns:
        Dictionary of processed results with summaries
    """
    summarized_results = {}
    
    for url, result in unique_results.items():
        # Use existing content if no raw content for summarization
        if not result.get("raw_content"):
            content = result['content']
        else:
            # Summarize raw content for better processing
            content = summarize_webpage_content(result['raw_content'])
        
        summarized_results[url] = {
            'title': result['title'],
            'content': content
        }
    
    return summarized_results

def process_medical_search_results(unique_results: dict, symptoms: str = "", conditions: list = None, focus_area: str = "") -> dict:
    """Process medical search results with clinical context.
    
    Args:
        unique_results: Dictionary of unique medical search results
        symptoms: Current patient symptoms for context
        conditions: List of possible conditions being considered
        focus_area: Medical focus area
        
    Returns:
        Dictionary of processed results with medical summaries
    """
    medical_results = {}
    
    for url, result in unique_results.items():
        # Use existing content if no raw content for summarization
        if not result.get("raw_content"):
            content = result['content']
        else:
            # Summarize raw content with medical context
            content = summarize_medical_content(
                result['raw_content'], 
                symptoms=symptoms,
                conditions=conditions,
                focus_area=focus_area
            )
        
        medical_results[url] = {
            'title': result['title'],
            'content': content
        }
    
    return medical_results

def format_search_output(summarized_results: dict) -> str:
    """Format search results into a well-structured string output.
    
    Args:
        summarized_results: Dictionary of processed search results
        
    Returns:
        Formatted string of search results with clear source separation
    """
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."
    
    formatted_output = "Search results: \n\n"
    
    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += "\n\n--- SOURCE " + str(i) + ": " + result['title'] + " ---\n"
        formatted_output += "URL: " + url + "\n\n"
        formatted_output += "SUMMARY:\n" + result['content'] + "\n\n"
        formatted_output += "-" * 80 + "\n"
    
    return formatted_output

def format_medical_search_output(medical_results: dict) -> str:
    """Format medical search results with clinical context.
    
    Args:
        medical_results: Dictionary of processed medical search results
        
    Returns:
        Formatted string of medical search results with clinical summaries
    """
    if not medical_results:
        return "No valid medical search results found. Please try different medical search terms."
    
    formatted_output = "Medical Research Results:\n\n"
    
    for i, (url, result) in enumerate(medical_results.items(), 1):
        formatted_output += "\n\n=== CLINICAL SOURCE " + str(i) + ": " + result['title'] + " ===\n"
        formatted_output += "URL: " + url + "\n\n"
        formatted_output += "MEDICAL SUMMARY:\n" + result['content'] + "\n\n"
        formatted_output += "=" * 80 + "\n"
    
    return formatted_output

