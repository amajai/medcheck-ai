
from typing_extensions import Annotated, Literal
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage

from utils import tavily_search_multiple, deduplicate_search_results, process_search_results, format_search_output, create_llm
from models import SymptomAnalysis, PatientRecommendation, EscalationAdvice
from prompts import symptom_analysis_prompt, recommendation_prompt, escalation_prompt

# Create medical analysis models
medical_llm = create_llm()
symptom_analyzer = medical_llm.with_structured_output(SymptomAnalysis)
recommendation_agent = medical_llm.with_structured_output(PatientRecommendation)  
escalation_agent = medical_llm.with_structured_output(EscalationAdvice)

@tool
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Fetch results from Tavily search API with content summarization.

    Args:
        query: A single search query to execute
        max_results: Maximum number of results to return
        topic: Topic to filter results by ('general', 'news', 'finance')

    Returns:
        Formatted string of search results with summaries
    """
    # Execute search for single query
    search_results = tavily_search_multiple(
        [query],  # Convert single query to list for the internal function
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    )

    # Deduplicate results by URL to avoid processing duplicate content
    unique_results = deduplicate_search_results(search_results)

    # Process results with summarization
    summarized_results = process_search_results(unique_results)

    # Format output for consumption
    return format_search_output(summarized_results)

@tool
def analyze_symptoms(
    symptoms: str,
    medical_history: str = "No medical history provided"
) -> SymptomAnalysis:
    """Analyze patient symptoms and classify urgency level.
    
    Args:
        symptoms: Patient's reported symptoms
        medical_history: Optional medical history information
        
    Returns:
        SymptomAnalysis object with analysis results
    """
    response = symptom_analyzer.invoke([
        HumanMessage(content=symptom_analysis_prompt.format(
            symptoms=symptoms,
            medical_history=medical_history
        ))
    ])
    return response

@tool
def create_recommendations(analysis: SymptomAnalysis) -> PatientRecommendation:
    """Generate patient-friendly recommendations based on symptom analysis.
    
    Args:
        analysis: SymptomAnalysis object from symptom analysis
        
    Returns:
        PatientRecommendation object with care guidance
    """
    response = recommendation_agent.invoke([
        HumanMessage(content=recommendation_prompt.format(
            analysis=analysis.model_dump_json(indent=2)
        ))
    ])
    return response

@tool
def create_escalation_advice(analysis: SymptomAnalysis, recommendations: PatientRecommendation) -> EscalationAdvice:
    """Generate urgent escalation advice for high-priority cases.
    
    Args:
        analysis: SymptomAnalysis object from symptom analysis
        recommendations: PatientRecommendation object from recommendation agent
        
    Returns:
        EscalationAdvice object with urgent guidance
    """
    response = escalation_agent.invoke([
        HumanMessage(content=escalation_prompt.format(
            analysis=analysis.model_dump_json(indent=2),
            recommendations=recommendations.model_dump_json(indent=2)
        ))
    ])
    return response