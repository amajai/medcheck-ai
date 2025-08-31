from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class MedicalState(TypedDict):
    """State for the MediCheck AI workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    symptoms: Optional[str]
    medical_history: Optional[str]
    symptom_analysis: Optional[dict]
    medical_research: Optional[str]
    recommendations: Optional[dict]
    escalation_advice: Optional[dict]
    urgency_level: Optional[str]
    final_report: Optional[str]
    report_filepath: Optional[str]

class MedicalInputState(TypedDict):
    """Input state for starting the medical workflow"""
    messages: List[BaseMessage]
    symptoms: Optional[str]
    medical_history: Optional[str]