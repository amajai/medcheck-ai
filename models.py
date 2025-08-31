from pydantic import BaseModel, Field
from typing import List, Optional
from typing_extensions import Literal

class Summary(BaseModel):
    """Summary model for webpage content."""
    summary: str = Field(description="Concise summary of the content")
    key_excerpts: str = Field(description="Key quotes or excerpts from the content")

class SymptomAnalysis(BaseModel):
    """Medical symptom analysis result."""
    symptom_summary: str = Field(description="Clear summary of reported symptoms")
    possible_conditions: List[str] = Field(description="List of possible medical conditions")
    urgency_level: Literal["low", "moderate", "urgent"] = Field(description="Urgency classification")
    reasoning: str = Field(description="Brief explanation of the analysis")

class PatientRecommendation(BaseModel):
    """Patient-friendly medical recommendations."""
    immediate_actions: List[str] = Field(description="Immediate steps the patient should take")
    general_care: List[str] = Field(description="General self-care recommendations")
    when_to_seek_help: str = Field(description="When to contact healthcare provider")
    follow_up: Optional[str] = Field(description="Follow-up care recommendations")

class EscalationAdvice(BaseModel):
    """Urgent medical escalation advice."""
    urgency_message: str = Field(description="Clear message about urgency level")
    immediate_action: str = Field(description="What to do immediately")
    warning_signs: List[str] = Field(description="Signs that require immediate attention")
    emergency_contact: str = Field(description="Emergency contact guidance")

class MedicalSummary(BaseModel):
    """Medical-focused webpage content summary."""
    medical_summary: str = Field(description="Comprehensive medical summary focusing on clinical relevance")
    key_clinical_excerpts: str = Field(description="Important clinical quotes and excerpts")
    relevance_to_symptoms: str = Field(description="How this information relates to current patient symptoms")
    reliability_assessment: str = Field(description="Assessment of source credibility")