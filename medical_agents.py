"""
Medical analysis agents as LangGraph nodes
"""
import os
import re
from datetime import datetime
from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from utils import create_llm
from medical_state import MedicalState
from models import SymptomAnalysis, PatientRecommendation
from tools import tavily_search, analyze_symptoms, create_recommendations, create_escalation_advice

# Create LLM with tool binding
medical_llm = create_llm()
medical_tools = [tavily_search, analyze_symptoms, create_recommendations, create_escalation_advice]
tool_enabled_llm = medical_llm.bind_tools(medical_tools)

def collect_patient_info(state: MedicalState) -> MedicalState:
    """
    Collect patient symptoms and medical history.
    This node extracts the patient input from messages.
    """
    messages = state.get("messages", [])
    
    # Extract symptoms and medical history from the latest message
    if messages:
        latest_message = messages[-1]
        content = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
        
        # For now, treat the entire message as symptoms
        # In a real implementation, you might parse structured input
        symptoms = content
        medical_history = state.get("medical_history", "No medical history provided")
        
        return {

            "symptoms": symptoms,
            "medical_history": medical_history,
            "messages": [AIMessage(content="Patient information collected successfully")]
        }
    
    return {
        "messages": [AIMessage(content="No patient information found")]
    }

def analyze_symptoms_node(state: MedicalState) -> MedicalState:
    """
    Intelligent Symptom Analysis Agent - Analyze symptoms with optional research capability.
    Uses tool-enabled LLM to decide when additional research is needed.
    """
    symptoms = state.get("symptoms", "")
    medical_history = state.get("medical_history", "No medical history provided")
    
    if not symptoms:
        return {
            "messages": [AIMessage(content="No symptoms provided for analysis")]
        }
    
    # Create analysis prompt that allows the LLM to use tools when needed
    analysis_prompt = (
        "You are a medical analysis assistant. Analyze the following patient symptoms and medical history. "
        "You have access to search tools if you need additional medical information to make a proper assessment. "
        "Use the analyze_symptoms tool to provide a structured analysis, and use tavily_search if you need "
        "to look up specific medical conditions, treatments, or diagnostic criteria.\n\n"
        "Patient Symptoms: " + symptoms + "\n"
        "Medical History: " + medical_history + "\n\n"
        "Please provide a comprehensive symptom analysis."
    )
    
    # Use tool-enabled LLM to allow intelligent tool usage
    response = tool_enabled_llm.invoke([
        AIMessage(content=analysis_prompt)
    ])
    
    # Check if the LLM used the analyze_symptoms tool
    analysis_result = None
    research_info = ""
    
    # Parse tool calls from response
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'analyze_symptoms':
                # Execute the analyze_symptoms tool with the LLM's parameters
                analysis_result = analyze_symptoms.invoke(tool_call['args'])
            elif tool_call['name'] == 'tavily_search':
                # Capture research information
                search_result = tavily_search.invoke(tool_call['args'])
                research_info += "Research: " + search_result + "\n"
    
    # If no tool was used or analysis failed, use direct tool call
    if not analysis_result:
        analysis_result = analyze_symptoms.invoke({
            "symptoms": symptoms,
            "medical_history": medical_history
        })
    
    # Convert SymptomAnalysis to dict for state storage
    analysis_dict = {
        "symptom_summary": analysis_result.symptom_summary,
        "possible_conditions": analysis_result.possible_conditions,
        "urgency_level": analysis_result.urgency_level,
        "reasoning": analysis_result.reasoning
    }
    
    return {
        "symptom_analysis": analysis_dict,
        "urgency_level": analysis_result.urgency_level,
        "medical_research": research_info if research_info else "No additional research needed",
        "messages": [AIMessage(content="Intelligent symptom analysis completed. Urgency level: " + analysis_result.urgency_level)]
    }


def collect_medical_history(state: MedicalState) -> MedicalState:
    """
    Human-in-the-loop node to collect medical history after initial analysis.
    Uses LangGraph interrupt to pause and ask for user input.
    """
    analysis = state.get("symptom_analysis", {})
    
    if not analysis:
        return {
            "messages": [AIMessage(content="No symptom analysis available for medical history collection")]
        }
    
    # Create interrupt request with analysis context
    interrupt_message = ("Based on the symptom analysis, do you have relevant medical history "
                        "(medications, conditions, allergies, surgeries, family history) "
                        "that might help improve the recommendations?\n\n"
                        "Current analysis:\n"
                        "- Summary: " + analysis.get("symptom_summary", "N/A") + "\n"
                        "- Urgency: " + analysis.get("urgency_level", "N/A") + "\n"
                        "- Possible conditions: " + str(analysis.get("possible_conditions", [])) + "\n\n"
                        "Enter your medical history (or 'no'/'none' if not applicable):")
    
    # Use LangGraph interrupt to pause execution and request user input
    user_input = interrupt(interrupt_message)
    
    # Process the user input
    if user_input and user_input.strip().lower() not in ["no", "none", "n/a", ""]:
        medical_history = user_input.strip()
        return {
            "medical_history": medical_history,
            "messages": [AIMessage(content="Medical history collected: " + medical_history)]
        }
    else:
        return {
            "medical_history": "No additional medical history provided",
            "messages": [AIMessage(content="No additional medical history provided")]
        }

def create_recommendations_node(state: MedicalState) -> MedicalState:
    """
    Intelligent Recommendation Agent - Generate comprehensive patient care recommendations.
    Uses tool-enabled LLM to gather additional information if needed for better recommendations.
    """
    analysis = state.get("symptom_analysis", {})
    medical_research = state.get("medical_research", "")
    
    if not analysis:
        return {
            "messages": [AIMessage(content="No symptom analysis available for recommendations")]
        }
    
    # Create comprehensive recommendation prompt
    recommendation_prompt = (
        "You are a medical recommendation assistant. Based on the symptom analysis and any research data, "
        "generate comprehensive patient-friendly care recommendations. You can use additional search tools "
        "if you need specific treatment information or care guidelines.\n\n"
        "Symptom Analysis: " + str(analysis) + "\n\n"
        "Previous Research: " + medical_research + "\n\n"
        "Use the create_recommendations tool to provide structured recommendations. "
        "If you need additional information about treatments or care guidelines, use tavily_search."
    )
    
    # Use tool-enabled LLM for intelligent recommendations
    response = tool_enabled_llm.invoke([
        AIMessage(content=recommendation_prompt)
    ])
    
    # Check if the LLM used tools
    recommendations_result = None
    additional_research = ""
    
    # Parse tool calls from response
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'create_recommendations':
                # Execute the create_recommendations tool
                symptom_analysis_obj = SymptomAnalysis(
                    symptom_summary=analysis.get("symptom_summary", ""),
                    possible_conditions=analysis.get("possible_conditions", []),
                    urgency_level=analysis.get("urgency_level", "low"),
                    reasoning=analysis.get("reasoning", "")
                )
                recommendations_result = create_recommendations.invoke({
                    "analysis": symptom_analysis_obj
                })
            elif tool_call['name'] == 'tavily_search':
                # Capture additional research
                search_result = tavily_search.invoke(tool_call['args'])
                additional_research += "Additional Research: " + search_result + "\n"
    
    # If no tool was used, use direct tool call
    if not recommendations_result:
        symptom_analysis_obj = SymptomAnalysis(
            symptom_summary=analysis.get("symptom_summary", ""),
            possible_conditions=analysis.get("possible_conditions", []),
            urgency_level=analysis.get("urgency_level", "low"),
            reasoning=analysis.get("reasoning", "")
        )
        recommendations_result = create_recommendations.invoke({
            "analysis": symptom_analysis_obj
        })
    
    # Convert to dict for state storage
    recommendations_dict = {
        "immediate_actions": recommendations_result.immediate_actions,
        "general_care": recommendations_result.general_care,
        "when_to_seek_help": recommendations_result.when_to_seek_help,
        "follow_up": recommendations_result.follow_up
    }
    
    # Update medical research if additional info was found
    updated_research = medical_research
    if additional_research:
        updated_research += "\n" + additional_research
    
    return {
        "recommendations": recommendations_dict,
        "medical_research": updated_research,
        "messages": [AIMessage(content="Comprehensive patient recommendations generated successfully")]
    }

def escalation_advice_node(state: MedicalState) -> MedicalState:
    """
    Escalation Agent - Generate urgent medical escalation advice for high-priority cases.
    """
    analysis = state.get("symptom_analysis", {})
    recommendations = state.get("recommendations", {})
    urgency_level = state.get("urgency_level", "low")
    
    if urgency_level != "urgent":
        return {
            "messages": [AIMessage(content="Case not urgent - escalation advice not needed")]
        }
    
    if not analysis or not recommendations:
        return {
            "messages": [AIMessage(content="Insufficient data for escalation advice")]
        }
    
    # Recreate objects for the tool
    symptom_analysis_obj = SymptomAnalysis(
        symptom_summary=analysis.get("symptom_summary", ""),
        possible_conditions=analysis.get("possible_conditions", []),
        urgency_level=analysis.get("urgency_level", "urgent"),
        reasoning=analysis.get("reasoning", "")
    )
    
    recommendations_obj = PatientRecommendation(
        immediate_actions=recommendations.get("immediate_actions", []),
        general_care=recommendations.get("general_care", []),
        when_to_seek_help=recommendations.get("when_to_seek_help", ""),
        follow_up=recommendations.get("follow_up", "")
    )
    
    # Use the create_escalation_advice tool
    response = create_escalation_advice.invoke({
        "analysis": symptom_analysis_obj,
        "recommendations": recommendations_obj
    })
    
    # Convert to dict for state storage
    escalation_dict = {
        "urgency_message": response.urgency_message,
        "immediate_action": response.immediate_action,
        "warning_signs": response.warning_signs,
        "emergency_contact": response.emergency_contact
    }
    
    return {
        "escalation_advice": escalation_dict,
        "messages": [AIMessage(content="Urgent escalation advice generated")]
    }

def generate_medical_report(state: MedicalState) -> MedicalState:
    """
    Generate final medical report and save to file.
    """
    analysis = state.get("symptom_analysis", {})
    recommendations = state.get("recommendations", {})
    escalation = state.get("escalation_advice", {})
    symptoms = state.get("symptoms", "")
    medical_history = state.get("medical_history", "")
    
    # Generate comprehensive report
    report_content = ("# MediCheck AI - Medical Analysis Report\n\n"
                     "**Generated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n"
                     "## Patient Information\n"
                     "**Reported Symptoms:** " + symptoms + "\n"
                     "**Medical History:** " + medical_history + "\n\n"
                     "## Symptom Analysis\n"
                     "**Summary:** " + analysis.get('symptom_summary', 'N/A') + "\n\n"
                     "**Possible Conditions:**\n")
    
    conditions = analysis.get('possible_conditions', [])
    for i, condition in enumerate(conditions, 1):
        report_content += str(i) + ". " + condition + "\n"
    
    report_content += ("\n**Urgency Level:** " + analysis.get('urgency_level', 'N/A').upper() + "\n"
                      "**Reasoning:** " + analysis.get('reasoning', 'N/A') + "\n\n"
                      "## Recommendations\n\n"
                      "**Immediate Actions:**\n")
    
    immediate_actions = recommendations.get('immediate_actions', [])
    for i, action in enumerate(immediate_actions, 1):
        report_content += str(i) + ". " + action + "\n"
    
    report_content += "\n**General Self-Care:**\n"
    general_care = recommendations.get('general_care', [])
    for i, care in enumerate(general_care, 1):
        report_content += str(i) + ". " + care + "\n"
    
    report_content += ("\n**When to Seek Help:** " + recommendations.get('when_to_seek_help', 'N/A') + "\n"
                      "**Follow-up:** " + recommendations.get('follow_up', 'N/A') + "\n")
    
    # Add escalation advice if urgent
    if escalation:
        report_content += ("\n## URGENT ESCALATION ADVICE\n"
                          "**Urgency Message:** " + escalation.get('urgency_message', 'N/A') + "\n"
                          "**Immediate Action:** " + escalation.get('immediate_action', 'N/A') + "\n\n"
                          "**Warning Signs:**\n")
        warning_signs = escalation.get('warning_signs', [])
        for i, sign in enumerate(warning_signs, 1):
            report_content += str(i) + ". " + sign + "\n"
        
        report_content += "\n**Emergency Contact:** " + escalation.get('emergency_contact', 'N/A') + "\n"
    
    report_content += ("\n---\n"
                      "**IMPORTANT DISCLAIMER:** This analysis is for educational purposes only and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult healthcare professionals for medical concerns.\n")
    
    # Create reports directory if it doesn't exist
    os.makedirs("medical_reports", exist_ok=True)
    
    # Generate filename based on timestamp and symptoms
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean symptoms for filename
    symptom_preview = re.sub(r'[^\w\s-]', '', symptoms[:30])
    symptom_preview = re.sub(r'[-\s]+', '_', symptom_preview)
    
    filename = "medical_analysis_" + timestamp + "_" + symptom_preview + ".md"
    filepath = os.path.join("medical_reports", filename)
    
    # Save the report
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return {
        "final_report": report_content,
        "report_filepath": filepath,
        "messages": [AIMessage(content="Medical analysis complete! Report saved to: " + filepath)]
    }

def should_escalate(state: MedicalState) -> str:
    """
    Conditional edge function to determine if escalation is needed.
    """
    urgency_level = state.get("urgency_level", "low")
    if urgency_level == "urgent":
        return "escalation_advice_node"
    else:
        return "generate_medical_report"