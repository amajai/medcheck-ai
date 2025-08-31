"""
MediCheck AI - LangGraph Workflow Implementation
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from medical_state import MedicalState, MedicalInputState
from medical_agents import (
    collect_patient_info,
    analyze_symptoms_node,
    collect_medical_history,
    create_recommendations_node,
    escalation_advice_node,
    generate_medical_report,
    should_escalate
)

# Build the medical analysis workflow
medcheck_builder = StateGraph(MedicalState, input_schema=MedicalInputState)

# Add workflow nodes
medcheck_builder.add_node("collect_patient_info", collect_patient_info)
medcheck_builder.add_node("analyze_symptoms_node", analyze_symptoms_node)
medcheck_builder.add_node("collect_medical_history", collect_medical_history)
medcheck_builder.add_node("create_recommendations_node", create_recommendations_node)
medcheck_builder.add_node("escalation_advice_node", escalation_advice_node)
medcheck_builder.add_node("generate_medical_report", generate_medical_report)

# Add workflow edges
medcheck_builder.add_edge(START, "collect_patient_info")
medcheck_builder.add_edge("collect_patient_info", "analyze_symptoms_node")
medcheck_builder.add_edge("analyze_symptoms_node", "collect_medical_history")
medcheck_builder.add_edge("collect_medical_history", "create_recommendations_node")

# Conditional edge: escalate if urgent, otherwise go to final report
medcheck_builder.add_conditional_edges(
    "create_recommendations_node",
    should_escalate,
    {
        "escalation_advice_node": "escalation_advice_node",
        "generate_medical_report": "generate_medical_report"
    }
)

# Both escalation and direct report generation lead to END
medcheck_builder.add_edge("escalation_advice_node", "generate_medical_report")
medcheck_builder.add_edge("generate_medical_report", END)

# Create memory checkpointer and compile the workflow
memory = MemorySaver()
medcheck_agent = medcheck_builder.compile(checkpointer=memory)