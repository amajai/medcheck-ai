"""
MediCheck AI - Streamlit Web Interface
"""
import streamlit as st
import sys
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from medical_workflow import medcheck_agent

def initialize_session_state():
    """Initialize session state variables"""
    if 'workflow_started' not in st.session_state:
        st.session_state.workflow_started = False
    if 'final_state' not in st.session_state:
        st.session_state.final_state = None
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = "streamlit_session"
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    # Track whether we're awaiting medical history input so we can hide the prompt after submission
    if 'awaiting_history' not in st.session_state:
        st.session_state.awaiting_history = False

def display_header():
    """Display the app header with medical disclaimer"""
    st.set_page_config(
        page_title="MediCheck AI",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🩺 MediCheck AI")
    st.subheader("Patient Symptom Analyzer")
    st.markdown("*AI-Powered Medical Assessment Tool*")
    
    with st.expander("⚠️ IMPORTANT MEDICAL DISCLAIMER", expanded=True):
        st.warning(
            "**MEDICAL DISCLAIMER**: This tool provides educational information only and is "
            "**NOT** a substitute for professional medical advice, diagnosis, or treatment. "
            "Always consult healthcare professionals for medical concerns."
        )

def display_analysis_results(analysis):
    """Display symptom analysis results"""
    st.subheader("📊 Analysis Results")
    
    # Summary
    if "symptom_summary" in analysis:
        st.write("**Summary:**")
        st.write(analysis["symptom_summary"])
    
    # Possible conditions
    conditions = analysis.get("possible_conditions", [])
    if conditions:
        st.write("**Possible Conditions:**")
        for i, condition in enumerate(conditions, 1):
            st.write(f"{i}. {condition}")
    
    # Urgency level
    urgency = analysis.get("urgency_level", "unknown").upper()
    if urgency == "LOW":
        st.success(f"**Urgency Level:** {urgency}")
    elif urgency == "MODERATE":
        st.warning(f"**Urgency Level:** {urgency}")
    else:
        st.error(f"**Urgency Level:** {urgency}")

def display_recommendations(recommendations):
    """Display medical recommendations"""
    st.subheader("💡 Medical Recommendations")
    
    # Immediate actions
    immediate_actions = recommendations.get("immediate_actions", [])
    if immediate_actions:
        st.write("**Immediate Actions:**")
        for i, action in enumerate(immediate_actions, 1):
            st.write(f"{i}. {action}")
    
    # General care
    general_care = recommendations.get("general_care", [])
    if general_care:
        st.write("**General Self-Care:**")
        for i, care in enumerate(general_care, 1):
            st.write(f"{i}. {care}")
    
    # When to seek help
    when_to_seek = recommendations.get("when_to_seek_help", "")
    if when_to_seek:
        st.write(f"**When to Seek Help:** {when_to_seek}")

def display_escalation_advice(escalation):
    """Display urgent escalation advice"""
    st.error("🚨 URGENT MEDICAL ALERT 🚨")
    
    # Urgency message
    urgency_msg = escalation.get("urgency_message", "")
    if urgency_msg:
        st.error(f"**URGENT:** {urgency_msg}")
    
    # Immediate action
    immediate_action = escalation.get("immediate_action", "")
    if immediate_action:
        st.error(f"**IMMEDIATE ACTION:** {immediate_action}")
    
    # Warning signs
    warning_signs = escalation.get("warning_signs", [])
    if warning_signs:
        st.error("**Critical Warning Signs:**")
        for i, sign in enumerate(warning_signs, 1):
            st.error(f"{i}. {sign}")
    
    # Emergency contact
    emergency_contact = escalation.get("emergency_contact", "")
    if emergency_contact:
        st.error(f"**Emergency Contact:** {emergency_contact}")

def run_workflow():
    """Run the medical analysis workflow"""
    if not st.session_state.workflow_started:
        return

    # If analysis already completed, render the final results from session state and exit
    if st.session_state.get("analysis_complete") and st.session_state.get("final_state"):
        final_state = st.session_state.final_state
        st.session_state.awaiting_history = False
        st.subheader("📊 Analysis Results")
        st.success("🎉 Medical Analysis Complete!")
        report_content = final_state.get("final_report", "")
        if report_content:
            report_filename = f"medical_analysis_{st.session_state.thread_id}.md"
            col1, col2 = st.columns([1, 1])
            with col1:
                st.download_button(
                    label="📁 Download Medical Report",
                    data=report_content,
                    file_name=report_filename,
                    mime="text/markdown",
                    type="primary"
                )
            with col2:
                if st.button("🔄 Start New Analysis", type="secondary"):
                    st.session_state.clear()
                    st.rerun()
        return
    
    # Prepare initial state
    symptoms = st.session_state.symptoms
    messages = [HumanMessage(content=symptoms)]
    thread = {"configurable": {"thread_id": st.session_state.thread_id, "recursion_limit": 50}}
    
    initial_state = {
        "messages": messages,
        "symptoms": symptoms,
        "medical_history": "No medical history provided"
    }
    
    # Progress bar and status
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    final_state = None
    step = 0
    total_steps = 6  # Estimated number of workflow steps
    
    with st.spinner("Running medical analysis workflow..."):
        try:
            # Start the workflow
            for event in medcheck_agent.stream(initial_state, config=thread):
                step += 1
                progress_bar.progress(min(step / total_steps, 0.8))
                
                for node, output in event.items():
                    final_state = output
                    
                    if node == "collect_patient_info":
                        status_text.text("✅ Patient information collected")
                        
                    elif node == "analyze_symptoms_node":
                        status_text.text("✅ Symptom analysis completed")
                        
                        # Display analysis results
                        if "symptom_analysis" in output:
                            with results_container:
                                display_analysis_results(output["symptom_analysis"])
                                
                    elif node == "collect_medical_history":
                        status_text.text("✅ Medical history processed")
                        
                    elif node == "create_recommendations_node":
                        status_text.text("✅ Recommendations generated")
                        
                        # Display recommendations
                        if "recommendations" in output:
                            with results_container:
                                display_recommendations(output["recommendations"])
                        
                        # Check if escalation needed
                        urgency = output.get("urgency_level", "low")
                        if urgency == "urgent":
                            status_text.text("⚠️ Urgent case detected - creating escalation advice")
                            
                    elif node == "escalation_advice_node":
                        status_text.text("🚨 URGENT ESCALATION ADVICE CREATED")
                        
                        # Display escalation advice
                        if "escalation_advice" in output:
                            with results_container:
                                display_escalation_advice(output["escalation_advice"])
                            
                    elif node == "generate_medical_report":
                        status_text.text("📋 Generating final medical report")
            
            # Check if we need human input
            current_state = medcheck_agent.get_state(config=thread)
            if current_state.next:
                # Handle human input requirement
                status_text.text("🤔 Human input required")
                st.session_state.awaiting_history = True
                
                # Get current analysis for context
                analysis = current_state.values.get("symptom_analysis", {})
                
                # Render the medical history prompt only when awaiting input
                if st.session_state.awaiting_history:
                    # Use a placeholder so we can clear it after submission
                    with results_container:
                        mh_placeholder = st.empty()
                        with mh_placeholder.container():
                            st.subheader("📝 Medical History Input Required")
                            st.write("**Current Analysis Summary:**")
                            st.write(f"- **Summary:** {analysis.get('symptom_summary', 'N/A')}")
                            st.write(f"- **Urgency:** {analysis.get('urgency_level', 'N/A')}")
                            st.write(f"- **Possible conditions:** {', '.join(analysis.get('possible_conditions', []))}")
                            # Get medical history input
                            st.text_area(
                                "Please provide your medical history (or type 'no' if not applicable):",
                                key="medical_history_input"
                            )
                            submitted = st.button("Continue Analysis", key="continue_btn")

                    if submitted:
                        medical_history = st.session_state.get("medical_history_input", "").strip()
                        if medical_history:
                            # Clear the input UI and mark history as submitted to avoid re-rendering
                            mh_placeholder.empty()
                            st.session_state.awaiting_history = False
                            status_text.text("▶️ Resuming medical analysis")

                            # Resume workflow with user input
                            for event in medcheck_agent.stream(Command(resume=medical_history), config=thread):
                                step += 1
                                progress_bar.progress(min(step / total_steps, 1.0))

                                for node, output in event.items():
                                    final_state = output

                                    if node == "collect_medical_history":
                                        status_text.text("✅ Medical history processed")

                                    elif node == "create_recommendations_node":
                                        status_text.text("✅ Recommendations generated")

                                        # Display recommendations
                                        if "recommendations" in output:
                                            with results_container:
                                                display_recommendations(output["recommendations"])

                                        # Check if escalation needed
                                        urgency = output.get("urgency_level", "low")
                                        if urgency == "urgent":
                                            status_text.text("⚠️ Urgent case detected - creating escalation advice")

                                    elif node == "escalation_advice_node":
                                        status_text.text("🚨 URGENT ESCALATION ADVICE CREATED")

                                        # Display escalation advice
                                        if "escalation_advice" in output:
                                            with results_container:
                                                display_escalation_advice(output["escalation_advice"])

                                    elif node == "generate_medical_report":
                                        status_text.text("📋 Final medical report generated")

                            st.session_state.analysis_complete = True
                            # Optionally clear the stored input
                            if "medical_history_input" in st.session_state:
                                del st.session_state["medical_history_input"]
                        else:
                            st.error("Please provide medical history to continue.")
            else:
                st.session_state.analysis_complete = True
            
            # Display final results
            if st.session_state.analysis_complete and final_state:
                progress_bar.progress(1.0)
                status_text.text("✅ Medical analysis complete")

                if "final_report" in final_state:
                    with results_container:
                        st.success("🎉 Medical Analysis Complete!")

                        # Create download button for the report
                        report_content = final_state.get("final_report", "")
                        if report_content:
                            report_filename = f"medical_analysis_{st.session_state.thread_id}.md"

                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.download_button(
                                    label="📁 Download Medical Report",
                                    data=report_content,
                                    file_name=report_filename,
                                    mime="text/markdown",
                                    type="primary"
                                )
                            with col2:
                                if st.button("🔄 Start New Analysis", type="secondary"):
                                    st.session_state.clear()
                                    st.rerun()
                    # Persist so reruns (e.g., after downloading) don't restart the workflow
                    st.session_state.final_state = final_state
                    st.session_state.awaiting_history = False
                    return

                # If final_state exists but no final_report key, still persist and exit
                st.session_state.final_state = final_state
                st.session_state.awaiting_history = False
                return

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.warning("Please try again or consult a healthcare professional directly.")

def main():
    """Main Streamlit application"""
    # Setup logger

    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Main content area - Patient Information
    if not st.session_state.workflow_started:
        st.header("Patient Information")
        
        # Symptom input
        symptoms = st.text_area(
            "🩺 Please describe your symptoms in detail:",
            height=150,
            key="symptoms_input",
            placeholder="Describe your symptoms, duration, severity, and any other relevant details..."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            # Start analysis button
            if st.button("Start Analysis", type="primary", disabled=not symptoms.strip()):
                if symptoms.strip():
                    st.session_state.symptoms = symptoms
                    st.session_state.workflow_started = True
                    st.session_state.analysis_complete = False
                    st.session_state.current_step = 0
                    st.rerun()
                else:
                    st.error("Please describe your symptoms to proceed.")
        
        with col2:
            # Reset button
            if st.button("Reset Analysis"):
                st.session_state.clear()
                st.rerun()
    else:
        # Analysis workflow area
        st.header("🔬 Medical Analysis Workflow")
        run_workflow()
    
    # Sidebar - Welcome and Information
    with st.sidebar:
        # Welcome screen
        st.markdown("""
        ## Welcome to MediCheck AI
        
        This AI-powered medical assessment tool helps analyze symptoms and provides educational 
        information about potential conditions and recommendations.
        
        ### How to use:
        1. **Describe your symptoms** in the main area
        2. **Click "Start Analysis"** to begin the medical workflow
        3. **Review the results** and follow any recommendations
        4. **Consult healthcare professionals** for proper medical care
        
        ### Features:
        - 🔍 **Symptom Analysis**: AI-powered symptom evaluation
        - 💡 **Recommendations**: Personalized care suggestions
        - 🚨 **Urgency Detection**: Identifies cases requiring immediate attention
        - 📋 **Medical Reports**: Comprehensive analysis documentation
        
        **Remember**: This tool is for educational purposes only and cannot replace professional medical advice.
        """)
        
        # Add some helpful information
        with st.expander("📖 About MediCheck AI"):
            st.markdown("""
            MediCheck AI uses advanced language models to analyze symptoms and provide 
            educational information. The system follows a structured workflow:
            
            1. **Patient Information Collection** - Gathers symptom descriptions
            2. **Symptom Analysis** - AI evaluation of reported symptoms
            3. **Medical History** - Optional medical background consideration
            4. **Recommendations** - Generates appropriate care suggestions
            5. **Escalation** - Identifies urgent cases requiring immediate attention
            6. **Report Generation** - Creates comprehensive analysis documentation
            
            The tool is designed to be helpful while maintaining appropriate medical safety standards.
            """)

if __name__ == "__main__":
    main()