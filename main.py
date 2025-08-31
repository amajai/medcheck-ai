import sys
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from medical_workflow import medcheck_agent
from niceterminalui import (
    print_banner, print_step, print_success, print_warning, print_error,
    rich_prompt, print_completion_message, console
)

def main():
    """Main MediCheck AI workflow"""
    try:
        # Print banner and disclaimer
        print_banner(
            title="MediCheck AI",
            subtitle="Patient Symptom Analyzer", 
            description="AI-Powered Medical Assessment Tool",
            subheader1="Educational Information Only",
            subheader2="Always Consult Healthcare Professionals"
        )
        
        print_warning("MEDICAL DISCLAIMER: This tool provides educational information only and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult healthcare professionals for medical concerns.")
        
        # Get patient input
        print_step("Patient Information Collection", "📝")
        
        symptoms = rich_prompt("🩺 Please describe your symptoms in detail")
        if not symptoms.strip():
            print_error("Symptoms are required to proceed")
            return
        
        # Prepare initial state with patient input
        messages = [HumanMessage(content=symptoms)]
        thread = {"configurable": {"thread_id": "1", "recursion_limit": 50}}
        
        initial_state = {
            "messages": messages,
            "symptoms": symptoms,
            "medical_history": "No medical history provided"
        }
        
        print_step("Starting Medical Analysis Workflow", "🔬")
        
        final_state = None
        
        # Start the workflow
        for event in medcheck_agent.stream(initial_state, config=thread):
            console.print("\n[dim]Processing: " + str(list(event.keys())) + "[/dim]")
            
            for node, output in event.items():
                final_state = output
                
                if "messages" in output and output["messages"]:
                    latest_message = output["messages"][-1]
                    message_content = latest_message if isinstance(latest_message, str) else latest_message.content
                    
                    # Handle different node outputs
                    if node == "collect_patient_info":
                        print_success("Patient information collected")
                        
                    elif node == "analyze_symptoms_node":
                        print_success("Symptom analysis completed")
                        
                        # Display analysis results
                        if "symptom_analysis" in output:
                            analysis = output["symptom_analysis"]
                            print_step("Analysis Results", "📊")
                            console.print("Summary: " + analysis.get("symptom_summary", "N/A"))
                            
                            conditions = analysis.get("possible_conditions", [])
                            if conditions:
                                console.print("Possible Conditions:")
                                for i, condition in enumerate(conditions, 1):
                                    console.print("  " + str(i) + ". " + condition)
                            
                            urgency = analysis.get("urgency_level", "unknown").upper()
                            if urgency == "LOW":
                                print_success("Urgency Level: " + urgency)
                            elif urgency == "MODERATE":
                                print_warning("Urgency Level: " + urgency)
                            else:
                                print_error("Urgency Level: " + urgency)
                    
                        
                    elif node == "__interrupt__":
                        break
                        
                    elif node == "collect_medical_history":
                        print_step("Medical History Collection", "🏥")
                        print_success("Medical history collected")
                        
                    elif node == "create_recommendations_node":
                        print_success("Recommendations generated")
                        
                        # Display recommendations
                        if "recommendations" in output:
                            recommendations = output["recommendations"]
                            print_step("Medical Recommendations", "💡")
                            
                            immediate_actions = recommendations.get("immediate_actions", [])
                            if immediate_actions:
                                console.print("Immediate Actions:")
                                for i, action in enumerate(immediate_actions, 1):
                                    console.print("  " + str(i) + ". " + action)
                            
                            general_care = recommendations.get("general_care", [])
                            if general_care:
                                console.print("General Self-Care:")
                                for i, care in enumerate(general_care, 1):
                                    console.print("  " + str(i) + ". " + care)
                            
                            when_to_seek = recommendations.get("when_to_seek_help", "")
                            if when_to_seek:
                                console.print("When to Seek Help: " + when_to_seek)
                        
                        # Check if escalation needed
                        urgency = output.get("urgency_level", "low")
                        if urgency == "urgent":
                            print_warning("Urgent case detected - creating escalation advice")
                            
                    elif node == "escalation_advice_node":
                        print_error("URGENT ESCALATION ADVICE CREATED")
                        
                        # Display escalation advice
                        if "escalation_advice" in output:
                            escalation = output["escalation_advice"]
                            print_error("🚨 URGENT MEDICAL ALERT 🚨")
                            
                            urgency_msg = escalation.get("urgency_message", "")
                            if urgency_msg:
                                print_error("URGENT: " + urgency_msg)
                            
                            immediate_action = escalation.get("immediate_action", "")
                            if immediate_action:
                                print_error("IMMEDIATE ACTION: " + immediate_action)
                            
                            warning_signs = escalation.get("warning_signs", [])
                            if warning_signs:
                                print_error("Critical Warning Signs:")
                                for i, sign in enumerate(warning_signs, 1):
                                    console.print("  " + str(i) + ". " + sign)
                            
                            emergency_contact = escalation.get("emergency_contact", "")
                            if emergency_contact:
                                print_error("Emergency Contact: " + emergency_contact)
                        
                    elif node == "generate_medical_report":
                        if "saved to:" in message_content or "Report saved to:" in message_content:
                            print_success(message_content)
                        else:
                            print_step("Generating final medical report", "📋")
        
        # Check if we're interrupted (paused for human input)
        current_state = medcheck_agent.get_state(config=thread)
        if current_state.next:
            # We're interrupted - handle the human input
            print_step("Human Input Required", "🤔")
            
            # Get current analysis for context
            analysis = current_state.values.get("symptom_analysis", {})
            console.print("\nBased on the symptom analysis:")
            console.print("- Summary: " + analysis.get("symptom_summary", "N/A"))
            console.print("- Urgency: " + analysis.get("urgency_level", "N/A"))
            console.print("- Possible conditions: " + str(analysis.get("possible_conditions", [])))
            console.print()
            
            # Get user input
            user_input = rich_prompt("Enter your medical history (or 'no' if not applicable)")
            
            # Resume workflow with user input
            print_step("Resuming Medical Analysis", "▶️")
            for event in medcheck_agent.stream(Command(resume=user_input), config=thread):
                console.print("\n[dim]Processing: " + str(list(event.keys())) + "[/dim]")
                
                for node, output in event.items():
                    final_state = output
                    
                    if "messages" in output and output["messages"]:
                        latest_message = output["messages"][-1]
                        message_content = latest_message if isinstance(latest_message, str) else latest_message.content
                        
                        # Handle remaining nodes
                        if node == "collect_medical_history":
                            print_success("Medical history processed")
                            
                        elif node == "create_recommendations_node":
                            print_success("Recommendations generated")
                            
                            # Display recommendations
                            if "recommendations" in output:
                                recommendations = output["recommendations"]
                                print_step("Medical Recommendations", "💡")
                                
                                immediate_actions = recommendations.get("immediate_actions", [])
                                if immediate_actions:
                                    console.print("Immediate Actions:")
                                    for i, action in enumerate(immediate_actions, 1):
                                        console.print("  " + str(i) + ". " + action)
                                
                                general_care = recommendations.get("general_care", [])
                                if general_care:
                                    console.print("General Self-Care:")
                                    for i, care in enumerate(general_care, 1):
                                        console.print("  " + str(i) + ". " + care)
                                
                                when_to_seek = recommendations.get("when_to_seek_help", "")
                                if when_to_seek:
                                    console.print("When to Seek Help: " + when_to_seek)
                            
                            # Check if escalation needed
                            urgency = output.get("urgency_level", "low")
                            if urgency == "urgent":
                                print_warning("Urgent case detected - creating escalation advice")
                                
                        elif node == "escalation_advice_node":
                            print_error("URGENT ESCALATION ADVICE CREATED")
                            
                            # Display escalation advice
                            if "escalation_advice" in output:
                                escalation = output["escalation_advice"]
                                print_error("🚨 URGENT MEDICAL ALERT 🚨")
                                
                                urgency_msg = escalation.get("urgency_message", "")
                                if urgency_msg:
                                    print_error("URGENT: " + urgency_msg)
                                
                                immediate_action = escalation.get("immediate_action", "")
                                if immediate_action:
                                    print_error("IMMEDIATE ACTION: " + immediate_action)
                                
                                warning_signs = escalation.get("warning_signs", [])
                                if warning_signs:
                                    print_error("Critical Warning Signs:")
                                    for i, sign in enumerate(warning_signs, 1):
                                        console.print("  " + str(i) + ". " + sign)
                                
                                emergency_contact = escalation.get("emergency_contact", "")
                                if emergency_contact:
                                    print_error("Emergency Contact: " + emergency_contact)
                            
                        elif node == "generate_medical_report":
                            if "saved to:" in message_content or "Report saved to:" in message_content:
                                print_success(message_content)
                            else:
                                print_step("Generating final medical report", "📋")
        
        # Display final results
        if final_state:
            if "final_report" in final_state:
                print_step("Medical Analysis Complete", "✅")
                console.print("\n[green]📁 Report saved to: " + final_state.get("report_filepath", "Unknown") + "[/green]")
            
            print_completion_message("MediCheck AI", "Your Health Analysis Assistant")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Thank you for using MediCheck AI! Stay healthy![/yellow]")
        sys.exit(0)
        
    except Exception as e:
        print_error("An unexpected error occurred: " + str(e))
        console.print("[yellow]Please try again or consult a healthcare professional directly.[/yellow]")
        sys.exit(1)

if __name__ == "__main__":
    main()
