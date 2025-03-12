import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple, Annotated
import time
import json
import re

# Define the state
class State(BaseModel):
    assessment_learned: Optional[str] = Field(default=None, description="Assessment learned from the evaluation")
    error: Optional[str] = Field(default=None, description="Error message if any")
    input_text: str = Field(default="", description="Input text to evaluate")
    rating: Optional[int] = Field(default=None, description="Overall rating from 1-5")
    rationale: Optional[str] = Field(default=None, description="Rationale for the rating")
    metrics_evaluation: Optional[Dict[str, int]] = Field(default=None, description="Evaluation scores per metric")
    status: str = Field(default="initialized", description="Current status of the process")
    reflection: Optional[str] = Field(default=None, description="Self-reflection on the assessment")
    final_assessment: Optional[str] = Field(default=None, description="Final assessment after reflection")
    final_rating: Optional[int] = Field(default=None, description="Final rating after reflection")

# Define the evaluation system
def evaluate_effectiveness(state: State) -> State:
    st.session_state.status_message = "Analyzing control effectiveness..."
    st.session_state.progress = 20
    
    try:
        # Define metrics prompt
        metrics_prompt = ChatPromptTemplate.from_template("""
        You are a control effectiveness evaluation agent. Evaluate the following control description based on these metrics:
        1. Clarity (1-5): How clearly the control is defined and communicated
        2. Appropriateness (1-5): How well the control addresses the identified risk
        3. Efficiency (1-5): How efficiently the control can be implemented
        4. Measurability (1-5): How easily the control's effectiveness can be measured
        5. Sustainability (1-5): How sustainable the control is over time

        Control: {input_text}
        
        Provide a JSON format response with:
        1. A score for each metric (1-5)
        2. A brief rationale for each score
        3. An overall score (1-5)
        4. Overall assessment
        
        Format:
        ```json
        {{
            "metrics": {{
                "clarity": {{
                    "score": <1-5>,
                    "rationale": "<brief explanation>"
                }},
                "appropriateness": {{
                    "score": <1-5>,
                    "rationale": "<brief explanation>"
                }},
                "efficiency": {{
                    "score": <1-5>,
                    "rationale": "<brief explanation>"
                }},
                "measurability": {{
                    "score": <1-5>,
                    "rationale": "<brief explanation>"
                }},
                "sustainability": {{
                    "score": <1-5>,
                    "rationale": "<brief explanation>"
                }}
            }},
            "overall_score": <1-5>,
            "overall_assessment": "<detailed assessment>"
        }}
        ```
        """)

        # LLM for evaluation
        model = ChatOpenAI(temperature=0, model="gpt-4")
        chain = metrics_prompt | model
        
        st.session_state.progress = 35
        st.session_state.status_message = "Calculating scores and generating assessment..."
        
        # Run the evaluation
        response = chain.invoke({"input_text": state.input_text})
        
        # Extract the JSON part from the response
        json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.content
            
        result = json.loads(json_str)
        
        # Update the state
        metrics_evaluation = {}
        for metric, data in result["metrics"].items():
            metrics_evaluation[metric] = data["score"]
            
        state.metrics_evaluation = metrics_evaluation
        state.rating = result["overall_score"]
        state.assessment_learned = result["overall_assessment"]
        state.rationale = "\n".join([f"{metric}: {data['rationale']}" for metric, data in result["metrics"].items()])
        state.status = "evaluated"
        
        st.session_state.progress = 50
        st.session_state.status_message = "Initial assessment complete. Starting self-reflection..."
        
    except Exception as e:
        state.error = str(e)
        state.status = "error"
        st.session_state.status_message = f"Error: {str(e)}"
        
    return state

# Define self-reflection system
def self_reflection(state: State) -> State:
    st.session_state.status_message = "Performing self-reflection on assessment..."
    st.session_state.progress = 65
    
    try:
        # Define reflection prompt
        reflection_prompt = ChatPromptTemplate.from_template("""
        You are a critical thinking agent that evaluates and provides feedback on control effectiveness assessments.
        
        Original Control: {input_text}
        
        Initial Assessment: {assessment_learned}
        Initial Rating: {rating}/5
        
        Detailed Rationale:
        {rationale}
        
        Your task:
        1. Critically evaluate the assessment for potential blind spots, biases, or areas that may have been overlooked
        2. Consider alternative perspectives that might change the evaluation
        3. Identify any potential inconsistencies between the rationale and the overall rating
        4. Suggest improvements to the assessment
        
        Provide a JSON format response with:
        1. Feedback points (3-5 critical reflections)
        2. Suggested perspective changes (if any)
        3. Overall reflection summary
        
        Format:
        ```json
        {{
            "feedback_points": [
                "<feedback point 1>",
                "<feedback point 2>",
                "<feedback point 3>"
            ],
            "perspective_changes": "<suggestions for alternative ways to look at the control>",
            "reflection_summary": "<overall summary of reflection>"
        }}
        ```
        """)

        # LLM for reflection
        model = ChatOpenAI(temperature=0, model="gpt-4")
        chain = reflection_prompt | model
        
        # Run the reflection
        response = chain.invoke({
            "input_text": state.input_text,
            "assessment_learned": state.assessment_learned,
            "rating": state.rating,
            "rationale": state.rationale
        })
        
        # Extract the JSON part from the response
        json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.content
            
        result = json.loads(json_str)
        
        # Update the state
        state.reflection = result["reflection_summary"]
        feedback_points = "\n".join([f"- {point}" for point in result["feedback_points"]])
        state.reflection = f"{result['reflection_summary']}\n\nKey feedback points:\n{feedback_points}\n\nPerspective changes to consider: {result['perspective_changes']}"
        state.status = "reflected"
        
        st.session_state.progress = 80
        st.session_state.status_message = "Self-reflection complete. Finalizing assessment..."
        
    except Exception as e:
        state.error = str(e)
        state.status = "error"
        st.session_state.status_message = f"Error during reflection: {str(e)}"
        
    return state

# Define reassessment system
def reassess(state: State) -> State:
    st.session_state.status_message = "Performing final assessment with reflection feedback..."
    st.session_state.progress = 90
    
    try:
        # Define reassessment prompt
        reassessment_prompt = ChatPromptTemplate.from_template("""
        You are a control effectiveness evaluation agent. You previously evaluated a control, and now you've received feedback from a self-reflection agent.
        
        Original Control: {input_text}
        
        Your Initial Assessment: {assessment_learned}
        Your Initial Rating: {rating}/5
        
        Self-Reflection Feedback:
        {reflection}
        
        Now, reassess the control taking into account the feedback you've received. Focus on providing only a single overall rating and assessment, not individual metric ratings.
        
        Provide a JSON format response with:
        1. A final overall score (1-5)
        2. A comprehensive final assessment that incorporates the reflection feedback
        
        Format:
        ```json
        {{
            "final_score": <1-5>,
            "final_assessment": "<comprehensive assessment>"
        }}
        ```
        """)

        # LLM for reassessment
        model = ChatOpenAI(temperature=0, model="gpt-4")
        chain = reassessment_prompt | model
        
        # Run the reassessment
        response = chain.invoke({
            "input_text": state.input_text,
            "assessment_learned": state.assessment_learned,
            "rating": state.rating,
            "reflection": state.reflection
        })
        
        # Extract the JSON part from the response
        json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.content
            
        result = json.loads(json_str)
        
        # Update the state
        state.final_rating = result["final_score"]
        state.final_assessment = result["final_assessment"]
        state.status = "completed"
        
        st.session_state.progress = 100
        st.session_state.status_message = "Assessment completed successfully!"
        
    except Exception as e:
        state.error = str(e)
        state.status = "error"
        st.session_state.status_message = f"Error during reassessment: {str(e)}"
        
    return state

# Define edges for the graph
def decide_after_evaluation(state: State) -> str:
    if state.error:
        return "error"
    return "reflect"

def decide_after_reflection(state: State) -> str:
    if state.error:
        return "error"
    return "reassess"

def decide_after_reassessment(state: State) -> str:
    if state.error:
        return "error"
    return "complete"

# Set up the graph
def create_graph():
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("evaluate", evaluate_effectiveness)
    workflow.add_node("reflect", self_reflection)
    workflow.add_node("reassess", reassess)
    
    # Add edges
    workflow.add_conditional_edges(
        "evaluate",
        decide_after_evaluation,
        {
            "error": END,
            "reflect": "reflect"
        }
    )
    
    workflow.add_conditional_edges(
        "reflect",
        decide_after_reflection,
        {
            "error": END,
            "reassess": "reassess"
        }
    )
    
    workflow.add_conditional_edges(
        "reassess",
        decide_after_reassessment,
        {
            "error": END,
            "complete": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("evaluate")
    
    return workflow.compile()

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Control Effectiveness Evaluator",
        page_icon="🛡️",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #f5f7f9;
    }
    .css-1p05t8e {
        border-radius: 15px;
        padding: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stProgress .st-bc {
        background-color: #4CAF50;
    }
    .metric-card {
        background-color: #f1f8ff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #3366ff;
    }
    .metric-header {
        font-weight: bold;
        color: #1a1a1a;
    }
    .overall-score {
        font-size: 24px;
        font-weight: bold;
        color: #3366ff;
    }
    .reflection-box {
        background-color: #fffde7;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    .improvement-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'progress' not in st.session_state:
        st.session_state.progress = 0
    if 'status_message' not in st.session_state:
        st.session_state.status_message = "Ready to evaluate"
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Header
    st.title("🛡️ Control Effectiveness Evaluator")
    st.markdown("Analyze and rate control effectiveness on a scale of 1-5 with self-reflection capabilities.")
    
    # Input section
    with st.container():
        st.subheader("Control Description")
        input_text = st.text_area(
            "Enter the control description to evaluate:",
            height=150,
            key="input_text"
        )
        
        col1, col2 = st.columns([1, 5])
        
        with col1:
            evaluate_button = st.button("Evaluate", type="primary")
        
    # Progress section
    if evaluate_button and input_text:
        # Create and run the graph
        graph = create_graph()
        
        # Status indicators
        status_container = st.container()
        
        with status_container:
            st.subheader("Evaluation Status")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize progress
            st.session_state.progress = 10
            st.session_state.status_message = "Starting evaluation..."
            
            progress_bar.progress(st.session_state.progress)
            status_text.info(st.session_state.status_message)
            
            # Run the graph
            initial_state = State(input_text=input_text)
            
            # Update progress
            for i in range(3):
                time.sleep(0.3)
                st.session_state.progress += 3
                progress_bar.progress(st.session_state.progress)
            
            result = graph.invoke(initial_state)
            
            # Final update
            progress_bar.progress(100)
            
            if result.error:
                status_text.error(f"Error: {result.error}")
            else:
                status_text.success("Evaluation completed successfully!")
                st.session_state.results = result
    
    # Display results
    if st.session_state.results and not st.session_state.results.error:
        result = st.session_state.results
        
        st.markdown("---")
        st.subheader("Evaluation Results")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Final Assessment", "Initial Assessment", "Self-Reflection"])
        
        with tab1:
            # Final assessment and rating
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Get color based on score
                if result.final_rating >= 4:
                    color = "#4CAF50"  # Green
                elif result.final_rating >= 3:
                    color = "#FFC107"  # Yellow
                else:
                    color = "#F44336"  # Red
                    
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background-color: #f1f8ff; border-radius: 10px; border: 1px solid {color};">
                    <div style="font-size: 14px; color: #666;">FINAL RATING</div>
                    <div style="font-size: 36px; font-weight: bold; color: {color};">{result.final_rating} / 5</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Final Assessment")
                st.markdown(f"> {result.final_assessment}")
            
            # Show improvement from initial to final
            rating_diff = result.final_rating - result.rating
            if rating_diff != 0:
                if rating_diff > 0:
                    diff_text = f"⬆️ Improved by {rating_diff} point{'s' if rating_diff > 1 else ''}"
                    diff_color = "#4CAF50"
                else:
                    diff_text = f"⬇️ Reduced by {abs(rating_diff)} point{'s' if abs(rating_diff) > 1 else ''}"
                    diff_color = "#F44336"
                    
                st.markdown(f"""
                <div style="padding: 10px; background-color: #f5f5f5; border-radius: 5px; margin-top: 15px;">
                    <span style="color: {diff_color}; font-weight: bold;">{diff_text}</span> after self-reflection
                </div>
                """, unsafe_allow_html=True)
                
        with tab2:
            # Initial assessment and rating
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Get color based on score
                if result.rating >= 4:
                    color = "#4CAF50"  # Green
                elif result.rating >= 3:
                    color = "#FFC107"  # Yellow
                else:
                    color = "#F44336"  # Red
                    
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 10px; border: 1px solid {color};">
                    <div style="font-size: 14px; color: #666;">INITIAL RATING</div>
                    <div style="font-size: 30px; font-weight: bold; color: {color};">{result.rating} / 5</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Initial Assessment")
                st.markdown(f"> {result.assessment_learned}")
            
            # Metrics breakdown (hidden by default)
            with st.expander("View Detailed Metrics Breakdown"):
                metric_cols = st.columns(len(result.metrics_evaluation))
                
                for i, (metric, score) in enumerate(result.metrics_evaluation.items()):
                    with metric_cols[i]:
                        # Get color based on score
                        if score >= 4:
                            color = "#4CAF50"  # Green
                        elif score >= 3:
                            color = "#FFC107"  # Yellow
                        else:
                            color = "#F44336"  # Red
                            
                        st.markdown(f"""
                        <div style="text-align: center; padding: 15px; background-color: #f9f9f9; border-radius: 8px; border-top: 5px solid {color};">
                            <div style="text-transform: uppercase; font-size: 12px; color: #666;">{metric}</div>
                            <div style="font-size: 24px; font-weight: bold; color: {color};">{score}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Detailed rationale
                rationale_lines = result.rationale.split("\n")
                for line in rationale_lines:
                    if line.strip():
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            metric, explanation = parts
                            st.markdown(f"**{metric.strip()}:** {explanation.strip()}")
                        else:
                            st.markdown(line)
                
        with tab3:
            # Self-reflection content
            st.markdown("### Self-Reflection Analysis")
            
            st.markdown(f"""
            <div class="reflection-box">
                <h4>Critical Reflection</h4>
                <p>{result.reflection}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show how reflection influenced the reassessment
            st.markdown("### How Reflection Influenced Final Assessment")
            
            # Differences between initial and final
            initial_lines = result.assessment_learned.split(". ")
            final_lines = result.final_assessment.split(". ")
            
            # Extract key differences (simplified approach)
            st.markdown(f"""
            <div class="improvement-box">
                <h4>Key Improvements</h4>
                <p>The self-reflection process helped refine the assessment by considering additional perspectives and addressing potential blind spots in the initial evaluation.</p>
                <p>Initial rating: <strong>{result.rating}/5</strong> → Final rating: <strong>{result.final_rating}/5</strong></p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
