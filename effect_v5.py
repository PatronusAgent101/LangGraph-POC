from typing import Dict, List, Any, Tuple, Optional
import json
from pydantic import BaseModel, Field
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


# Define data models for structured outputs
class ControlMetrics(BaseModel):
    total_controls: int = Field(description="Total number of controls tested")
    key_controls_passed: int = Field(description="Number of key controls passed")
    key_controls_failed: int = Field(description="Number of key controls failed")
    nonkey_controls_passed: int = Field(description="Number of non-key controls passed")
    nonkey_controls_failed: int = Field(description="Number of non-key controls failed")
    pass_rate: float = Field(description="Percentage of controls passed")
    sample_ratio_average: float = Field(description="Average ratio of sample to population")
    
class RiskAssessment(BaseModel):
    effectiveness_rating: int = Field(description="Rating from 1-5 where 1 is highly effective and 5 is highly ineffective")
    rating_description: str = Field(description="Description of the effectiveness rating")
    identified_gaps: List[str] = Field(description="List of identified gaps in risk mitigation")
    countries_of_concern: List[str] = Field(description="Countries with concerning test results")
    
class ControlAssessment(BaseModel):
    metrics: ControlMetrics = Field(description="Metrics about control effectiveness")
    assessment: RiskAssessment = Field(description="Assessment of control effectiveness and risks")
    recommendations: List[str] = Field(description="Recommendations for improving control effectiveness")


class ControlEffectivenessAgent:
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0):
        self.model = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.graph = self._build_graph()
        
        # System prompt for the LLM
        self.system_prompt = """
        You are a Control Effectiveness Assessor Agent specializing in evaluating control testing results and providing assessments of control effectiveness. Your task is to:

        1. Analyze the test results provided
        2. Extract key metrics about control effectiveness
        3. Identify patterns and anomalies in the test results
        4. Assess the overall effectiveness of controls on a scale of 1-5 where:
           - 1: Highly Effective (Controls working as intended with minimal failures)
           - 2: Effective (Controls generally working with minor issues)
           - 3: Moderately Effective (Controls working but with notable gaps)
           - 4: Ineffective (Controls have significant failures)
           - 5: Highly Ineffective (Controls routinely failing or not implemented)
        5. Identify gaps where risks are not adequately mitigated
        6. Provide recommendations for improving control effectiveness

        Use Bayesian inference principles where appropriate to assess probability of control failures based on the sample data available. Consider both failed tests and high deviation in sample ratios as indicators of potential control weaknesses.

        For countries with multiple test results, use the worst-case results for your assessment.

        Provide your analysis in a structured format with clear metrics, assessment, and recommendations.
        """
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph for the agent."""
        graph = StateGraph(StateGraph.construct_init({"input": "", "history": [], "assessment": {}}))
        
        # Add nodes to the graph
        graph.add_node("parse_input", self._parse_input)
        graph.add_node("extract_metrics", self._extract_metrics)
        graph.add_node("assess_effectiveness", self._assess_effectiveness)
        graph.add_node("generate_recommendations", self._generate_recommendations)
        graph.add_node("format_output", self._format_output)
        
        # Define graph edges
        graph.add_edge("parse_input", "extract_metrics")
        graph.add_edge("extract_metrics", "assess_effectiveness")
        graph.add_edge("assess_effectiveness", "generate_recommendations")
        graph.add_edge("generate_recommendations", "format_output")
        graph.add_edge("format_output", END)
        
        # Set entry point
        graph.set_entry_point("parse_input")
        
        return graph
    
    def _parse_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the input data from the Tester Agent."""
        # The parsing logic depends on the format of the input
        # For the sample data, we'll assume it's in JSON format
        try:
            data = json.loads(state["input"]) if isinstance(state["input"], str) else state["input"]
            return {"input": data, "history": state["history"], "parsed_data": data}
        except json.JSONDecodeError:
            # If the input is not valid JSON, we'll use the LLM to try to extract structured data
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Extract the key information from the following test results data into a structured format:"),
                ("human", state["input"])
            ])
            
            response = self.model.invoke(prompt)
            
            try:
                # Try to parse the LLM's response as JSON
                parsed_data = json.loads(response.content)
            except json.JSONDecodeError:
                # If that fails, use a best-effort approach
                parsed_data = {"raw_text": response.content}
            
            return {"input": state["input"], "history": state["history"], "parsed_data": parsed_data}
    
    def _extract_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from the parsed data."""
        parsed_data = state["parsed_data"]
        
        # Prepare the prompt for the LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt + "\n\nYour task now is to extract the following metrics from the test results data:\n"
                      "- Total number of controls tested\n"
                      "- Number of key controls passed/failed (if available)\n"
                      "- Number of non-key controls passed/failed (if available)\n"
                      "- Pass rate\n"
                      "- Sample ratio average\n\n"
                      "If any of this information is not available, indicate 'no data available'."),
            ("human", f"Here is the test results data: {json.dumps(parsed_data, indent=2)}")
        ])
        
        # Use JsonOutputParser to get structured output
        parser = JsonOutputParser(pydantic_object=ControlMetrics)
        
        # Combine prompt with parser
        chain = prompt | self.model | parser
        
        # Run the chain
        metrics = chain.invoke({})
        
        return {"input": state["input"], "history": state["history"], "parsed_data": parsed_data, "metrics": metrics}
    
    def _assess_effectiveness(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the effectiveness of controls based on the metrics."""
        metrics = state["metrics"]
        parsed_data = state["parsed_data"]
        
        # Prepare the prompt for the LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt + "\n\nYour task now is to assess the effectiveness of controls based on the metrics and test results data. "
                      "Rate the effectiveness on a scale of 1-5 where 1 is highly effective and 5 is highly ineffective. "
                      "Identify gaps where risks are not adequately mitigated and list countries with concerning test results."),
            ("human", f"Here are the metrics: {json.dumps(metrics, indent=2)}\n\n"
                      f"Here is the full test results data: {json.dumps(parsed_data, indent=2)}")
        ])
        
        # Use JsonOutputParser to get structured output
        parser = JsonOutputParser(pydantic_object=RiskAssessment)
        
        # Combine prompt with parser
        chain = prompt | self.model | parser
        
        # Run the chain
        assessment = chain.invoke({})
        
        return {
            "input": state["input"], 
            "history": state["history"], 
            "parsed_data": parsed_data, 
            "metrics": metrics, 
            "assessment": assessment
        }
    
    def _generate_recommendations(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for improving control effectiveness."""
        metrics = state["metrics"]
        assessment = state["assessment"]
        parsed_data = state["parsed_data"]
        
        # Prepare the prompt for the LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt + "\n\nYour task now is to generate actionable recommendations for improving control effectiveness "
                      "based on the metrics, assessment, and test results data."),
            ("human", f"Here are the metrics: {json.dumps(metrics, indent=2)}\n\n"
                      f"Here is the assessment: {json.dumps(assessment, indent=2)}\n\n"
                      f"Here is the full test results data: {json.dumps(parsed_data, indent=2)}")
        ])
        
        # Run the chain
        response = self.model.invoke(prompt)
        
        # Parse the response to extract recommendations
        try:
            recommendations = json.loads(response.content)
            if isinstance(recommendations, dict) and "recommendations" in recommendations:
                recommendations = recommendations["recommendations"]
            elif not isinstance(recommendations, list):
                recommendations = [response.content]
        except json.JSONDecodeError:
            # If parsing fails, extract recommendations from the text
            recommendations = [line.strip() for line in response.content.split("\n") if line.strip().startswith("- ")]
            if not recommendations:
                recommendations = [response.content]
        
        return {
            "input": state["input"], 
            "history": state["history"], 
            "parsed_data": parsed_data, 
            "metrics": metrics, 
            "assessment": assessment, 
            "recommendations": recommendations
        }
    
    def _format_output(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Format the final output of the agent."""
        metrics = state["metrics"]
        assessment = state["assessment"]
        recommendations = state["recommendations"]
        
        control_assessment = ControlAssessment(
            metrics=metrics,
            assessment=assessment,
            recommendations=recommendations
        )
        
        return {
            "input": state["input"], 
            "history": state["history"], 
            "assessment": control_assessment
        }
    
    def run(self, input_data: Any) -> ControlAssessment:
        """Run the Control Effectiveness Assessor Agent on the input data."""
        initial_state = {"input": input_data, "history": [], "assessment": {}}
        final_state = self.graph.invoke(initial_state)
        return final_state["assessment"]


# Example usage
if __name__ == "__main__":
    # Sample data from the Tester Agent
    sample_data = [
        {
            "Summary for Passed Tests": {
                "Key Observations": {
                    "Total Tests": 3120,
                    "Overall Population Average": "281.75%",
                    "Sample Size Average": "34.75%",
                    "Sample Ratio Average": "34.69%",
                    "Sample Ratio Standard Deviation": "28.92%"
                },
                "Potential Patterns or Anomalies": {
                    "Min Deviation": "1.42%",
                    "Max Deviation": "100%",
                    "By Country": {
                        "CHINA": "100%",
                        "BAHRAIN": "65.70%",
                        "GHANA": "54.64%",
                        "TAIWAN": "51.93%",
                        "INDONESIA": "50.57%",
                        "BANGLADESH": "36.67%",
                        "ZIMBABWE": "36.58%",
                        "JERSEY": "35.11%",
                        "BRUNEI DARUSSALAM": "34.10%",
                        "COTE D'IVOIRE": "34.10%",
                        "NEPAL": "34.10%",
                        "SRI LANKA": "34.10%",
                        "TANZANIA, UNITED REPUBLIC OF": "34.10%",
                        "GAMBIA": "33.57%",
                        "SIERRA LEONE": "33.57%",
                        "VIETNAM": "33.23%",
                        "UGANDA": "29.71%",
                        "ZAMBIA": "24.51%",
                        "PAKISTAN": "22.67%",
                        "INDIA": "15.39%",
                        "MALAYSIA": "12.10%",
                        "BOTSWANA": "8.19%",
                        "UNITED ARAB EMIRATES": "7.76%",
                        "NIGERIA": "7.27%",
                        "SINGAPORE": "5.32%",
                        "KENYA": "5.11%",
                        "HONG KONG": "4.18%"
                    }
                },
                "Recommendations for Further Analysis": [
                    "Investigate the high deviation in sample ratios for countries like China (100%) and Bahrain (65.70%) to understand the underlying reasons.",
                    "Analyze the low deviation in sample ratios for countries like Hong Kong (4.18%) and Kenya (5.11%) to ensure that the sampling process is consistent and representative.",
                    "Conduct a deeper analysis on countries with mid-range deviations (e.g., Ghana, Indonesia, Taiwan) to identify any specific factors contributing to these variations.",
                    "Review the sampling methodology to ensure that it is robust and can be applied uniformly across different countries and regions.",
                    "Consider implementing additional checks or controls for countries with high deviations to ensure data accuracy and reliability."
                ]
            }
        },
        {
            "Summary for Failed Tests": {
                "Key Observations": {
                    "Total Tests Conducted": 120,
                    "Countries Impacted": [
                        "SINGAPORE",
                        "HONG KONG"
                    ],
                    "Overall Population Average": "1457.5%",
                    "Defect Average Percentage": "6.0%",
                    "Sample Average Percentage": "31.5%",
                    "Sample to Population Average Percentage": "2.27%",
                    "Sample to Population Standard Deviation Percentage": "0.45%"
                },
                "Potential Patterns or Anomalies": {
                    "High Defect Count in Hong Kong": "Hong Kong has a higher defect count (10) compared to Singapore (2).",
                    "Daily Report Issues": "Falcon system had 9 exceptions where daily perm block reports were not sent on time, indicating a potential issue with daily monitoring.",
                    "Test Mode Issues": "Several defects were related to rules being in test mode and not live, leading to unauthorized updates and removals.",
                    "Consistent Issues in Singapore": "Singapore had consistent issues with blocklisting removal approvals not being available for multiple cases."
                },
                "Recommendations for Further Analysis": {
                    "Daily Monitoring Improvement": "Investigate and improve the daily monitoring process for the Falcon system to ensure timely report generation and sharing.",
                    "Test Mode Protocols": "Review and enhance protocols for handling rules in test mode to prevent unauthorized updates and ensure proper approvals are in place.",
                    "Cross-Market Support": "Implement and monitor the effectiveness of the backup manager identified for cross-market support to ensure timely report sign-offs.",
                    "Country-Specific Analysis": "Conduct a deeper analysis of the defects in Hong Kong and Singapore to identify root causes and implement targeted improvements."
                }
            }
        }
    ]
    
    # Create and run the agent
    agent = ControlEffectivenessAgent()
    result = agent.run(sample_data)
    
    # Print the result
    print(json.dumps(result.model_dump(), indent=2))
