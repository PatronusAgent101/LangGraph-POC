import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import langgraph as lg
from langgraph.graph import StateGraph, END


class ControlMetrics(BaseModel):
    """Model for control metrics results"""
    key_controls_total: int = Field(0, description="Total number of key controls")
    nonkey_controls_total: int = Field(0, description="Total number of non-key controls")
    key_controls_passed: int = Field(0, description="Number of key controls that passed")
    key_controls_failed: int = Field(0, description="Number of key controls that failed")
    nonkey_controls_passed: int = Field(0, description="Number of non-key controls that passed")
    nonkey_controls_failed: int = Field(0, description="Number of non-key controls that failed")
    controls_without_results: int = Field(0, description="Number of controls without test results")
    data_availability: str = Field("complete", description="Completeness of data")


class ControlEffectivenessAgent:
    """
    Agent for assessing control effectiveness based on risk and control data.
    Uses a 1-5 rating scale where 1 is lowest risk and 5 is highest risk.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.1,
        verbose: bool = False
    ):
        """
        Initialize the ControlEffectivenessAgent with model settings and system prompt.
        
        Args:
            model_name: The name of the LLM model to use
            temperature: Sampling temperature for the model
            verbose: Whether to print detailed outputs
        """
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )
        
        # Define the system prompt for risk assessment analysis
        self.system_prompt = """
        You are a Control Effectiveness Assessor Agent specializing in analyzing risk and control data.
        
        Your task is to analyze the provided risk and control data and determine the effectiveness
        of the control framework on a scale of 1 to 5, where:
        
        1 = Very Low Risk (Highly Effective Controls)
        2 = Low Risk (Effective Controls)
        3 = Moderate Risk (Somewhat Effective Controls)
        4 = High Risk (Less Effective Controls)
        5 = Very High Risk (Ineffective Controls)
        
        When determining the effectiveness rating, consider:
        1. The number of controls that passed vs. failed
        2. The ratio of key controls to non-key controls
        3. The number of controls without test results
        4. The alignment between risks and controls
        5. The overall coverage of the control framework
        
        Provide a detailed justification for your rating that includes:
        - A summary of the control framework
        - The strengths and weaknesses identified
        - The specific factors that informed your rating
        - Any recommendations for improvement
        
        Also provide the following metrics based on the data:
        - Number of key controls (total, passed, failed)
        - Number of non-key controls (total, passed, failed)
        - Number of controls without test results
        
        If any data is not available, indicate "no data available" for that metric.
        """
        
        # Create workflow graph
        self.workflow = self._create_workflow()
        
        # Store assessment results
        self.assessment_results = {}
        self.control_metrics = ControlMetrics()
        
    def _create_workflow(self) -> StateGraph:
        """
        Create the LangGraph workflow for control effectiveness assessment.
        
        Returns:
            StateGraph: Configured workflow for control effectiveness assessment
        """
        # Define the nodes in our graph
        nodes = {
            "parse_input_data": self.parse_input_data,
            "calculate_control_metrics": self.calculate_control_metrics,
            "evaluate_control_effectiveness": self.evaluate_control_effectiveness,
            "generate_assessment_report": self.generate_assessment_report
        }
        
        # Create the graph
        workflow = StateGraph(nodes=nodes)
        
        # Define the edges (the flow between nodes)
        workflow.add_edge("parse_input_data", "calculate_control_metrics")
        workflow.add_edge("calculate_control_metrics", "evaluate_control_effectiveness")
        workflow.add_edge("evaluate_control_effectiveness", "generate_assessment_report")
        workflow.add_edge("generate_assessment_report", END)
        
        # Compile the graph
        return workflow.compile()
    
    def parse_input_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the input data containing risk and control information.
        
        Args:
            state: Current state of the workflow containing input data
            
        Returns:
            Updated state with parsed data
        """
        if self.verbose:
            print("Parsing input data...")
        
        input_data = state.get("input_data")
        if not input_data:
            state["error"] = "Input data not provided"
            return state
        
        try:
            # If input_data is already a dictionary, use it directly
            if isinstance(input_data, dict):
                parsed_data = input_data
            # If it's a string, try to parse it as JSON
            elif isinstance(input_data, str):
                parsed_data = json.loads(input_data)
            else:
                state["error"] = f"Unsupported input data type: {type(input_data)}"
                return state
            
            # Extract risks and controls from parsed data
            risks = parsed_data.get("risks", [])
            controls = parsed_data.get("controls", [])
            
            # Update state with parsed data
            state["risks"] = risks
            state["controls"] = controls
            
            if self.verbose:
                print(f"Successfully parsed input data: {len(risks)} risks and {len(controls)} controls")
                
        except Exception as e:
            state["error"] = f"Error parsing input data: {e}"
            
        return state
    
    def calculate_control_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metrics related to controls.
        
        Args:
            state: Current state of the workflow containing parsed data
            
        Returns:
            Updated state with calculated metrics
        """
        if self.verbose:
            print("Calculating control metrics...")
        
        controls = state.get("controls", [])
        if not controls:
            state["error"] = "No controls available for metric calculation"
            state["metrics"] = {
                "key_controls_total": "no data available",
                "nonkey_controls_total": "no data available",
                "key_controls_passed": "no data available",
                "key_controls_failed": "no data available",
                "nonkey_controls_passed": "no data available",
                "nonkey_controls_failed": "no data available",
                "controls_without_results": "no data available",
                "data_availability": "incomplete"
            }
            return state
        
        # Initialize metrics
        metrics = ControlMetrics()
        
        # Calculate metrics
        for control in controls:
            # Check if control is key or non-key
            is_key = control.get("isKey", False)
            
            # Check if control has test results
            has_results = "testResult" in control
            
            # Check if control passed or failed
            test_result = control.get("testResult", "")
            passed = test_result.lower() == "passed" if has_results else False
            
            # Update metrics
            if is_key:
                metrics.key_controls_total += 1
                if has_results:
                    if passed:
                        metrics.key_controls_passed += 1
                    else:
                        metrics.key_controls_failed += 1
                else:
                    metrics.controls_without_results += 1
            else:
                metrics.nonkey_controls_total += 1
                if has_results:
                    if passed:
                        metrics.nonkey_controls_passed += 1
                    else:
                        metrics.nonkey_controls_failed += 1
                else:
                    metrics.controls_without_results += 1
        
        # Check data availability
        if metrics.key_controls_total == 0 and metrics.nonkey_controls_total == 0:
            metrics.data_availability = "incomplete"
        
        # Update state with calculated metrics
        state["metrics"] = metrics.dict()
        self.control_metrics = metrics
        
        if self.verbose:
            print(f"Calculated control metrics: {metrics}")
        
        return state
    
    def evaluate_control_effectiveness(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of the control framework.
        
        Args:
            state: Current state of the workflow containing parsed data and metrics
            
        Returns:
            Updated state with effectiveness evaluation
        """
        if self.verbose:
            print("Evaluating control effectiveness...")
        
        risks = state.get("risks", [])
        controls = state.get("controls", [])
        metrics = state.get("metrics", {})
        
        if not risks or not controls:
            state["error"] = "Insufficient data for effectiveness evaluation"
            state["effectiveness_rating"] = "no data available"
            state["effectiveness_justification"] = "Insufficient data to evaluate effectiveness"
            return state
        
        # Create a prompt for effectiveness evaluation
        evaluation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
            Please evaluate the effectiveness of the control framework based on the following data:
            
            Risks:
            {json.dumps(risks, indent=2)}
            
            Controls:
            {json.dumps(controls, indent=2)}
            
            Control Metrics:
            {json.dumps(metrics, indent=2)}
            
            Provide a rating on a scale of 1 to 5 (where 1 is lowest risk and 5 is highest risk)
            and a detailed justification for your rating.
            """)
        ])
        
        # Create a chain for effectiveness evaluation
        evaluation_chain = evaluation_prompt | self.llm | StrOutputParser()
        
        # Execute the chain
        try:
            evaluation_result = evaluation_chain.invoke({})
            
            # Extract rating and justification using another prompt
            extraction_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="Extract the effectiveness rating and justification from the following evaluation."),
                HumanMessage(content=f"""
                From the following evaluation, extract:
                1. The effectiveness rating (a number from 1 to 5)
                2. The justification for the rating
                
                Format the response as a JSON object with "rating" and "justification" fields.
                
                Evaluation:
                {evaluation_result}
                """)
            ])
            
            extraction_chain = extraction_prompt | self.llm | StrOutputParser()
            extraction_result = extraction_chain.invoke({})
            
            # Parse the extraction result
            try:
                extracted_data = json.loads(extraction_result)
                rating = extracted_data.get("rating")
                justification = extracted_data.get("justification")
                
                state["effectiveness_rating"] = rating
                state["effectiveness_justification"] = justification
                
                if self.verbose:
                    print(f"Effectiveness rating: {rating}")
                    
            except json.JSONDecodeError:
                state["effectiveness_rating"] = "no data available"
                state["effectiveness_justification"] = evaluation_result
                state["error"] = "Failed to parse effectiveness evaluation result as JSON"
                
        except Exception as e:
            state["error"] = f"Error evaluating control effectiveness: {e}"
            
        return state
    
    def generate_assessment_report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive assessment report.
        
        Args:
            state: Current state of the workflow containing parsed data, metrics, and effectiveness evaluation
            
        Returns:
            Updated state with assessment report
        """
        if self.verbose:
            print("Generating assessment report...")
        
        metrics = state.get("metrics", {})
        rating = state.get("effectiveness_rating", "no data available")
        justification = state.get("effectiveness_justification", "")
        
        # Format metrics for report
        formatted_metrics = {}
        for key, value in metrics.items():
            if key == "data_availability":
                formatted_metrics[key] = value
            elif value == 0 and key.endswith("_total"):
                formatted_metrics[key] = "no data available"
            else:
                formatted_metrics[key] = value
        
        # Create assessment report
        assessment_report = {
            "effectiveness_rating": rating,
            "effectiveness_justification": justification,
            "metrics": formatted_metrics
        }
        
        # Update state with assessment report
        state["assessment_report"] = assessment_report
        self.assessment_results = assessment_report
        
        if self.verbose:
            print("Assessment report generated successfully")
        
        return state
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the control effectiveness assessment workflow.
        
        Args:
            input_data: Risk and control data to assess
            
        Returns:
            Assessment results and metrics
        """
        # Initialize state with input data
        state = {"input_data": input_data}
        
        # Run workflow
        if self.verbose:
            print("Starting control effectiveness assessment workflow...")
            
        result = self.workflow.invoke(state)
        
        if self.verbose:
            print("Control effectiveness assessment workflow completed")
        
        return result.get("assessment_report", {})
    
    def get_metrics_report(self) -> Dict[str, Any]:
        """
        Get a formatted report of the control metrics.
        
        Returns:
            Dictionary containing control metrics
        """
        metrics = self.control_metrics.dict()
        
        # Format metrics with "no data available" where appropriate
        formatted_metrics = {}
        for key, value in metrics.items():
            if key == "data_availability":
                formatted_metrics[key] = value
            elif value == 0 and key.endswith("_total"):
                formatted_metrics[key] = "no data available"
            else:
                formatted_metrics[key] = value
        
        return formatted_metrics
    
    def get_sample_data(self) -> Dict[str, Any]:
        """
        Generate sample data for demonstration purposes.
        
        Returns:
            Dictionary with sample risks and controls
        """
        sample_data = {
            "risks": [
                {
                    "id": "R-001",
                    "description": "Unauthorized access to financial systems",
                    "likelihood": "Medium",
                    "impact": "High",
                    "inherentRiskLevel": "High"
                },
                {
                    "id": "R-002",
                    "description": "Incomplete financial reporting",
                    "likelihood": "Low",
                    "impact": "High",
                    "inherentRiskLevel": "Medium"
                },
                {
                    "id": "R-003",
                    "description": "Data loss due to system failure",
                    "likelihood": "Low",
                    "impact": "Very High",
                    "inherentRiskLevel": "High"
                }
            ],
            "controls": [
                {
                    "id": "C-001",
                    "description": "Multi-factor authentication for all financial systems",
                    "isKey": True,
                    "relatedRisks": ["R-001"],
                    "testResult": "Passed"
                },
                {
                    "id": "C-002",
                    "description": "Regular review of access permissions
