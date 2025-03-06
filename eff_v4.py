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
    """Model for tracking control test metrics"""
    total_tests: int = Field(0, description="Total number of tests conducted")
    passed_tests: int = Field(0, description="Number of tests that passed")
    failed_tests: int = Field(0, description="Number of tests that failed")
    countries_with_failures: List[str] = Field([], description="Countries with test failures")
    sample_ratio_average: float = Field(0.0, description="Average sample ratio")
    sample_ratio_deviation: float = Field(0.0, description="Standard deviation of sample ratios")
    high_deviation_countries: List[Dict[str, float]] = Field([], description="Countries with high deviation")
    low_deviation_countries: List[Dict[str, float]] = Field([], description="Countries with low deviation")
    defect_patterns: List[str] = Field([], description="Identified patterns in defects")


class ControlEffectivenessAgent:
    """
    Agent for assessing control effectiveness based on test results data.
    Uses a 1-5 rating scale where 1 is lowest risk (most effective controls) and 5 is highest risk.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.1,
        verbose: bool = False
    ):
        """Initialize the Control Effectiveness Agent with model settings and system prompt"""
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
        You are a Control Effectiveness Assessor Agent specializing in analyzing risk control test data.
        
        Your task is to analyze the provided test results data and determine the effectiveness
        of the control framework on a scale of 1 to 5, where:
        
        1 = Very Low Risk (Highly Effective Controls) - Controls are working as intended with minimal failures
        2 = Low Risk (Effective Controls) - Controls are generally working with some minor issues
        3 = Moderate Risk (Somewhat Effective Controls) - Controls have notable issues but are still partly effective
        4 = High Risk (Less Effective Controls) - Controls have significant issues and require substantial improvement
        5 = Very High Risk (Ineffective Controls) - Controls are failing to mitigate risks adequately
        
        When determining the effectiveness rating, consider:
        1. The ratio of passed to failed tests
        2. The severity and patterns of failures
        3. The consistency of control application across different regions
        4. The sample sizes and testing methodology
        5. Any identified anomalies or patterns in the data
        
        Provide a detailed justification for your rating that includes:
        - A summary of key test results
        - The strengths and weaknesses identified in the control framework
        - The specific factors that informed your rating
        - Recommendations for improvement
        
        Also provide the following metrics if available in the data:
        - Number of tests (passed and failed)
        - Sample ratios and deviations
        - Countries or regions with notable patterns
        
        If any data is not available, indicate "no data available" for that metric.
        """
        
        # Create workflow graph
        self.workflow = self._create_workflow()
        
        # Store assessment results
        self.assessment_results = {}
        self.control_metrics = ControlMetrics()
        
    def _create_workflow(self) -> StateGraph:
        """Create the workflow graph for control effectiveness assessment"""
        # Define the nodes in our graph
        nodes = {
            "parse_input_data": self.parse_input_data,
            "extract_test_metrics": self.extract_test_metrics,
            "identify_patterns": self.identify_patterns,
            "evaluate_control_effectiveness": self.evaluate_control_effectiveness,
            "generate_assessment_report": self.generate_assessment_report
        }
        
        # Create the graph
        workflow = StateGraph(nodes=nodes)
        
        # Define the edges (the flow between nodes)
        workflow.add_edge("parse_input_data", "extract_test_metrics")
        workflow.add_edge("extract_test_metrics", "identify_patterns")
        workflow.add_edge("identify_patterns", "evaluate_control_effectiveness")
        workflow.add_edge("evaluate_control_effectiveness", "generate_assessment_report")
        workflow.add_edge("generate_assessment_report", END)
        
        # Compile the graph
        return workflow.compile()
    
    def parse_input_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the input data containing test results"""
        if self.verbose:
            print("Parsing input data...")
        
        input_data = state.get("input_data")
        if not input_data:
            state["error"] = "Input data not provided"
            return state
        
        try:
            # If input_data is already a dictionary or list, use it directly
            if isinstance(input_data, (dict, list)):
                parsed_data = input_data
            # If it's a string, try to parse it as JSON
            elif isinstance(input_data, str):
                parsed_data = json.loads(input_data)
            else:
                state["error"] = f"Unsupported input data type: {type(input_data)}"
                return state
            
            # Update state with parsed data
            state["parsed_data"] = parsed_data
            
            if self.verbose:
                print(f"Successfully parsed input data")
                
        except Exception as e:
            state["error"] = f"Error parsing input data: {e}"
            
        return state
    
    def extract_test_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from test results data"""
        if self.verbose:
            print("Extracting test metrics...")
        
        parsed_data = state.get("parsed_data")
        if not parsed_data:
            state["error"] = "No parsed data available for metric extraction"
            return state
        
        metrics = ControlMetrics()
        
        try:
            # Extract passed test metrics
            passed_tests_data = next((item["Summary for Passed Tests"] for item in parsed_data if "Summary for Passed Tests" in item), None)
            if passed_tests_data:
                key_observations = passed_tests_data.get("Key Observations", {})
                metrics.total_tests += int(key_observations.get("Total Tests", 0))
                metrics.passed_tests = int(key_observations.get("Total Tests", 0))
                
                # Extract sample ratio
                sample_ratio_str = key_observations.get("Sample Ratio Average", "0%").replace("%", "")
                metrics.sample_ratio_average = float(sample_ratio_str) if sample_ratio_str else 0
                
                # Extract standard deviation
                std_dev_str = key_observations.get("Sample Ratio Standard Deviation", "0%").replace("%", "")
                metrics.sample_ratio_deviation = float(std_dev_str) if std_dev_str else 0
                
                # Extract country data
                country_data = passed_tests_data.get("Potential Patterns or Anomalies", {}).get("By Country", {})
                
                # Identify high and low deviation countries
                for country, deviation_str in country_data.items():
                    deviation = float(deviation_str.replace("%", ""))
                    country_info = {"country": country, "deviation": deviation}
                    
                    if deviation > 50:  # High deviation threshold
                        metrics.high_deviation_countries.append(country_info)
                    elif deviation < 10:  # Low deviation threshold
                        metrics.low_deviation_countries.append(country_info)
            
            # Extract failed test metrics
            failed_tests_data = next((item["Summary for Failed Tests"] for item in parsed_data if "Summary for Failed Tests" in item), None)
            if failed_tests_data:
                key_observations = failed_tests_data.get("Key Observations", {})
                failed_tests_count = int(key_observations.get("Total Tests Conducted", 0))
                metrics.total_tests += failed_tests_count
                metrics.failed_tests = failed_tests_count
                
                # Extract countries with failures
                countries_impacted = key_observations.get("Countries Impacted", [])
                metrics.countries_with_failures = countries_impacted
                
                # Extract defect patterns
                patterns = failed_tests_data.get("Potential Patterns or Anomalies", {})
                metrics.defect_patterns = [f"{key}: {value}" for key, value in patterns.items() if isinstance(value, str)]
            
            # Update state with metrics
            state["metrics"] = metrics.dict()
            self.control_metrics = metrics
            
            if self.verbose:
                print(f"Extracted test metrics: {metrics.total_tests} total tests, {metrics.passed_tests} passed, {metrics.failed_tests} failed")
            
        except Exception as e:
            state["error"] = f"Error extracting test metrics: {e}"
            
        return state
    
    def identify_patterns(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Identify patterns and anomalies in the test data"""
        if self.verbose:
            print("Identifying patterns in test data...")
        
        parsed_data = state.get("parsed_data")
        metrics = state.get("metrics", {})
        
        if not parsed_data:
            state["error"] = "No parsed data available for pattern identification"
            return state
        
        try:
            patterns = []
            
            # Extract patterns from passed tests
            passed_tests_data = next((item["Summary for Passed Tests"] for item in parsed_data if "Summary for Passed Tests" in item), None)
            if passed_tests_data:
                anomalies = passed_tests_data.get("Potential Patterns or Anomalies", {})
                min_dev = anomalies.get("Min Deviation", "N/A")
                max_dev = anomalies.get("Max Deviation", "N/A")
                
                if min_dev != "N/A" and max_dev != "N/A":
                    patterns.append(f"Sample ratio deviation ranges from {min_dev} to {max_dev}")
                
                # Add recommendations as patterns
                recommendations = passed_tests_data.get("Recommendations for Further Analysis", [])
                for rec in recommendations:
                    patterns.append(f"Recommendation: {rec}")
            
            # Extract patterns from failed tests
            failed_tests_data = next((item["Summary for Failed Tests"] for item in parsed_data if "Summary for Failed Tests" in item), None)
            if failed_tests_data:
                anomalies = failed_tests_data.get("Potential Patterns or Anomalies", {})
                for key, value in anomalies.items():
                    if isinstance(value, str):
                        patterns.append(f"{key}: {value}")
                
                # Add recommendations as patterns
                recommendations = failed_tests_data.get("Recommendations for Further Analysis", {})
                for key, value in recommendations.items():
                    if isinstance(value, str):
                        patterns.append(f"Recommendation: {value}")
            
            # Update state with identified patterns
            state["identified_patterns"] = patterns
            
            if self.verbose:
                print(f"Identified {len(patterns)} patterns in the test data")
            
        except Exception as e:
            state["error"] = f"Error identifying patterns: {e}"
            
        return state
    
    def evaluate_control_effectiveness(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate control effectiveness based on test data and metrics"""
        if self.verbose:
            print("Evaluating control effectiveness...")
        
        parsed_data = state.get("parsed_data")
        metrics = state.get("metrics", {})
        identified_patterns = state.get("identified_patterns", [])
        
        if not parsed_data:
            state["error"] = "Insufficient data for effectiveness evaluation"
            state["effectiveness_rating"] = "no data available"
            state["effectiveness_justification"] = "Insufficient data to evaluate effectiveness"
            return state
        
        # Create a prompt for effectiveness evaluation
        evaluation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
            Please evaluate the effectiveness of the control framework based on the following data:
            
            Parsed Test Data:
            {json.dumps(parsed_data, indent=2)}
            
            Extracted Metrics:
            {json.dumps(metrics, indent=2)}
            
            Identified Patterns:
            {json.dumps(identified_patterns, indent=2)}
            
            Provide a rating on a scale of 1 to 5 (where 1 is lowest risk and 5 is highest risk)
            and a detailed justification for your rating. Focus on the effectiveness of the controls
            based on the test results.
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
        """Generate comprehensive assessment report"""
        if self.verbose:
            print("Generating assessment report...")
        
        metrics = state.get("metrics", {})
        rating = state.get("effectiveness_rating", "no data available")
        justification = state.get("effectiveness_justification", "")
        identified_patterns = state.get("identified_patterns", [])
        
        # Calculate key metrics for the report
        total_tests = metrics.get("total_tests", 0)
        passed_tests = metrics.get("passed_tests", 0)
        failed_tests = metrics.get("failed_tests", 0)
        
        pass_rate = (passed_tests / total
