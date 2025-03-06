import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import langgraph as lg
from langgraph.graph import StateGraph, END


class ControlEffectivenessAgent:
    """
    Agent for assessing control effectiveness based on risk assessment documents.
    Uses LangGraph to orchestrate a workflow for analyzing controls and their effectiveness.
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
        You are a Control Effectiveness Assessor Agent specializing in analyzing risk assessments.
        Your task is to thoroughly analyze risk assessment documents and evaluate the effectiveness
        of the controls mentioned in these documents.
        
        When analyzing a risk assessment document, you should:
        
        1. Identify all risks mentioned in the document
        2. For each risk, identify:
           - The risk description
           - The potential impact of the risk
           - The likelihood of the risk occurring
           - The current risk level (e.g., High, Medium, Low)
           - All existing controls associated with the risk
        
        3. For each control:
           - Assess its design effectiveness (whether the control is appropriately designed to mitigate the risk)
           - Assess its operational effectiveness (whether the control is functioning as intended)
           - Identify any gaps or weaknesses in the control
           - Recommend improvements or additional controls if necessary
        
        4. Provide an overall assessment of the risk management approach:
           - Are the risks well-identified and described?
           - Are the controls comprehensive and addressing all aspects of the risks?
           - Is there a clear relationship between risks and controls?
           - Are there any significant gaps in the risk management approach?
        
        Your analysis should be detailed, comprehensive, and provide actionable insights to improve the risk management framework.
        Always provide a structured assessment with clear categories and explicit reasoning for your conclusions.
        """
        
        # Create workflow graph
        self.workflow = self._create_workflow()
        
        # Store extracted information
        self.risk_data = {}
        self.control_data = {}
        self.effectiveness_assessment = {}
        
    def _create_workflow(self) -> StateGraph:
        """
        Create the LangGraph workflow for control effectiveness assessment.
        
        Returns:
            StateGraph: Configured workflow for risk assessment analysis
        """
        # Define the nodes in our graph
        nodes = {
            "read_document": self.read_document,
            "extract_risks": self.extract_risks,
            "extract_controls": self.extract_controls,
            "assess_control_effectiveness": self.assess_control_effectiveness,
            "identify_gaps": self.identify_gaps,
            "generate_recommendations": self.generate_recommendations,
            "prepare_final_report": self.prepare_final_report
        }
        
        # Create the graph
        workflow = StateGraph(nodes=nodes)
        
        # Define the edges (the flow between nodes)
        workflow.add_edge("read_document", "extract_risks")
        workflow.add_edge("extract_risks", "extract_controls")
        workflow.add_edge("extract_controls", "assess_control_effectiveness")
        workflow.add_edge("assess_control_effectiveness", "identify_gaps")
        workflow.add_edge("identify_gaps", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "prepare_final_report")
        workflow.add_edge("prepare_final_report", END)
        
        # Compile the graph
        return workflow.compile()
        
    def read_document(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Read and process the risk assessment document.
        
        Args:
            state: Current state of the workflow
            
        Returns:
            Updated state with document content
        """
        if self.verbose:
            print("Reading risk assessment document...")
        
        document_path = state.get("document_path")
        if not document_path:
            raise ValueError("Document path not provided in the state")
        
        try:
            with open(document_path, 'r') as file:
                document_content = file.read()
                
            # Update state with document content
            state["document_content"] = document_content
            
            if self.verbose:
                print(f"Successfully read document: {document_path}")
                
        except Exception as e:
            print(f"Error reading document: {e}")
            state["error"] = f"Error reading document: {e}"
            
        return state
    
    def extract_risks(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract risks from the risk assessment document.
        
        Args:
            state: Current state of the workflow containing document content
            
        Returns:
            Updated state with extracted risks
        """
        if self.verbose:
            print("Extracting risks from document...")
        
        document_content = state.get("document_content")
        if not document_content:
            state["error"] = "Document content not available for risk extraction"
            return state
        
        # Create a prompt for risk extraction
        risk_extraction_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"{self.system_prompt}\n\nYour task is to identify and extract all risks mentioned in the document."),
            HumanMessage(content=f"Please extract all risks from the following risk assessment document. For each risk, provide: risk description, impact, likelihood, and current risk level.\n\nDocument content:\n{document_content}")
        ])
        
        # Create a chain for risk extraction
        risk_extraction_chain = risk_extraction_prompt | self.llm | StrOutputParser()
        
        # Execute the chain
        try:
            risk_extraction_result = risk_extraction_chain.invoke({})
            
            # Structured prompt to get JSON output
            risk_json_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="Convert the risk information into a structured JSON format."),
                HumanMessage(content=f"Convert the following risk information into a JSON structure with an array of risks, each containing 'description', 'impact', 'likelihood', and 'riskLevel':\n\n{risk_extraction_result}")
            ])
            
            risk_json_chain = risk_json_prompt | self.llm | StrOutputParser()
            risk_json_result = risk_json_chain.invoke({})
            
            # Parse the JSON result
            try:
                risks = json.loads(risk_json_result)
                state["risks"] = risks
                self.risk_data = risks
                
                if self.verbose:
                    print(f"Extracted {len(risks.get('risks', []))} risks from the document")
                    
            except json.JSONDecodeError:
                state["risks"] = {"risks": []}
                state["risk_extraction_raw"] = risk_json_result
                state["error"] = "Failed to parse risk extraction result as JSON"
                
        except Exception as e:
            state["error"] = f"Error extracting risks: {e}"
            
        return state
    
    def extract_controls(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract controls from the risk assessment document.
        
        Args:
            state: Current state of the workflow containing document content and risks
            
        Returns:
            Updated state with extracted controls
        """
        if self.verbose:
            print("Extracting controls from document...")
        
        document_content = state.get("document_content")
        risks = state.get("risks", {"risks": []})
        
        if not document_content:
            state["error"] = "Document content not available for control extraction"
            return state
        
        # Create a prompt for control extraction
        control_extraction_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"{self.system_prompt}\n\nYour task is to identify and extract all controls mentioned in the document."),
            HumanMessage(content=f"Please extract all controls from the following risk assessment document. For each control, provide: control description, the risk(s) it addresses, and its implementation status.\n\nDocument content:\n{document_content}\n\nIdentified risks:\n{json.dumps(risks, indent=2)}")
        ])
        
        # Create a chain for control extraction
        control_extraction_chain = control_extraction_prompt | self.llm | StrOutputParser()
        
        # Execute the chain
        try:
            control_extraction_result = control_extraction_chain.invoke({})
            
            # Structured prompt to get JSON output
            control_json_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="Convert the control information into a structured JSON format."),
                HumanMessage(content=f"Convert the following control information into a JSON structure with an array of controls, each containing 'description', 'relatedRisks' (array of risk descriptions), and 'implementationStatus':\n\n{control_extraction_result}")
            ])
            
            control_json_chain = control_json_prompt | self.llm | StrOutputParser()
            control_json_result = control_json_chain.invoke({})
            
            # Parse the JSON result
            try:
                controls = json.loads(control_json_result)
                state["controls"] = controls
                self.control_data = controls
                
                if self.verbose:
                    print(f"Extracted {len(controls.get('controls', []))} controls from the document")
                    
            except json.JSONDecodeError:
                state["controls"] = {"controls": []}
                state["control_extraction_raw"] = control_json_result
                state["error"] = "Failed to parse control extraction result as JSON"
                
        except Exception as e:
            state["error"] = f"Error extracting controls: {e}"
            
        return state
    
    def assess_control_effectiveness(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the effectiveness of controls based on the risk assessment document.
        
        Args:
            state: Current state of the workflow containing document content, risks, and controls
            
        Returns:
            Updated state with control effectiveness assessment
        """
        if self.verbose:
            print("Assessing control effectiveness...")
        
        document_content = state.get("document_content")
        risks = state.get("risks", {"risks": []})
        controls = state.get("controls", {"controls": []})
        
        if not document_content:
            state["error"] = "Document content not available for control effectiveness assessment"
            return state
        
        # Create a prompt for control effectiveness assessment
        assessment_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"{self.system_prompt}\n\nYour task is to assess the effectiveness of the controls identified in the document."),
            HumanMessage(content=f"""
            Please assess the effectiveness of each control in addressing the associated risks.
            For each control, provide:
            - An assessment of its design effectiveness (whether the control is appropriately designed to mitigate the risk)
            - An assessment of its operational effectiveness (whether the control is functioning as intended)
            - A rating of its overall effectiveness (High, Medium, Low)
            
            Document content:
            {document_content}
            
            Identified risks:
            {json.dumps(risks, indent=2)}
            
            Identified controls:
            {json.dumps(controls, indent=2)}
            """)
        ])
        
        # Create a chain for control effectiveness assessment
        assessment_chain = assessment_prompt | self.llm | StrOutputParser()
        
        # Execute the chain
        try:
            assessment_result = assessment_chain.invoke({})
            
            # Structured prompt to get JSON output
            assessment_json_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="Convert the control effectiveness assessment into a structured JSON format."),
                HumanMessage(content=f"Convert the following control effectiveness assessment into a JSON structure with an array of assessments, each containing 'controlDescription', 'designEffectiveness', 'operationalEffectiveness', and 'overallEffectivenessRating':\n\n{assessment_result}")
            ])
            
            assessment_json_chain = assessment_json_prompt | self.llm | StrOutputParser()
            assessment_json_result = assessment_json_chain.invoke({})
            
            # Parse the JSON result
            try:
                effectiveness_assessment = json.loads(assessment_json_result)
                state["effectiveness_assessment"] = effectiveness_assessment
                self.effectiveness_assessment = effectiveness_assessment
                
                if self.verbose:
                    print(f"Assessed effectiveness of {len(effectiveness_assessment.get('assessments', []))} controls")
                    
            except json.JSONDecodeError:
                state["effectiveness_assessment"] = {"assessments": []}
                state["assessment_raw"] = assessment_json_result
                state["error"] = "Failed to parse effectiveness assessment result as JSON"
                
        except Exception as e:
            state["error"] = f"Error assessing control effectiveness: {e}"
            
        return state
    
    def identify_gaps(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify gaps in the control framework based on the assessment.
        
        Args:
            state: Current state of the workflow containing document content, risks, controls, and effectiveness assessment
            
        Returns:
            Updated state with identified gaps
        """
        if self.verbose:
            print("Identifying gaps in control framework...")
        
        document_content = state.get("document_content")
        risks = state.get("risks", {"risks": []})
        controls = state.get("controls", {"controls": []})
        effectiveness_assessment = state.get("effectiveness_assessment", {"assessments": []})
        
        if not document_content:
            state["error"] = "Document content not available for gap identification"
            return state
        
        # Create a prompt for gap identification
        gap_identification_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"{self.system_prompt}\n\nYour task is to identify gaps in the control framework based on the assessment."),
            HumanMessage(content=f"""
            Please identify gaps in the control framework based on the risks, controls, and effectiveness assessment.
            Consider:
            - Risks that are not adequately addressed by existing controls
            - Controls with low effectiveness ratings
            - Areas where additional controls may be needed
            - Inconsistencies in the control framework
            
            Document content:
            {document_content}
            
            Identified risks:
            {json.dumps(risks, indent=2)}
            
            Identified controls:
            {json.dumps(controls, indent=2)}
            
            Control effectiveness assessment:
            {json.dumps(effectiveness_assessment, indent=2)}
            """)
        ])
        
        # Create a chain for gap identification
        gap_identification_chain = gap_identification_prompt | self.llm | StrOutputParser()
        
        # Execute the chain
        try:
            gap_identification_result = gap_identification_chain.invoke({})
            
            # Structured prompt to get JSON output
            gap_json_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="Convert the gap identification information into a structured JSON format."),
                HumanMessage(content=f"Convert the following gap identification information into a JSON structure with an array of gaps, each containing 'description', 'relatedRisks', 'severity', and 'impactOn
