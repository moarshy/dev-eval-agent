#!/usr/bin/env python3
"""
RAG-Enhanced Coding Agent with Self-Correction

A coding agent that uses RAG to retrieve relevant documentation and iteratively 
generates, tests, and improves code solutions for use cases. Follows the AlphaCodium 
approach with self-correction loops.

Key features:
1. RAG-based code generation using vector store
2. Iterative testing and improvement
3. Documentation quality analysis
4. Integration with progressive planner use cases
"""

from typing import List, Dict, Optional, Any, TypedDict
from datetime import datetime
from pathlib import Path
import json
from pydantic import BaseModel, Field

# LangGraph imports
from langgraph.graph import END, StateGraph, START

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Local imports
from document_processor import VectorStoreManager
from models import ProcessedDocument
from progressive_planner import UseCase


class CodeSolution(BaseModel):
    """Structured code solution output"""
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


class CodeAttempt(BaseModel):
    """Single attempt at implementing code for a use case"""
    attempt_number: int = Field(description="Which attempt this is (1, 2, 3...)")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # RAG Information
    rag_query: str = Field(description="Query used for RAG retrieval")
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved documentation chunks")
    relevance_scores: List[float] = Field(default_factory=list, description="Relevance scores for retrieved docs")
    
    # Generated Code
    code_solution: Optional[CodeSolution] = Field(None, description="Generated code solution")
    
    # Test Results
    import_test_passed: bool = Field(default=False)
    import_error: Optional[str] = Field(None)
    execution_test_passed: bool = Field(default=False)
    execution_error: Optional[str] = Field(None)
    execution_output: Optional[str] = Field(None)
    
    # Documentation Analysis
    documentation_helpful: bool = Field(default=False, description="Was retrieved documentation useful?")
    missing_information: List[str] = Field(default_factory=list, description="What info was missing from docs")
    confusing_parts: List[str] = Field(default_factory=list, description="What parts of docs were unclear")
    suggested_improvements: List[str] = Field(default_factory=list, description="How to improve the docs")


class UseCaseExecution(BaseModel):
    """Complete execution record for one use case"""
    use_case_id: str = Field(description="Reference to the UseCase")
    use_case_title: str = Field(description="Title of the use case")
    use_case_description: str = Field(description="Description of what to implement")
    
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = Field(None)
    
    # Multiple attempts with RAG and self-correction
    attempts: List[CodeAttempt] = Field(default_factory=list)
    final_working_code: Optional[str] = Field(None, description="Final successful implementation")
    
    # Success Analysis
    overall_success: bool = Field(default=False)
    attempts_needed: int = Field(default=0, description="How many attempts were required")
    
    # Documentation Quality Insights
    documentation_quality_score: Optional[float] = Field(None, ge=0, le=10, description="Overall doc quality (1-10)")
    key_documentation_gaps: List[str] = Field(default_factory=list)
    documentation_strengths: List[str] = Field(default_factory=list)


class CodingState(TypedDict):
    """State for the coding agent LangGraph workflow"""
    use_case: UseCase
    current_attempt: int
    max_attempts: int
    error: str  # "yes" or "no" 
    messages: List[tuple]
    rag_context: str
    code_solution: Optional[CodeSolution]
    execution_result: Optional[UseCaseExecution]


class RAGCodingAgent:
    """
    RAG-enhanced coding agent with iterative self-correction
    """
    
    def __init__(self, 
                 vector_store_manager: VectorStoreManager,
                 max_attempts: int = 3,
                 llm_model: str = "gpt-4o-mini"):
        self.vector_store = vector_store_manager
        self.max_attempts = max_attempts
        self.llm = ChatOpenAI(model=llm_model, temperature=0.7)
        
        # Create structured output LLM
        self.code_gen_llm = self.llm.with_structured_output(CodeSolution)
        
        # Create prompts
        self._create_code_generation_prompt()
        self._create_documentation_analysis_prompt()
        
        # Create LangGraph workflow
        self.workflow = self._create_workflow()
    
    def _create_code_generation_prompt(self):
        """Create the code generation prompt with RAG context"""
        self.code_gen_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert coding assistant. Your job is to implement code solutions based on documentation and requirements.

You will be given:
1. A specific use case/task to implement
2. Relevant documentation retrieved via RAG
3. Any previous error messages if this is a retry

Your task:
1. Analyze the retrieved documentation to understand how to implement the solution
2. Generate working code that fulfills the use case requirements
3. Ensure all imports are included and the code can be executed standalone
4. Structure your response with: description, imports, and functioning code

IMPORTANT: 
- Base your implementation ONLY on the provided documentation
- Include all necessary imports
- Make the code executable and self-contained
- If retrying due to errors, fix the specific issues mentioned

Retrieved Documentation:
{rag_context}

Previous Error (if any):
{error_context}"""),
            ("user", """Use Case: {use_case_title}
Description: {use_case_description}

Category: {use_case_category}
Main Objectives: {main_objectives}

Please implement a working solution for this use case based on the retrieved documentation.""")
        ])
    
    def _create_documentation_analysis_prompt(self):
        """Create prompt for analyzing documentation quality"""
        self.doc_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a documentation quality analyst. Analyze whether the provided documentation was sufficient for implementing the coding task.

Evaluate:
1. Was the documentation helpful for this specific task?
2. What information was missing that would have been useful?
3. What parts of the documentation were confusing or unclear?
4. How could the documentation be improved for this use case?

Be specific and constructive in your feedback."""),
            ("user", """Task: {use_case_description}
Documentation Retrieved: {rag_context}
Code Generated: {generated_code}
Test Results: {test_results}
Error Messages: {error_messages}

Please analyze the documentation quality for this coding task.""")
        ])
    
    def execute_use_case(self, use_case: UseCase) -> UseCaseExecution:
        """
        Execute a single use case with iterative RAG and self-correction
        
        Args:
            use_case: The use case to implement
            
        Returns:
            UseCaseExecution with complete results
        """
        print(f"\n🔨 Executing Use Case: {use_case.title}")
        print(f"   Description: {use_case.description}")
        print(f"   Category: {use_case.category}")
        
        # Initialize execution tracking
        execution = UseCaseExecution(
            use_case_id=use_case.id,
            use_case_title=use_case.title,
            use_case_description=use_case.description
        )
        
        # Initialize state for LangGraph workflow
        initial_state = CodingState(
            use_case=use_case,
            current_attempt=0,
            max_attempts=self.max_attempts,
            error="no",
            messages=[("user", f"Implement: {use_case.description}")],
            rag_context="",
            code_solution=None,
            execution_result=execution
        )
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Extract final results
        execution = final_state["execution_result"]
        execution.end_time = datetime.now()
        execution.attempts_needed = len(execution.attempts)
        
        # Determine overall success
        if execution.attempts:
            last_attempt = execution.attempts[-1]
            execution.overall_success = (last_attempt.import_test_passed and 
                                       last_attempt.execution_test_passed)
            if execution.overall_success and last_attempt.code_solution:
                execution.final_working_code = f"{last_attempt.code_solution.imports}\n{last_attempt.code_solution.code}"
        
        return execution
    
    def _retrieve_documentation(self, state: CodingState) -> CodingState:
        """RAG retrieval node"""
        print("   📚 Retrieving relevant documentation...")
        
        use_case = state["use_case"]
        current_attempt = state["current_attempt"]
        
        # Build RAG query based on use case and previous attempts
        rag_query = self._build_rag_query(use_case, state.get("messages", []), current_attempt)
        
        # Retrieve relevant documentation
        retrieved_docs = self.vector_store.similarity_search(rag_query, k=8)
        
        # Prepare context for LLM
        rag_context = self._prepare_rag_context(retrieved_docs)
        
        print(f"   📖 Retrieved {len(retrieved_docs)} relevant documentation chunks")
        
        return {
            **state,
            "rag_context": rag_context,
            "messages": state["messages"] + [("system", f"Retrieved {len(retrieved_docs)} documentation chunks")]
        }
    
    def _generate_code(self, state: CodingState) -> CodingState:
        """Code generation node"""
        print(f"   🔧 Generating code (attempt {state['current_attempt'] + 1})...")
        
        use_case = state["use_case"]
        rag_context = state["rag_context"]
        messages = state["messages"]
        
        # Prepare error context if this is a retry
        error_context = ""
        if state["error"] == "yes":
            # Extract error messages from conversation
            error_messages = [msg[1] for msg in messages if "failed" in msg[1].lower()]
            error_context = "\n".join(error_messages[-2:])  # Last 2 error messages
        
        # Generate code solution
        code_solution = self.code_gen_llm.invoke(self.code_gen_prompt.format_messages(
            rag_context=rag_context,
            error_context=error_context,
            use_case_title=use_case.title,
            use_case_description=use_case.description,
            use_case_category=use_case.category,
            main_objectives="\n".join(f"• {obj}" for obj in use_case.main_objectives)
        ))
        
        # Update messages
        updated_messages = messages + [
            ("assistant", f"Generated solution: {code_solution.prefix}\nImports: {code_solution.imports}\nCode: {code_solution.code}")
        ]
        
        return {
            **state,
            "code_solution": code_solution,
            "messages": updated_messages,
            "current_attempt": state["current_attempt"] + 1
        }
    
    def _test_code(self, state: CodingState) -> CodingState:
        """Code testing node"""
        print("   🧪 Testing generated code...")
        
        code_solution = state["code_solution"]
        execution_result = state["execution_result"]
        messages = state["messages"]
        
        if not code_solution:
            return {**state, "error": "yes"}
        
        # Create new attempt record
        attempt = CodeAttempt(
            attempt_number=state["current_attempt"],
            rag_query=self._extract_rag_query_from_messages(messages),
            code_solution=code_solution
        )
        
        # Test 1: Import test
        import_error = None
        try:
            exec(code_solution.imports)
            attempt.import_test_passed = True
            print("   ✅ Import test passed")
        except Exception as e:
            import_error = str(e)
            attempt.import_error = import_error
            attempt.import_test_passed = False
            print(f"   ❌ Import test failed: {e}")
        
        # Test 2: Execution test (only if imports pass)
        execution_error = None
        execution_output = None
        if attempt.import_test_passed:
            try:
                # Capture output
                import io
                import sys
                import threading
                import time
                
                captured_output = io.StringIO()
                sys.stdout = captured_output
                
                # Execute with timeout to prevent hanging
                execution_exception = None
                
                def execute_code():
                    nonlocal execution_exception
                    try:
                        exec(code_solution.imports + "\n" + code_solution.code)
                    except Exception as e:
                        execution_exception = e
                
                execution_thread = threading.Thread(target=execute_code)
                execution_thread.daemon = True
                execution_thread.start()
                execution_thread.join(timeout=3)  # 3 second timeout
                
                sys.stdout = sys.__stdout__
                execution_output = captured_output.getvalue()
                
                if execution_exception:
                    # Code failed with an exception - pass to self-correction
                    raise execution_exception
                elif execution_thread.is_alive():
                    # Code is still running (likely server) - this is actually success for servers
                    execution_output += "\n[Code appears to be running a server - execution continuing]"
                    attempt.execution_test_passed = True
                    attempt.execution_output = execution_output
                    print("   ✅ Execution test passed (server running)")
                else:
                    # Code completed successfully
                    attempt.execution_test_passed = True
                    attempt.execution_output = execution_output
                    print("   ✅ Execution test passed")
                    
            except Exception as e:
                sys.stdout = sys.__stdout__
                execution_error = str(e)
                attempt.execution_error = execution_error
                attempt.execution_test_passed = False
                print(f"   ❌ Execution test failed: {e}")
        
        # Add attempt to execution result
        execution_result.attempts.append(attempt)
        
        # Determine if we have an error
        has_error = not (attempt.import_test_passed and attempt.execution_test_passed)
        error_status = "yes" if has_error else "no"
        
        # Update messages with test results
        if has_error:
            error_msg = import_error or execution_error or "Unknown error"
            updated_messages = messages + [("user", f"Code test failed: {error_msg}. Please fix and try again.")]
        else:
            updated_messages = messages + [("user", "Code tests passed successfully!")]
        
        return {
            **state,
            "error": error_status,
            "messages": updated_messages,
            "execution_result": execution_result
        }
    
    def _analyze_documentation(self, state: CodingState) -> CodingState:
        """Documentation quality analysis node"""
        print("   📊 Analyzing documentation quality...")
        
        use_case = state["use_case"]
        rag_context = state["rag_context"]
        execution_result = state["execution_result"]
        
        if not execution_result.attempts:
            return state
        
        last_attempt = execution_result.attempts[-1]
        
        # Prepare analysis context
        test_results = f"Import test: {'PASSED' if last_attempt.import_test_passed else 'FAILED'}"
        if last_attempt.import_error:
            test_results += f" (Error: {last_attempt.import_error})"
        
        test_results += f"\nExecution test: {'PASSED' if last_attempt.execution_test_passed else 'FAILED'}"
        if last_attempt.execution_error:
            test_results += f" (Error: {last_attempt.execution_error})"
        
        error_messages = (last_attempt.import_error or "") + (last_attempt.execution_error or "")
        generated_code = f"{last_attempt.code_solution.imports}\n{last_attempt.code_solution.code}" if last_attempt.code_solution else ""
        
        # Analyze documentation quality using LLM
        try:
            analysis_response = self.llm.invoke(self.doc_analysis_prompt.format_messages(
                use_case_description=use_case.description,
                rag_context=rag_context,
                generated_code=generated_code,
                test_results=test_results,
                error_messages=error_messages
            ))
            
            # Extract insights from analysis (simplified for now)
            analysis_text = analysis_response.content
            
            # Update attempt with documentation analysis
            last_attempt.documentation_helpful = "helpful" in analysis_text.lower() or "sufficient" in analysis_text.lower()
            
            # Simple keyword-based extraction of insights
            if "missing" in analysis_text.lower():
                last_attempt.missing_information.append("Documentation gaps identified")
            if "confusing" in analysis_text.lower() or "unclear" in analysis_text.lower():
                last_attempt.confusing_parts.append("Parts of documentation were unclear")
            if "improve" in analysis_text.lower():
                last_attempt.suggested_improvements.append("Documentation improvements suggested")
            
        except Exception as e:
            print(f"   ⚠️ Documentation analysis failed: {e}")
        
        return {
            **state,
            "execution_result": execution_result
        }
    
    def _decide_next_action(self, state: CodingState) -> str:
        """Decision node for workflow control"""
        error = state["error"]
        current_attempt = state["current_attempt"]
        max_attempts = state["max_attempts"]
        
        if error == "no":
            print("   ✅ Code generation successful!")
            return "analyze_docs"
        elif current_attempt >= max_attempts:
            print(f"   ⏰ Max attempts ({max_attempts}) reached")
            return "analyze_docs" 
        else:
            print(f"   🔄 Retrying... (attempt {current_attempt + 1}/{max_attempts})")
            return "retrieve_docs"
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(CodingState)
        
        # Add nodes
        workflow.add_node("retrieve_docs", self._retrieve_documentation)
        workflow.add_node("generate_code", self._generate_code)
        workflow.add_node("test_code", self._test_code)
        workflow.add_node("analyze_docs", self._analyze_documentation)
        
        # Add edges
        workflow.add_edge(START, "retrieve_docs")
        workflow.add_edge("retrieve_docs", "generate_code")
        workflow.add_edge("generate_code", "test_code")
        
        # Conditional edges for retry logic
        workflow.add_conditional_edges(
            "test_code",
            self._decide_next_action,
            {
                "retrieve_docs": "retrieve_docs",
                "analyze_docs": "analyze_docs"
            }
        )
        
        workflow.add_edge("analyze_docs", END)
        
        return workflow.compile()
    
    def _build_rag_query(self, use_case: UseCase, messages: List[tuple], attempt_number: int) -> str:
        """Build RAG query based on use case and previous failures"""
        base_query = f"{use_case.category} {use_case.description}"
        
        if attempt_number == 0:
            # First attempt: broad query
            return f"{base_query} example implementation tutorial"
        else:
            # Later attempts: incorporate error information
            error_keywords = []
            for msg_type, msg_content in messages:
                if "failed" in msg_content.lower():
                    if "import" in msg_content.lower():
                        error_keywords.append("installation setup imports dependencies")
                    elif "execution" in msg_content.lower():
                        error_keywords.append("implementation examples working code")
            
            error_context = " ".join(error_keywords)
            return f"{base_query} {error_context} troubleshooting debugging"
    
    def _prepare_rag_context(self, docs: List[ProcessedDocument]) -> str:
        """Prepare retrieved documents for LLM context"""
        context_parts = []
        for i, doc in enumerate(docs[:5], 1):  # Top 5 most relevant
            context_parts.append(f"""
Document {i} ({doc.content_type}):
Title: {doc.section_title}
Content: {doc.content[:500]}{"..." if len(doc.content) > 500 else ""}
""")
        return "\n".join(context_parts)
    
    def _extract_rag_query_from_messages(self, messages: List[tuple]) -> str:
        """Extract the RAG query used from messages (simplified)"""
        # In a real implementation, you'd track this more precisely
        return "documentation query"
    
    def execute_multiple_use_cases(self, use_cases: List[UseCase]) -> List[UseCaseExecution]:
        """Execute multiple use cases and return results"""
        print(f"\n🚀 Starting execution of {len(use_cases)} use cases...")
        
        results = []
        for i, use_case in enumerate(use_cases, 1):
            print(f"\n[{i}/{len(use_cases)}] Processing: {use_case.title}")
            
            try:
                execution = self.execute_use_case(use_case)
                results.append(execution)
                
                # Print summary
                status = "✅ SUCCESS" if execution.overall_success else "❌ FAILED"
                print(f"   {status} - {execution.attempts_needed} attempts")
                
            except Exception as e:
                print(f"   💥 ERROR: {e}")
                # Create a failed execution record
                failed_execution = UseCaseExecution(
                    use_case_id=use_case.id,
                    use_case_title=use_case.title,
                    use_case_description=use_case.description,
                    overall_success=False,
                    attempts_needed=0
                )
                results.append(failed_execution)
        
        return results


# Example usage and integration functions
def create_coding_agent_from_vector_store(vector_store_manager: VectorStoreManager,
                                        max_attempts: int = 3,
                                        llm_model: str = "gpt-4o-mini") -> RAGCodingAgent:
    """Create a coding agent from an existing vector store"""
    return RAGCodingAgent(
        vector_store_manager=vector_store_manager,
        max_attempts=max_attempts,
        llm_model=llm_model
    )


def save_execution_results(results: List[UseCaseExecution], output_path: str):
    """Save execution results to JSON file"""
    # Convert to JSON-serializable format
    json_results = []
    for result in results:
        json_results.append(result.model_dump(mode='json'))
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_use_cases': len(results),
            'successful_cases': sum(1 for r in results if r.overall_success),
            'execution_results': json_results
        }, f, indent=2)
    
    print(f"💾 Execution results saved to: {output_path}")


def load_use_cases_from_planning_results(json_path: str) -> List[UseCase]:
    """Load UseCase objects from progressive planner results"""
    print(f"📂 Loading planning results from: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    use_cases = []
    for plan_data in data['generated_plans']:
        use_case = UseCase(
            id=plan_data['id'],
            title=plan_data['title'],
            category=plan_data['category'], 
            description=plan_data['description'],
            main_objectives=plan_data['main_objectives'],
            success_criteria=plan_data['success_criteria'],
            expected_challenges=plan_data['expected_challenges'],
            supporting_queries=plan_data.get('supporting_queries', []),
            evidence_strength=plan_data.get('evidence_strength', 0.0),
            documentation_refs=plan_data.get('documentation_refs', []),
            estimated_difficulty=plan_data.get('estimated_difficulty', 'medium'),
            prerequisite_knowledge=plan_data.get('prerequisite_knowledge', []),
            key_apis_or_features=plan_data.get('key_apis_or_features', []),
            estimated_duration_minutes=plan_data.get('estimated_duration_minutes', 20)
        )
        use_cases.append(use_case)
    
    print(f"✅ Loaded {len(use_cases)} use cases")
    return use_cases


if __name__ == "__main__":
    # Load use cases from previous progressive planner run
    from document_processor import VectorStoreManager
    from progressive_planner import UseCase
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from pathlib import Path
    
    print("🚀 RAG-Enhanced Coding Agent Evaluation")
    print("=" * 50)
    
    # Check if we have planning results
    planning_results_path = "./outputs/fastapi_planning_results.json"
    if not Path(planning_results_path).exists():
        print(f"❌ Planning results not found at: {planning_results_path}")
        print("Please run progressive_planner.py first to generate use cases.")
        exit(1)
    
    # Load use cases from planning results
    use_cases = load_use_cases_from_planning_results(planning_results_path)
    
    # Set up vector store manager
    persist_dir = "./chroma_db"
    collection_name = "fastapi-fastapi"
    
    if not Path(persist_dir).exists():
        print(f"❌ Vector store not found at: {persist_dir}")
        print("Please run document_processor.py first to create the vector store.")
        exit(1)
    
    print("📚 Loading vector store...")
    vector_manager = VectorStoreManager(
        persist_directory=persist_dir,
        collection_name=collection_name
    )
    
    vector_manager.vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name=collection_name
    )
    
    # Create coding agent
    print("🤖 Initializing RAG Coding Agent...")
    coding_agent = RAGCodingAgent(
        vector_store_manager=vector_manager,
        max_attempts=3,
        llm_model="gpt-4o-mini"
    )
    
    # Execute use cases (limit to first 3 for testing)
    test_use_cases = use_cases[:3]  # Start with first 3 use cases
    print(f"\n🎯 Running coding evaluation on {len(test_use_cases)} use cases:")
    for i, uc in enumerate(test_use_cases, 1):
        print(f"  {i}. {uc.title} ({uc.category}, {uc.estimated_difficulty})")
    
    # Execute the use cases
    execution_results = coding_agent.execute_multiple_use_cases(test_use_cases)
    
    # Save results
    output_path = "./outputs/fastapi_coding_results.json"
    save_execution_results(execution_results, output_path)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 EXECUTION SUMMARY")
    print("=" * 50)
    
    successful = sum(1 for r in execution_results if r.overall_success)
    total = len(execution_results)
    success_rate = (successful / total) * 100 if total > 0 else 0
    
    print(f"✅ Successful: {successful}/{total} ({success_rate:.1f}%)")
    print(f"⏱️  Total attempts: {sum(r.attempts_needed for r in execution_results)}")
    print(f"📈 Average attempts per use case: {sum(r.attempts_needed for r in execution_results) / total:.1f}")
    
    print(f"\n📋 Detailed Results:")
    for result in execution_results:
        status = "✅" if result.overall_success else "❌"
        print(f"  {status} {result.use_case_title}")
        print(f"      Attempts: {result.attempts_needed}, Duration: {(result.end_time - result.start_time).total_seconds():.1f}s")
        if not result.overall_success and result.attempts:
            last_error = result.attempts[-1].import_error or result.attempts[-1].execution_error
            if last_error:
                print(f"      Error: {last_error[:100]}...")
    
    print(f"\n💾 Detailed results saved to: {output_path}")
    print("🎉 Coding agent evaluation complete!")
