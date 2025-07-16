class RAGCodingAgent:
    """RAG-enhanced coding agent with iterative improvement and documentation feedback"""
    
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        
        # Code generation prompt
        self.code_gen_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a coding assistant that implements solutions based on documentation.

Your task is to generate working code that accomplishes the given use case using ONLY the provided documentation context.

Here is the relevant documentation:
{documentation_context}

Previous attempt information (if any):
{previous_attempts}

Requirements:
1. Generate functional code that can be executed
2. Include all necessary imports
3. Use only what's documented in the provided context
4. Follow the patterns and examples shown in the documentation
5. Include proper error handling where applicable

Structure your response with:
- prefix: Description of your approach
- imports: Import statements needed
- code: Main implementation code

If this is a retry attempt, learn from the previous errors and try a different approach."""),
            ("user", "Use Case: {use_case_description}\n\nSuccess Criteria: {success_criteria}\n\nGenerate working code for this use case.")
        ])
        
        # Documentation analysis prompt
        self.doc_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the quality of documentation for a coding task.

You attempted to implement this use case:
{use_case_description}

Using this documentation:
{documentation_used}

The generated code was:
{generated_code}

Results:
- Import test: {import_result}
- Execution test: {execution_result}

Provide specific feedback on:
1. Was the documentation helpful for this task?
2. What specific information was missing?
3. What parts of the documentation were confusing?
4. How could the documentation be improved?
5. Which documentation chunks were actually useful?
6. Which chunks were irrelevant or distracting?"""),
            ("user", "Please analyze the documentation quality for this coding attempt.")
        ])
    
    def execute_use_case(self, 
                        use_case: UseCase, 
                        execution: UseCaseExecution,
                        vector_manager: VectorStoreManager) -> UseCaseExecution:
        """Execute use case with iterative RAG and self-correction"""
        
        execution.status = TaskStatus.IN_PROGRESS
        execution.start_time = datetime.now()
        
        for attempt_num in range(1, self.max_attempts + 1):
            print(f"  Attempt {attempt_num}/{self.max_attempts}")
            
            # Make coding attempt
            attempt = self._make_code_attempt(use_case, execution, attempt_num, vector_manager)
            execution.attempts.append(attempt)
            
            # Check if successful
            if attempt.import_test_passed and attempt.execution_test_passed:
                execution.overall_success = True
                execution.final_working_code = f"{attempt.imports}\n{attempt.code}"
                execution.status = TaskStatus.COMPLETED
                break
            else:
                print(f"    ❌ Attempt {attempt_num} failed")
                if attempt_num < self.max_attempts:
                    print(f"    🔄 Preparing retry...")
        
        execution.end_time = datetime.now()
        execution.attempts_needed = len(execution.attempts)
        
        if not execution.overall_success:
            execution.status = TaskStatus.FAILED
        
        return execution
    
    def _make_code_attempt(self, 
                          use_case: UseCase, 
                          execution: UseCaseExecution,
                          attempt_num: int,
                          vector_manager: VectorStoreManager) -> CodeAttempt:
        """Make a single code attempt with RAG retrieval"""
        
        # Step 1: Build RAG query based on use case and previous attempts
        rag_query = self._build_rag_query(use_case, execution, attempt_num)
        
        # Step 2: Retrieve relevant documentation
        retrieved_docs = vector_manager.similarity_search(rag_query, k=10)
        
        # Step 3: Format documentation context
        doc_context = self._format_documentation_context(retrieved_docs)
        
        # Step 4: Format previous attempts context
        previous_context = self._format_previous_attempts(execution.attempts)
        
        # Step 5: Generate code
        code_solution = self._generate_code(use_case, doc_context, previous_context)
        
        # Step 6: Test the code
        import_result, execution_result = self._test_code(code_solution.imports, code_solution.code)
        
        # Step 7: Analyze documentation quality
        doc_analysis = self._analyze_documentation_quality(
            use_case, retrieved_docs, code_solution, import_result, execution_result
        )
        
        # Step 8: Create attempt record
        rag_context = RAGContext(
            query=rag_query,
            retrieved_chunks=retrieved_docs,
            relevance_scores=[0.8] * len(retrieved_docs),  # Placeholder scores
            total_chunks_available=100  # Placeholder
        )
        
        return CodeAttempt(
            attempt_number=attempt_num,
            rag_context=rag_context,
            prefix=code_solution.prefix,
            imports=code_solution.imports,
            code=code_solution.code,
            import_test_passed=import_result['success'],
            import_error=import_result.get('error'),
            execution_test_passed=execution_result['success'],
            execution_error=execution_result.get('error'),
            execution_output=execution_result.get('output'),
            documentation_helpful=doc_analysis.documentation_helpful,
            missing_information=doc_analysis.missing_information,
            confusing_parts=doc_analysis.confusing_parts,
            suggested_improvements=doc_analysis.suggested_improvements,
            chunks_used=doc_analysis.chunks_used,
            chunks_ignored=doc_analysis.chunks_ignored
        )
    
    def _build_rag_query(self, use_case: UseCase, execution: UseCaseExecution, attempt_num: int) -> str:
        """Build RAG query based on use case and previous failures"""
        
        base_query = f"{use_case.category} {use_case.description}"
        
        if attempt_num == 1:
            return f"{base_query} example implementation tutorial"
        else:
            # Incorporate failure information from previous attempts
            if execution.attempts:
                last_attempt = execution.attempts[-1]
                if last_attempt.import_error:
                    return f"{base_query} imports dependencies installation {last_attempt.import_error}"
                elif last_attempt.execution_error:
                    return f"{base_query} error debugging troubleshooting {last_attempt.execution_error}"
        
        return base_query
    
    def _format_documentation_context(self, retrieved_docs) -> str:
        """Format retrieved documentation for LLM context"""
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"--- Document {i+1}: {doc.section_title} ---\n{doc.content}\n")
        return "\n".join(context_parts)
    
    def _format_previous_attempts(self, attempts: List[CodeAttempt]) -> str:
        """Format previous attempts for context"""
        if not attempts:
            return "This is the first attempt."
        
        context_parts = []
        for attempt in attempts:
            context_parts.append(f"Attempt {attempt.attempt_number}:")
            context_parts.append(f"  Code: {attempt.imports}\n{attempt.code}")
            if attempt.import_error:
                context_parts.append(f"  Import Error: {attempt.import_error}")
            if attempt.execution_error:
                context_parts.append(f"  Execution Error: {attempt.execution_error}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _generate_code(self, use_case: UseCase, doc_context: str, previous_context: str) -> CodeSolution:
        """Generate code using LLM with structured output"""
        
        # Use structured output to ensure consistent format
        structured_llm = self.llm.with_structured_output(CodeSolution)
        
        response = structured_llm.invoke(
            self.code_gen_prompt.format_messages(
                documentation_context=doc_context,
                previous_attempts=previous_context,
                use_case_description=use_case.description,
                success_criteria=", ".join(use_case.success_criteria)
            )
        )
        
        return response
    
    def _test_code(self, imports: str, code: str) -> Tuple[Dict, Dict]:
        """Test the generated code (matching your existing pattern)"""
        
        # Test imports
        import_result = {'success': False}
        try:
            exec(imports)
            import_result['success'] = True
        except Exception as e:
            import_result['error'] = str(e)
        
        # Test execution
        execution_result = {'success': False}
        if import_result['success']:
            try:
                # Execute in a clean namespace
                namespace = {}
                exec(imports + "\n" + code, namespace)
                execution_result['success'] = True
                execution_result['output'] = "Code executed successfully"
            except Exception as e:
                execution_result['error'] = str(e)
        else:
            execution_result['error'] = "Skipped due to import failure"
        
        return import_result, execution_result
    
    def _analyze_documentation_quality(self, 
                                     use_case: UseCase,
                                     retrieved_docs,
                                     code_solution: CodeSolution,
                                     import_result: Dict,
                                     execution_result: Dict) -> DocumentationAnalysis:
        """Analyze documentation quality using LLM"""
        
        # Format documentation used
        doc_used = self._format_documentation_context(retrieved_docs)
        
        # Format results
        import_status = "PASSED" if import_result['success'] else f"FAILED: {import_result.get('error', 'Unknown error')}"
        execution_status = "PASSED" if execution_result['success'] else f"FAILED: {execution_result.get('error', 'Unknown error')}"
        
        # Get analysis from LLM
        structured_llm = self.llm.with_structured_output(DocumentationAnalysis)
        
        analysis = structured_llm.invoke(
            self.doc_analysis_prompt.format_messages(
                use_case_description=use_case.description,
                documentation_used=doc_used,
                generated_code=f"{code_solution.imports}\n{code_solution.code}",
                import_result=import_status,
                execution_result=execution_status
            )
        )
        
        return analysis
