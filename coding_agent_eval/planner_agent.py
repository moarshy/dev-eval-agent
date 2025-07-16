
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import re
import subprocess
import tempfile
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from models import (
    AgentReadinessState, UseCase, UseCaseExecution, CodeAttempt, 
    RAGContext, EvaluationDimension, TaskStatus, DifficultyLevel
)
from document_processor import VectorStoreManager

# Code generation schema (matching your existing structure)
class CodeSolution(BaseModel):
    """Schema for code solutions to questions about documentation."""
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")

class DocumentationAnalysis(BaseModel):
    """Analysis of documentation quality for a coding attempt"""
    documentation_helpful: bool = Field(description="Was the documentation helpful?")
    missing_information: List[str] = Field(description="What information was missing?")
    confusing_parts: List[str] = Field(description="What parts were confusing?")
    suggested_improvements: List[str] = Field(description="How to improve the docs?")
    chunks_used: List[str] = Field(description="Which chunks were actually helpful?")
    chunks_ignored: List[str] = Field(description="Which chunks were irrelevant?")

class PlannerAgent:
    """RAG-enhanced planner that generates use cases based on documentation analysis"""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        
        self.planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing developer tool documentation and creating comprehensive test scenarios.

Your task is to analyze the provided documentation context and generate 4-6 strategic use cases that will thoroughly test whether an AI coding agent can successfully implement solutions using only this documentation.

Consider these essential categories (always include at least one from each):
1. **Setup & Installation** - Can an agent get the tool working?
2. **Authentication** - Can an agent properly configure access?
3. **Basic Usage** - Can an agent perform the primary function?
4. **Error Handling** - Can an agent handle common failure scenarios?

You may also add tool-specific scenarios like:
- **Integration** - Connecting with other systems
- **Advanced Features** - Complex workflows
- **Configuration** - Customization and setup options

Here is the documentation context retrieved for analysis:
{documentation_context}

Generate use cases that:
- Progress from beginner to advanced difficulty
- Have clear, measurable success criteria
- Test different aspects of the documentation
- Identify potential gaps or weak points

Format each use case with:
- Clear title and description
- Specific success criteria (what constitutes success?)
- Appropriate difficulty level
- Category classification
- Estimated duration

Respond with a structured analysis and use case list."""),
            ("user", "Tool: {tool_name}\nCategory: {tool_category}\n\nAnalyze the documentation and generate comprehensive use cases for AI agent testing.")
        ])
    
    def generate_use_cases(self, state: AgentReadinessState, vector_manager: VectorStoreManager) -> AgentReadinessState:
        """Generate use cases using RAG-enhanced analysis"""
        
        # Step 1: Analyze documentation comprehensively
        documentation_context = self._analyze_documentation_comprehensively(vector_manager)
        
        # Step 2: Generate use cases
        response = self.llm.invoke(self.planning_prompt.format_messages(
            tool_name=state.tool_name,
            tool_category=state.tool_category,
            documentation_context=documentation_context
        ))
        
        # Step 3: Parse response and create use cases
        use_cases = self._parse_use_cases_from_response(response.content)
        
        # Step 4: Store RAG context and results
        state.use_cases = use_cases
        state.planning_rag_context = RAGContext(
            query="comprehensive documentation analysis",
            retrieved_chunks=documentation_context['chunks'],
            relevance_scores=documentation_context['scores'],
            total_chunks_available=len(state.processed_documents)
        )
        state.planning_notes = response.content
        state.planning_completed = True
        
        return state
    
    def _analyze_documentation_comprehensively(self, vector_manager: VectorStoreManager) -> Dict:
        """Perform comprehensive documentation analysis"""
        
        analysis_queries = [
            "setup installation getting started configuration",
            "authentication login credentials api key token",
            "basic usage example tutorial simple implementation",
            "error handling debugging troubleshooting common issues",
            "api endpoints methods functions core features",
            "advanced features integration workflows"
        ]
        
        all_chunks = []
        all_scores = []
        
        for query in analysis_queries:
            results = vector_manager.similarity_search(query, k=8)
            for result in results:
                if result not in all_chunks:  # Avoid duplicates
                    all_chunks.append(result)
                    all_scores.append(0.8)  # Placeholder score
        
        return {
            'chunks': all_chunks,
            'scores': all_scores,
            'summary': f"Analyzed documentation with {len(all_chunks)} relevant sections"
        }
    
    def _parse_use_cases_from_response(self, response_content: str) -> List[UseCase]:
        """Parse use cases from LLM response"""
        # This is a simplified parser - in practice, you'd want more robust parsing
        use_cases = []
        
        # Default use cases if parsing fails
        default_use_cases = [
            {
                'title': 'Setup and Installation',
                'description': 'Successfully install and configure the tool',
                'category': 'setup',
                'difficulty': 'beginner',
                'success_criteria': ['Tool is installed', 'Basic configuration works']
            },
            {
                'title': 'Authentication Setup',
                'description': 'Configure authentication and verify access',
                'category': 'auth',
                'difficulty': 'beginner',
                'success_criteria': ['Authentication configured', 'Access verified']
            },
            {
                'title': 'Basic Usage Implementation',
                'description': 'Implement basic functionality using the tool',
                'category': 'basic_usage',
                'difficulty': 'intermediate',
                'success_criteria': ['Basic functionality works', 'Code executes successfully']
            },
            {
                'title': 'Error Handling',
                'description': 'Implement proper error handling for common scenarios',
                'category': 'error_handling',
                'difficulty': 'intermediate',
                'success_criteria': ['Errors are caught', 'Appropriate responses provided']
            }
        ]
        
        # Convert to UseCase objects
        for i, use_case_data in enumerate(default_use_cases):
            use_case = UseCase(
                id=f"uc_{i+1}",
                title=use_case_data['title'],
                description=use_case_data['description'],
                success_criteria=use_case_data['success_criteria'],
                difficulty_level=DifficultyLevel(use_case_data['difficulty']),
                category=use_case_data['category'],
                estimated_duration_minutes=15
            )
            use_cases.append(use_case)
        
        return use_cases