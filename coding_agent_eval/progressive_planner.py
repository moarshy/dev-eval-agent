#!/usr/bin/env python3
"""
Progressive Planner Agent

A sophisticated planner that uses multi-stage exploration to deeply understand
a tool's capabilities before generating targeted project plans for testing agent readiness.

The planner works in three stages:
1. Progressive Exploration: Multiple rounds of targeted queries to build understanding
2. Evidence-Based Plan Generation: Generate diverse plans based on accumulated knowledge
3. Targeted Plan Refinement: Refine each plan with specific supporting evidence
"""

from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import json
import re
from pydantic import BaseModel, Field, field_validator

from document_processor import VectorStoreManager
from models import ProcessedDocument

# LLM imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class StructuredAnalysis(BaseModel):
    """Structured analysis output from LLM"""
    key_findings: List[str] = Field(default_factory=list, description="Key insights discovered")
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0, description="How well this query was answered (0-1)")
    documentation_gaps: List[str] = Field(default_factory=list, description="Missing or unclear documentation areas")
    insight_categories: Dict[str, List[str]] = Field(default_factory=dict, description="Categorized insights by type")


class ExplorationRound(BaseModel):
    """Single round of exploration with query, results, and insights"""
    round_number: int
    query: str
    results: List[ProcessedDocument]
    key_findings: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="How well this query was answered (0-1)")
    documentation_gaps: List[str] = Field(default_factory=list)
    structured_analysis: Optional[StructuredAnalysis] = None


class ProjectScratchpad(BaseModel):
    """Accumulates understanding about the tool across exploration rounds"""
    tool_name: str
    exploration_rounds: List[ExplorationRound] = Field(default_factory=list)
    
    # Accumulated insights
    core_capabilities: List[str] = Field(default_factory=list)
    common_patterns: List[str] = Field(default_factory=list)
    setup_requirements: List[str] = Field(default_factory=list)
    authentication_info: List[str] = Field(default_factory=list)
    integration_patterns: List[str] = Field(default_factory=list)
    advanced_features: List[str] = Field(default_factory=list)
    
    # Quality assessment
    well_documented_areas: List[str] = Field(default_factory=list)
    documentation_gaps: List[str] = Field(default_factory=list)
    confidence_areas: List[str] = Field(default_factory=list)
    uncertain_areas: List[str] = Field(default_factory=list)
    
    def add_exploration_round(self, round_data: ExplorationRound):
        """Add a new exploration round and update insights"""
        self.exploration_rounds.append(round_data)
        self._update_cumulative_insights(round_data)
    
    def _update_cumulative_insights(self, round_data: ExplorationRound):
        """Update cumulative understanding based on latest round"""
        
        # Use structured analysis if available, otherwise fall back to simple categorization
        if round_data.structured_analysis:
            categories = round_data.structured_analysis.insight_categories
            
            # Map structured categories to scratchpad fields
            self.core_capabilities.extend(categories.get("core_features", []))
            self.setup_requirements.extend(categories.get("setup", []))
            self.authentication_info.extend(categories.get("authentication", []))
            self.advanced_features.extend(categories.get("advanced", []))
            self.integration_patterns.extend(categories.get("integration", []))
            self.common_patterns.extend(categories.get("patterns", []))
            
            # Add any other categorized insights to common patterns
            for category, insights in categories.items():
                if category not in ["core_features", "setup", "authentication", "advanced", "integration", "patterns"]:
                    self.common_patterns.extend(insights)
        else:
            # Fallback to keyword-based categorization
            if any(keyword in round_data.query.lower() for keyword in ["core", "functionality", "features"]):
                self.core_capabilities.extend(round_data.key_findings)
            elif any(keyword in round_data.query.lower() for keyword in ["setup", "installation", "install", "getting started"]):
                self.setup_requirements.extend(round_data.key_findings)
            elif any(keyword in round_data.query.lower() for keyword in ["authentication", "auth", "security", "login"]):
                self.authentication_info.extend(round_data.key_findings)
            elif any(keyword in round_data.query.lower() for keyword in ["advanced", "complex", "customization"]):
                self.advanced_features.extend(round_data.key_findings)
            elif any(keyword in round_data.query.lower() for keyword in ["integration", "integrate", "connect"]):
                self.integration_patterns.extend(round_data.key_findings)
            else:
                self.common_patterns.extend(round_data.key_findings)
        
        # Assess documentation quality
        if round_data.confidence_score > 0.7:
            self.well_documented_areas.append(round_data.query)
            self.confidence_areas.extend(round_data.key_findings)
        elif round_data.confidence_score < 0.4:
            self.uncertain_areas.append(round_data.query)
            self.documentation_gaps.extend(round_data.documentation_gaps)
    
    def get_cumulative_understanding(self) -> str:
        """Get a comprehensive summary of accumulated understanding"""
        return f"""
        === TOOL UNDERSTANDING: {self.tool_name} ===
        
        Core Capabilities:
        {chr(10).join(f"• {cap}" for cap in self.core_capabilities[:10])}
        
        Setup & Installation:
        {chr(10).join(f"• {req}" for req in self.setup_requirements[:5])}
        
        Authentication & Security:
        {chr(10).join(f"• {auth}" for auth in self.authentication_info[:5])}
        
        Common Patterns:
        {chr(10).join(f"• {pattern}" for pattern in self.common_patterns[:8])}
        
        Advanced Features:
        {chr(10).join(f"• {feature}" for feature in self.advanced_features[:5])}
        
        Integration Patterns:
        {chr(10).join(f"• {integration}" for integration in self.integration_patterns[:5])}
        
        Well-Documented Areas:
        {chr(10).join(f"• {area}" for area in self.well_documented_areas)}
        
        Documentation Gaps:
        {chr(10).join(f"• {gap}" for gap in self.documentation_gaps[:10])}
        """


class ProjectPlan(BaseModel):
    """A specific project plan for testing agent capabilities"""
    plan_id: str
    title: str
    category: str = Field(..., description="Plan category: beginner, auth, integration, error_handling, performance, advanced")
    description: str
    main_objectives: List[str]
    success_criteria: List[str]
    expected_challenges: List[str]
    
    # Supporting evidence from documentation
    supporting_queries: List[str] = Field(default_factory=list)
    evidence_strength: float = Field(default=0.0, ge=0.0, le=1.0, description="How well documented this plan is (0-1)")
    documentation_refs: List[str] = Field(default_factory=list)
    
    # Implementation details
    estimated_difficulty: str = Field(default="medium", description="Difficulty level: easy, medium, hard")
    prerequisite_knowledge: List[str] = Field(default_factory=list)
    key_apis_or_features: List[str] = Field(default_factory=list)
    
    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        valid_categories = {'beginner', 'authentication', 'integration', 'error_handling', 'performance', 'advanced'}
        if v not in valid_categories:
            raise ValueError(f'Category must be one of {valid_categories}')
        return v
    
    @field_validator('estimated_difficulty')
    @classmethod
    def validate_difficulty(cls, v):
        valid_difficulties = {'easy', 'medium', 'hard'}
        if v not in valid_difficulties:
            raise ValueError(f'Difficulty must be one of {valid_difficulties}')
        return v
    
    @field_validator('main_objectives', 'success_criteria', 'expected_challenges')
    @classmethod
    def validate_non_empty_lists(cls, v):
        if not v:
            raise ValueError('List cannot be empty')
        return v


class ProgressivePlannerAgent:
    """
    Multi-stage planner that progressively understands a tool before generating plans
    """
    
    def __init__(self, 
                 vector_store_manager: VectorStoreManager,
                 max_exploration_rounds: int = 5,
                 plans_to_generate: int = 10,
                 llm_model: str = "gpt-4o-mini"):
        self.vector_store = vector_store_manager
        self.max_exploration_rounds = max_exploration_rounds
        self.plans_to_generate = plans_to_generate
        self.llm = ChatOpenAI(model=llm_model, temperature=0.7)
        self.scratchpad = None
        
        # Create structured LLM for analysis
        self.structured_llm = self.llm.with_structured_output(StructuredAnalysis)
        
        # Create the LLM prompts
        self._create_plan_generation_prompt()
        self._create_exploration_prompt()
        self._create_analysis_prompt()
    
    def _create_plan_generation_prompt(self):
        """Create the LLM prompt for generating project plans"""
        self.plan_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI agent readiness evaluator. Your job is to generate diverse, realistic project plans for testing whether AI coding agents can successfully implement solutions using a specific tool's documentation.

You have access to comprehensive exploration data about the tool, including:
- What capabilities were discovered
- Which areas are well-documented vs poorly documented  
- What types of content are available (tutorials, examples, API references)
- Specific patterns and requirements found

Based on this real exploration data, generate {num_plans} diverse project plans that will thoroughly test an AI agent's ability to work with this tool.

IMPORTANT GUIDELINES:
1. **Ground plans in discovered capabilities**: Only include features/workflows that were actually found during exploration
2. **Vary difficulty based on documentation quality**: Well-documented areas = easier plans, poorly documented = harder
3. **Test diverse aspects**: Cover different categories like setup, authentication, integration, error handling, performance, advanced features
4. **Be specific**: Include concrete objectives and success criteria based on actual documentation
5. **Consider documentation gaps**: Create plans that test edge cases where docs might be weak

OUTPUT FORMAT:
Return a JSON array of exactly {num_plans} plans. Each plan must have this structure:
{{
  "plan_id": "toolname_category",
  "title": "Descriptive Title",
  "category": "one of: beginner, authentication, integration, error_handling, performance, advanced",
  "description": "What this plan tests about the tool",
  "main_objectives": ["objective1", "objective2", "objective3"],
  "success_criteria": ["criteria1", "criteria2", "criteria3"],
  "expected_challenges": ["challenge1", "challenge2", "challenge3"],
  "estimated_difficulty": "easy, medium, or hard",
  "prerequisite_knowledge": ["prereq1", "prereq2"]
}}

Base your plans on the following exploration data:"""),
            ("user", """TOOL: {tool_name}

EXPLORATION SUMMARY:
Rounds completed: {rounds_completed}
Well-documented areas: {well_documented_areas}
Documentation gaps: {documentation_gaps}

DISCOVERED CAPABILITIES:
{capabilities_summary}

SETUP & INSTALLATION INSIGHTS:
{setup_insights}

AUTHENTICATION & SECURITY INSIGHTS:  
{auth_insights}

COMMON PATTERNS FOUND:
{pattern_insights}

ADVANCED FEATURES DISCOVERED:
{advanced_insights}

CONFIDENCE ASSESSMENT:
High confidence areas: {high_confidence_areas}
Low confidence areas: {low_confidence_areas}

Generate {num_plans} diverse, realistic project plans that test different aspects of this tool based on what was actually discovered during exploration. Make sure the difficulty and scope match the actual documentation quality found.""")
        ])
    
    def _create_exploration_prompt(self):
        """Create the LLM prompt for generating exploration queries"""
        self.exploration_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert documentation analyst. Your job is to generate strategic exploration queries for understanding a developer tool's capabilities and documentation quality.

You will be given information about a tool and need to generate {max_rounds} diverse, targeted queries that will help build comprehensive understanding of:
1. Core functionality and capabilities
2. Setup, installation, and configuration requirements  
3. Authentication and security patterns
4. Common usage patterns and workflows
5. Advanced features and customization options
6. Integration possibilities
7. Error handling and troubleshooting

GUIDELINES:
- Generate queries that progress from broad to specific
- Adapt queries based on the tool type (API, framework, library, CLI tool, platform, etc.)
- Include queries that test documentation completeness
- Focus on aspects most important for AI agent implementation
- Each query should be 5-15 words that would work well with semantic search

OUTPUT FORMAT:
Return a JSON array of exactly {max_rounds} query strings:
["query1", "query2", "query3", "query4", "query5"]

The queries should build on each other and cover different aspects of the tool comprehensively."""),
            ("user", """TOOL: {tool_name}
CONTEXT: {context}

Based on the tool name and context, first determine what type of tool this is (e.g., web framework, database, cloud service, ML library, CLI tool, etc.), then generate {max_rounds} strategic exploration queries that will help understand this tool's capabilities, documentation quality, and implementation requirements. Make the queries specific to this type of tool.""")
        ])
    
    def _create_analysis_prompt(self):
        """Create the LLM prompt for analyzing exploration results"""
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert documentation analyst analyzing search results from documentation.

Analyze the documentation and provide:
1. Key findings: List of 3-5 key insights about the tool's capabilities and features
2. Confidence score: How well the query was answered (0.0 = no info, 1.0 = comprehensive)
3. Documentation gaps: What information is missing or unclear
4. Insight categories: Group findings by category (setup, core_features, authentication, advanced, integration, patterns, error_handling)

Be specific and factual based only on the provided documentation."""),
            ("user", """Query: {query}
Tool: {tool_name}
Round: {round_number}

Documentation Retrieved:
{documentation_content}

Based on this documentation, analyze the query and provide insights.""")
        ])
    
    def plan_project(self, tool_name: str, documentation_context: Optional[str] = None) -> Tuple[List[ProjectPlan], ProjectScratchpad]:
        """
        Main planning workflow: explore → generate → refine
        
        Args:
            tool_name: Name of the tool to analyze
            documentation_context: Optional context about the tool
            
        Returns:
            Tuple of (refined_plans, scratchpad_with_exploration_data)
        """
        print(f"🔍 Starting progressive planning for: {tool_name}")
        
        # Initialize scratchpad
        self.scratchpad = ProjectScratchpad(tool_name=tool_name)
        
        # Stage 1: Progressive Exploration
        print("\n📚 Stage 1: Progressive Exploration")
        self._explore_tool_capabilities(tool_name, documentation_context)
        
        # Stage 2: Generate Multiple Candidate Plans
        print("\n💡 Stage 2: Generating Candidate Plans")
        candidate_plans = self._generate_candidate_plans()
        
        # Stage 3: Refine Each Plan with Targeted Research
        print("\n🎯 Stage 3: Refining Plans with Evidence")
        refined_plans = self._refine_all_plans(candidate_plans)
        
        print(f"\n✅ Generated {len(refined_plans)} refined project plans")
        return refined_plans, self.scratchpad
    
    def _explore_tool_capabilities(self, tool_name: str, context: Optional[str] = None):
        """Stage 1: Progressive exploration to build understanding using LLM-generated queries"""
        
        # Start with overview context from vector store
        print(f"  Getting overview context for {tool_name}")
        overview_query = f"{tool_name} overview what is documentation main features capabilities"
        overview_results = self.vector_store.similarity_search(overview_query, k=10)
        overview_context = self._prepare_docs_for_analysis(overview_results)
        print(f"Overview context: {overview_context}")
        # Generate exploration queries using LLM with overview context
        exploration_queries = self._generate_exploration_queries(tool_name, context, overview_context)
        print(f"Exploration queries: {exploration_queries}")
        
        for round_num, query in enumerate(exploration_queries[:self.max_exploration_rounds], 1):
            print(f"  Round {round_num}: {query}")
            
            # Retrieve relevant documentation
            results = self.vector_store.similarity_search(query, k=8)
            
            # Analyze results and extract insights
            round_data = self._analyze_exploration_results(round_num, query, results)
            print(f"Round data: {round_data}")
            
            # Add to scratchpad
            self.scratchpad.add_exploration_round(round_data)
            
            print(f"    Found {len(results)} docs, confidence: {round_data.confidence_score:.2f}")
    
    def _generate_exploration_queries(self, tool_name: str, context: Optional[str] = None, overview_context: Optional[str] = None) -> List[str]:
        """Generate strategic exploration queries using LLM with overview context"""
        
        # Enhance context with overview information
        enhanced_context = context or "General developer tool for building applications"
        if overview_context:
            enhanced_context += f"\n\nOverview from documentation:\n{overview_context}"
        
        response = self.llm.invoke(self.exploration_prompt.format_messages(
            tool_name=tool_name,
            context=enhanced_context,
            max_rounds=self.max_exploration_rounds
        ))
        
        # Parse LLM response
        queries = self._parse_exploration_queries_response(response.content)
        
        return queries
    
    
    def _parse_exploration_queries_response(self, response_content: str) -> List[str]:
        """Parse LLM response into exploration queries"""
        
        try:
            # Extract JSON array from response
            json_match = re.search(r'\[.*?\]', response_content, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON array found in response")
            
            json_str = json_match.group(0)
            queries = json.loads(json_str)
            
            if not isinstance(queries, list):
                raise ValueError("Response is not a JSON array")
            
            # Validate queries
            valid_queries = []
            for query in queries:
                if isinstance(query, str) and 5 <= len(query.split()) <= 15:
                    valid_queries.append(query.strip())
            
            return valid_queries[:self.max_exploration_rounds]
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️ Error parsing exploration queries: {e}")
            return []
    
    def _analyze_exploration_results(self, round_num: int, query: str, results: List[ProcessedDocument]) -> ExplorationRound:
        """Analyze exploration results using LLM to extract insights"""
        
        if not results:
            return ExplorationRound(
                round_number=round_num,
                query=query,
                results=[],
                key_findings=["No relevant documentation found"],
                confidence_score=0.0,
                documentation_gaps=[f"No documentation for: {query}"]
            )
        
        # Prepare documentation content for LLM analysis
        doc_content = self._prepare_docs_for_analysis(results)
        
        # Use structured LLM to analyze the results
        structured_analysis = self.structured_llm.invoke(self.analysis_prompt.format_messages(
            query=query,
            round_number=round_num,
            tool_name=self.scratchpad.tool_name,
            documentation_content=doc_content
        ))
        
        return ExplorationRound(
            round_number=round_num,
            query=query,
            results=results,
            key_findings=structured_analysis.key_findings,
            confidence_score=structured_analysis.confidence_score,
            documentation_gaps=structured_analysis.documentation_gaps,
            structured_analysis=structured_analysis
        )
    
    def _prepare_docs_for_analysis(self, results: List[ProcessedDocument]) -> str:
        """Prepare documentation for LLM analysis"""
        
        doc_summaries = []
        for i, doc in enumerate(results[:5], 1):  # Top 5 results
            content_preview = doc.content[:400]  # First 400 chars
            doc_summaries.append(f"""
Document {i}:
Title: {doc.section_title}
Type: {doc.content_type}
Source: {doc.file_path or 'Unknown'}
Content: {content_preview}{'...' if len(doc.content) > 400 else ''}
""")
        
        return "\n".join(doc_summaries)
    

    
    def _generate_candidate_plans(self) -> List[ProjectPlan]:
        """Stage 2: Generate diverse project plans based on accumulated knowledge using LLM"""
        
        # Prepare exploration data for the LLM
        exploration_data = self._prepare_exploration_data_for_llm()
        print(f"Exploration data: {exploration_data}")
        
        # Generate plans using LLM
        response = self.llm.invoke(self.plan_generation_prompt.format_messages(
            num_plans=self.plans_to_generate,
            tool_name=self.scratchpad.tool_name,
            **exploration_data
        ))
        
        # Parse LLM response into ProjectPlan objects
        candidate_plans = self._parse_llm_plans_response(response.content)
        print(f"Candidate plans: {candidate_plans}")
        
        return candidate_plans
    
    def _prepare_exploration_data_for_llm(self) -> Dict[str, Any]:
        """Prepare exploration data in a format suitable for the LLM prompt"""
        
        # Categorize insights by type
        capabilities = []
        setup_insights = []
        auth_insights = []
        pattern_insights = []
        advanced_insights = []
        
        # Extract insights from different categories
        for insight in self.scratchpad.core_capabilities:
            capabilities.append(insight)
        
        for insight in self.scratchpad.setup_requirements:
            setup_insights.append(insight)
        
        for insight in self.scratchpad.authentication_info:
            auth_insights.append(insight)
        
        for insight in self.scratchpad.common_patterns:
            pattern_insights.append(insight)
        
        for insight in self.scratchpad.advanced_features:
            advanced_insights.append(insight)
        
        # Determine confidence levels
        high_confidence_areas = []
        low_confidence_areas = []
        
        for round_data in self.scratchpad.exploration_rounds:
            if round_data.confidence_score > 0.7:
                high_confidence_areas.append(round_data.query)
            elif round_data.confidence_score < 0.4:
                low_confidence_areas.append(round_data.query)
        
        return {
            "rounds_completed": len(self.scratchpad.exploration_rounds),
            "well_documented_areas": self.scratchpad.well_documented_areas,
            "documentation_gaps": self.scratchpad.documentation_gaps,
            "capabilities_summary": "\n".join(f"• {cap}" for cap in capabilities[:10]),
            "setup_insights": "\n".join(f"• {insight}" for insight in setup_insights[:5]),
            "auth_insights": "\n".join(f"• {insight}" for insight in auth_insights[:5]),
            "pattern_insights": "\n".join(f"• {insight}" for insight in pattern_insights[:8]),
            "advanced_insights": "\n".join(f"• {insight}" for insight in advanced_insights[:5]),
            "high_confidence_areas": high_confidence_areas,
            "low_confidence_areas": low_confidence_areas
        }
    
    def _parse_llm_plans_response(self, response_content: str) -> List[ProjectPlan]:
        """Parse LLM response into ProjectPlan objects"""
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response_content, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON array found in response")
            
            json_str = json_match.group(0)
            plans_data = json.loads(json_str)
            
            if not isinstance(plans_data, list):
                raise ValueError("Response is not a JSON array")
            
            candidate_plans = []
            for plan_data in plans_data:
                try:
                    # Create ProjectPlan with validation
                    plan = ProjectPlan(
                        plan_id=plan_data.get("plan_id", f"{self.scratchpad.tool_name.lower()}_{len(candidate_plans)}"),
                        title=plan_data["title"],
                        category=plan_data["category"],
                        description=plan_data["description"],
                        main_objectives=plan_data["main_objectives"],
                        success_criteria=plan_data["success_criteria"],
                        expected_challenges=plan_data["expected_challenges"],
                        estimated_difficulty=plan_data.get("estimated_difficulty", "medium"),
                        prerequisite_knowledge=plan_data.get("prerequisite_knowledge", [])
                    )
                    candidate_plans.append(plan)
                except Exception as e:
                    print(f"⚠️ Skipping invalid plan: {e}")
                    continue
            
            return candidate_plans
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error: {e}")
            return []
        except Exception as e:
            print(f"⚠️ Error parsing LLM response: {e}")
            return []

    
    def _refine_all_plans(self, candidate_plans: List[ProjectPlan]) -> List[ProjectPlan]:
        """Stage 3: Refine each plan with targeted evidence gathering"""
        
        refined_plans = []
        
        for plan in candidate_plans:
            print(f"  Refining: {plan.title}")
            refined_plan = self._refine_single_plan(plan)
            refined_plans.append(refined_plan)
        
        return refined_plans
    
    def _refine_single_plan(self, plan: ProjectPlan) -> ProjectPlan:
        """Refine a single plan with targeted documentation research"""
        
        # Generate specific queries for this plan
        refinement_queries = [
            f"{self.scratchpad.tool_name} {plan.category} specific examples implementation",
            f"{self.scratchpad.tool_name} {plan.category} tutorial step by step",
            f"{self.scratchpad.tool_name} {plan.category} common issues errors troubleshooting",
            f"{self.scratchpad.tool_name} {plan.category} best practices patterns"
        ]
        
        all_evidence = []
        total_evidence_quality = 0.0
        
        for query in refinement_queries:
            results = self.vector_store.similarity_search(query, k=5)
            plan.supporting_queries.append(query)
            
            if results:
                # Calculate evidence quality
                evidence_quality = len(results) / 5.0  # Normalize to 0-1
                has_code = any(doc.content_type == "code_example" for doc in results)
                if has_code:
                    evidence_quality += 0.2
                
                total_evidence_quality += min(1.0, evidence_quality)
                all_evidence.extend(results)
                
                # Add documentation references
                for doc in results[:2]:  # Top 2 per query
                    if doc.file_path:
                        plan.documentation_refs.append(f"{doc.file_path}: {doc.section_title}")
        
        # Calculate overall evidence strength
        plan.evidence_strength = total_evidence_quality / len(refinement_queries)
        
        # Update difficulty based on evidence strength
        if plan.evidence_strength > 0.7:
            plan.estimated_difficulty = "easy"
        elif plan.evidence_strength > 0.4:
            plan.estimated_difficulty = "medium"
        else:
            plan.estimated_difficulty = "hard"
        
        # Extract key APIs/features mentioned in evidence
        for doc in all_evidence[:10]:
            content_lower = doc.content.lower()
            if "api" in content_lower or "function" in content_lower or "method" in content_lower:
                plan.key_apis_or_features.append(doc.section_title)
        
        return plan
    
    def save_planning_results(self, plans: List[ProjectPlan], scratchpad: ProjectScratchpad, output_path: str):
        """Save planning results to JSON file using Pydantic serialization"""
        
        # Create a wrapper model for better structure
        class PlanningResults(BaseModel):
            tool_name: str
            timestamp: str
            exploration_summary: Dict[str, Any]
            generated_plans: List[Dict[str, Any]]
        
        results = PlanningResults(
            tool_name=scratchpad.tool_name,
            timestamp=datetime.now().isoformat(),
            exploration_summary={
                "rounds_completed": len(scratchpad.exploration_rounds),
                "core_capabilities": scratchpad.core_capabilities,
                "well_documented_areas": scratchpad.well_documented_areas,
                "documentation_gaps": scratchpad.documentation_gaps
            },
            generated_plans=[plan.model_dump() for plan in plans]
        )
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(results.model_dump_json(indent=2))
        
        print(f"📝 Planning results saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # This would be used with the vector store from document_processor.py
    from document_processor import VectorStoreManager, create_documentation_vector_store
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from pathlib import Path
    
    # Check if vector store already exists
    persist_dir = "./chroma_db"
    collection_name = "fastapi-fastapi"
    
    if Path(persist_dir).exists():
        print("Loading existing vector store...")
        # Load existing vector store
        vector_manager = VectorStoreManager(
            persist_directory=persist_dir,
            collection_name=collection_name
        )
        
        vector_manager.vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            collection_name=collection_name
        )
    else:
        print("Creating new vector store...")
        # Create vector store from FastAPI docs
        vector_manager = create_documentation_vector_store(
            "https://github.com/fastapi/fastapi",
            persist_directory=persist_dir,
            collection_name=collection_name,
            include_folders=["docs/en"]
        )
    
    # Create planner
    planner = ProgressivePlannerAgent(vector_manager, llm_model="gpt-4o-mini")
    
    # Generate plans
    plans, scratchpad = planner.plan_project("FastAPI")
    
    # Save results
    planner.save_planning_results(plans, scratchpad, "./outputs/fastapi_planning_results.json")
    
    # Print summary
    print(f"\n🎯 Generated {len(plans)} project plans:")
    for plan in plans:
        print(f"  • {plan.title} (difficulty: {plan.estimated_difficulty}, evidence: {plan.evidence_strength:.2f})")