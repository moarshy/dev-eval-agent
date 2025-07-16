from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class TaskStatus(str, Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class RawContent(BaseModel):
    """Raw content fetched from various sources"""
    source_type: str = Field(..., description="Type of source: openapi, website, or github")
    source_url: str = Field(..., description="Original source URL or file path")
    content: Dict[str, str] = Field(..., description="Key-value pairs of extracted content")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional metadata about the source")
    fetch_timestamp: str = Field(..., description="ISO timestamp of when content was fetched")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class DocumentationSource(BaseModel):
    """Enhanced documentation source information"""
    source_type: str = Field(..., description="website, github, or openapi")
    source_url: str = Field(..., description="Original source URL")
    fetching_strategy: str = Field(default="auto", description="Fetching strategy used")
    raw_content: Optional[RawContent] = Field(None, description="Raw fetched content")
    fetch_completed: bool = Field(default=False)
    fetch_error: Optional[str] = Field(None, description="Error during fetching if any")

class ProcessedDocument(BaseModel):
    """Represents a processed documentation chunk ready for vector store"""
    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    source_url: str = Field(..., description="Original source URL/file path")
    content: str = Field(..., description="Processed markdown/text content")
    content_type: str = Field(..., description="code_example, explanation, api_reference, tutorial, etc.")
    section_title: str = Field(default="", description="Section or page title")
    file_path: Optional[str] = Field(None, description="File path for GitHub sources")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class VectorStoreMetadata(BaseModel):
    """Enhanced vector store metadata"""
    total_chunks: int = Field(..., description="Total number of documentation chunks")
    embedding_model: str = Field(..., description="Model used for embeddings")
    chunk_strategy: str = Field(..., description="How documentation was chunked")
    average_chunk_size: int = Field(..., description="Average characters per chunk")
    
    # Content type distribution
    content_type_distribution: Dict[str, int] = Field(default_factory=dict, description="Count by content type")
    source_distribution: Dict[str, int] = Field(default_factory=dict, description="Count by source file/page")
    
    # Quality metrics
    code_example_count: int = Field(default=0, description="Number of code examples found")
    api_endpoint_count: int = Field(default=0, description="Number of API endpoints documented")
    tutorial_section_count: int = Field(default=0, description="Number of tutorial sections")
    
    # GitHub-specific metadata
    repo_info: Optional[Dict[str, Any]] = Field(None, description="GitHub repository information")
    file_types_processed: List[str] = Field(default_factory=list, description="File extensions processed")

class RAGContext(BaseModel):
    """Context retrieved from vector store for a specific query"""
    query: str = Field(..., description="The query that was used for retrieval")
    retrieved_chunks: List[ProcessedDocument] = Field(..., description="Relevant documentation chunks")
    relevance_scores: List[float] = Field(..., description="Relevance scores for each chunk")
    total_chunks_available: int = Field(..., description="Total chunks in vector store")
    retrieval_strategy: str = Field(default="semantic", description="Retrieval strategy used")

class CodeAttempt(BaseModel):
    """Single attempt at implementing code for a use case"""
    attempt_number: int = Field(..., description="Which attempt this is (1, 2, 3...)")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # RAG Information
    rag_context: RAGContext = Field(..., description="Documentation retrieved for this attempt")
    
    # Generated Code (matching your existing code structure)
    prefix: str = Field(..., description="Description of the approach")
    imports: str = Field(..., description="Import statements")
    code: str = Field(..., description="Main code block")
    
    # Execution Results
    import_test_passed: bool = Field(default=False)
    import_error: Optional[str] = Field(None)
    execution_test_passed: bool = Field(default=False)
    execution_error: Optional[str] = Field(None)
    execution_output: Optional[str] = Field(None)
    
    # Documentation Analysis
    documentation_helpful: bool = Field(..., description="Was retrieved documentation useful?")
    missing_information: List[str] = Field(default_factory=list, description="What info was missing from docs")
    confusing_parts: List[str] = Field(default_factory=list, description="What parts of docs were unclear")
    suggested_improvements: List[str] = Field(default_factory=list, description="How to improve the docs")
    
    # Enhanced analysis
    chunks_used: List[str] = Field(default_factory=list, description="Which doc chunks were actually helpful")
    chunks_ignored: List[str] = Field(default_factory=list, description="Which chunks were irrelevant")

class UseCase(BaseModel):
    """Individual test scenario generated by the Planner Agent"""
    id: str = Field(..., description="Unique identifier for this use case")
    title: str = Field(..., description="Human-readable title")
    description: str = Field(..., description="Detailed description of what should be accomplished")
    success_criteria: List[str] = Field(..., description="Specific, measurable criteria for success")
    difficulty_level: DifficultyLevel = Field(..., description="Expected complexity level")
    dependencies: List[str] = Field(default_factory=list, description="IDs of use cases that must complete first")
    category: str = Field(..., description="setup, auth, basic_usage, integration, error_handling")
    estimated_duration_minutes: int = Field(default=15, description="Expected time to complete")

class UseCaseExecution(BaseModel):
    """Complete execution record for one use case with iterative attempts"""
    use_case_id: str = Field(..., description="Reference to the UseCase")
    status: TaskStatus = Field(default=TaskStatus.PLANNED)
    start_time: Optional[datetime] = Field(None)
    end_time: Optional[datetime] = Field(None)
    
    # Multiple attempts with RAG and self-correction
    attempts: List[CodeAttempt] = Field(default_factory=list)
    final_working_code: Optional[str] = Field(None, description="Final successful implementation")
    
    # Success Analysis
    success_criteria_met: Dict[str, bool] = Field(default_factory=dict)
    overall_success: bool = Field(default=False)
    attempts_needed: int = Field(default=0, description="How many attempts were required")
    
    # Documentation Quality Insights
    documentation_quality_score: Optional[float] = Field(None, ge=0, le=10)
    key_documentation_gaps: List[str] = Field(default_factory=list)
    documentation_strengths: List[str] = Field(default_factory=list)

class EvaluationDimension(BaseModel):
    """Score for one aspect of agent readiness"""
    name: str = Field(..., description="Dimension name")
    score: float = Field(..., ge=0, le=10, description="Score from 0-10")
    weight: float = Field(..., ge=0, le=1, description="Relative importance")
    reasoning: str = Field(..., description="Explanation of score")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")

class AgentReadinessState(BaseModel):
    """Complete shared state for the evaluation system"""
    
    # Input Information
    tool_name: str = Field(..., description="Name of the tool being evaluated")
    tool_category: str = Field(..., description="API, Framework, Library, Platform, etc.")
    
    # Enhanced Documentation Source
    documentation_source: DocumentationSource = Field(..., description="Source information and raw content")
    
    # Processed Documentation
    processed_documents: List[ProcessedDocument] = Field(default_factory=list)
    vector_store_metadata: Optional[VectorStoreMetadata] = Field(None)
    vector_store_ready: bool = Field(default=False)
    
    # Planner Agent Output
    use_cases: List[UseCase] = Field(default_factory=list)
    planning_rag_context: Optional[RAGContext] = Field(None, description="RAG context used for planning")
    planning_notes: str = Field(default="", description="Planner's analysis and reasoning")
    planning_completed: bool = Field(default=False)
    
    # Coding Agent Output  
    executions: List[UseCaseExecution] = Field(default_factory=list)
    environment_setup: Dict[str, Any] = Field(default_factory=dict)
    coding_completed: bool = Field(default=False)
    
    # Evaluator Agent Output
    dimension_scores: List[EvaluationDimension] = Field(default_factory=list)
    overall_score: Optional[float] = Field(None, ge=0, le=10)
    documentation_improvement_plan: List[str] = Field(default_factory=list)
    key_insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    evaluation_completed: bool = Field(default=False)
    
    # Meta Information
    evaluation_id: str = Field(..., description="Unique identifier for this evaluation")
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = Field(None)
    total_duration_minutes: Optional[float] = Field(None)

class Crawl4AIConfig(BaseModel):
    """Configuration for Crawl4AI-based fetching"""
    max_pages: int = Field(default=50, description="Maximum number of pages to crawl")
    max_depth: int = Field(default=2, description="Maximum crawl depth")
    delay: float = Field(default=1.0, description="Delay between requests in seconds")
    include_external: bool = Field(default=False, description="Whether to include external links")
    crawl_strategy: str = Field(default="bfs", description="Crawling strategy: bfs, dfs, best_first, or simple")
    stream_results: bool = Field(default=False, description="Whether to stream results as they come")
    keywords: List[str] = Field(default_factory=list, description="Keywords for relevance scoring (best_first only)")
    verbose: bool = Field(default=False, description="Enable verbose logging")
    cache_enabled: bool = Field(default=True, description="Enable caching")
    remove_overlay_elements: bool = Field(default=True, description="Remove popups and overlays")
    word_count_threshold: int = Field(default=10, description="Minimum words per content block")