import dspy
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Literal, Union
from enum import Enum
import json
import re
import concurrent.futures
from .fetcher import RawContent

# Default agent instructions
DEFAULT_AGENT_INSTRUCTIONS = """
You are extracting tool documentation for automated test planning and development.

Your goal is to extract actionable information that enables comprehensive testing and integration:

1. FOCUS ON TESTABLE ELEMENTS: Extract concrete operations, parameters, expected outputs, and error scenarios
2. PRIORITIZE EXAMPLES: Include working code examples, sample requests/responses, and real-world usage patterns
3. IDENTIFY FAILURE POINTS: Look for error conditions, validation rules, rate limits, and edge cases
4. EXTRACT SETUP REQUIREMENTS: Document authentication, dependencies, configuration, and prerequisites
5. CAPTURE WORKFLOWS: Extract multi-step processes, common usage patterns, and integration sequences

The extracted information will be used by test planning agents to:
- Generate comprehensive test cases covering normal and edge cases
- Create setup and teardown procedures
- Identify integration points and dependencies
- Plan error handling and recovery testing
- Validate API contracts and data flows

Be thorough, specific, and focus on information that enables robust automated testing.
"""

# Enums for validation
class ToolType(str, Enum):
    API = "api"
    LIBRARY = "library"
    FRAMEWORK = "framework"
    CLI_TOOL = "cli_tool"
    SERVICE = "service"

class OperationType(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SUBSCRIBE = "subscribe"
    CONFIGURE = "configure"

class ComplexityLevel(str, Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class AuthType(str, Enum):
    API_KEY = "api_key"
    OAUTH = "oauth"
    BASIC_AUTH = "basic_auth"
    BEARER_TOKEN = "bearer_token"
    NONE = "none"

class SetupType(str, Enum):
    INSTALL = "install"
    CONFIG = "config"
    ACCOUNT = "account"
    DEPENDENCY = "dependency"

# Core BaseModel classes
class ToolOverview(BaseModel):
    """Basic tool information"""
    name: str = Field(..., description="Name of the tool")
    type: ToolType = Field(..., description="Type of tool")
    purpose: str = Field(..., description="What problem does this solve")
    base_url: Optional[str] = Field(None, description="Base URL for APIs")
    package_name: Optional[str] = Field(None, description="Package name for libraries")
    version: Optional[str] = Field(None, description="Version information")

class Concept(BaseModel):
    """Core concept or entity in the tool"""
    name: str = Field(..., description="Name of the concept")
    description: str = Field(..., description="Description of the concept")
    attributes: List[str] = Field(default_factory=list, description="Key properties/fields")
    operations: List[str] = Field(default_factory=list, description="What you can do with it")

class Operation(BaseModel):
    """Individual operation or capability"""
    name: str = Field(..., description="Name of the operation")
    type: OperationType = Field(..., description="Type of operation")
    complexity: ComplexityLevel = Field(..., description="Complexity level")
    description: str = Field(..., description="What this operation does")
    concepts_involved: List[str] = Field(default_factory=list, description="Which concepts are used")
    parameters: List[str] = Field(default_factory=list, description="Required inputs")
    returns: Optional[str] = Field(None, description="What it returns")

class AuthMethod(BaseModel):
    """Authentication method"""
    type: AuthType = Field(..., description="Authentication type")
    description: str = Field(..., description="Description of auth method")
    required_for: List[str] = Field(default_factory=list, description="Operations that need this")
    setup_steps: List[str] = Field(default_factory=list, description="How to set it up")
    example_code: Optional[str] = Field(None, description="Example auth code")

class SetupRequirement(BaseModel):
    """Setup requirement"""
    type: SetupType = Field(..., description="Type of setup requirement")
    description: str = Field(..., description="Description of requirement")
    commands: List[str] = Field(default_factory=list, description="Commands to run")
    is_required: bool = Field(True, description="Whether this is mandatory")

class UsagePattern(BaseModel):
    """Usage pattern or workflow"""
    name: str = Field(..., description="Name of the pattern")
    complexity: ComplexityLevel = Field(..., description="Complexity level")
    description: str = Field(..., description="What this pattern accomplishes")
    steps: List[str] = Field(default_factory=list, description="Sequence of operations")
    example_code: Optional[str] = Field(None, description="Working example")
    concepts_used: List[str] = Field(default_factory=list, description="Concepts involved")

class ErrorScenario(BaseModel):
    """Error scenario"""
    error_type: str = Field(..., description="Type of error")
    trigger_conditions: List[str] = Field(default_factory=list, description="What causes this")
    expected_response: str = Field(..., description="Expected error message/code")
    handling_approach: str = Field(..., description="How to handle/recover")
    related_operations: List[str] = Field(default_factory=list, description="Operations that can trigger this")

class PageAnalysis(BaseModel):
    """Analysis results for a single page/URL"""
    overview: List[ToolOverview] = Field(default_factory=list)
    concepts: List[Concept] = Field(default_factory=list)
    operations: List[Operation] = Field(default_factory=list)
    auth_methods: List[AuthMethod] = Field(default_factory=list)
    setup_requirements: List[SetupRequirement] = Field(default_factory=list)
    patterns: List[UsagePattern] = Field(default_factory=list)
    error_scenarios: List[ErrorScenario] = Field(default_factory=list)
    source_type: str = Field(..., description="Source type: api, website, or github")
    content: str = Field(..., description="Raw documentation content")

class ToolDocumentation(BaseModel):
    """Complete tool documentation for test planning"""
    pages: Dict[str, PageAnalysis] = Field(default_factory=dict, description="Analysis organized by page/URL")

# DSPy signatures - intelligently combined into 3 extractors
class CoreExtractor(dspy.Signature):
    """Extract foundational tool information and core concepts"""
    content: str = dspy.InputField(desc="Raw documentation content")
    source_type: str = dspy.InputField(desc="Source type: api, website, or github")
    metadata: str = dspy.InputField(desc="Metadata from fetcher")
    agent_instructions: str = dspy.InputField(desc="Instructions for what to extract and how it will be used")
    tool_overview: List[ToolOverview] = dspy.OutputField(desc="Tool overview information")
    concepts: List[Concept] = dspy.OutputField(desc="Core concepts with attributes")

class CapabilityExtractor(dspy.Signature):
    """Extract operations, capabilities, authentication and setup requirements"""
    content: str = dspy.InputField(desc="Raw documentation content")
    tool_type: str = dspy.InputField(desc="Type of tool for context")
    concepts: List[Concept] = dspy.InputField(desc="Previously extracted concepts")
    agent_instructions: str = dspy.InputField(desc="Instructions for what to extract and how it will be used")
    operations: List[Operation] = dspy.OutputField(desc="Operations with complexity and parameters")
    auth_methods: List[AuthMethod] = dspy.OutputField(desc="Authentication methods")
    setup_requirements: List[SetupRequirement] = dspy.OutputField(desc="Setup requirements")

class WorkflowExtractor(dspy.Signature):
    """Extract usage patterns, workflows and error scenarios"""
    content: str = dspy.InputField(desc="Raw documentation content")
    operations: List[Operation] = dspy.InputField(desc="Previously extracted operations")
    agent_instructions: str = dspy.InputField(desc="Instructions for what to extract and how it will be used")
    patterns: List[UsagePattern] = dspy.OutputField(desc="Usage patterns with examples")
    error_scenarios: List[ErrorScenario] = dspy.OutputField(desc="Error scenarios and responses")



# Processing Pipeline
class DocumentProcessor:
    """Main processor that orchestrates the DSPy pipeline"""
    
    def __init__(self, agent_instructions: str, use_parallel: bool = True, max_workers: int = None):
        self.agent_instructions = agent_instructions
        self.use_parallel = use_parallel
        self.max_workers = max_workers or 4
        
        # DSPy extractors - 3 combined extractors
        self.core_extractor = dspy.ChainOfThought(CoreExtractor)
        self.capability_extractor = dspy.ChainOfThought(CapabilityExtractor)
        self.workflow_extractor = dspy.ChainOfThought(WorkflowExtractor)
    
    def process(self, raw_content: RawContent) -> Dict[str, any]:
        """Process raw content through DSPy pipeline"""
        
        metadata = json.dumps(raw_content.metadata)
        
        # Prepare pages for processing
        pages = [(url, content) for url, content in raw_content.content.items() if content.strip()]
        
        if not pages:
            return {'pages': {}}
        
        # Process pages in parallel or sequentially
        if self.use_parallel and len(pages) > 1:
            print(f"Processing {len(pages)} pages in parallel with {self.max_workers} workers")
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [executor.submit(self._process_single_page, url, content, raw_content.source_type, metadata) for url, content in pages]
                    page_results = [future.result() for future in futures if future.result() is not None]
            except Exception as e:
                print(f"Parallel processing failed: {e}, falling back to sequential")
                page_results = [self._process_single_page(url, content, raw_content.source_type, metadata) for url, content in pages]
                page_results = [result for result in page_results if result is not None]
        else:
            print(f"Processing {len(pages)} pages sequentially")
            page_results = [self._process_single_page(url, content, raw_content.source_type, metadata) for url, content in pages]
            page_results = [result for result in page_results if result is not None]
        
        # Organize results by page/URL
        pages_analysis = {}
        
        for page in page_results:
            pages_analysis[page['url']] = {
                'overview': page['overview'],
                'concepts': page['concepts'],
                'operations': page['operations'],
                'auth_methods': page['auth_methods'],
                'setup_requirements': page['setup_requirements'],
                'patterns': page['patterns'],
                'error_scenarios': page['error_scenarios'],
                'source_type': raw_content.source_type,
                'content': page['content']
            }
        
        return {'pages': pages_analysis}
    

    
    def _process_single_page(self, url: str, content: str, source_type: str, metadata: str) -> Optional[dict]:
        """Process a single page through the 3-step DSPy pipeline"""
        
        if not content.strip():  # Skip empty pages
            return None
            
        print(f"Processing page: {url}")
        
        # Step 1: Extract core information (overview + concepts)
        core_result = self.core_extractor(
            content=content,
            source_type=source_type,
            metadata=metadata,
            agent_instructions=self.agent_instructions
        )
        
        # Get tool type from overview
        tool_type = 'service'  # default to valid enum value
        if core_result.tool_overview and len(core_result.tool_overview) > 0:
            tool_type = str(core_result.tool_overview[0].type)  # convert enum to string
        
        # Step 2: Extract capabilities (operations + auth + setup)
        capability_result = self.capability_extractor(
            content=content,
            tool_type=tool_type,
            concepts=core_result.concepts,
            agent_instructions=self.agent_instructions
        )
        
        # Step 3: Extract workflows (patterns + errors)
        workflow_result = self.workflow_extractor(
            content=content,
            operations=capability_result.operations,
            agent_instructions=self.agent_instructions
        )
        
        # Return results for this page
        return {
            'url': url,
            'overview': core_result.tool_overview,
            'concepts': core_result.concepts,
            'operations': capability_result.operations,
            'auth_methods': capability_result.auth_methods,
            'setup_requirements': capability_result.setup_requirements,
            'patterns': workflow_result.patterns,
            'error_scenarios': workflow_result.error_scenarios,
            'source_type': source_type,
            'content': content
        }

# High-level analyzer
class DocumentAnalyzer:
    """High-level analyzer that converts DSPy output to structured models"""
    
    def __init__(self, agent_instructions: str = DEFAULT_AGENT_INSTRUCTIONS, use_parallel: bool = True, max_workers: int = None):
        self.processor = DocumentProcessor(agent_instructions, use_parallel, max_workers)
    
    def analyze(self, raw_content: RawContent) -> ToolDocumentation:
        """Analyze raw content and return structured documentation"""
        try:
            # Process with DSPy
            processed = self.processor.process(raw_content)
            
            # Convert to structured format
            tool_doc = self._convert_to_structured(processed, raw_content)
            
            return tool_doc
            
        except Exception as e:
            # Return error document
            return self._create_error_document(raw_content, str(e))
    
    def _convert_to_structured(self, processed: Dict[str, any], raw_content: RawContent) -> ToolDocumentation:
        """Convert DSPy output to structured format with validation"""
        try:
            # Convert page-organized data to PageAnalysis objects
            pages_analysis = {}
            for url, page_data in processed.get('pages', {}).items():
                pages_analysis[url] = PageAnalysis(
                    overview=page_data['overview'],
                    concepts=page_data['concepts'],
                    operations=page_data['operations'],
                    auth_methods=page_data['auth_methods'],
                    setup_requirements=page_data['setup_requirements'],
                    patterns=page_data['patterns'],
                    error_scenarios=page_data['error_scenarios'],
                    source_type=page_data['source_type'],
                    content=page_data['content']
                )
            
            return ToolDocumentation(pages=pages_analysis)
            
        except Exception as e:
            return self._create_error_document(raw_content, f"Conversion error: {str(e)}")
    
    def _create_error_document(self, raw_content: RawContent, error_message: str) -> ToolDocumentation:
        """Create an error document when processing fails"""
        return ToolDocumentation(pages={})