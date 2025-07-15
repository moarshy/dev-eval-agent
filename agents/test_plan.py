import dspy
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Set
from enum import Enum
import concurrent.futures
from doc_ingestion.analyzer import ToolDocumentation, PageAnalysis, ToolType, ComplexityLevel

# Test categories from project doc
class TestCategory(str, Enum):
    AUTHENTICATION = "authentication"
    BASIC_USAGE = "basic_usage"
    CORE_WORKFLOWS = "core_workflows"
    ERROR_HANDLING = "error_handling"

class TestPriority(str, Enum):
    CRITICAL = "critical"  # Must pass for tool to be usable
    HIGH = "high"         # Important functionality
    MEDIUM = "medium"     # Nice to have features
    LOW = "low"          # Edge cases or advanced features

class TestScenario(BaseModel):
    """Individual test scenario"""
    name: str = Field(..., description="Clear test scenario name")
    category: TestCategory = Field(..., description="Test category")
    priority: TestPriority = Field(..., description="Test priority level")
    description: str = Field(..., description="What this test validates")
    
    # What we're testing
    concepts_involved: List[str] = Field(default_factory=list, description="Concepts being tested")
    operations_tested: List[str] = Field(default_factory=list, description="Operations being tested")
    auth_methods_used: List[str] = Field(default_factory=list, description="Auth methods required")
    
    # Test execution plan
    setup_requirements: List[str] = Field(default_factory=list, description="Prerequisites for this test")
    test_steps: List[str] = Field(default_factory=list, description="Step-by-step test execution plan")
    expected_outcome: str = Field(..., description="What should happen if test passes")
    failure_scenarios: List[str] = Field(default_factory=list, description="Ways this test might fail")
    
    # Difficulty and dependencies
    complexity: ComplexityLevel = Field(..., description="Test complexity level")
    depends_on: List[str] = Field(default_factory=list, description="Other test scenarios this depends on")
    
    # Source tracking
    source_pages: List[str] = Field(default_factory=list, description="Pages that contributed to this test")

class PageTestPlan(BaseModel):
    """Test plan for a single documentation page"""
    page_url: str = Field(..., description="URL of the page")
    page_summary: str = Field(..., description="Brief summary of page content")
    
    # Generated test scenarios for this page
    scenarios: List[TestScenario] = Field(default_factory=list, description="Test scenarios from this page")
    
    # Page-specific insights
    coverage_areas: List[str] = Field(default_factory=list, description="What functionality this page covers")
    missing_info: List[str] = Field(default_factory=list, description="Information gaps that affect testing")


# DSPy signatures for test plan generation
class PageTestPlanExtractor(dspy.Signature):
    """Generate test scenarios for a single documentation page"""
    page_url: str = dspy.InputField(desc="URL of the page being analyzed")
    page_analysis: str = dspy.InputField(desc="JSON string of PageAnalysis data")
    raw_content: str = dspy.InputField(desc="Raw documentation content from the page")
    agent_instructions: str = dspy.InputField(desc="Instructions for test planning approach")
    
    test_scenarios: List[TestScenario] = dspy.OutputField(desc="Generated test scenarios for this page")
    page_summary: str = dspy.OutputField(desc="Brief summary of what this page covers")
    coverage_areas: List[str] = dspy.OutputField(desc="Functionality areas covered by this page")
    missing_info: List[str] = dspy.OutputField(desc="Information gaps that affect testing")

# Default instructions for test planning
DEFAULT_TEST_PLAN_INSTRUCTIONS = """
You are creating comprehensive test plans for developer tools to evaluate their "agent usability" - how well they work with AI-assisted development.

You have access to both:
1. STRUCTURED ANALYSIS: Parsed operations, concepts, auth methods, and patterns
2. RAW CONTENT: Original documentation with examples, code snippets, and detailed explanations

Use both sources to create comprehensive test scenarios:
- Use structured analysis for systematic coverage of all operations and concepts
- Use raw content for realistic examples, parameter values, and edge cases
- Extract actual code examples from raw content for test steps
- Identify specific error messages and response codes from documentation

FOCUS ON PRACTICAL TESTING:
1. Generate tests that a developer or AI agent would actually run
2. Assume setup has already been done. API keys are available.
3. Include both happy path and error scenarios
4. Test authentication, configuration, and integration points
5. Validate that documentation matches actual behavior

TEST CATEGORIES:
- AUTHENTICATION: API keys, OAuth, tokens, credentials management
- BASIC_USAGE: Simple operations, core functionality, getting started
- CORE_WORKFLOWS: Complex multi-step processes, real-world usage patterns
- ERROR_HANDLING: Invalid inputs, rate limits, failures, recovery

PRIORITIZATION:
- CRITICAL: Must work for tool to be usable (auth, basic operations)
- HIGH: Important functionality users expect (core workflows)
- MEDIUM: Nice-to-have features (advanced options)
- LOW: Edge cases or rarely used features

Create tests that are:
- Specific and actionable with real parameter values from examples
- Realistic for real developer workflows
- Progressive in complexity (simple to advanced)
- Well-documented with clear success criteria
- Include actual code snippets and expected responses from documentation
"""

class TestPlanGenerator:
    """Main test plan generator that creates comprehensive test plans"""
    
    def __init__(self, instructions: str = DEFAULT_TEST_PLAN_INSTRUCTIONS, use_parallel: bool = True, max_workers: int = None):
        self.instructions = instructions
        self.use_parallel = use_parallel
        self.max_workers = max_workers or 4
        self.page_extractor = dspy.ChainOfThought(PageTestPlanExtractor)
    
    def generate_test_plan(self, tool_docs: ToolDocumentation) -> PageTestPlan:
        """Generate comprehensive test plan from tool documentation"""
        
        # Step 1: Generate per-page test plans
        pages = list(tool_docs.pages.items())
        
        # Process pages in parallel or sequentially
        if self.use_parallel and len(pages) > 1:
            print(f"Processing {len(pages)} pages in parallel with {self.max_workers} workers")
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [executor.submit(self._generate_page_test_plan, page_url, page_analysis, tool_docs) 
                              for page_url, page_analysis in pages]
                    page_test_plans = [future.result() for future in futures if future.result() is not None]
            except Exception as e:
                print(f"Parallel processing failed: {e}, falling back to sequential")
                page_test_plans = [self._generate_page_test_plan(page_url, page_analysis, tool_docs) 
                                  for page_url, page_analysis in pages]
                page_test_plans = [plan for plan in page_test_plans if plan is not None]
        else:
            print(f"Processing {len(pages)} pages sequentially")
            page_test_plans = [self._generate_page_test_plan(page_url, page_analysis, tool_docs) 
                              for page_url, page_analysis in pages]
            page_test_plans = [plan for plan in page_test_plans if plan is not None]
        
        return page_test_plans
    
    def _generate_page_test_plan(self, page_url: str, page_analysis: PageAnalysis, tool_docs: ToolDocumentation) -> Optional[PageTestPlan]:
        """Generate test plan for a single page"""
        
        # Determine tool type
        tool_type = "service"  # default
        if page_analysis.overview:
            tool_type = str(page_analysis.overview[0].type)
        
        # Convert page analysis to JSON for DSPy
        import json
        page_data = page_analysis.dict()
        
        try:
            # Generate test scenarios for this page
            result = self.page_extractor(
                page_url=page_url,
                page_analysis=json.dumps(page_data),
                raw_content=page_analysis.content,
                tool_type=tool_type,
                agent_instructions=self.instructions
            )
            
            return PageTestPlan(
                page_url=page_url,
                page_summary=result.page_summary,
                scenarios=result.test_scenarios,
                coverage_areas=result.coverage_areas,
                missing_info=result.missing_info
            )
            
        except Exception as e:
            print(f"Error generating test plan for page {page_url}: {e}")
            return None
