#!/usr/bin/env python3
"""
Complete End-to-End Developer Tool Testing System

Orchestrates the entire testing pipeline:
1. Fetch documentation pages
2. Analyze content with DSPy
3. Generate test plans
4. Execute tests in parallel  
5. Generate intelligent reports

Tracks progress by page and provides comprehensive configuration options.
"""

import json
import time
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import traceback
import os
import logging
from dotenv import load_dotenv

# DSPy configuration
import dspy

# Import our components
from .doc_ingestion.analyzer import DocumentProcessor, ToolDocumentation
from .test_plan import TestPlanGenerator, PageTestPlan
from .code_execution import ParallelTestExecutor, TestResult
from .report import IntelligentReporter, PageReport, OverallReport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
dspy.configure(lm=dspy.LM("gemini/gemini-2.5-flash", max_tokens=30000))


class StageStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class PageProgress(BaseModel):
    """Tracks progress of a single page through the pipeline"""
    page_url: str = Field(..., description="URL of the page")
    
    # Stage statuses
    fetch_status: StageStatus = Field(default=StageStatus.PENDING, description="Page fetching status")
    analysis_status: StageStatus = Field(default=StageStatus.PENDING, description="Content analysis status")
    test_plan_status: StageStatus = Field(default=StageStatus.PENDING, description="Test plan generation status")
    execution_status: StageStatus = Field(default=StageStatus.PENDING, description="Test execution status")
    report_status: StageStatus = Field(default=StageStatus.PENDING, description="Report generation status")
    
    # Stage outputs
    raw_content: Optional[str] = Field(default=None, description="Fetched page content")
    analysis_result: Optional[Dict[str, Any]] = Field(default=None, description="Analysis output")
    test_plan: Optional[PageTestPlan] = Field(default=None, description="Generated test plan")
    test_results: List[TestResult] = Field(default_factory=list, description="Test execution results")
    page_report: Optional[PageReport] = Field(default=None, description="Generated page report")
    
    # Metadata
    fetch_time: Optional[float] = Field(default=None, description="Time to fetch page")
    analysis_time: Optional[float] = Field(default=None, description="Time to analyze content")
    test_plan_time: Optional[float] = Field(default=None, description="Time to generate test plan")
    execution_time: Optional[float] = Field(default=None, description="Time to execute tests")
    report_time: Optional[float] = Field(default=None, description="Time to generate report")
    
    # Error tracking
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Warnings generated")

class PipelineConfig(BaseModel):
    """Configuration for the testing pipeline"""
    
    # Target configuration
    base_url: str = Field(..., description="Base URL to start crawling from")
    tool_name: str = Field(..., description="Name of the tool being tested")
    
    # URL filtering
    urls_to_include: List[str] = Field(default_factory=list, description="Specific URLs to include (empty = include all)")
    urls_to_exclude: List[str] = Field(default_factory=list, description="URLs to exclude from processing")
    url_patterns_to_exclude: List[str] = Field(default_factory=list, description="URL patterns to exclude (regex)")
    
    # Stage controls
    skip_analysis: bool = Field(default=False, description="Skip content analysis stage")
    skip_test_plans: bool = Field(default=False, description="Skip test plan generation")
    skip_execution: bool = Field(default=False, description="Skip test execution")
    skip_reports: bool = Field(default=False, description="Skip report generation")
    
    # API Keys and context
    api_keys: Dict[str, str] = Field(default_factory=dict, description="API keys for testing")
    context: Dict[str, str] = Field(default_factory=dict, description="Additional context for tests")
    
    # Fetching options  
    max_depth: int = Field(default=3, description="Maximum crawl depth")
    keywords: List[str] = Field(default_factory=lambda: ["api", "documentation", "guide"], description="Keywords for crawling")
    
    # Processing options
    max_pages: Optional[int] = Field(default=20, description="Maximum number of pages to process")
    parallel_analysis: bool = Field(default=True, description="Use parallel processing for analysis")
    parallel_test_plans: bool = Field(default=True, description="Use parallel processing for test plans")
    parallel_execution: bool = Field(default=True, description="Use parallel test execution")
    max_workers: int = Field(default=8, description="Maximum number of parallel workers")
    
    # Output options
    save_intermediate: bool = Field(default=True, description="Save intermediate results")
    output_dir: str = Field(default="test_results", description="Directory for output files")

class TestingPipelineState(BaseModel):
    """Complete state of the testing pipeline"""
    
    config: PipelineConfig = Field(..., description="Pipeline configuration")
    
    # Overall progress
    start_time: Optional[datetime] = Field(default=None, description="Pipeline start time")
    end_time: Optional[datetime] = Field(default=None, description="Pipeline end time")
    current_stage: str = Field(default="initialized", description="Current pipeline stage")
    
    # Page tracking
    pages: Dict[str, PageProgress] = Field(default_factory=dict, description="Progress by page URL")
    
    # Stage summaries
    total_pages: int = Field(default=0, description="Total pages to process")
    completed_pages: int = Field(default=0, description="Pages fully completed")
    failed_pages: int = Field(default=0, description="Pages that failed")
    
    # Final outputs
    overall_report: Optional[OverallReport] = Field(default=None, description="Final overall report")
    output_files: List[str] = Field(default_factory=list, description="Generated output files")
    
    # Error tracking
    pipeline_errors: List[str] = Field(default_factory=list, description="Pipeline-level errors")

class DeveloperToolTestingPipeline:
    """Complete end-to-end testing pipeline for developer tools"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.state = TestingPipelineState(config=config)
                
        # Initialize components
        self.document_processor = DocumentProcessor(
            agent_instructions="Extract comprehensive tool documentation for testing",
            use_parallel=config.parallel_analysis,
            max_workers=config.max_workers
        )
        
        self.test_plan_generator = TestPlanGenerator(
            use_parallel=config.parallel_test_plans,
            max_workers=config.max_workers
        )
        
        self.test_executor = ParallelTestExecutor(
            max_workers=config.max_workers,
            use_parallel=config.parallel_execution
        )
        
        self.reporter = IntelligentReporter(tool_name=config.tool_name)
    
    def run_complete_pipeline(self) -> TestingPipelineState:
        """Run the complete testing pipeline from start to finish"""
        
        print(f"🚀 STARTING COMPLETE TESTING PIPELINE: {self.config.tool_name}")
        print("=" * 70)
        
        self.state.start_time = datetime.now()
        
        try:
            # Stage 1: Fetch pages
            if not self._fetch_pages():
                return self.state
            
            # Stage 2: Analyze content
            if not self.config.skip_analysis:
                if not self._analyze_content():
                    return self.state
            
            # Stage 3: Generate test plans
            if not self.config.skip_test_plans:
                if not self._generate_test_plans():
                    return self.state
            
            # Stage 4: Execute tests
            if not self.config.skip_execution:
                if not self._execute_tests():
                    return self.state
            
            # Stage 5: Generate reports
            if not self.config.skip_reports:
                if not self._generate_reports():
                    return self.state
            
            self.state.current_stage = "completed"
            print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self.state.pipeline_errors.append(error_msg)
            self.state.current_stage = "failed"
            print(f"\n❌ PIPELINE FAILED: {error_msg}")
            traceback.print_exc()
        
        finally:
            self.state.end_time = datetime.now()
            # Calculate final page statistics
            self._update_page_statistics()
            self._save_pipeline_state()
        
        return self.state
    
    def _fetch_pages(self) -> bool:
        """Stage 1: Fetch documentation pages"""
        
        self.state.current_stage = "fetching_pages"
        logger.info(f"\n📥 STAGE 1: FETCHING PAGES")
        logger.info("-" * 40)
        
        try:
            # Use crawl4ai fetcher (following fetcher.ipynb pattern)
            from .doc_ingestion.crawl4ai import create_crawl4ai_fetcher
            
            logger.info(f"🕷️  Crawling: {self.config.base_url}")
            logger.info(f"📊 Max pages: {self.config.max_pages}, Max depth: {self.config.max_depth}")
            
            # Create fetcher and fetch content
            start_time = time.time()
            fetcher = create_crawl4ai_fetcher(
                crawl_type="deep",
                max_pages=self.config.max_pages,
                max_depth=self.config.max_depth,
                keywords=self.config.keywords,
                verbose=True
            )
            
            raw_content = fetcher.fetch(self.config.base_url)
            fetch_duration = time.time() - start_time
            
            logger.info(f"✅ Fetching completed in {fetch_duration:.2f}s")
            logger.info(f"📄 Found {len(raw_content.content)} pages")
            
            # POST-PROCESSING: Deduplicate URLs by normalizing trailing slashes
            deduplicated_content = self._deduplicate_urls(raw_content.content)
            logger.info(f"🔄 After deduplication: {len(deduplicated_content)} pages")
            
            # Process pages directly (content already has URL->content mapping)
            processed_count = 0
            for url, content in deduplicated_content.items():
                if self._should_process_url(url):
                    page_progress = PageProgress(page_url=url)
                    page_progress.fetch_status = StageStatus.IN_PROGRESS
                    page_progress.raw_content = content
                    page_progress.fetch_time = fetch_duration
                    page_progress.fetch_status = StageStatus.COMPLETED
                    
                    self.state.pages[url] = page_progress
                    self.state.total_pages += 1
                    processed_count += 1
                    
                    logger.info(f"✅ Processed: {url}")
                else:
                    logger.info(f"⏭️  Skipped: {url}")
            
            logger.info(f"\n📊 Processed {processed_count} of {len(raw_content.content)} fetched pages")
            return True
            
        except Exception as e:
            error_msg = f"Failed to fetch pages: {str(e)}"
            self.state.pipeline_errors.append(error_msg)
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return False
    

    
    def _deduplicate_urls(self, content: Dict[str, str]) -> Dict[str, str]:
        """Deduplicate URLs by removing trailing slash variants"""
        from .doc_ingestion.crawl4ai import normalize_url
        
        # Track normalized URL -> original URL mapping
        normalized_to_original = {}
        deduplicated_content = {}
        
        for url, page_content in content.items():
            normalized_url = normalize_url(url)
            
            if normalized_url in normalized_to_original:
                original_url = normalized_to_original[normalized_url]
                logger.info(f"🔄 Deduplicating: {url} -> {normalized_url} (keeping {original_url})")
                
                # If the new content is longer, prefer it
                if len(page_content) > len(deduplicated_content.get(original_url, "")):
                    logger.info(f"   📝 Replacing with longer content ({len(page_content)} > {len(deduplicated_content.get(original_url, ''))} chars)")
                    deduplicated_content[original_url] = page_content
            else:
                # First time seeing this normalized URL
                normalized_to_original[normalized_url] = url
                deduplicated_content[url] = page_content
        
        return deduplicated_content
    
    def _update_page_statistics(self):
        """Update completed_pages and failed_pages based on actual page states"""
        completed = 0
        failed = 0
        
        for page in self.state.pages.values():
            # A page is completed if all stages that should run are completed
            stages_to_check = [page.fetch_status]
            
            if not self.config.skip_analysis:
                stages_to_check.append(page.analysis_status)
            if not self.config.skip_test_plans:
                stages_to_check.append(page.test_plan_status)
            if not self.config.skip_execution:
                stages_to_check.append(page.execution_status)
            if not self.config.skip_reports:
                stages_to_check.append(page.report_status)
            
            # Check if any stage failed
            if StageStatus.FAILED in stages_to_check:
                failed += 1
            # Check if all stages that should run are completed
            elif all(status == StageStatus.COMPLETED for status in stages_to_check):
                completed += 1
            # Otherwise, it's still in progress or failed
            else:
                failed += 1
        
        self.state.completed_pages = completed
        self.state.failed_pages = failed
        
        logger.info(f"📊 Final statistics: {completed} completed, {failed} failed out of {self.state.total_pages} total pages")
    
    def _should_process_url(self, url: str) -> bool:
        """Check if URL should be processed based on include/exclude rules"""
        
        # Check exclude list
        if url in self.config.urls_to_exclude:
            return False
        
        # Check exclude patterns
        import re
        for pattern in self.config.url_patterns_to_exclude:
            if re.search(pattern, url):
                return False
        
        # Check include list (if specified)
        if self.config.urls_to_include:
            return url in self.config.urls_to_include
        
        # Check max pages limit
        if self.config.max_pages and len(self.state.pages) >= self.config.max_pages:
            return False
        
        return True
    
    def _analyze_content(self) -> bool:
        """Stage 2: Analyze documentation content"""
        
        self.state.current_stage = "analyzing_content"
        logger.info(f"\n🔍 STAGE 2: ANALYZING CONTENT")
        logger.info("-" * 40)
        
        try:
            # Prepare raw content for analysis
            from .doc_ingestion.fetcher import RawContent
            content_dict = {}
            for url, page in self.state.pages.items():
                if page.raw_content:
                    content_dict[url] = page.raw_content
            
            if not content_dict:
                logger.info("⚠️  No content available for analysis")
                return True
                        
            raw_content = RawContent(
                source_type="website",
                source_url=self.config.base_url,
                content=content_dict,
                metadata={"base_url": self.config.base_url},
                fetch_timestamp=datetime.now().isoformat()
            )
            
            # Run analysis
            logger.info(f"🧠 Analyzing {len(content_dict)} pages with DSPy...")
            start_time = time.time()
            
            analysis_result = self.document_processor.process(raw_content)
            
            # Store results per page
            for url, page_analysis in analysis_result.get('pages', {}).items():
                if url in self.state.pages:
                    self.state.pages[url].analysis_status = StageStatus.IN_PROGRESS
                    self.state.pages[url].analysis_result = page_analysis
                    self.state.pages[url].analysis_time = time.time() - start_time
                    self.state.pages[url].analysis_status = StageStatus.COMPLETED
                    logger.info(f"✅ Analyzed: {url}")
            
            logger.info(f"\n📊 Analyzed {len(analysis_result.get('pages', {}))} pages")
            return True
            
        except Exception as e:
            error_msg = f"Failed to analyze content: {str(e)}"
            self.state.pipeline_errors.append(error_msg)
            logger.info(f"❌ {error_msg}")
            traceback.print_exc()
            return False
    
    def _generate_test_plans(self) -> bool:
        """Stage 3: Generate test plans"""
        
        self.state.current_stage = "generating_test_plans"
        logger.info(f"\n📋 STAGE 3: GENERATING TEST PLANS")
        logger.info("-" * 40)
        
        try:
            # Create tool documentation from analysis results
            tool_docs = ToolDocumentation()
            
            for url, page in self.state.pages.items():
                if page.analysis_result:
                    # Convert analysis result to PageAnalysis
                    from .doc_ingestion.analyzer import PageAnalysis
                    page_analysis = PageAnalysis(**page.analysis_result)
                    tool_docs.pages[url] = page_analysis
            
            # Generate test plans
            logger.info(f"📝 Generating test plans for {len(tool_docs.pages)} pages...")
            start_time = time.time()
            
            page_test_plans = self.test_plan_generator.generate_test_plan(tool_docs)
            
            # Store results per page
            for test_plan in page_test_plans:
                url = test_plan.page_url
                if url in self.state.pages:
                    self.state.pages[url].test_plan_status = StageStatus.IN_PROGRESS
                    self.state.pages[url].test_plan = test_plan
                    self.state.pages[url].test_plan_time = time.time() - start_time
                    self.state.pages[url].test_plan_status = StageStatus.COMPLETED
                    logger.info(f"✅ Test plan: {url} ({len(test_plan.scenarios)} scenarios)")
            
            logger.info(f"\n📊 Generated test plans for {len(page_test_plans)} pages")
            return True
            
        except Exception as e:
            error_msg = f"Failed to generate test plans: {str(e)}"
            self.state.pipeline_errors.append(error_msg)
            logger.info(f"❌ {error_msg}")
            traceback.print_exc()
            return False
    
    def _execute_tests(self) -> bool:
        """Stage 4: Execute tests in parallel"""
        
        self.state.current_stage = "executing_tests"
        logger.info(f"\n⚡ STAGE 4: EXECUTING TESTS")
        logger.info("-" * 40)
        
        try:
            # Collect all test scenarios
            all_scenarios = []
            scenario_to_page = {}
            
            for url, page in self.state.pages.items():
                if page.test_plan and page.test_plan.scenarios:
                    for scenario in page.test_plan.scenarios:
                        all_scenarios.append(scenario)
                        scenario_to_page[scenario.name] = url
            
            if not all_scenarios:
                print("⚠️ No test scenarios to execute")
                return True
            
            # Prepare context
            context = {**self.config.api_keys, **self.config.context}
            logger.info(f"🔍 Context: {context}")
            
            # Execute tests in parallel
            logger.info(f"🚀 Executing {len(all_scenarios)} test scenarios...")
            start_time = time.time()
            
            page_content = "Combined documentation content for testing"
            test_results = self.test_executor.run_tests_parallel(all_scenarios, page_content, context)
            
            # Distribute results back to pages
            for result in test_results:
                page_url = scenario_to_page.get(result.scenario_name)
                if page_url and page_url in self.state.pages:
                    self.state.pages[page_url].execution_status = StageStatus.IN_PROGRESS
                    self.state.pages[page_url].test_results.append(result)
                    self.state.pages[page_url].execution_time = time.time() - start_time
                    self.state.pages[page_url].execution_status = StageStatus.COMPLETED
            
            # Update execution status for all pages
            for url, page in self.state.pages.items():
                if page.test_plan and not page.test_results:
                    page.execution_status = StageStatus.FAILED
                    page.errors.append("No test results received")
            
            success_count = sum(1 for r in test_results if r.passed.value in ["PASSED", "MINOR_FAILURE"])
            logger.info(f"\n📊 Executed {len(test_results)} tests, {success_count} successful")
            return True
            
        except Exception as e:
            error_msg = f"Failed to execute tests: {str(e)}"
            self.state.pipeline_errors.append(error_msg)
            logger.info(f"❌ {error_msg}")
            traceback.print_exc()
            return False
    
    def _generate_reports(self) -> bool:
        """Stage 5: Generate intelligent reports"""
        
        self.state.current_stage = "generating_reports"
        logger.info(f"\n📊 STAGE 5: GENERATING REPORTS")
        logger.info("-" * 40)
        
        try:
            # Collect test plans and results
            page_plans = []
            all_test_results = []
            page_contents = {}
            
            for url, page in self.state.pages.items():
                if page.test_plan:
                    page_plans.append(page.test_plan)
                    all_test_results.extend(page.test_results)
                    if page.raw_content:
                        page_contents[url] = page.raw_content
            
            if not page_plans:
                print("⚠️ No test plans available for reporting")
                return True
            
            # Generate reports
            logger.info(f"🤖 Generating intelligent reports with DSPy...")
            start_time = time.time()
            
            page_reports, overall_report = self.reporter.generate_full_report(
                page_plans, all_test_results, page_contents
            )
            
            # Store page reports
            for page_report in page_reports:
                url = page_report.page_url
                if url in self.state.pages:
                    self.state.pages[url].report_status = StageStatus.IN_PROGRESS
                    self.state.pages[url].page_report = page_report
                    self.state.pages[url].report_time = time.time() - start_time
                    self.state.pages[url].report_status = StageStatus.COMPLETED
            
            # Store overall report
            self.state.overall_report = overall_report
            
            # Print summary
            self.reporter.print_summary_report(page_reports, overall_report)
            
            # Save detailed report
            report_filename = self.reporter.save_json_report(page_reports, overall_report)
            self.state.output_files.append(report_filename)
            
            logger.info(f"\n📊 Generated reports for {len(page_reports)} pages")
            return True
            
        except Exception as e:
            error_msg = f"Failed to generate reports: {str(e)}"
            self.state.pipeline_errors.append(error_msg)
            logger.info(f"❌ {error_msg}")
            traceback.print_exc()
            return False
    
    def _save_pipeline_state(self):
        """Save complete pipeline state to file"""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_state_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(self.state.model_dump(), f, indent=2, default=str)
            
            self.state.output_files.append(filename)
            logger.info(f"\n💾 Pipeline state saved to: {filename}")
            
        except Exception as e:
            logger.info(f"⚠️ Failed to save pipeline state: {e}")
    
    def print_pipeline_summary(self):
        """Print a summary of the pipeline execution"""
        
        logger.info(f"\n📈 PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Tool: {self.config.tool_name}")
        logger.info(f"Total Pages: {self.state.total_pages}")
        logger.info(f"Completed: {sum(1 for p in self.state.pages.values() if p.report_status == StageStatus.COMPLETED)}")
        logger.info(f"Failed: {sum(1 for p in self.state.pages.values() if StageStatus.FAILED in [p.fetch_status, p.analysis_status, p.test_plan_status, p.execution_status, p.report_status])}")
        
        if self.state.start_time and self.state.end_time:
            duration = self.state.end_time - self.state.start_time
            logger.info(f"Duration: {duration.total_seconds():.1f} seconds")
        
        if self.state.output_files:
            logger.info(f"\nOutput Files:")
            for file in self.state.output_files:
                logger.info(f"  📄 {file}")

# Convenience function for quick pipeline execution
def run_complete_testing_pipeline(
    base_url: str,
    tool_name: str,
    api_keys: Dict[str, str] = None,
    urls_to_exclude: List[str] = None,
    max_pages: int = None,
    max_depth: int = None,
    keywords: List[str] = None,
    **kwargs
) -> TestingPipelineState:
    """Quick way to run the complete testing pipeline
    
    Args:
        base_url: Base URL to start crawling from
        tool_name: Name of the tool being tested
        api_keys: Dictionary of API keys for testing
        urls_to_exclude: List of URLs to exclude from processing
        max_pages: Maximum number of pages to process
        max_depth: Maximum crawl depth
        keywords: Keywords for crawling
        **kwargs: Additional configuration options
    """
    
    config = PipelineConfig(
        base_url=base_url,
        tool_name=tool_name,
        api_keys=api_keys or {},
        urls_to_exclude=urls_to_exclude or [],
        max_pages=max_pages or 20,
        max_depth=max_depth or 3,
        keywords=keywords or ["api", "documentation", "guide"],
        **kwargs
    )
    
    pipeline = DeveloperToolTestingPipeline(config)
    state = pipeline.run_complete_pipeline()
    pipeline.print_pipeline_summary()
    
    return state

def main():
    """Main entry point for CLI testing"""
    run_complete_testing_pipeline(
        base_url="https://openweathermap.org/api",
        tool_name="OpenWeatherMap",
        api_keys={"OPEN_WEATHER_API_KEY": "7a4834a9d4b666e30261978ec5950ab6"},
        urls_to_exclude=["https://openweathermap.org/api"],
        max_pages=20,
        max_depth=3,
        keywords=["api", "documentation", "guide"],
    )

if __name__ == "__main__":
    main()
