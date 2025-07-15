#!/usr/bin/env python3
"""
DSPy-Powered Reporting System for Developer Tool Testing

Uses LLM analysis to generate intelligent page-level reports and overall summaries
that provide actionable insights based on test failures and documentation content.
"""

import json
import dspy
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field

from code_execution import TestResult, TestStatus
from test_plan import PageTestPlan, TestScenario, TestCategory, TestPriority

class PageReport(BaseModel):
    """AI-generated report for a single documentation page"""
    page_url: str = Field(..., description="URL of the page")
    
    # Calculated statistics
    total_tests: int = Field(default=0, description="Total tests run for this page")
    passed_tests: int = Field(default=0, description="Tests that passed") 
    minor_failure_tests: int = Field(default=0, description="Tests with minor failures")
    major_failure_tests: int = Field(default=0, description="Tests with major failures")
    success_rate: float = Field(default=0.0, description="Success rate percentage")
    total_execution_time: float = Field(default=0.0, description="Total execution time")
    
    # AI-generated insights
    page_summary: str = Field(default="", description="AI summary of page content and testing")
    documentation_quality: str = Field(default="", description="Assessment of documentation quality")
    main_issues: List[str] = Field(default_factory=list, description="Key issues identified")
    success_factors: List[str] = Field(default_factory=list, description="What worked well")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Specific improvements for this page")
    missing_examples: List[str] = Field(default_factory=list, description="Examples that should be added")
    
    # Category insights
    category_analysis: Dict[str, str] = Field(default_factory=dict, description="Analysis by test category")
    priority_issues: List[str] = Field(default_factory=list, description="Critical and high priority issues")

class OverallReport(BaseModel):
    """AI-generated overall summary report"""
    tool_name: str = Field(default="", description="Name of the tool being tested")
    generation_time: str = Field(default="", description="When this report was generated")
    
    # Calculated statistics
    total_pages: int = Field(default=0, description="Total pages analyzed")
    total_tests: int = Field(default=0, description="Total tests executed")
    overall_success_rate: float = Field(default=0.0, description="Overall success rate")
    total_execution_time: float = Field(default=0.0, description="Total execution time")
    
    # AI-generated insights
    executive_summary: str = Field(default="", description="High-level summary of testing results")
    overall_documentation_assessment: str = Field(default="", description="Assessment of overall documentation quality")
    systemic_issues: List[str] = Field(default_factory=list, description="Issues affecting multiple pages")
    strength_areas: List[str] = Field(default_factory=list, description="Areas where the tool excels")
    strategic_recommendations: List[str] = Field(default_factory=list, description="High-level strategic recommendations")
    
    # Comparative analysis
    best_performing_pages: List[str] = Field(default_factory=list, description="Pages with best results")
    most_problematic_pages: List[str] = Field(default_factory=list, description="Pages needing attention")
    
    # Priority insights
    immediate_actions: List[str] = Field(default_factory=list, description="Actions to take immediately")
    medium_term_improvements: List[str] = Field(default_factory=list, description="Improvements for next iteration")

# DSPy Signatures for intelligent report generation
class PageReportGenerator(dspy.Signature):
    """Generate intelligent page-level report by analyzing test results against documentation content"""
    
    page_url: str = dspy.InputField(desc="URL of the documentation page")
    page_content: str = dspy.InputField(desc="Raw content of the documentation page")
    test_scenarios: str = dspy.InputField(desc="JSON string of test scenarios for this page")
    test_results: str = dspy.InputField(desc="JSON string of test execution results")
    calculated_stats: str = dspy.InputField(desc="JSON string of calculated statistics")
    
    page_summary: str = dspy.OutputField(desc="Concise summary of page content and testing approach")
    documentation_quality: str = dspy.OutputField(desc="Assessment of documentation quality and completeness")
    main_issues: List[str] = dspy.OutputField(desc="Key issues identified from test failures")
    success_factors: List[str] = dspy.OutputField(desc="What worked well and contributed to successful tests")
    improvement_suggestions: List[str] = dspy.OutputField(desc="Specific actionable improvements for this page")
    missing_examples: List[str] = dspy.OutputField(desc="Code examples or use cases that should be added")
    category_analysis: Dict[str, str] = dspy.OutputField(desc="Analysis of performance by test category")
    priority_issues: List[str] = dspy.OutputField(desc="Critical and high priority issues requiring immediate attention")

class OverallReportGenerator(dspy.Signature):
    """Generate overall summary report from all page-level reports"""
    
    tool_name: str = dspy.InputField(desc="Name of the tool being tested")
    page_reports: str = dspy.InputField(desc="JSON string of all page-level reports")
    overall_stats: str = dspy.InputField(desc="JSON string of aggregated statistics")
    
    executive_summary: str = dspy.OutputField(desc="High-level executive summary of testing results and tool readiness")
    overall_documentation_assessment: str = dspy.OutputField(desc="Assessment of documentation quality across all pages")
    systemic_issues: List[str] = dspy.OutputField(desc="Issues that affect multiple pages or represent systemic problems")
    strength_areas: List[str] = dspy.OutputField(desc="Areas where the tool documentation and functionality excel")
    strategic_recommendations: List[str] = dspy.OutputField(desc="High-level strategic recommendations for improvement")
    immediate_actions: List[str] = dspy.OutputField(desc="Actions that should be taken immediately")
    medium_term_improvements: List[str] = dspy.OutputField(desc="Improvements to plan for future iterations")

class IntelligentReporter:
    """DSPy-powered intelligent test reporter"""
    
    def __init__(self, tool_name: str = "Unknown Tool"):
        self.tool_name = tool_name
        self.generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Initialize DSPy modules
        self.page_report_generator = dspy.ChainOfThought(PageReportGenerator)
        self.overall_report_generator = dspy.ChainOfThought(OverallReportGenerator)
    
    def _calculate_page_stats(self, page_plan: PageTestPlan, test_results: List[TestResult]) -> Dict:
        """Calculate basic statistics for a page"""
        
        # Filter results for this page's scenarios
        page_scenario_names = {scenario.name for scenario in page_plan.scenarios}
        page_results = [r for r in test_results if r.scenario_name in page_scenario_names]
        
        if not page_results:
            return {
                "total_tests": 0,
                "passed_tests": 0,
                "minor_failure_tests": 0,
                "major_failure_tests": 0,
                "success_rate": 0.0,
                "total_execution_time": 0.0,
                "failed_scenarios": [],
                "category_breakdown": {},
                "priority_breakdown": {}
            }
        
        # Basic counts
        total_tests = len(page_results)
        passed = sum(1 for r in page_results if r.passed == TestStatus.PASSED)
        minor_failures = sum(1 for r in page_results if r.passed == TestStatus.MINOR_FAILURE)
        major_failures = sum(1 for r in page_results if r.passed == TestStatus.MAJOR_FAILURE)
        
        # Success rate (PASSED + MINOR_FAILURE = success)
        success_rate = ((passed + minor_failures) / total_tests * 100) if total_tests > 0 else 0.0
        
        # Execution time
        total_time = sum(r.execution_time for r in page_results)
        
        # Failed scenarios
        failed_scenarios = [r.scenario_name for r in page_results if r.passed == TestStatus.MAJOR_FAILURE]
        
        # Category and priority breakdown
        scenario_map = {s.name: s for s in page_plan.scenarios}
        category_breakdown = {}
        priority_breakdown = {}
        
        for result in page_results:
            scenario = scenario_map.get(result.scenario_name)
            if scenario:
                cat = scenario.category.value
                pri = scenario.priority.value
                category_breakdown[cat] = category_breakdown.get(cat, 0) + 1
                priority_breakdown[pri] = priority_breakdown.get(pri, 0) + 1
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed,
            "minor_failure_tests": minor_failures,
            "major_failure_tests": major_failures,
            "success_rate": success_rate,
            "total_execution_time": total_time,
            "failed_scenarios": failed_scenarios,
            "category_breakdown": category_breakdown,
            "priority_breakdown": priority_breakdown
        }
    
    def generate_page_report(self, page_plan: PageTestPlan, test_results: List[TestResult], page_content: str = None) -> PageReport:
        """Generate AI-powered page-level report"""
        
        # Calculate statistics
        stats = self._calculate_page_stats(page_plan, test_results)
        
        # Use page content from plan if not provided
        if page_content is None:
            page_content = getattr(page_plan, 'page_content', '') or "No page content available"
        
        # Prepare data for DSPy
        page_scenario_names = {scenario.name for scenario in page_plan.scenarios}
        page_test_results = [r for r in test_results if r.scenario_name in page_scenario_names]
        
        scenarios_json = json.dumps([scenario.dict() for scenario in page_plan.scenarios])
        results_json = json.dumps([
            {
                "scenario_name": r.scenario_name,
                "passed": r.passed.value,
                "execution_time": r.execution_time,
                "final_reasoning": r.final_reasoning
            } for r in page_test_results
        ])
        stats_json = json.dumps(stats)
        
        try:
            # Generate AI report
            result = self.page_report_generator(
                page_url=page_plan.page_url,
                page_content=page_content,
                test_scenarios=scenarios_json,
                test_results=results_json,
                calculated_stats=stats_json
            )
            
            return PageReport(
                page_url=page_plan.page_url,
                total_tests=stats["total_tests"],
                passed_tests=stats["passed_tests"],
                minor_failure_tests=stats["minor_failure_tests"],
                major_failure_tests=stats["major_failure_tests"],
                success_rate=stats["success_rate"],
                total_execution_time=stats["total_execution_time"],
                page_summary=result.page_summary,
                documentation_quality=result.documentation_quality,
                main_issues=result.main_issues,
                success_factors=result.success_factors,
                improvement_suggestions=result.improvement_suggestions,
                missing_examples=result.missing_examples,
                category_analysis=result.category_analysis,
                priority_issues=result.priority_issues
            )
            
        except Exception as e:
            print(f"Error generating AI report for page {page_plan.page_url}: {e}")
            # Fallback to basic statistical report
            return PageReport(
                page_url=page_plan.page_url,
                total_tests=stats["total_tests"],
                passed_tests=stats["passed_tests"],
                minor_failure_tests=stats["minor_failure_tests"],
                major_failure_tests=stats["major_failure_tests"],
                success_rate=stats["success_rate"],
                total_execution_time=stats["total_execution_time"],
                page_summary=f"Page with {stats['total_tests']} tests, {stats['success_rate']:.1f}% success rate",
                documentation_quality="Analysis failed - check DSPy configuration",
                main_issues=[f"Failed to generate AI analysis: {str(e)}"]
            )
    
    def generate_overall_report(self, page_reports: List[PageReport]) -> OverallReport:
        """Generate AI-powered overall summary report"""
        
        if not page_reports:
            return OverallReport(
                tool_name=self.tool_name,
                generation_time=self.generation_time,
                executive_summary="No page reports available for analysis"
            )
        
        # Calculate overall statistics
        total_pages = len(page_reports)
        total_tests = sum(p.total_tests for p in page_reports)
        total_passed = sum(p.passed_tests for p in page_reports)
        total_minor_failures = sum(p.minor_failure_tests for p in page_reports)
        total_major_failures = sum(p.major_failure_tests for p in page_reports)
        overall_success_rate = ((total_passed + total_minor_failures) / total_tests * 100) if total_tests > 0 else 0.0
        total_time = sum(p.total_execution_time for p in page_reports)
        
        # Identify best and worst pages
        pages_with_tests = [p for p in page_reports if p.total_tests > 0]
        best_pages = sorted(pages_with_tests, key=lambda p: p.success_rate, reverse=True)[:3]
        worst_pages = sorted(pages_with_tests, key=lambda p: p.success_rate)[:3]
        
        overall_stats = {
            "total_pages": total_pages,
            "total_tests": total_tests,
            "overall_success_rate": overall_success_rate,
            "total_execution_time": total_time,
            "best_pages": [p.page_url for p in best_pages],
            "worst_pages": [p.page_url for p in worst_pages]
        }
        
        # Prepare data for DSPy
        page_reports_json = json.dumps([report.dict() for report in page_reports])
        stats_json = json.dumps(overall_stats)
        
        try:
            # Generate AI overall report
            result = self.overall_report_generator(
                tool_name=self.tool_name,
                page_reports=page_reports_json,
                overall_stats=stats_json
            )
            
            return OverallReport(
                tool_name=self.tool_name,
                generation_time=self.generation_time,
                total_pages=total_pages,
                total_tests=total_tests,
                overall_success_rate=overall_success_rate,
                total_execution_time=total_time,
                executive_summary=result.executive_summary,
                overall_documentation_assessment=result.overall_documentation_assessment,
                systemic_issues=result.systemic_issues,
                strength_areas=result.strength_areas,
                strategic_recommendations=result.strategic_recommendations,
                best_performing_pages=[p.page_url for p in best_pages if p.success_rate >= 80],
                most_problematic_pages=[p.page_url for p in worst_pages if p.success_rate < 50],
                immediate_actions=result.immediate_actions,
                medium_term_improvements=result.medium_term_improvements
            )
            
        except Exception as e:
            print(f"Error generating overall AI report: {e}")
            # Fallback to basic report
            return OverallReport(
                tool_name=self.tool_name,
                generation_time=self.generation_time,
                total_pages=total_pages,
                total_tests=total_tests,
                overall_success_rate=overall_success_rate,
                total_execution_time=total_time,
                executive_summary=f"Testing completed for {self.tool_name} with {overall_success_rate:.1f}% success rate",
                overall_documentation_assessment="AI analysis failed - check DSPy configuration",
                systemic_issues=[f"Failed to generate AI analysis: {str(e)}"]
            )
    
    def generate_full_report(self, page_plans: List[PageTestPlan], test_results: List[TestResult], page_contents: Dict[str, str] = None) -> Tuple[List[PageReport], OverallReport]:
        """Generate complete intelligent report with both page-level and overall summaries"""
        
        page_contents = page_contents or {}
        
        # Generate page reports
        page_reports = []
        for page_plan in page_plans:
            page_content = page_contents.get(page_plan.page_url, None)
            page_report = self.generate_page_report(page_plan, test_results, page_content)
            page_reports.append(page_report)
        
        # Generate overall report
        overall_report = self.generate_overall_report(page_reports)
        
        return page_reports, overall_report
    
    def print_summary_report(self, page_reports: List[PageReport], overall_report: OverallReport):
        """Print a formatted summary to console"""
        
        print("\n" + "=" * 70)
        print(f"🤖 AI-POWERED TEST REPORT: {overall_report.tool_name}")
        print("=" * 70)
        print(f"📅 Generated: {overall_report.generation_time}")
        print(f"📊 Pages: {overall_report.total_pages} | Tests: {overall_report.total_tests}")
        print(f"✅ Success Rate: {overall_report.overall_success_rate:.1f}%")
        print(f"⏱️  Total Time: {overall_report.total_execution_time:.2f}s")
        
        # Executive summary
        print(f"\n📋 EXECUTIVE SUMMARY:")
        print(f"   {overall_report.executive_summary}")
        
        # Documentation assessment
        if overall_report.overall_documentation_assessment:
            print(f"\n📚 DOCUMENTATION ASSESSMENT:")
            print(f"   {overall_report.overall_documentation_assessment}")
        
        # Systemic issues
        if overall_report.systemic_issues:
            print(f"\n🚨 SYSTEMIC ISSUES:")
            for issue in overall_report.systemic_issues:
                print(f"   • {issue}")
        
        # Strengths
        if overall_report.strength_areas:
            print(f"\n🏆 STRENGTH AREAS:")
            for strength in overall_report.strength_areas:
                print(f"   • {strength}")
        
        # Immediate actions
        if overall_report.immediate_actions:
            print(f"\n⚡ IMMEDIATE ACTIONS:")
            for action in overall_report.immediate_actions:
                print(f"   • {action}")
        
        # Strategic recommendations
        if overall_report.strategic_recommendations:
            print(f"\n💡 STRATEGIC RECOMMENDATIONS:")
            for rec in overall_report.strategic_recommendations:
                print(f"   • {rec}")
        
        print("\n" + "=" * 70)
    
    def save_json_report(self, page_reports: List[PageReport], overall_report: OverallReport, filename: str = None):
        """Save detailed reports to JSON file"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_test_report_{timestamp}.json"
        
        report_data = {
            "overall_report": overall_report.dict(),
            "page_reports": [report.dict() for report in page_reports],
            "metadata": {
                "generated_at": self.generation_time,
                "tool_name": self.tool_name,
                "report_type": "ai_powered",
                "total_pages": len(page_reports),
                "total_tests": overall_report.total_tests
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 AI-powered report saved to: {filename}")
        return filename
