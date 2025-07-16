import dspy
import subprocess
import sys
import time
import traceback
import json
import os
import concurrent.futures
import threading
from enum import Enum
from typing import Dict, Any, List
from pydantic import BaseModel, Field

class TestStatus(Enum):
    PASSED = "PASSED"
    MINOR_FAILURE = "MINOR_FAILURE"
    MAJOR_FAILURE = "MAJOR_FAILURE"

class TestResult(BaseModel):
    """Enhanced test execution result with generic tracking"""
    scenario_name: str
    passed: TestStatus = Field(description="Minor failure if the test passed but there were some minor issues, major failure if the test failed")
    execution_time: float

    # Enhanced tracking - GENERIC for all tool types
    trajectory: Any
    final_reasoning: str = ""  # Agent's final reasoning

# Thread-local storage for context isolation
_thread_local = threading.local()

def get_thread_context():
    """Get thread-local context"""
    if not hasattr(_thread_local, 'context'):
        _thread_local.context = {}
    return _thread_local.context

def set_thread_context(context: dict):
    """Set thread-local context"""
    if not hasattr(_thread_local, 'context'):
        _thread_local.context = {}
    _thread_local.context.update(context.copy() if context else {})

def extract_code(code_string: str) -> str:
    """Extract clean Python code from markdown-wrapped code string"""
    if code_string.startswith("'") and code_string.endswith("'"):
        code_string = code_string[1:-1]
    
    if code_string.startswith('```python'):
        code_string = code_string[9:]
    elif code_string.startswith('```'):
        code_string = code_string[3:]
    
    if code_string.endswith('```'):
        code_string = code_string[:-3]
    
    code_string = code_string.replace('\\n', '\n')
    return code_string.strip()

# Global context storage for backward compatibility
_global_context = {}

def set_global_context(context: dict):
    """Set global context for tools to access"""
    global _global_context
    _global_context = context.copy() if context else {}

def set_environment_variables_direct(context: dict):
    """Set environment variables directly from context keys"""
    for env_key, value in context.items():
        if value:
            os.environ[env_key] = str(value)
            print(f"Set {env_key} = {str(value)[:8]}***")

def set_environment_variables_thread_safe(context: dict):
    """Set environment variables from thread-local context"""
    thread_context = get_thread_context()
    combined_context = {**_global_context, **thread_context, **context}
    
    for env_key, value in combined_context.items():
        if value:
            os.environ[env_key] = str(value)

# DSPy ReAct Tools - updated to use thread-safe context
def check_available_api_keys():
    """Check what API keys are available in the context"""
    # Check both global and thread-local context
    global_context = _global_context
    thread_context = get_thread_context()
    context = {**global_context, **thread_context}
    
    if not context:
        return "⚠️  NO CONTEXT PROVIDED - No API keys available"
    
    available_keys = []
    for key, value in context.items():
        if value:  # Only show keys that have values
            masked_value = str(value)[:8] + "***" if len(str(value)) > 8 else "***"
            available_keys.append(f"✅ {key}: {masked_value}")
    
    if available_keys:
        result = "🔑 AVAILABLE API KEYS:\n\n"
        result += "\n".join(available_keys) + "\n\n"
        result += "💡 Use os.getenv('KEY_NAME') to access these in your code\n"
        result += f"Available keys: {', '.join(context.keys())}"
    else:
        result = "⚠️  NO API KEYS IN CONTEXT!"
    
    return result

def execute_code(code: str):
    """Execute Python code with environment variables from context"""
    # Use both global and thread-local context
    global_context = _global_context
    thread_context = get_thread_context()
    context = {**global_context, **thread_context}
    
    start_time = time.time()
    
    try:
        # Clean the code first
        clean_code = extract_code(code)
        
        # Set environment variables from combined context
        if context:
            set_environment_variables_thread_safe(context)
        
        # Create a safe exit function that doesn't terminate the application
        def safe_exit(code=0):
            raise SystemExit(code)
        
        # Basic execution environment
        exec_globals = {
            '__builtins__': __builtins__,
            'time': time,
            'json': json,
            'subprocess': subprocess,
            'sys': sys,
            'exit': safe_exit,  # Override exit to prevent app termination
        }
        
        # Capture output
        captured_output = []
        
        def capture_print(*args, **kwargs):
            line = ' '.join(str(arg) for arg in args)
            captured_output.append(line)
            print(line)  # Still print to console
        
        exec_globals['print'] = capture_print
        
        # Execute the code
        exec(clean_code, exec_globals)
        
        execution_time = time.time() - start_time
        output = '\n'.join(captured_output) if captured_output else "Code executed successfully (no output)"
        
        return f"SUCCESS: Code executed in {execution_time:.2f}s\nOutput:\n{output}"
        
    except SystemExit as e:
        execution_time = time.time() - start_time
        output = '\n'.join(captured_output) if captured_output else "No output before exit"
        return f"EXIT: Code called exit({e.code})\nTime: {execution_time:.2f}s\nOutput:\n{output}"
        
    except ImportError as e:
        execution_time = time.time() - start_time
        missing_module = str(e).split("'")[1] if "'" in str(e) else str(e)
        return f"MISSING_DEPENDENCY: {missing_module}\nError: {str(e)}\nTime: {execution_time:.2f}s"
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_trace = traceback.format_exc()
        return f"ERROR: Code execution failed in {execution_time:.2f}s\nError: {str(e)}\nTrace:\n{error_trace}"

def install_package(package_name: str):
    """Try to install a Python package with tracking"""
    
    try:
        print(f"Attempting to install {package_name}...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', package_name],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        success = result.returncode == 0
        
        if success:
            return f"Successfully installed {package_name}"
        else:
            return f"Failed to install {package_name}: {result.stderr}"
            
    except Exception as e:
        return f"Installation error: {str(e)}"

def check_environment():
    """Check what packages are available with tracking"""
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        return f"Environment check completed. Installed packages:\n{result.stdout[:500]}..."
        
    except Exception as e:
        return f"Environment check failed: {str(e)}"

class TestExecutorSignature(dspy.Signature):
    """You are a test execution agent that runs test scenarios for developer tools.
    
    You can execute code, install dependencies, check the environment, and check available API keys.
    Your goal is to successfully execute the test scenario and determine if it PASSED, MINOR_FAILURE, or MAJOR_FAILURE.

    CLASSIFICATION CRITERIA:
    
    PASSED: 
    - Test scenario executed successfully without any issues
    - All assertions passed as expected
    - No code modifications or workarounds were needed
    - API responses matched expected format exactly
    
    MINOR_FAILURE: 
    - Test scenario succeeded but required workarounds or adjustments
    - API responses were in different format than expected (but still valid)
    - Had to modify test logic due to documentation inconsistencies  
    - Missing dependencies that were easily installable
    - Minor authentication or configuration issues that were resolved
    - Expected format didn't match actual API behavior
    
    MAJOR_FAILURE:
    - Test scenario completely failed to execute
    - Authentication failed and couldn't be resolved
    - Critical dependencies missing and couldn't be installed
    - API endpoints don't exist or are completely non-functional
    - Code errors that couldn't be resolved
    
    Available tools:
    - check_available_api_keys(): See what API keys are available
    - execute_code(code): Run Python code 
    - install_package(name): Install missing packages
    - check_environment(): List installed packages
    
    IMPORTANT: If you need to modify test expectations or make workarounds due to 
    documentation inconsistencies, classify as MINOR_FAILURE, not PASSED.
    """
    
    scenario: str = dspy.InputField(desc="JSON string of the test scenario to execute")
    context: str = dspy.InputField(desc="JSON string of context with API keys and configurations")
    
    execution_result: str = dspy.OutputField(desc="Final result: PASSED, MINOR_FAILURE, or MAJOR_FAILURE with explanation")

# Create the ReAct agent
test_executor_agent = dspy.ReAct(
    TestExecutorSignature,
    tools=[
        check_available_api_keys,
        execute_code,
        install_package, 
        check_environment
    ],
    max_iters=10
)

class ParallelTestExecutor:
    """Parallel test execution manager"""
    
    def __init__(self, max_workers: int = None, use_parallel: bool = True):
        self.max_workers = max_workers or 4
        self.use_parallel = use_parallel
    
    def run_tests_parallel(self, scenarios: List[Any], page_content: str, context: Dict[str, str] = None) -> List[TestResult]:
        """Run multiple test scenarios in parallel"""
        
        if not scenarios:
            return []
        
        context = context or {}
        
        # Set global context once
        set_global_context(context)
        
        if self.use_parallel and len(scenarios) > 1:
            print(f"Running {len(scenarios)} tests in parallel with {self.max_workers} workers")
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all test scenarios
                    futures = {
                        executor.submit(self._run_single_test_thread_safe, scenario, page_content, context): scenario 
                        for scenario in scenarios
                    }
                    
                    # Collect results as they complete
                    results = []
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            if result:
                                results.append(result)
                        except Exception as e:
                            scenario = futures[future]
                            print(f"Test {scenario.name} failed with exception: {e}")
                            # Create failure result
                            results.append(TestResult(
                                scenario_name=scenario.name,
                                passed=TestStatus.MAJOR_FAILURE,
                                execution_time=0.0,
                                trajectory={},
                                final_reasoning=f"Parallel execution error: {str(e)}"
                            ))
                    
                    return results
                    
            except Exception as e:
                print(f"Parallel execution failed: {e}, falling back to sequential")
                return self._run_tests_sequential(scenarios, page_content, context)
        else:
            print(f"Running {len(scenarios)} tests sequentially")
            return self._run_tests_sequential(scenarios, page_content, context)
    
    def _run_tests_sequential(self, scenarios: List[Any], page_content: str, context: Dict[str, str]) -> List[TestResult]:
        """Run tests sequentially (fallback)"""
        results = []
        for scenario in scenarios:
            try:
                result = run_test_with_react(scenario, page_content, context)
                results.append(result)
            except Exception as e:
                print(f"Sequential test {scenario.name} failed: {e}")
                results.append(TestResult(
                    scenario_name=scenario.name,
                    passed=TestStatus.MAJOR_FAILURE,
                    execution_time=0.0,
                    trajectory={},
                    final_reasoning=f"Sequential execution error: {str(e)}"
                ))
        return results
    
    def _run_single_test_thread_safe(self, scenario: Any, page_content: str, context: Dict[str, str]) -> TestResult:
        """Run a single test in a thread-safe manner"""
        # Set thread-local context for this test
        set_thread_context(context)
        
        try:
            return run_test_with_react(scenario, page_content, context)
        except Exception as e:
            print(f"Thread-safe test {scenario.name} failed: {e}")
            return TestResult(
                scenario_name=scenario.name,
                passed=TestStatus.MAJOR_FAILURE,
                execution_time=0.0,
                trajectory={},
                final_reasoning=f"Thread execution error: {str(e)}"
            )

def run_test_with_react(scenario, page_content: str, context: Dict[str, str] = None) -> TestResult:
    """Run a test scenario using the ReAct agent - preserved for backward compatibility"""
    context = context or {}
    start_time = time.time()
    
    try:
        # Set both global and thread-local context
        set_global_context(context)
        set_thread_context(context)
        
        # Run the ReAct agent
        result = test_executor_agent(
            scenario=scenario.model_dump_json(),
            context=json.dumps(page_content)
        )
        
        print("=== REACT AGENT RESULT ===")
        print(f"Execution Result: {result.execution_result}")
        if hasattr(result, 'reasoning'):
            print(f"Reasoning: {result.reasoning}")
        
        # Parse the result
        if "PASSED" in result.execution_result.upper():
            passed = TestStatus.PASSED
        elif "MINOR_FAILURE" in result.execution_result.upper():
            passed = TestStatus.MINOR_FAILURE
        elif "MAJOR_FAILURE" in result.execution_result.upper():
            passed = TestStatus.MAJOR_FAILURE
        else:
            passed = TestStatus.MAJOR_FAILURE  # Default to MAJOR_FAILURE if unknown
        

        reasoning = ""
        if hasattr(result, 'reasoning'):
            reasoning = result.reasoning
            print(f"Reasoning: {reasoning}")
        
        # Extract steps from trajectory
        steps = []
        if hasattr(result, 'trajectory'):
            for i in range(20):  # Check up to 20 steps
                thought_key = f'thought_{i}'
                tool_key = f'tool_name_{i}'
                if thought_key in result.trajectory and tool_key in result.trajectory:
                    thought = result.trajectory[thought_key]
                    tool = result.trajectory[tool_key]
                    steps.append(f"Step {i+1}: {thought} -> Used {tool}")
                else:
                    break
        
        return TestResult(
            scenario_name=scenario.name,
            passed=passed,
            execution_time=time.time() - start_time,
            trajectory=result.trajectory,
            final_reasoning=reasoning
        )
        
    except Exception as e:
        print(f"ERROR in run_test_with_react: {str(e)}")
        traceback.print_exc()
        return TestResult(
            scenario_name=scenario.name,
            passed=TestStatus.MAJOR_FAILURE,
            execution_time=time.time() - start_time,
            trajectory={},
            final_reasoning=f"ReAct agent error: {str(e)}"
        )