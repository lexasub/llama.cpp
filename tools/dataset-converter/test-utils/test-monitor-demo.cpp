/**
 * @file test-monitor-demo.cpp
 * @brief Test Monitor Demonstration Tool and Usage Examples
 * 
 * This demonstration tool showcases the comprehensive test execution monitoring
 * capabilities provided by the llama.cpp dataset converter test suite. It serves
 * as both a practical utility for running monitored test executions and a
 * comprehensive example of how to use the TestExecutionMonitor API effectively.
 * 
 * ## Purpose and Functionality
 * 
 * ### Primary Objectives
 * - Demonstrate real-world usage of the TestExecutionMonitor class
 * - Provide a practical tool for executing and monitoring dataset converter tests
 * - Showcase advanced monitoring features including memory tracking and crash detection
 * - Serve as a reference implementation for test monitoring integration
 * 
 * ### Key Capabilities
 * - Automatic test discovery in common build directories
 * - Configurable monitoring parameters (timeouts, memory sampling, etc.)
 * - Comprehensive test execution with full performance analytics
 * - Detailed reporting and summary statistics
 * - Error handling and graceful failure recovery
 * 
 * ## Usage Scenarios
 * 
 * ### Basic Test Discovery and Execution
 * The tool can automatically discover and execute all test executables in a directory:
 * ```bash
 * ./test-monitor-demo
 * # Discovers tests in current directory and common build locations
 * ```
 * 
 * ### Custom Working Directory
 * Specify a custom directory for test discovery:
 * ```bash
 * ./test-monitor-demo /path/to/test/directory
 * # Discovers and runs tests in the specified directory
 * ```
 * 
 * ### Specific Test Execution
 * Run specific test executables with monitoring:
 * ```bash
 * ./test-monitor-demo . ./test_core ./test_formats ./test_streaming
 * # Runs the specified tests with full monitoring
 * ```
 * 
 * ### Build System Integration
 * The tool automatically checks common build directories:
 * - `bb/` (typical llama.cpp build directory)
 * - `build/` (standard CMake build directory)
 * - `cmake-build-debug/` (CLion debug builds)
 * - `cmake-build-release/` (CLion release builds)
 * 
 * ## Monitoring Features Demonstrated
 * 
 * ### Performance Monitoring
 * - High-resolution execution time measurement
 * - Real-time memory usage tracking with 50ms sampling intervals
 * - Peak memory usage detection and reporting
 * - Performance trend analysis across multiple test runs
 * 
 * ### Error Detection and Recovery
 * - Comprehensive crash detection with signal handling
 * - Timeout detection with configurable limits (default: 60 seconds)
 * - Stack trace capture for debugging crashed tests
 * - Graceful handling of missing or non-executable test files
 * 
 * ### Reporting and Analytics
 * - Individual test result reporting with detailed metrics
 * - Comprehensive execution reports with statistical analysis
 * - Human-readable formatting for memory sizes and durations
 * - Summary statistics including pass/fail rates and performance metrics
 * 
 * ## Configuration Options
 * 
 * The demonstration tool uses a comprehensive configuration that showcases
 * all major monitoring capabilities:
 * 
 * ### Execution Control
 * - **Timeout**: 60 seconds maximum execution time per test
 * - **Working Directory**: Configurable via command line or auto-detection
 * - **Verbose Output**: Enabled for detailed execution information
 * 
 * ### Monitoring Settings
 * - **Memory Monitoring**: Enabled with 50ms sampling intervals
 * - **Crash Detection**: Enabled with signal handler installation
 * - **Performance Tracking**: Full execution time and resource monitoring
 * 
 * ## Output and Reporting
 * 
 * ### Real-time Progress Information
 * The tool provides real-time feedback during test execution:
 * - Test discovery progress and results
 * - Individual test execution status and metrics
 * - Error messages and diagnostic information
 * - Performance metrics for each completed test
 * 
 * ### Comprehensive Final Report
 * After all tests complete, the tool generates:
 * - Detailed execution report with statistical analysis
 * - Summary statistics including pass/fail counts
 * - Performance analysis and memory usage patterns
 * - Success rate calculation and overall assessment
 * 
 * ## Error Handling and Edge Cases
 * 
 * ### Robust Error Recovery
 * - Graceful handling of missing test executables
 * - Proper cleanup after crashed or timed-out tests
 * - Informative error messages for troubleshooting
 * - Fallback test discovery in multiple build directories
 * 
 * ### Platform Compatibility
 * - Cross-platform file system operations using std::filesystem
 * - Platform-agnostic process monitoring and signal handling
 * - Consistent behavior across different operating systems
 * - Proper handling of platform-specific executable formats
 * 
 * ## Integration Examples
 * 
 * ### CI/CD Pipeline Integration
 * ```bash
 * # Run all tests with monitoring and capture results
 * ./test-monitor-demo /build/tests > test_results.log 2>&1
 * EXIT_CODE=$?
 * 
 * # Parse results for CI reporting
 * if [ $EXIT_CODE -eq 0 ]; then
 *     echo "All tests passed with monitoring"
 * else
 *     echo "Some tests failed - check test_results.log"
 * fi
 * ```
 * 
 * ### Development Workflow
 * ```bash
 * # Quick test run during development
 * ./test-monitor-demo . ./my_new_test
 * 
 * # Full regression testing
 * ./test-monitor-demo /build/tests
 * ```
 * 
 * ### Performance Analysis
 * ```bash
 * # Run tests multiple times for performance analysis
 * for i in {1..5}; do
 *     echo "Run $i:"
 *     ./test-monitor-demo . ./performance_test
 * done
 * ```
 * 
 * ## Implementation Architecture
 * 
 * ### Main Function Structure
 * The main function demonstrates a complete monitoring workflow:
 * 1. **Configuration Setup**: Initialize monitoring parameters
 * 2. **Test Discovery**: Find test executables using multiple strategies
 * 3. **Execution Loop**: Run each test with comprehensive monitoring
 * 4. **Result Collection**: Gather and analyze execution results
 * 5. **Report Generation**: Create detailed reports and summaries
 * 
 * ### Key Design Patterns
 * - **Configuration-driven behavior**: All monitoring aspects are configurable
 * - **Graceful degradation**: Continues execution even when individual tests fail
 * - **Comprehensive logging**: Detailed progress and diagnostic information
 * - **Resource cleanup**: Proper cleanup of monitoring resources
 * 
 * ## Educational Value
 * 
 * This demonstration tool serves as an excellent learning resource for:
 * - Understanding the TestExecutionMonitor API and capabilities
 * - Learning best practices for test monitoring integration
 * - Seeing real-world examples of performance monitoring
 * - Understanding error handling in test execution environments
 * 
 * ## Performance Considerations
 * 
 * ### Monitoring Overhead
 * - Memory monitoring uses separate threads to minimize impact
 * - 50ms sampling interval provides good accuracy with low overhead
 * - Efficient process management reduces system resource usage
 * - Optimized file system operations for test discovery
 * 
 * ### Scalability
 * - Sequential test execution prevents resource conflicts
 * - Configurable timeouts prevent runaway tests
 * - Memory monitoring adapts to different test characteristics
 * - Efficient cleanup prevents resource leaks
 * 
 * @author llama.cpp dataset-converter team
 * @version 1.0
 * @since 2024
 * 
 * @see TestExecutionMonitor
 * @see TestExecutionConfig
 * @see TestExecutionResult
 * @see MemoryMonitor
 * 
 * ## Example Output
 * 
 * ```
 * Dataset Test Execution Monitor Demo
 * ===================================
 * 
 * Working directory: ./build/tests
 * Timeout: 60 seconds
 * Memory monitoring: enabled
 * Crash detection: enabled
 * 
 * Found 3 test executable(s):
 *   - ./test_core
 *   - ./test_formats
 *   - ./test_streaming
 * 
 * Executing tests...
 * ==================
 * 
 * Running: ./test_core
 *   Result: PASSED
 *   Exit code: 0
 *   Execution time: 2.45s
 *   Peak memory: 15.2 MB
 * 
 * Running: ./test_formats
 *   Result: PASSED
 *   Exit code: 0
 *   Execution time: 1.83s
 *   Peak memory: 22.7 MB
 * 
 * Running: ./test_streaming
 *   Result: PASSED
 *   Exit code: 0
 *   Execution time: 3.12s
 *   Peak memory: 31.4 MB
 * 
 * [Detailed execution report follows...]
 * 
 * Final Summary:
 * =============
 * Total tests executed: 3
 * Passed: 3
 * Failed: 0
 * Success rate: 100%
 * ```
 */

#include <filesystem>
#include <iostream>
#include <vector>

#include "src/llama-impl.h"
#include "test-execution-monitor.h"

using namespace llama_dataset;

/**
 * @brief Main function demonstrating comprehensive test execution monitoring
 * 
 * This function serves as the primary entry point for the test monitor demonstration
 * tool. It showcases a complete workflow for discovering, configuring, executing,
 * and reporting on test executions using the TestExecutionMonitor framework.
 * 
 * ## Workflow Overview
 * 
 * The function implements a comprehensive test monitoring workflow:
 * 1. **Configuration Setup**: Initialize monitoring parameters based on command-line arguments
 * 2. **Test Discovery**: Automatically discover test executables using multiple strategies
 * 3. **Validation**: Verify that discovered executables are valid and executable
 * 4. **Execution**: Run each test with full monitoring and performance tracking
 * 5. **Reporting**: Generate detailed reports and summary statistics
 * 
 * ## Command-Line Interface
 * 
 * ### Usage Patterns
 * ```bash
 * # Auto-discovery in current directory
 * ./test-monitor-demo
 * 
 * # Custom working directory
 * ./test-monitor-demo /path/to/tests
 * 
 * # Specific test executables
 * ./test-monitor-demo . ./test1 ./test2 ./test3
 * ```
 * 
 * ### Argument Processing
 * - **argv[0]**: Program name (used for usage messages)
 * - **argv[1]**: Working directory (optional, defaults to current directory)
 * - **argv[2+]**: Specific test executables to run (optional, triggers auto-discovery if not provided)
 * 
 * ## Configuration Demonstration
 * 
 * The function demonstrates comprehensive configuration of the TestExecutionMonitor:
 * 
 * ### Monitoring Parameters
 * - **Timeout**: 60-second limit per test to prevent runaway executions
 * - **Memory Monitoring**: Enabled with 50ms sampling for accurate peak detection
 * - **Crash Detection**: Full signal handling and stack trace capture
 * - **Verbose Output**: Detailed logging for demonstration purposes
 * 
 * ### Directory Management
 * - Configurable working directory for test execution
 * - Automatic fallback to current directory if not specified
 * - Proper path resolution and validation
 * 
 * ## Test Discovery Strategy
 * 
 * ### Multi-tier Discovery Process
 * 1. **Direct Specification**: Use tests provided as command-line arguments
 * 2. **Working Directory Scan**: Search specified or current directory
 * 3. **Build Directory Fallback**: Check common build locations if no tests found
 * 4. **Subdirectory Search**: Look in dataset-converter specific subdirectories
 * 
 * ### Build Directory Detection
 * The function automatically checks these common build directories:
 * - `bb/`: Standard llama.cpp build directory
 * - `build/`: Common CMake build directory
 * - `cmake-build-debug/`: CLion debug build directory
 * - `cmake-build-release/`: CLion release build directory
 * 
 * Each build directory is also checked for dataset-converter specific subdirectories
 * to find tests in modular build layouts.
 * 
 * ## Execution and Monitoring
 * 
 * ### Individual Test Execution
 * For each discovered test executable, the function:
 * - Validates executable availability and permissions
 * - Executes with comprehensive monitoring enabled
 * - Collects detailed performance and diagnostic metrics
 * - Provides real-time progress feedback
 * 
 * ### Real-time Feedback
 * During execution, the function provides:
 * - Test name and execution status
 * - Pass/fail results with exit codes
 * - Performance metrics (execution time, memory usage)
 * - Error information for failed or crashed tests
 * - Timeout detection and reporting
 * 
 * ## Error Handling and Recovery
 * 
 * ### Graceful Error Management
 * - **Missing Executables**: Continues with remaining tests, reports errors
 * - **Execution Failures**: Captures error details, continues test suite
 * - **Crashes**: Collects crash information, performs cleanup
 * - **Timeouts**: Handles gracefully, terminates runaway processes
 * 
 * ### Diagnostic Information
 * - Detailed error messages for troubleshooting
 * - Stack traces for crashed tests
 * - Performance metrics even for failed tests
 * - Comprehensive logging for debugging
 * 
 * ## Reporting and Analysis
 * 
 * ### Individual Test Results
 * For each test, the function reports:
 * - Pass/fail status with clear indicators
 * - Exit code for detailed failure analysis
 * - Execution time with human-readable formatting
 * - Peak memory usage with appropriate units
 * - Crash and timeout indicators
 * - Error messages and diagnostic information
 * 
 * ### Comprehensive Final Report
 * After all tests complete:
 * - Detailed execution report with statistical analysis
 * - Summary statistics including counts and success rates
 * - Performance analysis across all tests
 * - Memory usage patterns and trends
 * 
 * ## Return Value Semantics
 * 
 * The function returns:
 * - **0**: All tests passed successfully
 * - **1**: One or more tests failed, or no tests could be executed
 * 
 * This return value convention makes the tool suitable for use in automated
 * testing environments and CI/CD pipelines.
 * 
 * ## Performance Characteristics
 * 
 * ### Execution Efficiency
 * - Sequential test execution prevents resource conflicts
 * - Efficient test discovery with filesystem optimizations
 * - Minimal overhead from monitoring infrastructure
 * - Proper resource cleanup between test executions
 * 
 * ### Memory Management
 * - Controlled memory usage during test discovery
 * - Efficient result collection and storage
 * - Proper cleanup of monitoring resources
 * - Memory leak prevention in long test runs
 * 
 * ## Integration Examples
 * 
 * ### Development Workflow
 * ```cpp
 * // This function can be called programmatically:
 * char* args[] = {"test-monitor-demo", "./build/tests", nullptr};
 * int result = main(2, args);
 * if (result == 0) {
 *     std::cout << "All tests passed!\n";
 * }
 * ```
 * 
 * ### Automated Testing
 * The function's design makes it suitable for:
 * - Continuous integration pipelines
 * - Automated regression testing
 * - Performance monitoring in development
 * - Quality assurance workflows
 * 
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return int Exit code: 0 for success, 1 for failure
 * 
 * @throws std::filesystem::filesystem_error If directory operations fail
 * @throws std::runtime_error If monitor initialization fails
 * 
 * @see TestExecutionMonitor
 * @see TestExecutionConfig
 * @see discover_test_executables()
 * @see format_duration()
 * @see format_memory_size()
 * 
 * ## Example Execution Flow
 * 
 * 1. **Initialization**
 *    ```
 *    Dataset Test Execution Monitor Demo
 *    ===================================
 *    
 *    Working directory: ./build/tests
 *    Timeout: 60 seconds
 *    Memory monitoring: enabled
 *    Crash detection: enabled
 *    ```
 * 
 * 2. **Test Discovery**
 *    ```
 *    Discovering test executables in: ./build/tests
 *    Found 3 test executable(s):
 *      - ./test_core
 *      - ./test_formats
 *      - ./test_streaming
 *    ```
 * 
 * 3. **Test Execution**
 *    ```
 *    Executing tests...
 *    ==================
 *    
 *    Running: ./test_core
 *      Result: PASSED
 *      Exit code: 0
 *      Execution time: 2.45s
 *      Peak memory: 15.2 MB
 *    ```
 * 
 * 4. **Final Summary**
 *    ```
 *    Final Summary:
 *    =============
 *    Total tests executed: 3
 *    Passed: 3
 *    Failed: 0
 *    Success rate: 100%
 *    ```
 */
int main(int argc, char* argv[]) {
    LLAMA_LOG_INFO("Dataset Test Execution Monitor Demo\n");
    LLAMA_LOG_INFO("===================================\n\n");

    // Configure the test execution monitor
    TestExecutionConfig config;
    config.timeout_seconds = 60;  // 1 minute timeout
    config.capture_memory_usage = true;
    config.enable_crash_detection = true;
    config.verbose_output = true;
    config.memory_sample_interval_ms = 50;  // Sample every 50ms

    // Set working directory to current directory if not specified
    if (argc > 1) {
        config.working_directory = argv[1];
    } else {
        config.working_directory = ".";
    }

    LLAMA_LOG_INFO("Working directory: %s\n", config.working_directory.c_str());
    LLAMA_LOG_INFO("Timeout: %d seconds\n", config.timeout_seconds);
    LLAMA_LOG_INFO("Memory monitoring: %s\n", config.capture_memory_usage ? "enabled" : "disabled");
    LLAMA_LOG_INFO("Crash detection: %s\n\n", config.enable_crash_detection ? "enabled" : "disabled");

    TestExecutionMonitor monitor(config);

    // Discover test executables in the working directory
    std::vector<std::string> test_executables;

    if (argc > 2) {
        // Use specific test executables provided as arguments
        for (int i = 2; i < argc; i++) {
            test_executables.push_back(argv[i]);
        }
    } else {
        // Auto-discover test executables
        LLAMA_LOG_INFO("Discovering test executables in: %s\n", config.working_directory.c_str());
        test_executables = discover_test_executables(config.working_directory);

        if (test_executables.empty()) {
            LLAMA_LOG_INFO("No test executables found. Checking common build directories...\n");

            // Check common build directories
            std::vector<std::string> build_dirs = {"bb", "build", "cmake-build-debug", "cmake-build-release"};
            for (const auto& build_dir : build_dirs) {
                if (std::filesystem::exists(build_dir)) {
                    LLAMA_LOG_INFO("Checking %s...\n", build_dir.c_str());
                    auto found_tests = discover_test_executables(build_dir);
                    test_executables.insert(test_executables.end(), found_tests.begin(), found_tests.end());

                    // Also check subdirectories
                    auto tools_dir = build_dir + "/tools/dataset-converter";
                    if (std::filesystem::exists(tools_dir)) {
                        auto tools_tests = discover_test_executables(tools_dir);
                        test_executables.insert(test_executables.end(), tools_tests.begin(), tools_tests.end());
                    }
                }
            }
        }
    }

    if (test_executables.empty()) {
        LLAMA_LOG_INFO("No test executables found!\n");
        LLAMA_LOG_INFO("Usage: %s [working_directory] [test_executable1] [test_executable2] ...\n", argv[0]);
        return 1;
    }

    LLAMA_LOG_INFO("Found %lu test executable(s):\n", test_executables.size());
    for (const auto& test : test_executables) {
        LLAMA_LOG_INFO("  - %s\n", test.c_str());
    }
    LLAMA_LOG_INFO("\n");

    // Execute tests
    LLAMA_LOG_INFO("Executing tests...\n");
    LLAMA_LOG_INFO("==================\n\n");

    std::vector<TestExecutionResult> results;

    for (const auto& test_executable : test_executables) {
        LLAMA_LOG_INFO("Running: %s\n", test_executable.c_str());

        if (!monitor.is_executable_available(test_executable)) {
            LLAMA_LOG_INFO("  ERROR: Executable not found or not executable\n\n");
            continue;
        }

        auto result = monitor.execute_test(test_executable);
        results.push_back(result);

        LLAMA_LOG_INFO("  Result: %s\n", result.passed ? "PASSED" : "FAILED");
        LLAMA_LOG_INFO("  Exit code: %d\n", result.exit_code);
        LLAMA_LOG_INFO("  Execution time: %s\n", format_duration(result.execution_time_ms).c_str());
        LLAMA_LOG_INFO("  Peak memory: %s\n", format_memory_size(result.peak_memory_usage_bytes).c_str());

        if (result.crashed) {
            LLAMA_LOG_INFO("  CRASHED: %s\n", result.error_message.c_str());
        }

        if (result.timeout_occurred) {
            LLAMA_LOG_INFO("  TIMEOUT: Test exceeded time limit\n");
        }

        if (!result.error_message.empty() && !result.crashed) {
            LLAMA_LOG_INFO("  Error: %s\n", result.error_message.c_str());
        }

        LLAMA_LOG_INFO("\n");
    }

    // Generate comprehensive report
    LLAMA_LOG_INFO("\n%s\n", monitor.generate_execution_report(results).c_str());

    // Summary statistics
    int passed = 0, failed = 0;
    for (const auto& result : results) {
        if (result.passed) passed++;
        else failed++;
    }

    LLAMA_LOG_INFO("Final Summary:\n");
    LLAMA_LOG_INFO("=============\n");
    LLAMA_LOG_INFO("Total tests executed: %lu\n", results.size());
    LLAMA_LOG_INFO("Passed: %d\n", passed);
    LLAMA_LOG_INFO("Failed: %d\n", failed);
    LLAMA_LOG_INFO("Success rate: %lu\n", results.empty() ? 0 : passed * 100 / results.size());

    return failed > 0 ? 1 : 0;
}
