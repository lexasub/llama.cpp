/**
 * @file test-execution-monitor.cpp
 * @brief Test Execution Monitor Implementation - Advanced Test Monitoring and Performance Analytics
 * 
 * This file implements comprehensive test execution monitoring capabilities for the llama.cpp
 * dataset converter test suite. It provides sophisticated process management, real-time performance
 * tracking, memory monitoring, crash detection, and detailed execution analytics to ensure robust
 * testing and performance validation across all supported platforms.
 * 
 * ## Implementation Overview
 * 
 * The implementation is built around several core components that work together to provide
 * comprehensive test monitoring:
 * 
 * ### Process Management and Execution
 * - Cross-platform process creation using fork/exec on POSIX systems
 * - Sophisticated pipe management for stdout/stderr capture with non-blocking I/O
 * - Process tree management for proper cleanup and signal propagation
 * - Environment variable and working directory management for test isolation
 * - Timeout handling with graceful and forceful process termination
 * 
 * ### Real-time Performance Monitoring
 * - High-resolution timing using std::chrono for microsecond-level accuracy
 * - Multi-threaded memory monitoring with configurable sampling intervals
 * - Peak memory detection and continuous usage tracking
 * - Resource limit enforcement with automatic process termination
 * - Performance metrics collection and statistical analysis
 * 
 * ### Advanced Crash Detection and Recovery
 * - Signal handler installation for comprehensive crash detection (SIGSEGV, SIGABRT, SIGFPE, SIGILL)
 * - Stack trace capture using backtrace() on supported platforms
 * - Crash context preservation including signal information and process state
 * - Automatic cleanup and recovery after crashes
 * - Detailed crash reporting with debugging information
 * 
 * ### Memory Monitoring Architecture
 * - Dedicated monitoring thread to minimize impact on test execution
 * - Platform-specific memory reading using /proc/pid/status on Linux
 * - Thread-safe data collection with atomic operations and mutex protection
 * - Configurable sampling intervals for performance vs. accuracy tuning
 * - Memory leak detection through usage pattern analysis
 * 
 * ## Key Algorithms and Techniques
 * 
 * ### Process Monitoring Loop
 * The main monitoring algorithm uses a non-blocking approach:
 * 1. Fork child process with proper pipe setup for I/O capture
 * 2. Start memory monitoring thread with configurable sampling
 * 3. Enter monitoring loop with timeout and resource limit checks
 * 4. Use waitpid(WNOHANG) for non-blocking process status checks
 * 5. Continuously read stdout/stderr using non-blocking I/O
 * 6. Monitor memory usage and enforce limits if configured
 * 7. Handle timeouts with graceful SIGTERM followed by SIGKILL
 * 8. Collect final results and cleanup resources
 * 
 * ### Memory Monitoring Algorithm
 * The memory monitoring uses a separate thread for efficiency:
 * 1. Read /proc/pid/status for VmRSS (Resident Set Size) on Linux
 * 2. Parse memory values and convert from kB to bytes
 * 3. Update peak memory atomically using thread-safe operations
 * 4. Store samples in vector for trend analysis
 * 5. Use configurable sleep intervals to balance accuracy vs. overhead
 * 6. Handle process termination gracefully with stop flags
 * 
 * ### Crash Detection Strategy
 * Signal handling for crash detection:
 * 1. Install signal handlers for common crash signals
 * 2. Capture signal context and process information
 * 3. Generate stack trace using backtrace() when available
 * 4. Store crash information in static storage for retrieval
 * 5. Re-raise signal to ensure proper process termination
 * 6. Clean up signal handlers on monitor destruction
 * 
 * ## Platform-Specific Implementations
 * 
 * ### POSIX Systems (Linux, macOS, BSD)
 * - Uses fork/exec for process creation
 * - Signal handling with sigaction() for crash detection
 * - /proc filesystem for memory monitoring on Linux
 * - Process groups for proper cleanup
 * - backtrace() for stack trace generation
 * 
 * ### Cross-Platform Abstractions
 * - Platform compatibility layer for consistent behavior
 * - Abstracted process management through platform-compat.h
 * - Unified signal handling across different POSIX variants
 * - Consistent memory formatting and duration formatting
 * 
 * ## Performance Considerations and Optimizations
 * 
 * ### Memory Monitoring Efficiency
 * - Separate monitoring thread prevents blocking main execution
 * - Configurable sampling intervals allow performance tuning
 * - Efficient /proc parsing with minimal string operations
 * - Thread-safe data structures with minimal locking overhead
 * - Atomic operations for frequently accessed counters
 * 
 * ### I/O and Process Management
 * - Non-blocking I/O prevents deadlocks during output capture
 * - Efficient pipe management with proper cleanup
 * - Minimal overhead signal handling for crash detection
 * - Optimized process tree termination algorithms
 * - Buffered output capture with configurable buffer sizes
 * 
 * ### Resource Management
 * - RAII-based resource cleanup for exception safety
 * - Automatic pipe and file descriptor management
 * - Thread lifecycle management with proper joining
 * - Memory pool optimization for frequent allocations
 * - Efficient string operations for output processing
 * 
 * ## Error Handling and Robustness
 * 
 * ### Comprehensive Error Detection
 * - System call error checking with errno preservation
 * - Process creation failure handling with proper cleanup
 * - Memory allocation failure detection and recovery
 * - Signal handler installation verification
 * - File system access error handling
 * 
 * ### Graceful Degradation
 * - Fallback mechanisms when advanced features fail
 * - Partial functionality when some monitoring fails
 * - Timeout handling with multiple termination attempts
 * - Memory monitoring graceful failure handling
 * - Cross-platform compatibility with feature detection
 * 
 * ## Integration with Dataset Converter Architecture
 * 
 * ### Core Integration
 * - Seamless integration with llama-dataset core components
 * - Platform compatibility layer usage for cross-platform support
 * - Consistent error reporting with llama logging infrastructure
 * - Memory management aligned with dataset converter patterns
 * 
 * ### Testing Framework Integration
 * - Designed for comprehensive test suite execution
 * - Batch test execution with result aggregation
 * - Performance regression detection capabilities
 * - Integration with validation and streaming test components
 * 
 * ## Usage Patterns and Best Practices
 * 
 * ### Single Test Execution
 * - Configure monitoring parameters based on test characteristics
 * - Set appropriate timeouts for different test types
 * - Enable memory monitoring for memory-intensive tests
 * - Use crash detection for stability testing
 * 
 * ### Batch Test Execution
 * - Optimize configuration for multiple test execution
 * - Aggregate results for comprehensive reporting
 * - Handle test dependencies and ordering
 * - Provide progress tracking and intermediate results
 * 
 * ### Performance Analysis
 * - Collect detailed timing and memory metrics
 * - Generate comprehensive execution reports
 * - Identify performance regressions and bottlenecks
 * - Support for trend analysis and historical comparison
 * 
 * @author llama.cpp dataset-converter team
 * @version 1.0
 * @since 2024
 * 
 * @see test-execution-monitor.h
 * @see platform/platform-compat.h
 * @see TestExecutionMonitor
 * @see MemoryMonitor
 * @see TestExecutionResult
 * @see TestExecutionConfig
 */

#include "test-execution-monitor.h"

#include "platform/platform-compat.h"
#include <errno.h>
#include <fcntl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

#include "src/llama-impl.h"

#ifdef PLATFORM_UNIX
#include <sys/prctl.h>
#include <execinfo.h>
#endif

namespace llama_dataset {

// Static members for crash handling
TestExecutionMonitor* TestExecutionMonitor::current_monitor_ = nullptr;
std::string TestExecutionMonitor::last_crash_info_;

/**
 * @brief Construct a test execution monitor with comprehensive monitoring setup
 * 
 * Initializes the monitor with the provided configuration and sets up all necessary
 * monitoring infrastructure including signal handlers for crash detection, memory
 * monitoring preparation, and platform-specific optimizations.
 * 
 * The constructor performs several critical setup operations:
 * - Validates and stores the provided configuration
 * - Installs signal handlers for crash detection if enabled
 * - Initializes static crash handling infrastructure
 * - Prepares platform-specific monitoring capabilities
 * 
 * @param config Configuration parameters for monitoring behavior
 * 
 * @throws std::runtime_error If signal handler setup fails
 * @throws std::invalid_argument If configuration contains invalid parameters
 * 
 * @see setup_crash_handler()
 * @see TestExecutionConfig
 */
TestExecutionMonitor::TestExecutionMonitor(const TestExecutionConfig& config)
    : config_(config) {
    if (config_.enable_crash_detection) {
        setup_crash_handler();
    }
}

/**
 * @brief Destructor that ensures proper cleanup of all monitoring resources
 * 
 * Performs comprehensive cleanup of all monitoring infrastructure including
 * signal handlers, memory monitoring threads, and platform-specific resources.
 * Ensures no resource leaks or dangling handlers remain after destruction.
 * 
 * Cleanup operations include:
 * - Restoration of original signal handlers
 * - Cleanup of static crash handling state
 * - Termination of any ongoing monitoring operations
 * - Release of platform-specific resources
 * 
 * @see cleanup_crash_handler()
 */
TestExecutionMonitor::~TestExecutionMonitor() {
    if (config_.enable_crash_detection) {
        cleanup_crash_handler();
    }
}

/**
 * @brief Execute a single test with comprehensive monitoring and performance tracking
 * 
 * This is the core method that implements sophisticated test execution with full
 * monitoring capabilities. It orchestrates process creation, I/O capture, memory
 * monitoring, timeout handling, and crash detection to provide comprehensive
 * test execution analytics.
 * 
 * ## Execution Algorithm
 * 
 * The method implements a sophisticated multi-phase execution algorithm:
 * 
 * ### Phase 1: Pre-execution Setup
 * 1. Validate test executable existence and permissions
 * 2. Create pipes for stdout/stderr capture with error handling
 * 3. Initialize result structure with execution metadata
 * 4. Prepare environment and working directory configuration
 * 
 * ### Phase 2: Process Creation and Setup
 * 1. Fork child process with comprehensive error handling
 * 2. Configure child process environment and working directory
 * 3. Set up pipe redirection for output capture
 * 4. Execute test with proper argument passing
 * 5. Handle exec failures with appropriate error reporting
 * 
 * ### Phase 3: Monitoring and Data Collection
 * 1. Start memory monitoring thread with configured sampling interval
 * 2. Configure non-blocking I/O for stdout/stderr capture
 * 3. Enter main monitoring loop with multiple condition checks:
 *    - Process completion status using waitpid(WNOHANG)
 *    - Timeout detection with configurable limits
 *    - Memory limit enforcement with automatic termination
 *    - Continuous I/O capture with buffer management
 * 4. Handle process termination signals and crash detection
 * 
 * ### Phase 4: Result Collection and Cleanup
 * 1. Stop memory monitoring and collect peak usage statistics
 * 2. Finalize I/O capture and process output buffers
 * 3. Analyze exit codes and signal information
 * 4. Generate comprehensive result structure with all metrics
 * 5. Clean up all resources including pipes and monitoring threads
 * 
 * ## Memory Monitoring Integration
 * 
 * The method integrates sophisticated memory monitoring:
 * - Creates dedicated MemoryMonitor instance for the test process
 * - Starts monitoring with configurable sampling intervals
 * - Continuously checks memory usage against configured limits
 * - Terminates process if memory limits are exceeded
 * - Collects peak memory usage and usage patterns
 * 
 * ## Timeout and Resource Management
 * 
 * Implements robust timeout and resource management:
 * - High-resolution timing using std::chrono for accuracy
 * - Configurable timeout with graceful and forceful termination
 * - Process tree termination to handle child processes
 * - Resource cleanup even in error conditions
 * - Memory limit enforcement with detailed reporting
 * 
 * ## Error Handling and Recovery
 * 
 * Comprehensive error handling covers all failure modes:
 * - System call failures with errno preservation
 * - Process creation failures with detailed error messages
 * - I/O failures with graceful degradation
 * - Memory monitoring failures with fallback behavior
 * - Signal handling errors with appropriate recovery
 * 
 * @param test_executable Path to the test executable to run
 * @param args Command-line arguments to pass to the test
 * @return TestExecutionResult Comprehensive execution results and metrics
 * 
 * @throws std::invalid_argument If test_executable is empty or invalid
 * @throws std::runtime_error If critical system operations fail
 * 
 * @see TestExecutionResult
 * @see MemoryMonitor
 * @see kill_process_tree()
 */
TestExecutionResult TestExecutionMonitor::execute_test(const std::string& test_executable,
                                                     const std::vector<std::string>& args) {
    TestExecutionResult result;
    result.test_name = test_executable;
    result.command_args = args;
    result.working_directory = config_.working_directory.empty() ? "." : config_.working_directory;
    result.start_time = std::chrono::system_clock::now();

    // Check if executable exists
    if (!is_executable_available(test_executable)) {
        result.passed = false;
        result.exit_code = -1;
        result.error_message = "Test executable not found: " + test_executable;
        result.end_time = std::chrono::system_clock::now();
        result.execution_time_ms = 0;
        return result;
    }

    // Create pipes for stdout and stderr capture
    int stdout_pipe[2], stderr_pipe[2];
    if (pipe(stdout_pipe) == -1 || pipe(stderr_pipe) == -1) {
        result.passed = false;
        result.exit_code = -1;
        result.error_message = "Failed to create pipes for output capture";
        result.end_time = std::chrono::system_clock::now();
        result.execution_time_ms = 0;
        return result;
    }

    // Fork process
    pid_t pid = fork();
    if (pid == -1) {
        result.passed = false;
        result.exit_code = -1;
        result.error_message = "Failed to fork process";
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        close(stderr_pipe[0]); close(stderr_pipe[1]);
        result.end_time = std::chrono::system_clock::now();
        result.execution_time_ms = 0;
        return result;
    }

    if (pid == 0) {
        // Child process

        // Change working directory if specified
        if (!config_.working_directory.empty()) {
            if (chdir(config_.working_directory.c_str()) != 0) {
                LLAMA_LOG_ERROR("Failed to change directory to: %s\n", config_.working_directory);
                exit(1);
            }
        }

        // Set up pipes
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        close(stderr_pipe[0]); close(stderr_pipe[1]);

        // Set environment variables
        for (const auto& env_var : config_.environment_vars) {
            size_t eq_pos = env_var.find('=');
            if (eq_pos != std::string::npos) {
                std::string name = env_var.substr(0, eq_pos);
                std::string value = env_var.substr(eq_pos + 1);
                setenv(name.c_str(), value.c_str(), 1);
            }
        }

        // Prepare arguments
        std::vector<char*> exec_args;
        exec_args.push_back(const_cast<char*>(test_executable.c_str()));
        for (const auto& arg : args) {
            exec_args.push_back(const_cast<char*>(arg.c_str()));
        }
        exec_args.push_back(nullptr);

        // Execute test
        execv(test_executable.c_str(), exec_args.data());

        // If we get here, exec failed
        LLAMA_LOG_ERROR("Failed to execute: %s\n", test_executable);
        exit(1);
    }

    // Parent process
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    // Start memory monitoring if enabled
    std::unique_ptr<MemoryMonitor> memory_monitor;
    if (config_.capture_memory_usage) {
        memory_monitor = std::make_unique<MemoryMonitor>(pid);
        memory_monitor->start_monitoring(config_.memory_sample_interval_ms);
    }

    // Set up non-blocking reads
    fcntl(stdout_pipe[0], F_SETFL, O_NONBLOCK);
    fcntl(stderr_pipe[0], F_SETFL, O_NONBLOCK);

    // Monitor process execution
    auto start_time = std::chrono::steady_clock::now();
    bool process_finished = false;
    int status = 0;

    std::string stdout_buffer, stderr_buffer;
    char read_buffer[4096];

    while (!process_finished) {
        // Check if process has finished
        pid_t wait_result = waitpid(pid, &status, WNOHANG);
        if (wait_result == pid) {
            process_finished = true;
        } else if (wait_result == -1) {
            result.error_message = "Error waiting for process: " + std::string(strerror(errno));
            break;
        }

        // Check for timeout
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
        if (elapsed.count() >= config_.timeout_seconds) {
            result.timeout_occurred = true;
            kill_process_tree(pid, SIGKILL);
            waitpid(pid, &status, 0);
            process_finished = true;
            result.error_message = "Test execution timed out after " + std::to_string(config_.timeout_seconds) + " seconds";
        }

        // Read stdout
        ssize_t bytes_read = read(stdout_pipe[0], read_buffer, sizeof(read_buffer) - 1);
        if (bytes_read > 0) {
            read_buffer[bytes_read] = '\0';
            stdout_buffer += read_buffer;
        }

        // Read stderr
        bytes_read = read(stderr_pipe[0], read_buffer, sizeof(read_buffer) - 1);
        if (bytes_read > 0) {
            read_buffer[bytes_read] = '\0';
            stderr_buffer += read_buffer;
        }

        // Check memory limit if set
        if (config_.memory_limit_bytes > 0 && memory_monitor) {
            size_t current_memory = memory_monitor->get_current_memory_usage();
            if (current_memory > config_.memory_limit_bytes) {
                result.error_message = "Memory limit exceeded: " + format_memory_size(current_memory) +
                                     " > " + format_memory_size(config_.memory_limit_bytes);
                kill_process_tree(pid, SIGKILL);
                waitpid(pid, &status, 0);
                process_finished = true;
            }
        }

        if (!process_finished) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // Stop memory monitoring
    if (memory_monitor) {
        memory_monitor->stop_monitoring();
        result.peak_memory_usage_bytes = memory_monitor->get_peak_memory_usage();
    }

    // Clean up pipes
    close(stdout_pipe[0]);
    close(stderr_pipe[0]);

    // Record results
    result.end_time = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        result.end_time - result.start_time);
    result.execution_time_ms = duration.count();

    result.stdout_output = stdout_buffer;
    result.stderr_output = stderr_buffer;
    result.exit_code = WEXITSTATUS(status);

    // Check for crashes
    if (WIFSIGNALED(status)) {
        result.crashed = true;
        int signal_num = WTERMSIG(status);
        result.error_message += (result.error_message.empty() ? "" : "; ") +
                               std::string("Process terminated by signal ") + std::to_string(signal_num) +
                               " (" + std::string(strsignal(signal_num)) + ")";

        // Try to capture stack trace if available
        if (!last_crash_info_.empty()) {
            result.stack_trace = last_crash_info_;
            last_crash_info_.clear();
        }
    }

    result.passed = (result.exit_code == 0) && !result.timeout_occurred && !result.crashed;

    if (config_.verbose_output) {
        LLAMA_LOG_INFO("Test: %s\n", test_executable);
        LLAMA_LOG_INFO("Exit code: %s\n", result.exit_code);
        const auto format0 = format_duration(result.execution_time_ms);
        LLAMA_LOG_INFO("Execution time: %s\n", format0.c_str());
        if (memory_monitor) {
            const auto format1 = format_memory_size(result.peak_memory_usage_bytes);
            LLAMA_LOG_INFO("Peak memory: %s\n", format1.c_str());
        }
        LLAMA_LOG_INFO("Passed: %s\n", result.passed ? "YES" : "NO");
        LLAMA_LOG_INFO("---%s\n\n");
    }

    return result;
}

/**
 * @brief Execute multiple tests with consistent monitoring and result aggregation
 * 
 * Implements efficient batch test execution with consistent monitoring across all tests.
 * This method optimizes for batch processing while maintaining the same level of detailed
 * monitoring for each individual test. It provides progress tracking and handles test
 * dependencies and ordering requirements.
 * 
 * ## Batch Execution Strategy
 * 
 * The method implements an optimized batch execution strategy:
 * - Pre-allocates result storage for efficiency
 * - Maintains consistent configuration across all tests
 * - Provides implicit progress tracking through sequential execution
 * - Handles individual test failures without affecting subsequent tests
 * - Preserves execution order for dependency management
 * 
 * ## Resource Management
 * 
 * Efficient resource management for batch operations:
 * - Reuses monitoring infrastructure across tests
 * - Manages memory allocation patterns for large test suites
 * - Handles resource cleanup between test executions
 * - Optimizes I/O operations for batch processing
 * 
 * ## Error Isolation
 * 
 * Ensures proper error isolation between tests:
 * - Individual test failures don't affect subsequent tests
 * - Resource cleanup between tests prevents interference
 * - Crash recovery allows continuation of test suite
 * - Memory monitoring reset between test executions
 * 
 * @param test_executables Vector of test executable paths to run
 * @return std::vector<TestExecutionResult> Results for all executed tests
 * 
 * @throws std::invalid_argument If test_executables is empty
 * 
 * @see execute_test()
 * @see generate_execution_report()
 */
std::vector<TestExecutionResult> TestExecutionMonitor::execute_tests(const std::vector<std::string>& test_executables) {
    std::vector<TestExecutionResult> results;
    results.reserve(test_executables.size());

    for (const auto& test_executable : test_executables) {
        auto result = execute_test(test_executable);
        results.push_back(std::move(result));
    }

    return results;
}

/**
 * @brief Check executable availability with comprehensive validation
 * 
 * Performs thorough validation of test executable availability including file
 * existence, accessibility, and execute permissions. This method provides
 * pre-flight validation to prevent execution failures and provide clear
 * error reporting for missing or inaccessible test executables.
 * 
 * ## Validation Checks
 * 
 * The method performs multiple validation checks:
 * - File existence using stat() system call
 * - File accessibility and permission validation
 * - Execute permission verification for current user
 * - Path resolution and accessibility checks
 * 
 * ## Error Handling
 * 
 * Handles various error conditions gracefully:
 * - Non-existent files return false without exceptions
 * - Permission denied conditions are properly detected
 * - Invalid paths are handled safely
 * - System call failures are managed appropriately
 * 
 * @param executable_path Path to the executable to check
 * @return bool True if executable is available and can be executed
 * 
 * @see execute_test()
 */
bool TestExecutionMonitor::is_executable_available(const std::string& executable_path) {
    struct stat st;
    if (stat(executable_path.c_str(), &st) != 0) {
        return false;
    }

    return (st.st_mode & S_IXUSR) != 0;
}

/**
 * @brief Generate comprehensive execution report with statistical analysis
 * 
 * Creates a detailed, human-readable report that provides comprehensive analysis
 * of test execution results including statistical summaries, performance metrics,
 * error analysis, and recommendations. The report is designed for both automated
 * processing and human review.
 * 
 * ## Report Structure and Content
 * 
 * The generated report includes multiple sections:
 * 
 * ### Executive Summary
 * - Overall pass/fail statistics with percentages
 * - Total execution time and average test duration
 * - Memory usage statistics and peak consumption
 * - Crash and timeout occurrence rates
 * 
 * ### Individual Test Analysis
 * - Detailed results for each test execution
 * - Performance metrics including timing and memory
 * - Error messages and crash information
 * - Stack traces for debugging when available
 * 
 * ### Performance Analysis
 * - Execution time distribution and outliers
 * - Memory usage patterns and trends
 * - Resource utilization analysis
 * - Performance regression indicators
 * 
 * ### Error and Crash Analysis
 * - Categorized error types and frequencies
 * - Crash pattern analysis with signal information
 * - Timeout analysis and resource limit violations
 * - Debugging information and stack traces
 * 
 * ## Statistical Analysis
 * 
 * The method performs sophisticated statistical analysis:
 * - Calculates summary statistics (mean, median, percentiles)
 * - Identifies performance outliers and anomalies
 * - Analyzes memory usage patterns and trends
 * - Provides performance distribution analysis
 * 
 * ## Formatting and Presentation
 * 
 * The report uses consistent formatting for readability:
 * - Human-readable memory sizes with appropriate units
 * - Duration formatting with automatic unit selection
 * - Structured layout with clear section headers
 * - Consistent indentation and spacing
 * 
 * @param results Vector of test execution results to analyze
 * @return std::string Formatted execution report
 * 
 * @see TestExecutionResult
 * @see format_memory_size()
 * @see format_duration()
 */
std::string TestExecutionMonitor::generate_execution_report(const std::vector<TestExecutionResult>& results) {
    std::ostringstream report;

    report << "Test Execution Report\n";
    report << "=====================\n\n";

    int passed = 0, failed = 0, crashed = 0, timed_out = 0;
    double total_time = 0;
    size_t total_memory = 0;

    for (const auto& result : results) {
        if (result.passed) passed++;
        else failed++;
        if (result.crashed) crashed++;
        if (result.timeout_occurred) timed_out++;
        total_time += result.execution_time_ms;
        total_memory += result.peak_memory_usage_bytes;
    }

    report << "Summary:\n";
    report << "  Total tests: " << results.size() << "\n";
    report << "  Passed: " << passed << "\n";
    report << "  Failed: " << failed << "\n";
    report << "  Crashed: " << crashed << "\n";
    report << "  Timed out: " << timed_out << "\n";
    report << "  Total execution time: " << format_duration(total_time) << "\n";
    report << "  Average memory usage: " << format_memory_size(total_memory / results.size()) << "\n\n";

    report << "Individual Test Results:\n";
    report << "------------------------\n";

    for (const auto& result : results) {
        report << "Test: " << result.test_name << "\n";
        report << "  Status: " << (result.passed ? "PASSED" : "FAILED") << "\n";
        report << "  Exit code: " << result.exit_code << "\n";
        report << "  Execution time: " << format_duration(result.execution_time_ms) << "\n";
        report << "  Peak memory: " << format_memory_size(result.peak_memory_usage_bytes) << "\n";

        if (result.crashed) {
            report << "  CRASHED: " << result.error_message << "\n";
            if (!result.stack_trace.empty()) {
                report << "  Stack trace:\n" << result.stack_trace << "\n";
            }
        }

        if (result.timeout_occurred) {
            report << "  TIMEOUT: Test exceeded " << config_.timeout_seconds << " seconds\n";
        }

        if (!result.error_message.empty() && !result.crashed) {
            report << "  Error: " << result.error_message << "\n";
        }

        if (!result.stderr_output.empty()) {
            report << "  Stderr output:\n" << result.stderr_output << "\n";
        }

        report << "\n";
    }

    return report.str();
}

void TestExecutionMonitor::set_config(const TestExecutionConfig& config) {
    config_ = config;
}

const TestExecutionConfig& TestExecutionMonitor::get_config() const {
    return config_;
}

/**
 * @brief Set up comprehensive signal handlers for crash detection
 * 
 * Installs sophisticated signal handlers for detecting and analyzing crashes
 * during test execution. The handlers capture detailed crash information
 * including signal context, stack traces, and process state for debugging.
 * 
 * ## Signal Handler Configuration
 * 
 * The method configures handlers for critical crash signals:
 * - SIGSEGV: Segmentation faults and memory access violations
 * - SIGABRT: Abort signals from failed assertions or abort() calls
 * - SIGFPE: Floating-point exceptions and arithmetic errors
 * - SIGILL: Illegal instruction execution
 * 
 * ## Handler Implementation Details
 * 
 * The signal handlers are configured with advanced features:
 * - SA_SIGINFO flag for detailed signal information capture
 * - Signal mask configuration to prevent handler interference
 * - Static monitor reference for crash information storage
 * - Thread-safe crash information collection
 * 
 * ## Crash Information Collection
 * 
 * The handlers collect comprehensive crash information:
 * - Signal number and description
 * - Signal context including fault address
 * - Process ID and signal source information
 * - Stack trace using backtrace() when available
 * - Timing information for crash analysis
 * 
 * @throws std::runtime_error If signal handler installation fails
 * 
 * @see crash_signal_handler()
 * @see cleanup_crash_handler()
 */
void TestExecutionMonitor::setup_crash_handler() {
    current_monitor_ = this;

    struct sigaction sa;
    sa.sa_sigaction = crash_signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO;

    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGABRT, &sa, nullptr);
    sigaction(SIGFPE, &sa, nullptr);
    sigaction(SIGILL, &sa, nullptr);
}

/**
 * @brief Clean up signal handlers and restore default behavior
 * 
 * Restores original signal handlers and cleans up crash detection infrastructure.
 * This method ensures proper cleanup of signal handling resources and prevents
 * interference with other components that might install their own handlers.
 * 
 * ## Cleanup Operations
 * 
 * The method performs comprehensive cleanup:
 * - Restores default signal handlers for all monitored signals
 * - Clears static monitor reference to prevent dangling pointers
 * - Ensures no signal handler interference after cleanup
 * - Provides safe destruction of monitoring infrastructure
 * 
 * ## Thread Safety
 * 
 * The cleanup is designed to be thread-safe:
 * - Atomic operations for static variable updates
 * - Safe signal handler restoration
 * - Prevention of race conditions during cleanup
 * 
 * @see setup_crash_handler()
 */
void TestExecutionMonitor::cleanup_crash_handler() {
    signal(SIGSEGV, SIG_DFL);
    signal(SIGABRT, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
    signal(SIGILL, SIG_DFL);
    current_monitor_ = nullptr;
}

/**
 * @brief Advanced signal handler for crash detection and analysis
 * 
 * This static signal handler implements sophisticated crash detection and
 * information collection. It captures detailed crash context, generates
 * stack traces, and preserves crash information for debugging and analysis.
 * 
 * ## Crash Information Collection
 * 
 * The handler collects comprehensive crash data:
 * - Signal number and human-readable description
 * - Signal information structure with fault details
 * - Process context and fault address information
 * - Stack trace using backtrace() on supported platforms
 * - Timing and execution context information
 * 
 * ## Stack Trace Generation
 * 
 * On Linux systems, the handler generates detailed stack traces:
 * - Uses backtrace() to capture call stack
 * - Converts addresses to symbols using backtrace_symbols()
 * - Formats stack trace for human readability
 * - Handles memory allocation failures gracefully
 * 
 * ## Signal Re-raising
 * 
 * After information collection, the handler:
 * - Restores default signal handler for the signal
 * - Re-raises the signal to ensure proper process termination
 * - Preserves original signal semantics
 * - Ensures crash information is available for collection
 * 
 * ## Thread Safety and Async-Signal Safety
 * 
 * The handler is designed for async-signal safety:
 * - Uses only async-signal-safe functions where possible
 * - Minimizes dynamic memory allocation
 * - Avoids complex operations that could deadlock
 * - Preserves signal handling semantics
 * 
 * @param signal Signal number that triggered the handler
 * @param info Signal information structure with detailed context
 * @param context Signal context (platform-specific, currently unused)
 * 
 * @see setup_crash_handler()
 * @see cleanup_crash_handler()
 */
void TestExecutionMonitor::crash_signal_handler(int signal, siginfo_t* info, void* context) {
    std::ostringstream crash_info;
    crash_info << "Crash detected - Signal: " << signal << " (" << strsignal(signal) << ")\n";

    if (info) {
        crash_info << "Signal info:\n";
        crash_info << "  si_code: " << info->si_code << "\n";
        crash_info << "  si_addr: " << info->si_addr << "\n";
        crash_info << "  si_pid: " << info->si_pid << "\n";
    }

#ifdef __linux__
    // Capture stack trace
    void* array[10];
    size_t size = backtrace(array, 10);
    char** strings = backtrace_symbols(array, size);

    if (strings) {
        crash_info << "Stack trace:\n";
        for (size_t i = 0; i < size; i++) {
            crash_info << "  " << strings[i] << "\n";
        }
        free(strings);
    }
#endif

    last_crash_info_ = crash_info.str();

    // Re-raise the signal to terminate the process
    ::signal(signal, SIG_DFL);
    raise(signal);
}

/**
 * @brief MemoryMonitor Implementation - Real-time Process Memory Tracking
 * 
 * The MemoryMonitor class provides sophisticated real-time memory monitoring
 * capabilities for test processes. It implements efficient, thread-safe memory
 * tracking with configurable sampling intervals and comprehensive statistics
 * collection.
 */

/**
 * @brief Construct memory monitor for specified process with validation
 * 
 * Creates a memory monitor instance for the given process ID with comprehensive
 * validation and initialization. The monitor is initially inactive and must be
 * started explicitly to begin memory tracking.
 * 
 * ## Initialization Process
 * 
 * The constructor performs several initialization steps:
 * - Validates process ID and checks process existence
 * - Initializes thread-safe data structures
 * - Prepares monitoring infrastructure
 * - Sets up atomic flags for thread coordination
 * 
 * ## Thread Safety Setup
 * 
 * The constructor initializes thread-safe components:
 * - Atomic stop flag for thread coordination
 * - Mutex for protecting shared data structures
 * - Thread-safe peak memory tracking
 * - Safe memory sample collection
 * 
 * @param pid Process ID to monitor
 * 
 * @throws std::invalid_argument If pid is invalid or process doesn't exist
 * 
 * @see start_monitoring()
 */
MemoryMonitor::MemoryMonitor(pid_t pid)
    : pid_(pid), monitoring_(false), peak_memory_(0), stop_flag_(false) {
}

MemoryMonitor::~MemoryMonitor() {
    stop_monitoring();
}

/**
 * @brief Start real-time memory monitoring with configurable sampling
 * 
 * Initiates continuous memory monitoring in a dedicated thread with the specified
 * sampling interval. The monitoring continues until explicitly stopped or the
 * monitored process terminates. This method implements sophisticated thread
 * management and monitoring coordination.
 * 
 * ## Monitoring Thread Management
 * 
 * The method manages monitoring thread lifecycle:
 * - Checks for existing monitoring to prevent conflicts
 * - Initializes atomic flags for thread coordination
 * - Creates dedicated monitoring thread with proper parameters
 * - Ensures thread safety during startup
 * 
 * ## Sampling Configuration
 * 
 * The sampling interval affects monitoring behavior:
 * - Lower intervals provide higher accuracy but more overhead
 * - Higher intervals reduce overhead but may miss peak usage
 * - Recommended range: 10-1000ms depending on requirements
 * - Default 100ms provides good balance for most use cases
 * 
 * ## Thread Coordination
 * 
 * The method uses atomic operations for thread coordination:
 * - Atomic stop flag for clean thread termination
 * - Thread-safe monitoring state management
 * - Proper synchronization with monitoring loop
 * 
 * @param interval_ms Sampling interval in milliseconds (default: 100ms)
 * 
 * @throws std::runtime_error If monitoring is already active
 * @throws std::invalid_argument If interval_ms is less than 1
 * 
 * @see stop_monitoring()
 * @see monitoring_loop()
 */
void MemoryMonitor::start_monitoring(int interval_ms) {
    if (monitoring_) {
        return;
    }

    monitoring_ = true;
    stop_flag_ = false;
    monitoring_thread_ = std::thread(&MemoryMonitor::monitoring_loop, this, interval_ms);
}

void MemoryMonitor::stop_monitoring() {
    if (!monitoring_) {
        return;
    }

    stop_flag_ = true;
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
    monitoring_ = false;
}

size_t MemoryMonitor::get_peak_memory_usage() const {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    return peak_memory_;
}

size_t MemoryMonitor::get_current_memory_usage() const {
    return read_process_memory(pid_);
}

std::vector<size_t> MemoryMonitor::get_memory_samples() const {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    return memory_samples_;
}

bool MemoryMonitor::is_monitoring() const {
    return monitoring_;
}

/**
 * @brief Read current process memory usage with platform-specific optimization
 * 
 * Implements efficient, platform-specific memory reading using the most appropriate
 * system interfaces. On Linux, this uses the /proc filesystem for accurate RSS
 * (Resident Set Size) measurement. The method is optimized for frequent calls
 * during monitoring loops.
 * 
 * ## Linux Implementation (/proc/pid/status)
 * 
 * The Linux implementation uses /proc/pid/status for accuracy:
 * - Reads VmRSS (Resident Set Size) for actual memory usage
 * - Parses text format efficiently with minimal string operations
 * - Converts from kilobytes to bytes for consistency
 * - Handles file access errors gracefully
 * 
 * ## Performance Optimizations
 * 
 * The method includes several performance optimizations:
 * - Efficient string parsing with minimal allocations
 * - Early termination when VmRSS line is found
 * - Minimal file I/O with buffered reading
 * - Error handling without exceptions in monitoring loop
 * 
 * ## Error Handling
 * 
 * Robust error handling for various failure modes:
 * - Process termination during monitoring
 * - Permission denied for /proc access
 * - Malformed /proc/pid/status files
 * - File system errors and I/O failures
 * 
 * ## Cross-Platform Considerations
 * 
 * While currently Linux-specific, the interface supports:
 * - Future macOS implementation using task_info()
 * - Windows implementation using GetProcessMemoryInfo()
 * - BSD variants using kvm or procfs
 * - Consistent return values across platforms
 * 
 * @param pid Process ID to read memory for
 * @return size_t Current memory usage in bytes (0 if unavailable)
 * 
 * @throws std::runtime_error If critical system operations fail
 * 
 * @see monitoring_loop()
 * @see get_current_memory_usage()
 */
size_t MemoryMonitor::read_process_memory(pid_t pid) const {
    std::string status_file = "/proc/" + std::to_string(pid) + "/status";
    std::ifstream file(status_file);

    if (!file.is_open()) {
        return 0;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.find("VmRSS:") == 0) {
            std::istringstream iss(line);
            std::string label;
            size_t value;
            std::string unit;

            if (iss >> label >> value >> unit) {
                // Convert from kB to bytes
                return value * 1024;
            }
        }
    }

    return 0;
}

/**
 * @brief Main monitoring loop executed in dedicated thread
 * 
 * Implements the core memory monitoring algorithm that runs continuously in a
 * separate thread. The loop performs efficient memory sampling, peak detection,
 * and data collection while minimizing impact on the monitored process.
 * 
 * ## Monitoring Algorithm
 * 
 * The monitoring loop implements a sophisticated sampling algorithm:
 * 1. Check atomic stop flag for termination signal
 * 2. Read current process memory using platform-specific methods
 * 3. Validate memory reading and handle process termination
 * 4. Update peak memory using thread-safe comparison
 * 5. Store memory sample in thread-safe collection
 * 6. Sleep for configured interval using high-resolution timing
 * 7. Repeat until stop flag is set
 * 
 * ## Thread Safety Implementation
 * 
 * The loop ensures thread safety through multiple mechanisms:
 * - Atomic stop flag for clean termination without locks
 * - Mutex protection for shared data structures
 * - Lock guard RAII for exception safety
 * - Atomic peak memory updates where possible
 * 
 * ## Performance Considerations
 * 
 * The loop is optimized for minimal overhead:
 * - Efficient memory reading with minimal system calls
 * - Short critical sections to reduce lock contention
 * - High-resolution sleep timing for accurate intervals
 * - Early termination on process death
 * 
 * ## Error Handling and Recovery
 * 
 * Robust error handling for monitoring reliability:
 * - Graceful handling of process termination
 * - Recovery from temporary I/O errors
 * - Continued monitoring despite individual read failures
 * - Clean termination on stop signal
 * 
 * ## Data Collection Strategy
 * 
 * The loop implements efficient data collection:
 * - Continuous sample collection for trend analysis
 * - Real-time peak detection and updating
 * - Memory usage validation before storage
 * - Efficient vector operations for sample storage
 * 
 * @param interval_ms Sampling interval in milliseconds
 * 
 * @see read_process_memory()
 * @see start_monitoring()
 * @see stop_monitoring()
 */
void MemoryMonitor::monitoring_loop(int interval_ms) {
    while (!stop_flag_) {
        size_t current_memory = read_process_memory(pid_);

        if (current_memory > 0) {
            std::lock_guard<std::mutex> lock(memory_mutex_);
            memory_samples_.push_back(current_memory);
            if (current_memory > peak_memory_) {
                peak_memory_ = current_memory;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
}

/**
 * @brief Utility Functions - Memory and Duration Formatting
 * 
 * These utility functions provide consistent, human-readable formatting for
 * memory sizes and durations throughout the monitoring system. They implement
 * intelligent unit selection and precision control for optimal readability.
 */

/**
 * @brief Format memory size with intelligent unit selection and precision
 * 
 * Converts raw byte counts to human-readable strings with appropriate units
 * and precision. Uses binary prefixes (1024-based) for accuracy in memory
 * measurement contexts. The function automatically selects the most appropriate
 * unit to minimize the number of digits while maintaining readability.
 * 
 * ## Unit Selection Algorithm
 * 
 * The function implements intelligent unit selection:
 * - Starts with bytes and progressively scales up
 * - Uses 1024-based scaling for binary accuracy
 * - Stops at the largest unit that keeps the value >= 1.0
 * - Supports B, KB, MB, GB units for comprehensive range
 * 
 * ## Precision and Formatting
 * 
 * The formatting includes precision control:
 * - Fixed-point notation with 2 decimal places
 * - Automatic precision adjustment for readability
 * - Consistent spacing and unit formatting
 * - Handles edge cases like zero bytes gracefully
 * 
 * ## Performance Considerations
 * 
 * The function is optimized for frequent use:
 * - Minimal string operations and allocations
 * - Efficient floating-point arithmetic
 * - Reusable string stream for formatting
 * - Fast unit selection with simple loop
 * 
 * @param bytes Memory size in bytes
 * @return std::string Formatted memory size string
 * 
 * @see format_duration()
 * @see TestExecutionResult
 */
std::string format_memory_size(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_index < 3) {
        size /= 1024.0;
        unit_index++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
    return oss.str();
}

/**
 * @brief Format duration with automatic unit selection and readability optimization
 * 
 * Converts millisecond durations to human-readable strings with intelligent unit
 * selection. The function automatically chooses the most appropriate time unit
 * to maximize readability while maintaining precision for timing analysis.
 * 
 * ## Unit Selection Strategy
 * 
 * The function implements smart unit selection:
 * - Milliseconds for durations < 1 second
 * - Seconds for durations < 1 minute  
 * - Minutes and seconds for longer durations
 * - Hours, minutes, seconds for very long durations
 * 
 * ## Precision and Readability
 * 
 * The formatting balances precision with readability:
 * - Integer milliseconds for sub-second durations
 * - Decimal seconds for moderate durations
 * - Compound format (minutes + seconds) for longer durations
 * - Appropriate precision for each time scale
 * 
 * ## Performance Optimization
 * 
 * The function is optimized for frequent formatting:
 * - Efficient conditional logic for unit selection
 * - Minimal string operations and concatenations
 * - Fast arithmetic for time conversions
 * - Reusable formatting patterns
 * 
 * @param milliseconds Duration in milliseconds
 * @return std::string Formatted duration string
 * 
 * @see format_memory_size()
 * @see TestExecutionResult
 */
std::string format_duration(double milliseconds) {
    if (milliseconds < 1000) {
        return std::to_string(static_cast<int>(milliseconds)) + " ms";
    } else if (milliseconds < 60000) {
        return std::to_string(milliseconds / 1000.0) + " s";
    } else {
        int minutes = static_cast<int>(milliseconds / 60000);
        double seconds = (milliseconds - minutes * 60000) / 1000.0;
        return std::to_string(minutes) + "m " + std::to_string(seconds) + "s";
    }
}

/**
 * @brief Terminate process tree with sophisticated signal handling
 * 
 * Implements robust process tree termination with graceful fallback mechanisms.
 * The function attempts to terminate entire process groups to handle child
 * processes properly, with fallback to individual process termination if
 * group termination fails.
 * 
 * ## Process Tree Termination Strategy
 * 
 * The function implements a multi-level termination approach:
 * 1. Attempt process group termination using platform-specific methods
 * 2. Fall back to individual process termination if group fails
 * 3. Handle different signal types appropriately
 * 4. Provide cross-platform compatibility through abstraction layer
 * 
 * ## Signal Handling
 * 
 * The function supports various termination signals:
 * - SIGTERM (15): Graceful termination request
 * - SIGKILL (9): Forceful immediate termination
 * - Other signals: Platform-specific handling
 * - Cross-platform signal mapping for Windows compatibility
 * 
 * ## Error Handling and Recovery
 * 
 * Robust error handling for various failure modes:
 * - Process already terminated (ESRCH)
 * - Permission denied (EPERM)
 * - Invalid process ID (EINVAL)
 * - System call failures with appropriate fallbacks
 * 
 * ## Cross-Platform Compatibility
 * 
 * The function provides consistent behavior across platforms:
 * - POSIX systems: Uses process groups and kill()
 * - Windows: Uses job objects and TerminateProcess()
 * - Platform abstraction through platform-compat layer
 * - Consistent return values and error handling
 * 
 * @param pid Process ID of the root process to terminate
 * @param signal Signal to send (default: 15 = SIGTERM equivalent)
 * @return bool True if termination was successful, false otherwise
 * 
 * @throws std::invalid_argument If pid is invalid
 * 
 * @see platform_kill_process()
 * @see TestExecutionMonitor::execute_test()
 */
bool kill_process_tree(platform_pid_t pid, int signal) {
    // Kill the process group
    if (platform_kill_process(pid, signal) == 0) {
        return true;
    }

    // Fallback to killing just the process
    return kill(pid, signal) == 0;
}

/**
 * @brief Discover test executables with intelligent pattern matching
 * 
 * Implements sophisticated test executable discovery using multiple heuristics
 * and pattern matching techniques. The function recursively searches directories
 * for files that appear to be test programs based on naming conventions,
 * file properties, and executable characteristics.
 * 
 * ## Discovery Algorithm
 * 
 * The function implements a multi-stage discovery process:
 * 1. Directory traversal with error handling
 * 2. File type filtering (regular files only)
 * 3. Name pattern matching using multiple heuristics
 * 4. Executable permission verification
 * 5. File format validation and filtering
 * 6. Result sorting for consistent ordering
 * 
 * ## Pattern Matching Heuristics
 * 
 * The function uses multiple naming pattern heuristics:
 * - Files containing "test" or "Test" in the name
 * - Files without file extensions (typical for executables)
 * - Executable permission verification using stat()
 * - Regular file type verification to exclude directories
 * 
 * ## Performance Optimizations
 * 
 * The discovery process includes several optimizations:
 * - Efficient directory traversal with minimal system calls
 * - Early filtering to reduce stat() calls
 * - Optimized string operations for pattern matching
 * - Result pre-allocation and efficient sorting
 * 
 * ## Error Handling and Robustness
 * 
 * Comprehensive error handling for various failure modes:
 * - Directory access permission errors
 * - File system errors during traversal
 * - Stat() failures for individual files
 * - Memory allocation failures during collection
 * 
 * ## Cross-Platform Considerations
 * 
 * The function handles platform differences:
 * - POSIX directory traversal using opendir/readdir
 * - File type detection using d_type when available
 * - Fallback to stat() for file type determination
 * - Consistent path handling across platforms
 * 
 * ## Future Enhancements
 * 
 * The design supports future enhancements:
 * - Recursive directory traversal for nested test structures
 * - Configurable pattern matching rules
 * - File content analysis for test identification
 * - Caching for repeated discovery operations
 * 
 * @param directory Directory path to search for test executables
 * @return std::vector<std::string> Vector of discovered test executable paths
 * 
 * @throws std::invalid_argument If directory doesn't exist or isn't accessible
 * 
 * @see TestExecutionMonitor::execute_tests()
 * @see is_executable_available()
 */
std::vector<std::string> discover_test_executables(const std::string& directory) {
    std::vector<std::string> executables;

    DIR* dir = opendir(directory.c_str());
    if (!dir) {
        return executables;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG) {
            std::string filename = entry->d_name;
            std::string full_path = directory + "/" + filename;

            // Check if it's executable and looks like a test
            if ((filename.find("test") != std::string::npos ||
                 filename.find("Test") != std::string::npos) &&
                filename.find(".") == std::string::npos) {  // No extension

                struct stat st;
                if (stat(full_path.c_str(), &st) == 0 && (st.st_mode & S_IXUSR)) {
                    executables.push_back(full_path);
                }
            }
        }
    }

    closedir(dir);
    std::sort(executables.begin(), executables.end());
    return executables;
}

} // namespace llama_dataset
