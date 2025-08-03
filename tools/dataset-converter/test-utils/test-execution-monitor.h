/**
 * @file test-execution-monitor.h
 * @brief Test Execution Monitoring and Performance Analytics Module
 * 
 * This module provides comprehensive test execution monitoring capabilities for the
 * llama.cpp dataset converter test suite. It offers real-time performance tracking,
 * memory usage monitoring, crash detection, and detailed execution analytics to
 * ensure robust testing and performance validation.
 * 
 * ## Key Features
 * 
 * ### Test Execution Monitoring
 * - Real-time test execution tracking with configurable timeouts
 * - Comprehensive result collection including exit codes, output capture, and timing
 * - Support for both single test and batch test execution
 * - Cross-platform process management and signal handling
 * 
 * ### Performance Analytics
 * - High-resolution execution time measurement
 * - Peak memory usage tracking with configurable sampling intervals
 * - Memory leak detection and reporting
 * - Performance trend analysis and reporting
 * 
 * ### Crash Detection and Recovery
 * - Advanced crash detection using signal handlers
 * - Stack trace capture for debugging
 * - Process tree termination for cleanup
 * - Timeout handling with graceful process termination
 * 
 * ### Memory Monitoring
 * - Real-time memory usage tracking
 * - Configurable memory limits and alerts
 * - Memory sampling with customizable intervals
 * - Peak memory usage detection and reporting
 * 
 * ## Usage Examples
 * 
 * ### Basic Test Execution
 * ```cpp
 * TestExecutionConfig config;
 * config.timeout_seconds = 60;
 * config.capture_memory_usage = true;
 * 
 * TestExecutionMonitor monitor(config);
 * auto result = monitor.execute_test("./my_test", {"--verbose"});
 * 
 * if (result.passed) {
 *     std::cout << "Test passed in " << result.execution_time_ms << "ms\n";
 *     std::cout << "Peak memory: " << format_memory_size(result.peak_memory_usage_bytes) << "\n";
 * }
 * ```
 * 
 * ### Batch Test Execution
 * ```cpp
 * std::vector<std::string> tests = discover_test_executables("./tests/");
 * auto results = monitor.execute_tests(tests);
 * std::string report = monitor.generate_execution_report(results);
 * ```
 * 
 * ### Memory Monitoring
 * ```cpp
 * MemoryMonitor mem_monitor(process_pid);
 * mem_monitor.start_monitoring(50); // 50ms intervals
 * // ... run test ...
 * mem_monitor.stop_monitoring();
 * size_t peak = mem_monitor.get_peak_memory_usage();
 * ```
 * 
 * ## Architecture
 * 
 * The module is designed with a modular architecture:
 * - `TestExecutionMonitor`: Main orchestrator for test execution and monitoring
 * - `MemoryMonitor`: Specialized component for memory usage tracking
 * - `TestExecutionResult`: Comprehensive result structure with all metrics
 * - `TestExecutionConfig`: Flexible configuration system
 * 
 * ## Platform Compatibility
 * 
 * This module provides cross-platform support through the platform compatibility layer:
 * - POSIX-compliant systems (Linux, macOS, BSD)
 * - Windows with appropriate signal handling adaptations
 * - Process management abstractions for different operating systems
 * 
 * ## Performance Considerations
 * 
 * - Memory monitoring uses separate threads to minimize impact on test execution
 * - Configurable sampling intervals allow balancing accuracy vs. overhead
 * - Efficient process tree management for cleanup operations
 * - Minimal overhead signal handling for crash detection
 * 
 * @author llama.cpp dataset-converter team
 * @version 1.0
 * @since 2024
 * 
 * @see TestExecutionMonitor
 * @see MemoryMonitor
 * @see TestExecutionResult
 * @see TestExecutionConfig
 */

#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include "platform/platform-compat.h"

namespace llama_dataset {

/**
 * @struct TestExecutionResult
 * @brief Comprehensive test execution result with performance metrics and diagnostics
 * 
 * This structure contains all information collected during test execution, including
 * basic execution results, performance metrics, error information, and diagnostic data.
 * It provides a complete picture of test execution for analysis and reporting.
 * 
 * ## Result Categories
 * 
 * ### Basic Execution Results
 * - Test name and pass/fail status
 * - Exit code and output capture
 * - Error messages and stack traces
 * 
 * ### Performance Metrics
 * - High-precision execution timing
 * - Memory usage statistics
 * - Timeout and crash detection
 * 
 * ### Diagnostic Information
 * - Execution environment details
 * - Command-line arguments
 * - Timing information for analysis
 * 
 * @see TestExecutionMonitor::execute_test()
 * @see TestExecutionMonitor::generate_execution_report()
 */
struct TestExecutionResult {
    std::string test_name;              ///< Name of the executed test
    bool passed;                        ///< Whether the test passed (exit code 0)
    int exit_code;                      ///< Process exit code
    std::string stdout_output;          ///< Captured standard output
    std::string stderr_output;          ///< Captured standard error
    std::string error_message;          ///< Human-readable error description
    std::string stack_trace;            ///< Stack trace if crash occurred

    // Performance metrics
    double execution_time_ms;           ///< Total execution time in milliseconds
    size_t peak_memory_usage_bytes;     ///< Peak memory usage during execution
    bool timeout_occurred;              ///< Whether execution timed out
    bool crashed;                       ///< Whether the process crashed
    bool memory_leak_detected;          ///< Whether memory leaks were detected

    // Additional diagnostic info
    std::string working_directory;      ///< Working directory during execution
    std::vector<std::string> command_args; ///< Command-line arguments used
    std::chrono::system_clock::time_point start_time; ///< Execution start timestamp
    std::chrono::system_clock::time_point end_time;   ///< Execution end timestamp
};

/**
 * @struct TestExecutionConfig
 * @brief Configuration parameters for test execution monitoring
 * 
 * This structure defines all configurable aspects of test execution monitoring,
 * including timeouts, memory monitoring settings, crash detection options, and
 * environment configuration. It allows fine-tuning of monitoring behavior for
 * different testing scenarios.
 * 
 * ## Configuration Categories
 * 
 * ### Execution Control
 * - Timeout settings for test execution
 * - Working directory and environment variables
 * - Output capture and verbosity control
 * 
 * ### Monitoring Options
 * - Memory usage tracking configuration
 * - Crash detection and signal handling
 * - Performance monitoring intervals
 * 
 * ### Resource Limits
 * - Memory usage limits and alerts
 * - Execution time constraints
 * - System resource monitoring
 * 
 * @see TestExecutionMonitor::TestExecutionMonitor()
 * @see TestExecutionMonitor::set_config()
 */
struct TestExecutionConfig {
    int timeout_seconds = 300;              ///< Maximum execution time (default: 5 minutes)
    bool capture_memory_usage = true;       ///< Enable memory usage monitoring
    bool enable_crash_detection = true;     ///< Enable crash detection and stack traces
    bool verbose_output = false;            ///< Enable verbose output capture
    std::string working_directory;          ///< Working directory for test execution
    std::vector<std::string> environment_vars; ///< Additional environment variables

    // Memory monitoring settings
    int memory_sample_interval_ms = 100;    ///< Memory sampling interval in milliseconds
    size_t memory_limit_bytes = 0;          ///< Memory usage limit (0 = no limit)
};

/**
 * @class TestExecutionMonitor
 * @brief Main orchestrator for comprehensive test execution monitoring
 * 
 * This class provides the primary interface for executing tests with comprehensive
 * monitoring capabilities. It coordinates all aspects of test execution including
 * process management, performance tracking, crash detection, and result collection.
 * 
 * ## Core Capabilities
 * 
 * ### Test Execution
 * - Single and batch test execution with full monitoring
 * - Configurable timeouts and resource limits
 * - Cross-platform process management
 * - Environment and working directory control
 * 
 * ### Performance Monitoring
 * - High-resolution timing measurement
 * - Real-time memory usage tracking
 * - Resource utilization monitoring
 * - Performance trend analysis
 * 
 * ### Error Detection and Recovery
 * - Advanced crash detection with signal handling
 * - Stack trace capture for debugging
 * - Timeout detection and graceful termination
 * - Memory leak detection and reporting
 * 
 * ### Reporting and Analytics
 * - Comprehensive execution reports
 * - Performance metrics aggregation
 * - Test result analysis and formatting
 * - Diagnostic information collection
 * 
 * ## Thread Safety
 * 
 * This class is designed to be thread-safe for concurrent test execution monitoring.
 * Internal synchronization ensures safe access to shared resources and crash handlers.
 * 
 * ## Signal Handling
 * 
 * The monitor installs signal handlers for crash detection. These handlers are
 * automatically managed during the monitor's lifetime and cleaned up on destruction.
 * 
 * @see TestExecutionResult
 * @see TestExecutionConfig
 * @see MemoryMonitor
 */
class TestExecutionMonitor {
public:
    /**
     * @brief Construct a new test execution monitor with specified configuration
     * 
     * Initializes the monitor with the provided configuration and sets up
     * necessary signal handlers for crash detection if enabled.
     * 
     * @param config Configuration parameters for monitoring behavior
     * 
     * @throws std::runtime_error If signal handler setup fails
     * 
     * @see TestExecutionConfig
     * @see setup_crash_handler()
     */
    explicit TestExecutionMonitor(const TestExecutionConfig& config = TestExecutionConfig{});
    
    /**
     * @brief Destructor that cleans up signal handlers and resources
     * 
     * Ensures proper cleanup of signal handlers and any ongoing monitoring
     * operations before destruction.
     * 
     * @see cleanup_crash_handler()
     */
    ~TestExecutionMonitor();

    /**
     * @brief Execute a single test with comprehensive monitoring
     * 
     * Executes the specified test executable with full monitoring including
     * performance tracking, memory usage monitoring, crash detection, and
     * output capture. Returns detailed execution results for analysis.
     * 
     * @param test_executable Path to the test executable to run
     * @param args Command-line arguments to pass to the test
     * @return TestExecutionResult Comprehensive execution results and metrics
     * 
     * @throws std::invalid_argument If test_executable is empty or invalid
     * @throws std::runtime_error If process creation fails
     * 
     * @see TestExecutionResult
     * @see MemoryMonitor
     * 
     * ## Example Usage
     * ```cpp
     * TestExecutionMonitor monitor;
     * auto result = monitor.execute_test("./my_test", {"--verbose", "--iterations=100"});
     * 
     * if (result.passed) {
     *     std::cout << "Test completed in " << result.execution_time_ms << "ms\n";
     *     std::cout << "Peak memory: " << format_memory_size(result.peak_memory_usage_bytes) << "\n";
     * } else {
     *     std::cerr << "Test failed: " << result.error_message << "\n";
     *     if (result.crashed) {
     *         std::cerr << "Stack trace:\n" << result.stack_trace << "\n";
     *     }
     * }
     * ```
     */
    TestExecutionResult execute_test(const std::string& test_executable,
                                   const std::vector<std::string>& args = {});

    /**
     * @brief Execute multiple tests with monitoring
     * 
     * Executes a batch of test executables sequentially, collecting comprehensive
     * results for each test. Provides efficient batch processing with consistent
     * monitoring across all tests.
     * 
     * @param test_executables Vector of test executable paths to run
     * @return std::vector<TestExecutionResult> Results for all executed tests
     * 
     * @throws std::invalid_argument If test_executables is empty
     * 
     * @see execute_test()
     * @see generate_execution_report()
     * 
     * ## Example Usage
     * ```cpp
     * std::vector<std::string> tests = {
     *     "./test_core",
     *     "./test_formats",
     *     "./test_streaming"
     * };
     * 
     * auto results = monitor.execute_tests(tests);
     * std::string report = monitor.generate_execution_report(results);
     * std::cout << report << std::endl;
     * ```
     */
    std::vector<TestExecutionResult> execute_tests(const std::vector<std::string>& test_executables);

    /**
     * @brief Check if a test executable is available and executable
     * 
     * Verifies that the specified executable exists, is accessible, and has
     * execute permissions. Useful for pre-flight checks before test execution.
     * 
     * @param executable_path Path to the executable to check
     * @return bool True if executable is available and can be executed
     * 
     * @see execute_test()
     * 
     * ## Example Usage
     * ```cpp
     * if (monitor.is_executable_available("./my_test")) {
     *     auto result = monitor.execute_test("./my_test");
     * } else {
     *     std::cerr << "Test executable not found or not executable\n";
     * }
     * ```
     */
    bool is_executable_available(const std::string& executable_path);
    
    /**
     * @brief Generate a comprehensive execution report from test results
     * 
     * Creates a detailed, human-readable report summarizing test execution
     * results, performance metrics, and any issues encountered. The report
     * includes statistical analysis and formatted output for easy review.
     * 
     * @param results Vector of test execution results to analyze
     * @return std::string Formatted execution report
     * 
     * @see TestExecutionResult
     * @see execute_tests()
     * 
     * ## Report Contents
     * - Summary statistics (pass/fail counts, total time, etc.)
     * - Individual test results with performance metrics
     * - Error analysis and crash reports
     * - Memory usage statistics and trends
     * - Performance analysis and recommendations
     * 
     * ## Example Usage
     * ```cpp
     * auto results = monitor.execute_tests(test_list);
     * std::string report = monitor.generate_execution_report(results);
     * 
     * // Save to file
     * std::ofstream report_file("test_report.txt");
     * report_file << report;
     * 
     * // Or print to console
     * std::cout << report << std::endl;
     * ```
     */
    std::string generate_execution_report(const std::vector<TestExecutionResult>& results);

    /**
     * @brief Update the monitor configuration
     * 
     * Updates the monitoring configuration, affecting subsequent test executions.
     * Changes take effect immediately for new test executions.
     * 
     * @param config New configuration parameters
     * 
     * @see TestExecutionConfig
     * @see get_config()
     * 
     * ## Example Usage
     * ```cpp
     * TestExecutionConfig config;
     * config.timeout_seconds = 120;
     * config.memory_sample_interval_ms = 50;
     * config.verbose_output = true;
     * 
     * monitor.set_config(config);
     * ```
     */
    void set_config(const TestExecutionConfig& config);
    
    /**
     * @brief Get the current monitor configuration
     * 
     * Returns a const reference to the current configuration parameters.
     * 
     * @return const TestExecutionConfig& Current configuration
     * 
     * @see TestExecutionConfig
     * @see set_config()
     */
    const TestExecutionConfig& get_config() const;

private:
    TestExecutionConfig config_;            ///< Current monitoring configuration

    /**
     * @brief Set up signal handlers for crash detection
     * 
     * Installs signal handlers for common crash signals (SIGSEGV, SIGABRT, etc.)
     * to enable crash detection and stack trace capture.
     * 
     * @throws std::runtime_error If signal handler installation fails
     */
    void setup_crash_handler();
    
    /**
     * @brief Clean up signal handlers
     * 
     * Restores original signal handlers and cleans up crash detection resources.
     */
    void cleanup_crash_handler();

    /**
     * @brief Static signal handler for crash detection
     * 
     * Handles crash signals and captures stack trace information for debugging.
     * This is a static function to comply with signal handler requirements.
     * 
     * @param signal Signal number that triggered the handler
     * @param info Signal information structure
     * @param context Signal context (platform-specific)
     */
    static void crash_signal_handler(int signal, siginfo_t* info, void* context);
    
    static TestExecutionMonitor* current_monitor_;  ///< Current monitor instance for signal handling
    static std::string last_crash_info_;            ///< Last crash information captured
};

/**
 * @class MemoryMonitor
 * @brief Specialized utility for real-time memory usage monitoring
 * 
 * This class provides dedicated memory monitoring capabilities for tracking
 * process memory usage in real-time. It runs in a separate thread to minimize
 * impact on the monitored process and provides detailed memory statistics.
 * 
 * ## Key Features
 * 
 * ### Real-time Monitoring
 * - Continuous memory usage tracking with configurable intervals
 * - Peak memory usage detection and recording
 * - Memory usage sampling for trend analysis
 * - Thread-safe access to monitoring data
 * 
 * ### Performance Optimization
 * - Separate monitoring thread to minimize overhead
 * - Efficient memory reading using platform-specific APIs
 * - Configurable sampling intervals for performance tuning
 * - Atomic operations for thread safety
 * 
 * ### Data Collection
 * - Current and peak memory usage tracking
 * - Historical memory usage samples
 * - Memory usage trends and patterns
 * - Statistical analysis support
 * 
 * ## Thread Safety
 * 
 * This class is fully thread-safe and designed for concurrent access.
 * All public methods can be safely called from multiple threads.
 * 
 * @see TestExecutionMonitor
 * @see TestExecutionResult
 */
class MemoryMonitor {
public:
    /**
     * @brief Construct a memory monitor for the specified process
     * 
     * Creates a memory monitor that will track memory usage for the given
     * process ID. The monitor is initially inactive and must be started
     * explicitly using start_monitoring().
     * 
     * @param pid Process ID to monitor
     * 
     * @throws std::invalid_argument If pid is invalid or process doesn't exist
     * 
     * @see start_monitoring()
     * 
     * ## Example Usage
     * ```cpp
     * pid_t test_pid = fork_test_process();
     * MemoryMonitor monitor(test_pid);
     * monitor.start_monitoring(50); // 50ms intervals
     * ```
     */
    explicit MemoryMonitor(pid_t pid);
    
    /**
     * @brief Destructor that ensures monitoring is stopped
     * 
     * Automatically stops monitoring if still active and cleans up resources.
     * Waits for the monitoring thread to complete before destruction.
     * 
     * @see stop_monitoring()
     */
    ~MemoryMonitor();

    /**
     * @brief Start real-time memory monitoring
     * 
     * Begins continuous memory monitoring in a separate thread with the
     * specified sampling interval. Monitoring continues until explicitly
     * stopped or the monitored process terminates.
     * 
     * @param interval_ms Sampling interval in milliseconds (default: 100ms)
     * 
     * @throws std::runtime_error If monitoring is already active
     * @throws std::invalid_argument If interval_ms is less than 1
     * 
     * @see stop_monitoring()
     * @see is_monitoring()
     * 
     * ## Performance Considerations
     * - Lower intervals provide more accurate peak detection but higher overhead
     * - Recommended range: 10-1000ms depending on test duration and accuracy needs
     * - Default 100ms provides good balance for most use cases
     * 
     * ## Example Usage
     * ```cpp
     * monitor.start_monitoring(25); // High-frequency monitoring
     * run_memory_intensive_test();
     * monitor.stop_monitoring();
     * 
     * size_t peak = monitor.get_peak_memory_usage();
     * std::cout << "Peak memory: " << format_memory_size(peak) << "\n";
     * ```
     */
    void start_monitoring(int interval_ms = 100);
    
    /**
     * @brief Stop memory monitoring
     * 
     * Stops the monitoring thread and finalizes memory usage data collection.
     * This method blocks until the monitoring thread has completely stopped.
     * 
     * @see start_monitoring()
     * @see is_monitoring()
     * 
     * ## Example Usage
     * ```cpp
     * monitor.start_monitoring();
     * // ... run test ...
     * monitor.stop_monitoring();
     * 
     * // Now safe to access final results
     * auto samples = monitor.get_memory_samples();
     * ```
     */
    void stop_monitoring();

    /**
     * @brief Get the peak memory usage observed during monitoring
     * 
     * Returns the highest memory usage value recorded since monitoring began.
     * This value is updated continuously during active monitoring.
     * 
     * @return size_t Peak memory usage in bytes
     * 
     * @see get_current_memory_usage()
     * @see get_memory_samples()
     * 
     * ## Thread Safety
     * This method is thread-safe and can be called during active monitoring.
     * 
     * ## Example Usage
     * ```cpp
     * monitor.start_monitoring();
     * // ... test execution ...
     * 
     * // Check peak usage during execution
     * size_t current_peak = monitor.get_peak_memory_usage();
     * if (current_peak > memory_limit) {
     *     std::cout << "Memory limit exceeded!\n";
     * }
     * ```
     */
    size_t get_peak_memory_usage() const;
    
    /**
     * @brief Get the current memory usage of the monitored process
     * 
     * Returns the most recent memory usage reading. If monitoring is active,
     * this reflects the latest sample. If monitoring is stopped, this reflects
     * the last recorded value.
     * 
     * @return size_t Current memory usage in bytes
     * 
     * @throws std::runtime_error If process no longer exists
     * 
     * @see get_peak_memory_usage()
     * @see get_memory_samples()
     * 
     * ## Example Usage
     * ```cpp
     * size_t current = monitor.get_current_memory_usage();
     * std::cout << "Current memory: " << format_memory_size(current) << "\n";
     * ```
     */
    size_t get_current_memory_usage() const;
    
    /**
     * @brief Get all memory usage samples collected during monitoring
     * 
     * Returns a vector containing all memory usage samples collected since
     * monitoring began. Useful for trend analysis and detailed memory profiling.
     * 
     * @return std::vector<size_t> Vector of memory usage samples in bytes
     * 
     * @see get_peak_memory_usage()
     * @see get_current_memory_usage()
     * 
     * ## Data Analysis
     * The returned samples can be used for:
     * - Memory usage trend analysis
     * - Memory leak detection
     * - Performance profiling
     * - Statistical analysis of memory patterns
     * 
     * ## Example Usage
     * ```cpp
     * auto samples = monitor.get_memory_samples();
     * 
     * // Calculate average memory usage
     * size_t total = 0;
     * for (size_t sample : samples) {
     *     total += sample;
     * }
     * size_t average = total / samples.size();
     * 
     * // Detect memory growth
     * bool growing = samples.back() > samples.front() * 1.1; // 10% growth
     * ```
     */
    std::vector<size_t> get_memory_samples() const;

    /**
     * @brief Check if memory monitoring is currently active
     * 
     * Returns true if the monitoring thread is running and actively collecting
     * memory usage data.
     * 
     * @return bool True if monitoring is active, false otherwise
     * 
     * @see start_monitoring()
     * @see stop_monitoring()
     * 
     * ## Example Usage
     * ```cpp
     * if (!monitor.is_monitoring()) {
     *     monitor.start_monitoring();
     * }
     * 
     * // ... test execution ...
     * 
     * if (monitor.is_monitoring()) {
     *     monitor.stop_monitoring();
     * }
     * ```
     */
    bool is_monitoring() const;

private:
    pid_t pid_;                             ///< Process ID being monitored
    bool monitoring_;                       ///< Whether monitoring is currently active
    size_t peak_memory_;                    ///< Peak memory usage observed
    std::vector<size_t> memory_samples_;    ///< All memory usage samples collected
    std::thread monitoring_thread_;         ///< Background monitoring thread
    mutable std::mutex memory_mutex_;       ///< Mutex for thread-safe access
    std::atomic<bool> stop_flag_;           ///< Atomic flag to stop monitoring

    /**
     * @brief Read current memory usage for the specified process
     * 
     * Platform-specific implementation to read process memory usage.
     * Uses /proc/pid/status on Linux, task_info on macOS, etc.
     * 
     * @param pid Process ID to read memory for
     * @return size_t Current memory usage in bytes
     * 
     * @throws std::runtime_error If memory reading fails
     */
    size_t read_process_memory(pid_t pid) const;
    
    /**
     * @brief Main monitoring loop executed in separate thread
     * 
     * Continuously samples memory usage at the specified interval until
     * the stop flag is set. Updates peak memory and sample collection.
     * 
     * @param interval_ms Sampling interval in milliseconds
     */
    void monitoring_loop(int interval_ms);
};

/**
 * @brief Format memory size in human-readable format
 * 
 * Converts a byte count to a human-readable string with appropriate units
 * (B, KB, MB, GB, TB). Uses binary prefixes (1024-based) for accuracy.
 * 
 * @param bytes Memory size in bytes
 * @return std::string Formatted memory size string
 * 
 * ## Example Usage
 * ```cpp
 * size_t memory = 1536 * 1024 * 1024; // 1.5 GB
 * std::cout << "Memory usage: " << format_memory_size(memory) << "\n";
 * // Output: "Memory usage: 1.50 GB"
 * ```
 * 
 * ## Format Examples
 * - 1024 bytes → "1.00 KB"
 * - 1536 KB → "1.50 MB"
 * - 2048 MB → "2.00 GB"
 * - 512 bytes → "512 B"
 */
std::string format_memory_size(size_t bytes);

/**
 * @brief Format duration in human-readable format
 * 
 * Converts a duration in milliseconds to a human-readable string with
 * appropriate units (ms, s, min, h). Automatically selects the most
 * appropriate unit for readability.
 * 
 * @param milliseconds Duration in milliseconds
 * @return std::string Formatted duration string
 * 
 * ## Example Usage
 * ```cpp
 * double duration = 125000.5; // 125.0005 seconds
 * std::cout << "Test took: " << format_duration(duration) << "\n";
 * // Output: "Test took: 2m 5.00s"
 * ```
 * 
 * ## Format Examples
 * - 1500.5 ms → "1.50s"
 * - 125000 ms → "2m 5s"
 * - 3661000 ms → "1h 1m 1s"
 * - 500 ms → "500ms"
 */
std::string format_duration(double milliseconds);

/**
 * @brief Terminate a process tree with the specified signal
 * 
 * Sends the specified signal to a process and all its child processes.
 * Provides cross-platform process tree termination with graceful fallback
 * to forceful termination if needed.
 * 
 * @param pid Process ID of the root process to terminate
 * @param signal Signal to send (default: 15 = SIGTERM equivalent)
 * @return bool True if termination was successful, false otherwise
 * 
 * @throws std::invalid_argument If pid is invalid
 * 
 * ## Signal Handling
 * - Default signal (15) attempts graceful termination
 * - Signal 9 (SIGKILL equivalent) forces immediate termination
 * - Cross-platform signal mapping for Windows compatibility
 * 
 * ## Example Usage
 * ```cpp
 * // Graceful termination
 * if (!kill_process_tree(test_pid)) {
 *     // Force termination if graceful fails
 *     kill_process_tree(test_pid, 9);
 * }
 * ```
 * 
 * ## Platform Support
 * - POSIX systems: Uses process groups and signal handling
 * - Windows: Uses job objects and TerminateProcess
 * - Handles zombie processes and cleanup
 */
bool kill_process_tree(platform_pid_t pid, int signal = 15); // SIGTERM equivalent

/**
 * @brief Discover test executables in the specified directory
 * 
 * Recursively searches the given directory for executable files that appear
 * to be test programs. Uses heuristics to identify test executables based on
 * naming patterns and file properties.
 * 
 * @param directory Directory path to search for test executables
 * @return std::vector<std::string> Vector of discovered test executable paths
 * 
 * @throws std::invalid_argument If directory doesn't exist or isn't accessible
 * 
 * ## Discovery Heuristics
 * - Files with execute permissions
 * - Names containing "test", "spec", or similar patterns
 * - Executable file format detection
 * - Exclusion of scripts and non-binary files
 * 
 * ## Search Patterns
 * The function looks for files matching these patterns:
 * - `*test*` (e.g., "unit_test", "test_core")
 * - `*_test` (e.g., "memory_test", "format_test")
 * - `test_*` (e.g., "test_streaming", "test_validation")
 * - `*spec*` (e.g., "core_spec", "format_spec")
 * 
 * ## Example Usage
 * ```cpp
 * auto tests = discover_test_executables("./tests/");
 * std::cout << "Found " << tests.size() << " test executables:\n";
 * for (const auto& test : tests) {
 *     std::cout << "  " << test << "\n";
 * }
 * 
 * // Execute all discovered tests
 * TestExecutionMonitor monitor;
 * auto results = monitor.execute_tests(tests);
 * ```
 * 
 * ## Performance Considerations
 * - Recursive directory traversal may be slow for large directory trees
 * - File system access patterns optimized for common test layouts
 * - Caching of directory contents for repeated calls
 */
std::vector<std::string> discover_test_executables(const std::string& directory);

} // namespace llama_dataset
