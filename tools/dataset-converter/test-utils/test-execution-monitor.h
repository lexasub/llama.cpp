#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <signal.h>
#include <sys/types.h>

namespace llama_dataset {

struct TestExecutionResult {
    std::string test_name;
    bool passed;
    int exit_code;
    std::string stdout_output;
    std::string stderr_output;
    std::string error_message;
    std::string stack_trace;
    
    // Performance metrics
    double execution_time_ms;
    size_t peak_memory_usage_bytes;
    bool timeout_occurred;
    bool crashed;
    bool memory_leak_detected;
    
    // Additional diagnostic info
    std::string working_directory;
    std::vector<std::string> command_args;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
};

struct TestExecutionConfig {
    int timeout_seconds = 300;  // 5 minutes default
    bool capture_memory_usage = true;
    bool enable_crash_detection = true;
    bool verbose_output = false;
    std::string working_directory;
    std::vector<std::string> environment_vars;
    
    // Memory monitoring settings
    int memory_sample_interval_ms = 100;
    size_t memory_limit_bytes = 0;  // 0 = no limit
};

class TestExecutionMonitor {
public:
    explicit TestExecutionMonitor(const TestExecutionConfig& config = TestExecutionConfig{});
    ~TestExecutionMonitor();
    
    // Execute a single test with full monitoring
    TestExecutionResult execute_test(const std::string& test_executable, 
                                   const std::vector<std::string>& args = {});
    
    // Execute multiple tests
    std::vector<TestExecutionResult> execute_tests(const std::vector<std::string>& test_executables);
    
    // Utility functions
    bool is_executable_available(const std::string& executable_path);
    std::string generate_execution_report(const std::vector<TestExecutionResult>& results);
    
    // Configuration
    void set_config(const TestExecutionConfig& config);
    const TestExecutionConfig& get_config() const;

private:
    TestExecutionConfig config_;
    
    // Internal monitoring functions
    void setup_crash_handler();
    void cleanup_crash_handler();
    size_t monitor_memory_usage(pid_t pid, std::chrono::milliseconds duration);
    bool check_process_timeout(pid_t pid, int timeout_seconds);
    std::string capture_stack_trace(pid_t pid);
    std::string get_process_status(pid_t pid);
    
    // Signal handling for crash detection
    static void crash_signal_handler(int signal, siginfo_t* info, void* context);
    static TestExecutionMonitor* current_monitor_;
    static std::string last_crash_info_;
};

// Utility class for memory monitoring
class MemoryMonitor {
public:
    explicit MemoryMonitor(pid_t pid);
    ~MemoryMonitor();
    
    void start_monitoring(int interval_ms = 100);
    void stop_monitoring();
    
    size_t get_peak_memory_usage() const;
    size_t get_current_memory_usage() const;
    std::vector<size_t> get_memory_samples() const;
    
    bool is_monitoring() const;

private:
    pid_t pid_;
    bool monitoring_;
    size_t peak_memory_;
    std::vector<size_t> memory_samples_;
    std::thread monitoring_thread_;
    mutable std::mutex memory_mutex_;
    std::atomic<bool> stop_flag_;
    
    size_t read_process_memory(pid_t pid) const;
    void monitoring_loop(int interval_ms);
};

// Utility functions
std::string format_memory_size(size_t bytes);
std::string format_duration(double milliseconds);
bool kill_process_tree(pid_t pid, int signal = SIGTERM);
std::vector<std::string> discover_test_executables(const std::string& directory);

} // namespace llama_dataset