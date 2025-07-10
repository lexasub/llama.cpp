#include "test-execution-monitor.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <filesystem>

#include <unistd.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <signal.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <dirent.h>

#ifdef __linux__
#include <sys/prctl.h>
#include <execinfo.h>
#endif

namespace llama_dataset {

// Static members for crash handling
TestExecutionMonitor* TestExecutionMonitor::current_monitor_ = nullptr;
std::string TestExecutionMonitor::last_crash_info_;

TestExecutionMonitor::TestExecutionMonitor(const TestExecutionConfig& config) 
    : config_(config) {
    if (config_.enable_crash_detection) {
        setup_crash_handler();
    }
}

TestExecutionMonitor::~TestExecutionMonitor() {
    if (config_.enable_crash_detection) {
        cleanup_crash_handler();
    }
}

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
                std::cerr << "Failed to change directory to: " << config_.working_directory << std::endl;
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
        std::cerr << "Failed to execute: " << test_executable << std::endl;
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
        std::cout << "Test: " << test_executable << std::endl;
        std::cout << "Exit code: " << result.exit_code << std::endl;
        std::cout << "Execution time: " << format_duration(result.execution_time_ms) << std::endl;
        if (memory_monitor) {
            std::cout << "Peak memory: " << format_memory_size(result.peak_memory_usage_bytes) << std::endl;
        }
        std::cout << "Passed: " << (result.passed ? "YES" : "NO") << std::endl;
        std::cout << "---" << std::endl;
    }
    
    return result;
}

std::vector<TestExecutionResult> TestExecutionMonitor::execute_tests(const std::vector<std::string>& test_executables) {
    std::vector<TestExecutionResult> results;
    results.reserve(test_executables.size());
    
    for (const auto& test_executable : test_executables) {
        auto result = execute_test(test_executable);
        results.push_back(std::move(result));
    }
    
    return results;
}

bool TestExecutionMonitor::is_executable_available(const std::string& executable_path) {
    struct stat st;
    if (stat(executable_path.c_str(), &st) != 0) {
        return false;
    }
    
    return (st.st_mode & S_IXUSR) != 0;
}

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

void TestExecutionMonitor::cleanup_crash_handler() {
    signal(SIGSEGV, SIG_DFL);
    signal(SIGABRT, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
    signal(SIGILL, SIG_DFL);
    current_monitor_ = nullptr;
}

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

// MemoryMonitor implementation
MemoryMonitor::MemoryMonitor(pid_t pid) 
    : pid_(pid), monitoring_(false), peak_memory_(0), stop_flag_(false) {
}

MemoryMonitor::~MemoryMonitor() {
    stop_monitoring();
}

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

// Utility functions
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

bool kill_process_tree(pid_t pid, int signal) {
    // Kill the process group
    if (killpg(pid, signal) == 0) {
        return true;
    }
    
    // Fallback to killing just the process
    return kill(pid, signal) == 0;
}

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