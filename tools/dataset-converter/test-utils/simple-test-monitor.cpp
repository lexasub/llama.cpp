#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#include <cstring>

struct SimpleTestResult {
    std::string test_name;
    bool passed;
    int exit_code;
    std::string stdout_output;
    std::string stderr_output;
    double execution_time_ms;
    bool timeout_occurred;
    bool crashed;
    std::string error_message;
};

class SimpleTestMonitor {
public:
    SimpleTestMonitor(int timeout_seconds = 60) : timeout_seconds_(timeout_seconds) {}
    
    SimpleTestResult execute_test(const std::string& test_executable) {
        SimpleTestResult result;
        result.test_name = test_executable;
        result.passed = false;
        result.exit_code = -1;
        result.timeout_occurred = false;
        result.crashed = false;
        
        auto start_time = std::chrono::steady_clock::now();
        
        // Check if executable exists
        if (access(test_executable.c_str(), X_OK) != 0) {
            result.error_message = "Test executable not found or not executable: " + test_executable;
            return result;
        }
        
        // Create pipes for stdout and stderr
        int stdout_pipe[2], stderr_pipe[2];
        if (pipe(stdout_pipe) == -1 || pipe(stderr_pipe) == -1) {
            result.error_message = "Failed to create pipes";
            return result;
        }
        
        // Fork process
        pid_t pid = fork();
        if (pid == -1) {
            result.error_message = "Failed to fork process";
            close(stdout_pipe[0]); close(stdout_pipe[1]);
            close(stderr_pipe[0]); close(stderr_pipe[1]);
            return result;
        }
        
        if (pid == 0) {
            // Child process
            dup2(stdout_pipe[1], STDOUT_FILENO);
            dup2(stderr_pipe[1], STDERR_FILENO);
            close(stdout_pipe[0]); close(stdout_pipe[1]);
            close(stderr_pipe[0]); close(stderr_pipe[1]);
            
            execl(test_executable.c_str(), test_executable.c_str(), nullptr);
            exit(1); // If exec fails
        }
        
        // Parent process
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);
        
        // Set non-blocking reads
        fcntl(stdout_pipe[0], F_SETFL, O_NONBLOCK);
        fcntl(stderr_pipe[0], F_SETFL, O_NONBLOCK);
        
        // Monitor execution
        bool process_finished = false;
        int status = 0;
        std::string stdout_buffer, stderr_buffer;
        char read_buffer[4096];
        
        while (!process_finished) {
            // Check if process finished
            pid_t wait_result = waitpid(pid, &status, WNOHANG);
            if (wait_result == pid) {
                process_finished = true;
            } else if (wait_result == -1) {
                result.error_message = "Error waiting for process: " + std::string(strerror(errno));
                break;
            }
            
            // Check timeout
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
            if (elapsed.count() >= timeout_seconds_) {
                result.timeout_occurred = true;
                kill(pid, SIGKILL);
                waitpid(pid, &status, 0);
                process_finished = true;
                result.error_message = "Test timed out after " + std::to_string(timeout_seconds_) + " seconds";
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
            
            if (!process_finished) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        
        // Calculate execution time
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.execution_time_ms = duration.count();
        
        result.stdout_output = stdout_buffer;
        result.stderr_output = stderr_buffer;
        result.exit_code = WEXITSTATUS(status);
        
        // Check for crashes
        if (WIFSIGNALED(status)) {
            result.crashed = true;
            int signal_num = WTERMSIG(status);
            result.error_message += (result.error_message.empty() ? "" : "; ") + 
                                   std::string("Process terminated by signal ") + std::to_string(signal_num);
        }
        
        result.passed = (result.exit_code == 0) && !result.timeout_occurred && !result.crashed;
        
        return result;
    }
    
    void print_result(const SimpleTestResult& result) {
        std::cout << "Test: " << result.test_name << std::endl;
        std::cout << "  Status: " << (result.passed ? "PASSED" : "FAILED") << std::endl;
        std::cout << "  Exit code: " << result.exit_code << std::endl;
        std::cout << "  Execution time: " << result.execution_time_ms << " ms" << std::endl;
        
        if (result.crashed) {
            std::cout << "  CRASHED: " << result.error_message << std::endl;
        }
        
        if (result.timeout_occurred) {
            std::cout << "  TIMEOUT: Test exceeded " << timeout_seconds_ << " seconds" << std::endl;
        }
        
        if (!result.error_message.empty() && !result.crashed) {
            std::cout << "  Error: " << result.error_message << std::endl;
        }
        
        if (!result.stdout_output.empty()) {
            std::cout << "  Stdout: " << result.stdout_output << std::endl;
        }
        
        if (!result.stderr_output.empty()) {
            std::cout << "  Stderr: " << result.stderr_output << std::endl;
        }
        
        std::cout << std::endl;
    }

private:
    int timeout_seconds_;
};

int main(int argc, char* argv[]) {
    std::cout << "Simple Dataset Test Monitor\n";
    std::cout << "===========================\n\n";
    
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <test_executable> [test_executable2] ...\n";
        std::cout << "Example: " << argv[0] << " ./test-scenarios success\n";
        return 1;
    }
    
    SimpleTestMonitor monitor(30); // 30 second timeout
    
    std::vector<SimpleTestResult> results;
    
    for (int i = 1; i < argc; i++) {
        std::string test_executable = argv[i];
        std::cout << "Running: " << test_executable << std::endl;
        
        auto result = monitor.execute_test(test_executable);
        results.push_back(result);
        monitor.print_result(result);
    }
    
    // Summary
    int passed = 0, failed = 0;
    for (const auto& result : results) {
        if (result.passed) passed++;
        else failed++;
    }
    
    std::cout << "Summary:\n";
    std::cout << "========\n";
    std::cout << "Total tests: " << results.size() << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << "Success rate: " << (results.empty() ? 0 : (passed * 100 / results.size())) << "%" << std::endl;
    
    return failed > 0 ? 1 : 0;
}