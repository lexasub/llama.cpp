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
        LLAMA_LOG_INFO("Test: %s\n", result.test_name);
        LLAMA_LOG_INFO("  Status: %s\n", result.passed ? "PASSED" : "FAILED");
        LLAMA_LOG_INFO("  Exit code: %s\n", result.exit_code);
        LLAMA_LOG_INFO("  Execution time: ms%s\n\n", result.execution_time_ms);

        if (result.crashed) {
            LLAMA_LOG_INFO("  CRASHED: %s\n", result.error_message);
        }

        if (result.timeout_occurred) {
            LLAMA_LOG_INFO("  TIMEOUT: Test exceeded %s seconds \n\n", timeout_seconds_);
        }

        if (!result.error_message.empty() && !result.crashed) {
            LLAMA_LOG_INFO("  Error: %s\n", result.error_message);
        }

        if (!result.stdout_output.empty()) {
            LLAMA_LOG_INFO("  Stdout: %s\n", result.stdout_output);
        }

        if (!result.stderr_output.empty()) {
            LLAMA_LOG_INFO("  Stderr: %s\n", result.stderr_output);
        }

        LLAMA_LOG_INFO("\n");
    }

private:
    int timeout_seconds_;
};

int main(int argc, char* argv[]) {
    LLAMA_LOG_INFO("Simple Dataset Test Monitor\n");
    LLAMA_LOG_INFO("===========================\n\n");

    if (argc < 2) {
        LLAMA_LOG_INFO("Usage: %s <test_executable> [test_executable2] ...\n", argv[0]);
        LLAMA_LOG_INFO("Example: %s ./test-scenarios success\n", argv[0]);
        return 1;
    }

    SimpleTestMonitor monitor(30); // 30 second timeout

    std::vector<SimpleTestResult> results;

    for (int i = 1; i < argc; i++) {
        std::string test_executable = argv[i];
        LLAMA_LOG_INFO("Running: %s\n", test_executable);

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

    LLAMA_LOG_INFO("Summary:\n");
    LLAMA_LOG_INFO("========\n");
    LLAMA_LOG_INFO("Total tests: %s\n", results.size());
    LLAMA_LOG_INFO("Passed: %s\n", passed);
    LLAMA_LOG_INFO("Failed: %s\n", failed);
    LLAMA_LOG_INFO("Success rate: %s %%\n", results.empty() ? 0 : passed * 100 / results.size());

    return failed > 0 ? 1 : 0;
}
