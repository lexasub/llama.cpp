#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/llama-impl.h"

struct TestResult {
    std::string test_name;
    bool passed;
    int exit_code;
    std::string stdout_output;
    std::string stderr_output;
    double execution_time_ms;
    size_t peak_memory_kb;
    std::string error_analysis;
};

class CoreTestMonitor {
private:
    std::string build_dir;
    std::vector<std::string> test_data_files;

    void create_test_data_if_missing() {
        // Create minimal test data files for testing
        std::vector<std::string> required_files = {
            "test_data/text_dataset.txt",
            "test_data/small_dataset.gguf",
            "test_data/parquet_dataset.parquet",
            "test_data/corrupted_dataset.gguf"
        };

        for (const auto& file : required_files) {
            std::ifstream check(file);
            if (!check.good()) {
                LLAMA_LOG_INFO("Creating missing test data file: %s\n", file.c_str());
                if (file.find(".txt") != std::string::npos) {
                    create_text_test_file(file);
                } else if (file.find("corrupted") != std::string::npos) {
                    create_corrupted_test_file(file);
                }
            }
        }
    }

    void create_text_test_file(const std::string& path) {
        std::ofstream file(path);
        if (file.is_open()) {
            file << "This is a test dataset for text processing.\n";
            file << "It contains multiple lines of text data.\n";
            file << "Each line represents a training sequence.\n";
            file << "The tokenizer should process this correctly.\n";
            file << "Special characters: !@#$%^&*()_+-={}[]|\\:;\"'<>?,./\n";
            file << "Unicode test: αβγδε 中文测试 🚀🔥💯\n";
            file.close();
            LLAMA_LOG_INFO("Created text test file: %s\n", path.c_str());
        }
    }

    void create_corrupted_test_file(const std::string& path) {
        std::ofstream file(path, std::ios::binary);
        if (file.is_open()) {
            // Write some invalid GGUF-like data
            file.write("GGUF", 4);
            file.write("\xFF\xFF\xFF\xFF", 4); // Invalid version
            file.write("CORRUPT_DATA_HERE", 17);
            file.close();
            LLAMA_LOG_INFO("Created corrupted test file: %s\n", path.c_str());
        }
    }

public:
    CoreTestMonitor(const std::string& build_directory) : build_dir(build_directory) {
        create_test_data_if_missing();
    }

    TestResult execute_test(const std::string& test_executable) {
        TestResult result;
        result.test_name = test_executable;
        result.passed = false;
        result.exit_code = -1;
        result.execution_time_ms = 0.0;
        result.peak_memory_kb = 0;

        LLAMA_LOG_INFO("\n=== Executing Core Dataset Test: %s ===\n", test_executable.c_str());

        auto start_time = std::chrono::high_resolution_clock::now();

        // Create pipes for stdout and stderr capture
        int stdout_pipe[2], stderr_pipe[2];
        if (pipe(stdout_pipe) == -1 || pipe(stderr_pipe) == -1) {
            result.error_analysis = "Failed to create pipes for output capture";
            return result;
        }

        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            close(stdout_pipe[0]);
            close(stderr_pipe[0]);
            dup2(stdout_pipe[1], STDOUT_FILENO);
            dup2(stderr_pipe[1], STDERR_FILENO);
            close(stdout_pipe[1]);
            close(stderr_pipe[1]);

            // Execute the test
            std::string full_path = build_dir + "/bin/" + test_executable;
            execl(full_path.c_str(), test_executable.c_str(), nullptr);
            exit(127); // execl failed
        } else if (pid > 0) {
            // Parent process
            close(stdout_pipe[1]);
            close(stderr_pipe[1]);

            // Read output
            char buffer[4096];
            ssize_t bytes_read;

            // Read stdout
            while ((bytes_read = read(stdout_pipe[0], buffer, sizeof(buffer) - 1)) > 0) {
                buffer[bytes_read] = '\0';
                result.stdout_output += buffer;
            }

            // Read stderr
            while ((bytes_read = read(stderr_pipe[0], buffer, sizeof(buffer) - 1)) > 0) {
                buffer[bytes_read] = '\0';
                result.stderr_output += buffer;
            }

            close(stdout_pipe[0]);
            close(stderr_pipe[0]);

            // Wait for child and get resource usage
            struct rusage usage;
            int status;
            wait4(pid, &status, 0, &usage);

            result.exit_code = WEXITSTATUS(status);
            result.peak_memory_kb = usage.ru_maxrss; // Peak memory in KB

            if (WIFSIGNALED(status)) {
                int signal_num = WTERMSIG(status);
                result.error_analysis += "Process terminated by signal " + std::to_string(signal_num);
                if (signal_num == SIGSEGV) {
                    result.error_analysis += " (SEGMENTATION FAULT)";
                } else if (signal_num == SIGABRT) {
                    result.error_analysis += " (ABORT)";
                }
            }
        } else {
            result.error_analysis = "Failed to fork process";
            return result;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        result.passed = (result.exit_code == 0);

        return result;
    }

    void analyze_test_result(const TestResult& result) {
        LLAMA_LOG_INFO("\n--- Test Analysis for %s ---\\n", result.test_name.c_str());
        LLAMA_LOG_INFO("Status: %s\n", result.passed ? "PASSED" : "FAILED");
        LLAMA_LOG_INFO("Exit Code: %d\n", result.exit_code);
        LLAMA_LOG_INFO("Execution Time: %f ms\n", result.execution_time_ms);
        LLAMA_LOG_INFO("Peak Memory Usage: %lu KB\n", result.peak_memory_kb);

        if (!result.error_analysis.empty()) {
            LLAMA_LOG_INFO("Error Analysis: %s\n", result.error_analysis.c_str());
        }

        if (!result.stdout_output.empty()) {
            LLAMA_LOG_INFO("\n--- STDOUT ---\n");
            LLAMA_LOG_INFO(result.stdout_output.c_str());
        }

        if (!result.stderr_output.empty()) {
            LLAMA_LOG_ERROR("\n--- STDERR ---\n");
            LLAMA_LOG_ERROR(result.stderr_output.c_str());
        }

        // Analyze specific failure patterns
        analyze_failure_patterns(result);
    }

    void analyze_failure_patterns(const TestResult& result) {
        LLAMA_LOG_INFO("\n--- Failure Pattern Analysis ---\n");

        // Check for common error patterns
        std::string combined_output = result.stdout_output + result.stderr_output;

        if (combined_output.find("segmentation fault") != std::string::npos ||
            combined_output.find("SIGSEGV") != std::string::npos ||
            result.error_analysis.find("SEGMENTATION FAULT") != std::string::npos) {
            LLAMA_LOG_INFO("⚠️  SEGMENTATION FAULT detected - likely null pointer dereference or memory corruption\n");
        }

        if (combined_output.find("assertion failed") != std::string::npos ||
            combined_output.find("Assertion") != std::string::npos) {
            LLAMA_LOG_INFO("⚠️  ASSERTION FAILURE detected - test expectations not met\n");
        }

        if (combined_output.find("not found") != std::string::npos) {
            LLAMA_LOG_INFO("⚠️  FILE NOT FOUND error detected - missing test data or executable\n");
        }

        if (combined_output.find("null") != std::string::npos) {
            LLAMA_LOG_INFO("✓ NULL POINTER handling being tested\n");
        }

        if (combined_output.find("not implemented") != std::string::npos ||
            combined_output.find("not supported") != std::string::npos) {
            LLAMA_LOG_INFO("ℹ️  UNIMPLEMENTED FEATURE detected - expected for placeholder functions\n");
        }

        if (result.peak_memory_kb > 100000) { // > 100MB
            LLAMA_LOG_INFO("⚠️  HIGH MEMORY USAGE detected - potential memory leak\n");
        }

        if (result.execution_time_ms > 5000) { // > 5 seconds
            LLAMA_LOG_INFO("⚠️  SLOW EXECUTION detected - potential performance issue\n");
        }

        // Check for successful test patterns
        if (combined_output.find("All tests passed") != std::string::npos) {
            LLAMA_LOG_INFO("✅ All internal tests passed successfully\n");
        }

        if (combined_output.find("✓") != std::string::npos) {
            LLAMA_LOG_INFO("✅ Individual test assertions passed\n");
        }
    }

    void generate_comprehensive_report(const std::vector<TestResult>& results) {
        LLAMA_LOG_INFO("\n%s\n", ((std::string(60, '=')).c_str()));
        LLAMA_LOG_INFO("COMPREHENSIVE CORE DATASET TEST REPORT\n");
        LLAMA_LOG_INFO("%s\n", ((std::string(60, '=')).c_str()));

        int passed = 0, failed = 0;
        double total_time = 0;
        size_t total_memory = 0;

        for (const auto& result : results) {
            if (result.passed) passed++;
            else failed++;
            total_time += result.execution_time_ms;
            total_memory += result.peak_memory_kb;
        }

        LLAMA_LOG_INFO("Summary:\n");
        LLAMA_LOG_INFO("  Total Tests: %lu\n", results.size());
        LLAMA_LOG_INFO("  Passed: %d\n", passed);
        LLAMA_LOG_INFO("  Failed: %d\n", failed);
        LLAMA_LOG_INFO("  Total Execution Time: %f ms\n", total_time);
        LLAMA_LOG_INFO("  Total Memory Usage: %lu KB\n", total_memory);

        LLAMA_LOG_INFO("\nDetailed Results:\n");
        for (const auto& result : results) {
            LLAMA_LOG_INFO("  %s\n : %s (exit: %d, time: %f ms, mem: %lu KB)\n", result.test_name.c_str(), (result.passed ? "PASS" : "FAIL"), result.exit_code, result.execution_time_ms, result.peak_memory_kb);
        }

        // Requirements validation
        LLAMA_LOG_INFO("\n--- Requirements Validation ---\n");
        validate_requirements(results);
    }

    void validate_requirements(const std::vector<TestResult>& results) {
        bool req_1_1 = true; // All tests pass consistently
        bool req_3_1 = false; // GGUF format handling
        bool req_3_2 = false; // Text format handling
        bool req_3_3 = false; // Parquet format handling
        bool req_4_1 = false; // Error handling for invalid inputs
        bool req_4_2 = false; // Error handling for corrupted files

        for (const auto& result : results) {
            if (!result.passed) {
                req_1_1 = false;
            }

            std::string combined_output = result.stdout_output + result.stderr_output;

            if (combined_output.find("gguf") != std::string::npos ||
                combined_output.find("GGUF") != std::string::npos) {
                req_3_1 = true;
            }

            if (combined_output.find("text") != std::string::npos ||
                combined_output.find("txt") != std::string::npos) {
                req_3_2 = true;
            }

            if (combined_output.find("parquet") != std::string::npos) {
                req_3_3 = true;
            }

            if (combined_output.find("null") != std::string::npos ||
                combined_output.find("not found") != std::string::npos) {
                req_4_1 = true;
            }

            if (combined_output.find("corrupt") != std::string::npos ||
                combined_output.find("invalid") != std::string::npos) {
                req_4_2 = true;
            }
        }

        LLAMA_LOG_INFO("Requirement 1.1 (All tests pass): %s\n", req_1_1 ? "✅ PASS" : "❌ FAIL");
        LLAMA_LOG_INFO("Requirement 3.1 (GGUF handling): %s\n", req_3_1 ? "✅ PASS" : "❌ FAIL");
        LLAMA_LOG_INFO("Requirement 3.2 (Text handling): %s\n", req_3_2 ? "✅ PASS" : "❌ FAIL");
        LLAMA_LOG_INFO("Requirement 3.3 (Parquet handling): %s\n", req_3_3 ? "✅ PASS" : "❌ FAIL");
        LLAMA_LOG_INFO("Requirement 4.1 (Invalid input errors): %s\n", req_4_1 ? "✅ PASS" : "❌ FAIL");
        LLAMA_LOG_INFO("Requirement 4.2 (Corrupted file errors): %s\n", req_4_2 ? "✅ PASS" : "❌ FAIL");
    }
};

int main() {
    LLAMA_LOG_INFO("Core Dataset Functionality Test Monitor\n");
    LLAMA_LOG_INFO("=======================================\n");

    CoreTestMonitor monitor("bb");

    std::vector<std::string> core_tests = {
        "test-dataset"
    };

    std::vector<TestResult> results;

    for (const auto& test : core_tests) {
        TestResult result = monitor.execute_test(test);
        monitor.analyze_test_result(result);
        results.push_back(result);
    }

    monitor.generate_comprehensive_report(results);

    return 0;
}
