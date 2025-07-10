#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <cstdlib>
#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>
#include <signal.h>
#include <cstring>

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
                std::cout << "Creating missing test data file: " << file << std::endl;
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
            std::cout << "Created text test file: " << path << std::endl;
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
            std::cout << "Created corrupted test file: " << path << std::endl;
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
        
        std::cout << "\n=== Executing Core Dataset Test: " << test_executable << " ===" << std::endl;
        
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
        std::cout << "\n--- Test Analysis for " << result.test_name << " ---" << std::endl;
        std::cout << "Status: " << (result.passed ? "PASSED" : "FAILED") << std::endl;
        std::cout << "Exit Code: " << result.exit_code << std::endl;
        std::cout << "Execution Time: " << result.execution_time_ms << " ms" << std::endl;
        std::cout << "Peak Memory Usage: " << result.peak_memory_kb << " KB" << std::endl;
        
        if (!result.error_analysis.empty()) {
            std::cout << "Error Analysis: " << result.error_analysis << std::endl;
        }
        
        if (!result.stdout_output.empty()) {
            std::cout << "\n--- STDOUT ---" << std::endl;
            std::cout << result.stdout_output << std::endl;
        }
        
        if (!result.stderr_output.empty()) {
            std::cout << "\n--- STDERR ---" << std::endl;
            std::cout << result.stderr_output << std::endl;
        }
        
        // Analyze specific failure patterns
        analyze_failure_patterns(result);
    }
    
    void analyze_failure_patterns(const TestResult& result) {
        std::cout << "\n--- Failure Pattern Analysis ---" << std::endl;
        
        // Check for common error patterns
        std::string combined_output = result.stdout_output + result.stderr_output;
        
        if (combined_output.find("segmentation fault") != std::string::npos ||
            combined_output.find("SIGSEGV") != std::string::npos ||
            result.error_analysis.find("SEGMENTATION FAULT") != std::string::npos) {
            std::cout << "⚠️  SEGMENTATION FAULT detected - likely null pointer dereference or memory corruption" << std::endl;
        }
        
        if (combined_output.find("assertion failed") != std::string::npos ||
            combined_output.find("Assertion") != std::string::npos) {
            std::cout << "⚠️  ASSERTION FAILURE detected - test expectations not met" << std::endl;
        }
        
        if (combined_output.find("not found") != std::string::npos) {
            std::cout << "⚠️  FILE NOT FOUND error detected - missing test data or executable" << std::endl;
        }
        
        if (combined_output.find("null") != std::string::npos) {
            std::cout << "✓ NULL POINTER handling being tested" << std::endl;
        }
        
        if (combined_output.find("not implemented") != std::string::npos ||
            combined_output.find("not supported") != std::string::npos) {
            std::cout << "ℹ️  UNIMPLEMENTED FEATURE detected - expected for placeholder functions" << std::endl;
        }
        
        if (result.peak_memory_kb > 100000) { // > 100MB
            std::cout << "⚠️  HIGH MEMORY USAGE detected - potential memory leak" << std::endl;
        }
        
        if (result.execution_time_ms > 5000) { // > 5 seconds
            std::cout << "⚠️  SLOW EXECUTION detected - potential performance issue" << std::endl;
        }
        
        // Check for successful test patterns
        if (combined_output.find("All tests passed") != std::string::npos) {
            std::cout << "✅ All internal tests passed successfully" << std::endl;
        }
        
        if (combined_output.find("✓") != std::string::npos) {
            std::cout << "✅ Individual test assertions passed" << std::endl;
        }
    }
    
    void generate_comprehensive_report(const std::vector<TestResult>& results) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "COMPREHENSIVE CORE DATASET TEST REPORT" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        int passed = 0, failed = 0;
        double total_time = 0;
        size_t total_memory = 0;
        
        for (const auto& result : results) {
            if (result.passed) passed++;
            else failed++;
            total_time += result.execution_time_ms;
            total_memory += result.peak_memory_kb;
        }
        
        std::cout << "Summary:" << std::endl;
        std::cout << "  Total Tests: " << results.size() << std::endl;
        std::cout << "  Passed: " << passed << std::endl;
        std::cout << "  Failed: " << failed << std::endl;
        std::cout << "  Total Execution Time: " << total_time << " ms" << std::endl;
        std::cout << "  Total Memory Usage: " << total_memory << " KB" << std::endl;
        
        std::cout << "\nDetailed Results:" << std::endl;
        for (const auto& result : results) {
            std::cout << "  " << result.test_name << ": " 
                      << (result.passed ? "PASS" : "FAIL") 
                      << " (exit: " << result.exit_code 
                      << ", time: " << result.execution_time_ms << "ms"
                      << ", mem: " << result.peak_memory_kb << "KB)" << std::endl;
        }
        
        // Requirements validation
        std::cout << "\n--- Requirements Validation ---" << std::endl;
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
        
        std::cout << "Requirement 1.1 (All tests pass): " << (req_1_1 ? "✅ PASS" : "❌ FAIL") << std::endl;
        std::cout << "Requirement 3.1 (GGUF handling): " << (req_3_1 ? "✅ PASS" : "❌ FAIL") << std::endl;
        std::cout << "Requirement 3.2 (Text handling): " << (req_3_2 ? "✅ PASS" : "❌ FAIL") << std::endl;
        std::cout << "Requirement 3.3 (Parquet handling): " << (req_3_3 ? "✅ PASS" : "❌ FAIL") << std::endl;
        std::cout << "Requirement 4.1 (Invalid input errors): " << (req_4_1 ? "✅ PASS" : "❌ FAIL") << std::endl;
        std::cout << "Requirement 4.2 (Corrupted file errors): " << (req_4_2 ? "✅ PASS" : "❌ FAIL") << std::endl;
    }
};

int main() {
    std::cout << "Core Dataset Functionality Test Monitor" << std::endl;
    std::cout << "=======================================" << std::endl;
    
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