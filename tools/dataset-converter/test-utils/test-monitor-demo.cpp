#include "test-execution-monitor.h"
#include <iostream>
#include <vector>
#include <filesystem>

using namespace llama_dataset;

int main(int argc, char* argv[]) {
    std::cout << "Dataset Test Execution Monitor Demo\n";
    std::cout << "===================================\n\n";
    
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
    
    std::cout << "Working directory: " << config.working_directory << "\n";
    std::cout << "Timeout: " << config.timeout_seconds << " seconds\n";
    std::cout << "Memory monitoring: " << (config.capture_memory_usage ? "enabled" : "disabled") << "\n";
    std::cout << "Crash detection: " << (config.enable_crash_detection ? "enabled" : "disabled") << "\n\n";
    
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
        std::cout << "Discovering test executables in: " << config.working_directory << "\n";
        test_executables = discover_test_executables(config.working_directory);
        
        if (test_executables.empty()) {
            std::cout << "No test executables found. Checking common build directories...\n";
            
            // Check common build directories
            std::vector<std::string> build_dirs = {"bb", "build", "cmake-build-debug", "cmake-build-release"};
            for (const auto& build_dir : build_dirs) {
                if (std::filesystem::exists(build_dir)) {
                    std::cout << "Checking " << build_dir << "...\n";
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
        std::cout << "No test executables found!\n";
        std::cout << "Usage: " << argv[0] << " [working_directory] [test_executable1] [test_executable2] ...\n";
        return 1;
    }
    
    std::cout << "Found " << test_executables.size() << " test executable(s):\n";
    for (const auto& test : test_executables) {
        std::cout << "  - " << test << "\n";
    }
    std::cout << "\n";
    
    // Execute tests
    std::cout << "Executing tests...\n";
    std::cout << "==================\n\n";
    
    std::vector<TestExecutionResult> results;
    
    for (const auto& test_executable : test_executables) {
        std::cout << "Running: " << test_executable << "\n";
        
        if (!monitor.is_executable_available(test_executable)) {
            std::cout << "  ERROR: Executable not found or not executable\n\n";
            continue;
        }
        
        auto result = monitor.execute_test(test_executable);
        results.push_back(result);
        
        std::cout << "  Result: " << (result.passed ? "PASSED" : "FAILED") << "\n";
        std::cout << "  Exit code: " << result.exit_code << "\n";
        std::cout << "  Execution time: " << format_duration(result.execution_time_ms) << "\n";
        std::cout << "  Peak memory: " << format_memory_size(result.peak_memory_usage_bytes) << "\n";
        
        if (result.crashed) {
            std::cout << "  CRASHED: " << result.error_message << "\n";
        }
        
        if (result.timeout_occurred) {
            std::cout << "  TIMEOUT: Test exceeded time limit\n";
        }
        
        if (!result.error_message.empty() && !result.crashed) {
            std::cout << "  Error: " << result.error_message << "\n";
        }
        
        std::cout << "\n";
    }
    
    // Generate comprehensive report
    std::cout << "\n" << monitor.generate_execution_report(results) << "\n";
    
    // Summary statistics
    int passed = 0, failed = 0;
    for (const auto& result : results) {
        if (result.passed) passed++;
        else failed++;
    }
    
    std::cout << "Final Summary:\n";
    std::cout << "=============\n";
    std::cout << "Total tests executed: " << results.size() << "\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n";
    std::cout << "Success rate: " << (results.empty() ? 0 : (passed * 100 / results.size())) << "%\n";
    
    return failed > 0 ? 1 : 0;
}