#include <filesystem>
#include <iostream>
#include <vector>

#include "src/llama-impl.h"
#include "test-execution-monitor.h"

using namespace llama_dataset;

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

        LLAMA_LOG_INFO("\n";
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
