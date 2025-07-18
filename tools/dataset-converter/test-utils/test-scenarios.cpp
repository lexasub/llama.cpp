#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "src/llama-impl.h"

// Simple test program that can simulate various test scenarios
int main(int argc, char* argv[]) {
    if (argc < 2) {
        LLAMA_LOG_INFO("Usage: %s  <scenario>\n", argv[0]);
        LLAMA_LOG_INFO("Scenarios:\n");
        LLAMA_LOG_INFO("  success - Normal successful execution\n");
        LLAMA_LOG_INFO("  failure - Exit with non-zero code\n");
        LLAMA_LOG_INFO("  crash - Segmentation fault\n");
        LLAMA_LOG_INFO("  timeout - Run for a long time\n");
        LLAMA_LOG_INFO("  memory - Allocate lots of memory\n");
        LLAMA_LOG_INFO("  output - Generate stdout/stderr output\n");
        return 1;
    }

    std::string scenario = argv[1];

    if (scenario == "success") {
        LLAMA_LOG_INFO("Test executing successfully...\n");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        LLAMA_LOG_INFO("Test completed successfully!\n");
        return 0;

    } else if (scenario == "failure") {
        LLAMA_LOG_INFO("Test executing...\n");
        LLAMA_LOG_ERROR("ERROR: Simulated test failure\n");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        return 1;

    } else if (scenario == "crash") {
        LLAMA_LOG_INFO("Test executing...\n");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        LLAMA_LOG_INFO("About to crash...\n");
        // Cause segmentation fault
        int* p = nullptr;
        *p = 42;
        return 0;  // Never reached

    } else if (scenario == "timeout") {
        LLAMA_LOG_INFO("Test executing (will run for a long time)...\n");
        for (int i = 0; i < 1000; i++) {
            LLAMA_LOG_INFO("Working... %d/1000\n", i);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        return 0;

    } else if (scenario == "memory") {
        LLAMA_LOG_INFO("Test executing (allocating memory)...\n");
        std::vector<std::vector<char>> memory_hog;

        for (int i = 0; i < 100; i++) {
            // Allocate 1MB chunks
            memory_hog.emplace_back(1024 * 1024, 'A' + (i % 26));
            LLAMA_LOG_INFO("Allocated %d MB\n", i + 1);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        LLAMA_LOG_INFO("Memory allocation test completed\n");
        return 0;

    } else if (scenario == "output") {
        LLAMA_LOG_INFO("=== STDOUT OUTPUT ===\n");
        LLAMA_LOG_INFO("This is normal output\n");
        LLAMA_LOG_INFO("Line 2 of output\n");
        LLAMA_LOG_INFO("Line 3 with some data: 12345\n");

        LLAMA_LOG_ERROR("=== STDERR OUTPUT ===\n");
        LLAMA_LOG_ERROR("This is error output\n");;
        LLAMA_LOG_ERROR("Warning: Something might be wrong\n");
        LLAMA_LOG_ERROR("Debug info: test_value=42\n");

        LLAMA_LOG_INFO("Mixed output test completed\n");
        return 0;

    } else {
        LLAMA_LOG_ERROR("Unknown scenario: %s\n", scenario.c_str());
        return 1;
    }
}
