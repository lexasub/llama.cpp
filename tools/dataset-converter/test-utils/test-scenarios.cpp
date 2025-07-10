#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <signal.h>

// Simple test program that can simulate various test scenarios
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <scenario>\n";
        std::cout << "Scenarios:\n";
        std::cout << "  success - Normal successful execution\n";
        std::cout << "  failure - Exit with non-zero code\n";
        std::cout << "  crash - Segmentation fault\n";
        std::cout << "  timeout - Run for a long time\n";
        std::cout << "  memory - Allocate lots of memory\n";
        std::cout << "  output - Generate stdout/stderr output\n";
        return 1;
    }
    
    std::string scenario = argv[1];
    
    if (scenario == "success") {
        std::cout << "Test executing successfully...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cout << "Test completed successfully!\n";
        return 0;
        
    } else if (scenario == "failure") {
        std::cout << "Test executing...\n";
        std::cerr << "ERROR: Simulated test failure\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        return 1;
        
    } else if (scenario == "crash") {
        std::cout << "Test executing...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::cout << "About to crash...\n";
        // Cause segmentation fault
        int* p = nullptr;
        *p = 42;
        return 0;  // Never reached
        
    } else if (scenario == "timeout") {
        std::cout << "Test executing (will run for a long time)...\n";
        for (int i = 0; i < 1000; i++) {
            std::cout << "Working... " << i << "/1000\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        return 0;
        
    } else if (scenario == "memory") {
        std::cout << "Test executing (allocating memory)...\n";
        std::vector<std::vector<char>> memory_hog;
        
        for (int i = 0; i < 100; i++) {
            // Allocate 1MB chunks
            memory_hog.emplace_back(1024 * 1024, 'A' + (i % 26));
            std::cout << "Allocated " << (i + 1) << " MB\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        std::cout << "Memory allocation test completed\n";
        return 0;
        
    } else if (scenario == "output") {
        std::cout << "=== STDOUT OUTPUT ===\n";
        std::cout << "This is normal output\n";
        std::cout << "Line 2 of output\n";
        std::cout << "Line 3 with some data: 12345\n";
        
        std::cerr << "=== STDERR OUTPUT ===\n";
        std::cerr << "This is error output\n";
        std::cerr << "Warning: Something might be wrong\n";
        std::cerr << "Debug info: test_value=42\n";
        
        std::cout << "Mixed output test completed\n";
        return 0;
        
    } else {
        std::cerr << "Unknown scenario: " << scenario << "\n";
        return 1;
    }
}