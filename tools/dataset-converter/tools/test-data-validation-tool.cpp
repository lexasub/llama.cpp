#include "test-data-validator.h"
#include "../../common/log.h"

#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Standalone tool for validating and creating test data files.
 * 
 * This tool can be used to:
 * - Validate all test data files
 * - Create missing test data files
 * - Fix file permissions
 * - Generate validation reports
 */

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --validate          Validate all test data files\n";
    std::cout << "  --create-missing    Create missing test data files\n";
    std::cout << "  --fix-permissions   Fix file permissions\n";
    std::cout << "  --report            Generate and print validation report\n";
    std::cout << "  --all               Perform all operations (validate, create, fix)\n";
    std::cout << "  --test-dir <dir>    Specify test data directory (default: current)\n";
    std::cout << "  --help              Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << program_name << " --validate\n";
    std::cout << "  " << program_name << " --create-missing --fix-permissions\n";
    std::cout << "  " << program_name << " --all --test-dir /path/to/test/data\n";
}

int main(int argc, char* argv[]) {
    bool validate = false;
    bool create_missing = false;
    bool fix_permissions = false;
    bool generate_report = false;
    bool all_operations = false;
    std::string test_dir = ".";
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--validate") {
            validate = true;
        } else if (arg == "--create-missing") {
            create_missing = true;
        } else if (arg == "--fix-permissions") {
            fix_permissions = true;
        } else if (arg == "--report") {
            generate_report = true;
        } else if (arg == "--all") {
            all_operations = true;
        } else if (arg == "--test-dir" && i + 1 < argc) {
            test_dir = argv[++i];
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // If no specific operations requested, default to validation and report
    if (!validate && !create_missing && !fix_permissions && !generate_report && !all_operations) {
        validate = true;
        generate_report = true;
    }
    
    // If all operations requested, enable everything
    if (all_operations) {
        validate = true;
        create_missing = true;
        fix_permissions = true;
        generate_report = true;
    }
    
    std::cout << "Test Data Validation Tool\n";
    std::cout << "========================\n";
    std::cout << "Working directory: " << test_dir << "\n\n";
    
    // Initialize validator
    TestDataValidator validator(test_dir);
    
    // Check if test directory is accessible
    if (!validator.IsDirectoryAccessible()) {
        std::cerr << "Error: Test directory is not accessible: " << test_dir << std::endl;
        return 1;
    }
    
    bool success = true;
    
    // Create missing files first
    if (create_missing) {
        std::cout << "Creating missing test data files...\n";
        if (validator.CreateMissingFiles()) {
            std::cout << "✓ Successfully created missing files\n";
        } else {
            std::cout << "✗ Failed to create some missing files\n";
            success = false;
        }
        std::cout << "\n";
    }
    
    // Fix permissions
    if (fix_permissions) {
        std::cout << "Fixing file permissions...\n";
        if (validator.FixPermissions()) {
            std::cout << "✓ Successfully fixed file permissions\n";
        } else {
            std::cout << "✗ Failed to fix some file permissions\n";
            success = false;
        }
        std::cout << "\n";
    }
    
    // Validate files
    if (validate) {
        std::cout << "Validating test data files...\n";
        if (validator.ValidateAllFiles()) {
            std::cout << "✓ All required test data files are valid\n";
        } else {
            std::cout << "✗ Some test data files are invalid or missing\n";
            success = false;
        }
        std::cout << "\n";
    }
    
    // Generate report
    if (generate_report) {
        std::cout << "Generating validation report...\n";
        auto report = validator.GenerateReport();
        validator.PrintReport(report);
        std::cout << "\n";
    }
    
    // Show missing and corrupted files
    auto missing_files = validator.GetMissingFiles();
    if (!missing_files.empty()) {
        std::cout << "Missing files:\n";
        for (const auto& file : missing_files) {
            std::cout << "  - " << file << "\n";
        }
        std::cout << "\n";
    }
    
    auto corrupted_files = validator.GetCorruptedFiles();
    if (!corrupted_files.empty()) {
        std::cout << "Corrupted files:\n";
        for (const auto& file : corrupted_files) {
            std::cout << "  - " << file << "\n";
        }
        std::cout << "\n";
    }
    
    // Summary
    if (success) {
        std::cout << "✓ Test data validation completed successfully\n";
        return 0;
    } else {
        std::cout << "✗ Test data validation completed with errors\n";
        return 1;
    }
}