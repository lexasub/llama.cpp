#include <iostream>
#include <string>
#include <vector>

#include "llama-impl.h"
#include "test-data-validator.h"

/**
 * @brief Standalone tool for validating and creating test data files.
 *
 * This tool can be used to:
 * - Validate all test data files
 * - Create missing test data files
 * - Fix file permissions
 * - Generate validation reports
 */
void print_usage(const char* program_name);
void print_usage(const char* program_name) {
    LLAMA_LOG_INFO("Usage: %s  [options]\n", program_name);
    LLAMA_LOG_INFO("\nOptions:\n");
    LLAMA_LOG_INFO("  --validate          Validate all test data files\n");
    LLAMA_LOG_INFO("  --create-missing    Create missing test data files\n");
    LLAMA_LOG_INFO("  --fix-permissions   Fix file permissions\n");
    LLAMA_LOG_INFO("  --report            Generate and print validation report\n");
    LLAMA_LOG_INFO("  --all               Perform all operations (validate, create, fix)\n");
    LLAMA_LOG_INFO("  --test-dir <dir>    Specify test data directory (default: current)\n");
    LLAMA_LOG_INFO("  --help              Show this help message\n");
    LLAMA_LOG_INFO("\nExamples:\n");
    LLAMA_LOG_INFO("  %s --validate\n", program_name);
    LLAMA_LOG_INFO("  %s --create-missing --fix-permissions\n", program_name);
    LLAMA_LOG_INFO("  %s --all --test-dir /path/to/test/data\n", program_name);
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
            LLAMA_LOG_ERROR("Unknown option: %s \n", arg.c_str());
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

    LLAMA_LOG_INFO("Test Data Validation Tool\n");
    LLAMA_LOG_INFO("========================\n");
    LLAMA_LOG_INFO("Working directory: %s\n\n", test_dir.c_str());

    // Initialize validator
    TestDataValidator validator(test_dir);

    // Check if test directory is accessible
    if (!validator.IsDirectoryAccessible()) {
        LLAMA_LOG_ERROR("Test directory is not accessible: %s\n", test_dir.c_str());
        return 1;
    }

    bool success = true;

    // Create missing files first
    if (create_missing) {
        LLAMA_LOG_INFO("Creating missing test data files...\n");
        if (validator.CreateMissingFiles()) {
            LLAMA_LOG_INFO("✓ Successfully created missing files\n");
        } else {
            LLAMA_LOG_INFO("✗ Failed to create some missing files\n");
            success = false;
        }
        LLAMA_LOG_INFO("\n");
    }

    // Fix permissions
    if (fix_permissions) {
        LLAMA_LOG_INFO("Fixing file permissions...\n");
        if (validator.FixPermissions()) {
            LLAMA_LOG_INFO("✓ Successfully fixed file permissions\n");
        } else {
            LLAMA_LOG_INFO("✗ Failed to fix some file permissions\n");
            success = false;
        }
        LLAMA_LOG_INFO("\n");
    }

    // Validate files
    if (validate) {
        LLAMA_LOG_INFO("Validating test data files...\n");
        if (validator.ValidateAllFiles()) {
            LLAMA_LOG_INFO("✓ All required test data files are valid\n");
        } else {
            LLAMA_LOG_INFO("✗ Some test data files are invalid or missing\n");
            success = false;
        }
        LLAMA_LOG_INFO("\n");
    }

    // Generate report
    if (generate_report) {
        LLAMA_LOG_INFO("Generating validation report...\n");
        auto report = validator.GenerateReport();
        validator.PrintReport(report);
        LLAMA_LOG_INFO("\n");
    }

    // Show missing and corrupted files
    auto missing_files = validator.GetMissingFiles();
    if (!missing_files.empty()) {
        LLAMA_LOG_INFO("Missing files:\n");
        for (const auto& file : missing_files) {
            LLAMA_LOG_INFO("  - %s\n", file.c_str());
        }
        LLAMA_LOG_INFO("\n");
    }

    auto corrupted_files = validator.GetCorruptedFiles();
    if (!corrupted_files.empty()) {
        LLAMA_LOG_INFO("Corrupted files:\n");
        for (const auto& file : corrupted_files) {
            LLAMA_LOG_INFO("  - %s\n", file.c_str());
        }
        LLAMA_LOG_INFO("\n");
    }

    // Summary
    if (success) {
        LLAMA_LOG_INFO("✓ Test data validation completed successfully\n");
        return 0;
    } else {
        LLAMA_LOG_INFO("✗ Test data validation completed with errors\n");
        return 1;
    }
}
