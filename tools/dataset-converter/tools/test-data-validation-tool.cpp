/**
 * @file test-data-validation-tool.cpp
 * @brief Comprehensive test data validation and management command-line tool.
 *
 * This standalone tool provides complete test data validation, creation, and management
 * capabilities for the dataset converter test suite. It ensures data integrity across
 * all supported formats (GGUF, Parquet, Text) and provides detailed validation reports
 * to maintain test data quality and consistency.
 *
 * ## Tool Purpose
 *
 * ### Primary Functions
 * - **Data Validation**: Comprehensive validation of test dataset files for integrity,
 *   format compliance, and content consistency across all supported formats
 * - **Missing File Creation**: Automatic generation of missing test data files with
 *   appropriate content and format specifications
 * - **Permission Management**: File permission fixing and access control management
 *   for test data directories and files
 * - **Validation Reporting**: Detailed validation reports with error diagnostics,
 *   statistics, and recommendations for issue resolution
 *
 * ### Validation Capabilities
 *
 * #### Format-Specific Validation
 * - **GGUF Files**: Header validation, metadata consistency checks, tensor data
 *   integrity verification, and format version compatibility
 * - **Parquet Files**: Schema validation, column type verification, data consistency
 *   checks, and Apache Arrow compatibility validation
 * - **Text Files**: Encoding validation, content structure verification, tokenization
 *   compatibility, and character set consistency
 *
 * #### Integrity Checks
 * - File existence and accessibility verification
 * - File size and format validation against expected specifications
 * - Corruption detection using format-specific algorithms
 * - Cross-file consistency and compatibility validation
 * - Permission and ownership verification
 *
 * #### Content Validation
 * - Data type consistency and range validation
 * - Sequence length and structure verification
 * - Cross-format compatibility checks
 * - Performance benchmark data validation
 * - Test scenario coverage verification
 *
 * ## Command-Line Interface
 *
 * ### Available Operations
 * - `--validate`: Perform comprehensive validation of all test data files
 * - `--create-missing`: Generate missing test data files with appropriate content
 * - `--fix-permissions`: Fix file permissions and access control settings
 * - `--report`: Generate and display detailed validation reports
 * - `--all`: Execute all operations in sequence (validate, create, fix)
 * - `--test-dir <dir>`: Specify custom test data directory path
 * - `--help`: Display usage information and command examples
 *
 * ### Usage Examples
 * ```bash
 * # Basic validation of current directory
 * ./test-data-validation-tool --validate
 *
 * # Create missing files and fix permissions
 * ./test-data-validation-tool --create-missing --fix-permissions
 *
 * # Complete validation and management with custom directory
 * ./test-data-validation-tool --all --test-dir /path/to/test/data
 *
 * # Generate detailed validation report
 * ./test-data-validation-tool --report --test-dir /path/to/test/data
 * ```
 *
 * ## Validation Criteria
 *
 * ### File-Level Validation
 * - File existence and accessibility checks
 * - File size validation against expected ranges
 * - Basic corruption detection using checksums and format headers
 * - Permission and ownership verification
 * - Directory structure validation
 *
 * ### Format-Specific Criteria
 * - **GGUF**: Magic number validation, header structure, metadata consistency,
 *   tensor data integrity, alignment requirements
 * - **Parquet**: Schema validation, column types, data consistency, metadata
 *   validation, compression integrity
 * - **Text**: Character encoding validation, content structure, line endings,
 *   tokenization compatibility, size constraints
 *
 * ### Content Quality Checks
 * - Data range and type validation
 * - Sequence structure and length verification
 * - Cross-format compatibility validation
 * - Performance characteristics validation
 * - Test coverage completeness
 *
 * ## Error Handling and Reporting
 *
 * ### Validation Results
 * The tool provides comprehensive validation results including:
 * - Overall validation status (pass/fail)
 * - File-by-file validation results with detailed status
 * - Error messages with specific failure reasons
 * - Warnings for potential issues or inconsistencies
 * - Statistics on file counts, sizes, and validation coverage
 *
 * ### Error Categories
 * - **Critical Errors**: Missing required files, severe corruption, access failures
 * - **Format Errors**: Invalid file formats, schema mismatches, encoding issues
 * - **Content Warnings**: Unusual data patterns, size anomalies, performance concerns
 * - **Permission Issues**: Access control problems, ownership mismatches
 *
 * ### Diagnostic Information
 * - Detailed error descriptions with context
 * - Suggested remediation steps for common issues
 * - File paths and line numbers for specific problems
 * - Performance metrics and validation timing
 *
 * ## Integration with Test Framework
 *
 * ### Automated Validation
 * - Integration with continuous integration pipelines
 * - Pre-test validation to ensure data integrity
 * - Post-test validation to detect data corruption
 * - Automated test data regeneration when needed
 *
 * ### Test Data Management
 * - Automatic creation of missing test files
 * - Version control integration for test data tracking
 * - Test data cleanup and maintenance operations
 * - Cross-platform compatibility validation
 *
 * ## Performance Considerations
 *
 * ### Optimization Features
 * - Parallel validation of multiple files when possible
 * - Incremental validation for large datasets
 * - Memory-efficient processing for large files
 * - Caching of validation results for repeated operations
 *
 * ### Scalability
 * - Support for large test data directories
 * - Efficient handling of numerous small files
 * - Progress reporting for long-running operations
 * - Resource usage monitoring and limits
 *
 * ## Dependencies
 *
 * ### Core Dependencies
 * - `test-data-validator.h`: Core validation framework and TestDataValidator class
 * - `llama-impl.h`: Logging and utility functions from llama.cpp core
 * - Standard C++ libraries: iostream, string, vector for basic operations
 *
 * ### Validation Modules
 * - Format-specific validators for GGUF, Parquet, and Text formats
 * - Core validation infrastructure for common operations
 * - Platform compatibility layer for cross-platform support
 *
 * @author Dataset Converter Team
 * @version 1.0
 * @since 2024
 *
 * @see test-data-validator.h for the core validation framework
 * @see llama-dataset-validation.h for low-level validation functions
 * @see TestDataValidator class for C++ validation interface
 */

#include <iostream>
#include <string>
#include <vector>

#include "llama-impl.h"
#include "test-data-validator.h"
/**
 * @brief Display comprehensive usage information and command-line help.
 *
 * Prints detailed usage information including all available command-line options,
 * their descriptions, and practical usage examples. The help text covers all
 * validation operations, file management features, and common usage patterns.
 *
 * The usage information includes:
 * - Complete list of command-line options with descriptions
 * - Default behavior when no options are specified
 * - Practical examples for common use cases
 * - Information about default directories and file paths
 * - Exit codes and error handling behavior
 *
 * @param program_name The name of the program executable (typically argv[0])
 *                     Used to display context-appropriate usage examples
 *
 * @note This function outputs to LLAMA_LOG_INFO for consistent logging
 * @note The function provides examples using the actual program name
 */
void print_usage(const char* program_name);

void print_usage(const char* program_name) {
    LLAMA_LOG_INFO("Usage: %s [options] [test_data_directories...]\n", program_name);
    LLAMA_LOG_INFO("\nOptions:\n");
    LLAMA_LOG_INFO("  --validate          Validate all test data files\n");
    LLAMA_LOG_INFO("  --create-missing    Create missing test data files\n");
    LLAMA_LOG_INFO("  --fix-permissions   Fix file permissions\n");
    LLAMA_LOG_INFO("  --report            Generate and print validation report\n");
    LLAMA_LOG_INFO("  --all               Perform all operations (validate, create, fix)\n");
    LLAMA_LOG_INFO("  --test-dir <dir>    Specify test data directory (can be used multiple times)\n");
    LLAMA_LOG_INFO("  --help              Show this help message\n");
    LLAMA_LOG_INFO("\nArguments:\n");
    LLAMA_LOG_INFO("  test_data_directories   One or more test data directory paths (default: current directory)\n");
    LLAMA_LOG_INFO("\nExamples:\n");
    LLAMA_LOG_INFO("  %s --validate\n", program_name);
    LLAMA_LOG_INFO("  %s --create-missing --fix-permissions\n", program_name);
    LLAMA_LOG_INFO("  %s --all --test-dir /path/to/test/data\n", program_name);
    LLAMA_LOG_INFO("  %s --validate test_data tools/test_data\n", program_name);
    LLAMA_LOG_INFO("  %s --all /path/to/test1 /path/to/test2 /path/to/test3\n", program_name);
}

/**
 * @brief Main entry point for the test data validation tool.
 *
 * Processes command-line arguments and executes the requested validation and
 * management operations on test data files. The function coordinates all
 * validation activities, manages the TestDataValidator instance, and provides
 * comprehensive reporting of results.
 *
 * ## Operation Flow
 *
 * ### Argument Processing
 * 1. Parses command-line arguments to determine requested operations
 * 2. Validates argument combinations and sets default behaviors
 * 3. Configures test data directory path and operation flags
 * 4. Handles help requests and invalid argument combinations
 *
 * ### Validation Operations
 * The operations are performed in the following order:
 * 1. **File Creation**: Creates missing test data files if requested
 * 2. **Permission Fixing**: Corrects file permissions and access control
 * 3. **Validation**: Performs comprehensive validation of all test files
 * 4. **Reporting**: Generates and displays detailed validation reports
 *
 * ### Error Handling
 * - Validates test directory accessibility before operations
 * - Tracks success/failure status for each operation
 * - Provides detailed error messages for operation failures
 * - Returns appropriate exit codes for automation integration
 *
 * ## Default Behavior
 * When no specific operations are requested, the tool defaults to:
 * - Validation of all test data files
 * - Generation of a validation report
 * - Working in the current directory
 *
 * ## Exit Codes
 * - **0**: All operations completed successfully, all files valid
 * - **1**: Operation failures or validation errors detected
 *
 * ## Output Format
 * The tool provides structured output including:
 * - Operation status indicators (✓ for success, ✗ for failure)
 * - Detailed validation reports with file-by-file results
 * - Lists of missing and corrupted files
 * - Summary statistics and final status
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return Exit code: 0 for success, 1 for errors or validation failures
 *
 * @note The function uses LLAMA_LOG_INFO and LLAMA_LOG_ERROR for output
 * @note All operations are performed through the TestDataValidator class
 * @see TestDataValidator for detailed validation functionality
 * @see print_usage() for command-line argument documentation
 */
int main(int argc, char* argv[]) {
    bool validate = false;
    bool create_missing = false;
    bool fix_permissions = false;
    bool generate_report = false;
    bool all_operations = false;
    std::vector<std::string> test_dirs;

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
            test_dirs.push_back(argv[++i]);
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg.substr(0, 2) != "--") {
            // Treat non-option arguments as test data directory paths
            test_dirs.push_back(arg);
        } else {
            LLAMA_LOG_ERROR("Unknown option: %s \n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    // Use default directory if none provided
    if (test_dirs.empty()) {
        test_dirs.push_back(".");
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
    LLAMA_LOG_INFO("Processing %zu test data directories:\n", test_dirs.size());
    for (const auto& dir : test_dirs) {
        LLAMA_LOG_INFO("  %s\n", dir.c_str());
    }
    LLAMA_LOG_INFO("\n");

    bool overall_success = true;

    // Process each test data directory
    for (const auto& test_dir : test_dirs) {
        LLAMA_LOG_INFO("=== Processing directory: %s ===\n", test_dir.c_str());
        
        // Initialize validator for this directory
        TestDataValidator validator(test_dir);

        // Check if test directory is accessible
        if (!validator.IsDirectoryAccessible()) {
            LLAMA_LOG_ERROR("Test directory is not accessible: %s\n", test_dir.c_str());
            overall_success = false;
            continue;
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

        if (!success) {
            overall_success = false;
        }

        LLAMA_LOG_INFO("=== Completed directory: %s ===\n\n", test_dir.c_str());
    }

    // Summary
    if (overall_success) {
        LLAMA_LOG_INFO("✓ Test data validation completed successfully for all directories\n");
        return 0;
    } else {
        LLAMA_LOG_INFO("✗ Test data validation completed with errors in one or more directories\n");
        return 1;
    }
}
