#pragma once

/**
 * @file test-data-validator-core.h
 * @brief Core orchestration functions for test data validation.
 *
 * This header provides the main coordination functions that orchestrate
 * validation across different format-specific modules. It contains the
 * core logic for managing validation reports, coordinating between
 * format-specific validators, and handling common validation workflows.
 */

#include "test-data-validator-common.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// Core validation orchestration functions
//

/**
 * @brief Initialize a validation report structure.
 *
 * @param report Pointer to the validation report to initialize
 */
void init_validation_report(struct test_data_validation_report* report);

/**
 * @brief Validate a single test file using the appropriate format-specific validator.
 *
 * This function coordinates validation by delegating to the appropriate
 * format-specific validation function based on the file specification.
 *
 * @param spec Pointer to the file specification
 * @param report Pointer to the validation report (for error logging)
 * @return Validation result code
 */
enum test_data_validation_result validate_test_file_core(const struct test_data_file_info* spec, 
                                                        struct test_data_validation_report* report);

/**
 * @brief Create a test file using the appropriate format-specific creator.
 *
 * This function coordinates file creation by delegating to the appropriate
 * format-specific creation function based on the file specification.
 *
 * @param spec Pointer to the file specification
 * @param report Pointer to the validation report (for tracking created files)
 * @return true if file was created successfully, false otherwise
 */
bool create_test_file_core(const struct test_data_file_info* spec, 
                          struct test_data_validation_report* report);

/**
 * @brief Fix file permissions using core permission handling.
 *
 * @param path Path to the file
 * @return true if permissions were fixed successfully, false otherwise
 */
bool fix_file_permissions_core(const char* path);

/**
 * @brief Check if a file is readable using core permission checking.
 *
 * @param path Path to the file
 * @return true if file is readable, false otherwise
 */
bool check_file_readable_core(const char* path);

//
// Core error handling and utility functions
//

/**
 * @brief Get a string representation of a validation result.
 *
 * @param result Validation result code
 * @return String representation of the result
 */
const char* test_data_validation_result_to_string(enum test_data_validation_result result);

/**
 * @brief Validate a file specification structure.
 *
 * @param spec Pointer to the file specification to validate
 * @return true if specification is valid, false otherwise
 */
bool validate_file_spec_core(const struct test_data_file_info* spec);

/**
 * @brief Log a validation error using the core logging system.
 *
 * @param path Path to the file that failed validation
 * @param result Validation result code
 */
void log_validation_error_core(const char* path, enum test_data_validation_result result);

#ifdef __cplusplus
}
#endif