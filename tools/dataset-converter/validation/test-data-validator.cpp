#include "test-data-validator.h"
#include "test-data-validator-core.h"
#include "test-data-validator-gguf.h"
#include "test-data-validator-parquet.h"
#include "test-data-validator-text.h"

#include "common/log.h"
#include "llama-dataset-internal.h"
#include "llama-impl.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

// Default test data file specifications
static const struct test_data_file_info default_test_data_specs[] = {
    // Required files
    {"test_data/text_dataset.txt", DATASET_TEXT, 50, 10240, 0, true, true},
    {"test_data/small_dataset.gguf", DATASET_GGUF, 1024, 1048576, 0, true, true},

    // Optional but useful files
    {"test_data/parquet_dataset.parquet", DATASET_PARQUET, 1024, 1048576, 0, false, true},
    {"test_data/large_text_dataset.txt", DATASET_TEXT, 1024, 102400, 0, false, true},

    // Corrupted files for error testing
    {"test_data/corrupted_dataset.gguf", DATASET_GGUF, 1, 1024, 0, false, true},
    {"test_data/corrupted_dataset.parquet", DATASET_PARQUET, 1, 1024, 0, false, true},

    // Test data in tools directory
    {"tools/dataset-converter/tests/test_data/text_dataset.txt", DATASET_TEXT, 50, 10240, 0, true, true},
    {"tools/dataset-converter/tests/test_data/corrupted_dataset.gguf", DATASET_GGUF, 1, 1024, 0, false, true},
    {"tools/dataset-converter/tests/test_data/corrupted_dataset.parquet", DATASET_PARQUET, 1, 1024, 0, false, true}
};

static constexpr int default_test_data_specs_count = std::size(default_test_data_specs);

// Note: Format-specific validation functions have been moved to separate modules:
// - test-data-validator-gguf.cpp for GGUF format functions
// - test-data-validator-parquet.cpp for Parquet format functions
// - test-data-validator-text.cpp for Text format functions
// - test-data-validator-common.cpp for shared helper functions

//
// Generic wrapper functions for format-specific operations
//

bool create_corrupted_test_file(const char* path, enum dataset_type type) {
    switch (type) {
        case DATASET_GGUF:
            return create_corrupted_gguf_test_file(path);
        case DATASET_PARQUET:
            return create_corrupted_parquet_test_file(path);
        case DATASET_TEXT:
            return create_corrupted_text_test_file(path);
        default:
            return false;
    }
}

//
// Local helper functions for high-level orchestration
// Note: file_exists and directory_exists are now available from test-data-validator-common.h
//

//
// Comprehensive validation functions implementation
//

bool validate_all_test_data(const char* test_data_dir, struct test_data_validation_report* report) {
    if (!test_data_dir || !report) {
        return false;
    }

    // Initialize report using core function
    init_validation_report(report);

    bool all_valid = true;

    for (int i = 0; i < default_test_data_specs_count; i++) {
        const struct test_data_file_info* spec = &default_test_data_specs[i];
        report->total_files_checked++;

        // Use core validation function
        enum test_data_validation_result result = validate_test_file_core(spec, report);

        switch (result) {
            case TEST_DATA_VALID:
                report->valid_files++;
                break;
            case TEST_DATA_MISSING:
                report->missing_files++;
                if (spec->required) {
                    all_valid = false;
                }
                break;
            case TEST_DATA_CORRUPTED:
            case TEST_DATA_INVALID_FORMAT:
            case TEST_DATA_CONTENT_INVALID:
            case TEST_DATA_SIZE_INVALID:
                report->corrupted_files++;
                if (spec->required) {
                    all_valid = false;
                }
                break;
            case TEST_DATA_PERMISSION_ERROR:
                // Try to fix permissions using core function
                if (fix_file_permissions_core(spec->path)) {
                    report->permission_fixes++;
                    report->valid_files++;
                } else {
                    if (spec->required) {
                        all_valid = false;
                    }
                }
                break;
        }
    }

    return all_valid;
}

bool create_missing_test_data(const char* test_data_dir, struct test_data_validation_report* report) {
    if (!test_data_dir || !report) {
        return false;
    }

    bool all_created = true;

    for (int i = 0; i < default_test_data_specs_count; i++) {
        const struct test_data_file_info* spec = &default_test_data_specs[i];

        // Use core creation function
        if (!create_test_file_core(spec, report)) {
            all_created = false;
        }
    }

    return all_created;
}

// Note: test_data_validation_result_to_string function is now in test-data-validator-core.cpp

void print_validation_report(const struct test_data_validation_report* report) {
    if (!report) {
        return;
    }

    printf("=== Test Data Validation Report ===\n");
    printf("Total files checked: %d\n", report->total_files_checked);
    printf("Valid files: %d\n", report->valid_files);
    printf("Missing files: %d\n", report->missing_files);
    printf("Corrupted files: %d\n", report->corrupted_files);
    printf("Created files: %d\n", report->created_files);
    printf("Permission fixes: %d\n", report->permission_fixes);

    if (strlen(report->error_messages) > 0) {
        printf("\nErrors:\n%s", report->error_messages);
    }

    printf("===================================\n");
}

const struct test_data_file_info* get_default_test_data_specs(int* count) {
    if (count) {
        *count = default_test_data_specs_count;
    }
    return default_test_data_specs;
}

//
// C++ wrapper implementation
//

#ifdef __cplusplus

TestDataValidator::TestDataValidator(const std::string& test_data_dir)
    : test_data_dir_(test_data_dir) {

    // Copy default specs
    int count;
    const struct test_data_file_info* specs = get_default_test_data_specs(&count);
    file_specs_.assign(specs, specs + count);
}

bool TestDataValidator::ValidateAllFiles() {
    struct test_data_validation_report report;
    return validate_all_test_data(test_data_dir_.c_str(), &report);
}

bool TestDataValidator::CreateMissingFiles() {
    struct test_data_validation_report report;
    return create_missing_test_data(test_data_dir_.c_str(), &report);
}

bool TestDataValidator::FixPermissions() {
    bool all_fixed = true;

    for (const auto& spec : file_specs_) {
        if (file_exists(spec.path) && !check_file_readable_core(spec.path)) {
            if (!fix_file_permissions_core(spec.path)) {
                all_fixed = false;
            }
        }
    }

    return all_fixed;
}

test_data_validation_result TestDataValidator::ValidateFile(const std::string& path, dataset_type type) {
    switch (type) {
        case DATASET_GGUF:
            return validate_gguf_test_file(path.c_str());
        case DATASET_TEXT:
            return validate_text_test_file(path.c_str());
        case DATASET_PARQUET:
            return validate_parquet_test_file(path.c_str());
        default:
            return TEST_DATA_INVALID_FORMAT;
    }
}

test_data_validation_report TestDataValidator::GenerateReport() {
    struct test_data_validation_report report;
    validate_all_test_data(test_data_dir_.c_str(), &report);
    return report;
}

void TestDataValidator::PrintReport(const test_data_validation_report& report) {
    print_validation_report(&report);
}

bool TestDataValidator::CreateMinimalDataset(const std::string& path, dataset_type type,
                                           uint64_t num_sequences, int32_t sequence_length) {
    switch (type) {
        case DATASET_GGUF:
            return create_minimal_gguf_dataset(path.c_str(), num_sequences, sequence_length);
        case DATASET_TEXT:
            return create_minimal_text_dataset(path.c_str(), num_sequences);
        case DATASET_PARQUET:
            return create_minimal_parquet_dataset(path.c_str(), num_sequences, sequence_length);
        default:
            return false;
    }
}

std::vector<std::string> TestDataValidator::GetMissingFiles() {
    std::vector<std::string> missing;

    for (const auto& spec : file_specs_) {
        if (!file_exists(spec.path)) {
            missing.push_back(spec.path);
        }
    }

    return missing;
}

std::vector<std::string> TestDataValidator::GetCorruptedFiles() {
    std::vector<std::string> corrupted;

    for (const auto& spec : file_specs_) {
        if (file_exists(spec.path)) {
            test_data_validation_result result = ValidateFile(spec.path, spec.expected_type);
            if (result == TEST_DATA_CORRUPTED || result == TEST_DATA_INVALID_FORMAT ||
                result == TEST_DATA_CONTENT_INVALID || result == TEST_DATA_SIZE_INVALID) {
                corrupted.push_back(spec.path);
            }
        }
    }

    return corrupted;
}

bool TestDataValidator::IsDirectoryAccessible() {
    return directory_exists(test_data_dir_.c_str()) && access(test_data_dir_.c_str(), R_OK | W_OK) == 0;
}

#endif // __cplusplus
