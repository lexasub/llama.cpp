#include "test-data-validator-core.h"
#include "test-data-validator-common.h"
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

//
// Core validation orchestration functions
//

void init_validation_report(struct test_data_validation_report* report) {
    if (!report) {
        return;
    }
    
    memset(report, 0, sizeof(*report));
}

enum test_data_validation_result validate_test_file_core(const struct test_data_file_info* spec, 
                                                        struct test_data_validation_report* report) {
    if (!spec || !report) {
        return TEST_DATA_INVALID_FORMAT;
    }

    enum test_data_validation_result result;

    // Delegate to format-specific validation functions
    switch (spec->expected_type) {
        case DATASET_GGUF:
            result = validate_gguf_test_file(spec->path);
            break;
        case DATASET_TEXT:
            result = validate_text_test_file(spec->path);
            break;
        case DATASET_PARQUET:
            result = validate_parquet_test_file(spec->path);
            break;
        default:
            result = TEST_DATA_INVALID_FORMAT;
            break;
    }

    // Add error message if validation failed
    if (result != TEST_DATA_VALID) {
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg), "%s: %s\n",
                spec->path, test_data_validation_result_to_string(result));
        
        // Append to error messages if there's space
        size_t current_len = strlen(report->error_messages);
        size_t available_space = sizeof(report->error_messages) - current_len - 1;
        if (available_space > strlen(error_msg)) {
            strncat(report->error_messages, error_msg, available_space);
        }
    }

    return result;
}

bool create_test_file_core(const struct test_data_file_info* spec, 
                          struct test_data_validation_report* report) {
    if (!spec || !report) {
        return false;
    }

    if (!spec->create_if_missing) {
        return true; // Not supposed to create this file
    }

    // Check if file already exists
    if (file_exists(spec->path)) {
        return true; // Already exists
    }

    bool created = false;

    // Delegate to format-specific creation functions
    switch (spec->expected_type) {
        case DATASET_GGUF:
            if (strstr(spec->path, "corrupted") != nullptr) {
                created = create_corrupted_gguf_test_file(spec->path);
            } else {
                created = create_minimal_gguf_dataset(spec->path, 5, 10);
            }
            break;

        case DATASET_TEXT:
            if (strstr(spec->path, "large") != nullptr) {
                created = create_minimal_text_dataset(spec->path, 100);
            } else if (strstr(spec->path, "corrupted") != nullptr) {
                created = create_corrupted_text_test_file(spec->path);
            } else {
                created = create_minimal_text_dataset(spec->path, 5);
            }
            break;

        case DATASET_PARQUET:
            if (strstr(spec->path, "corrupted") != nullptr) {
                created = create_corrupted_parquet_test_file(spec->path);
            } else {
                created = create_minimal_parquet_dataset(spec->path, 5, 10);
            }
            break;

        default:
            created = false;
            break;
    }

    if (created) {
        report->created_files++;
    }

    return created;
}

bool fix_file_permissions_core(const char* path) {
    if (!path) {
        return false;
    }

    // Try to fix file permissions by making it readable
    if (chmod(path, 0644) == 0) {
        return true;
    }

    return false;
}

bool check_file_readable_core(const char* path) {
    if (!path) {
        return false;
    }

    // Check if file is readable
    return access(path, R_OK) == 0;
}

//
// Core error handling and utility functions
//

const char* test_data_validation_result_to_string(enum test_data_validation_result result) {
    switch (result) {
        case TEST_DATA_VALID:
            return "Valid";
        case TEST_DATA_MISSING:
            return "Missing";
        case TEST_DATA_CORRUPTED:
            return "Corrupted";
        case TEST_DATA_INVALID_FORMAT:
            return "Invalid format";
        case TEST_DATA_PERMISSION_ERROR:
            return "Permission error";
        case TEST_DATA_SIZE_INVALID:
            return "Invalid size";
        case TEST_DATA_CONTENT_INVALID:
            return "Invalid content";
        default:
            return "Unknown error";
    }
}

bool validate_file_spec_core(const struct test_data_file_info* spec) {
    if (!spec) {
        return false;
    }

    // Basic validation of file specification
    if (!spec->path || strlen(spec->path) == 0) {
        return false;
    }

    if (spec->min_size_bytes > spec->max_size_bytes && spec->max_size_bytes > 0) {
        return false;
    }

    return true;
}

void log_validation_error_core(const char* path, enum test_data_validation_result result) {
    if (!path) {
        return;
    }

    LLAMA_LOG_DEBUG("Validation error for %s: %s\n", path, test_data_validation_result_to_string(result));
}