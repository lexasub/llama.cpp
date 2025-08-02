#include "test-data-validator-text.h"
#include "test-data-validator-common.h"
#include "test-data-validator-core.h"

#include "common/log.h"
#include "llama-dataset-internal.h"
#include "llama-impl.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

//
// Text-specific validation functions
//

static bool is_text_file_valid(const char* path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    // Check if file has at least one line of text
    std::string line;
    if (!std::getline(file, line)) {
        return false;
    }

    // Check if line is not empty and contains printable characters
    if (line.empty()) {
        return false;
    }

    // Basic check for printable ASCII characters
    for (char c : line) {
        if (c < 32 && c != '\t' && c != '\n' && c != '\r') {
            return false;
        }
    }

    return true;
}

enum test_data_validation_result validate_text_test_file(const char* path) {
    if (!path) {
        return TEST_DATA_INVALID_FORMAT;
    }

    if (!file_exists(path)) {
        return TEST_DATA_MISSING;
    }

    if (!check_file_readable_core(path)) {
        return TEST_DATA_PERMISSION_ERROR;
    }

    uint64_t size = get_file_size(path);
    if (size == 0) {
        return TEST_DATA_SIZE_INVALID;
    }

    if (!is_text_file_valid(path)) {
        return TEST_DATA_CONTENT_INVALID;
    }

    return TEST_DATA_VALID;
}

bool create_minimal_text_dataset(const char* path, uint64_t num_lines) {
    if (!path || num_lines == 0) {
        return false;
    }

    // Ensure directory exists
    std::string path_str(path);
    size_t pos = path_str.find_last_of('/');
    if (pos != std::string::npos) {
        std::string dir = path_str.substr(0, pos);
        if (!create_directory_if_missing(dir.c_str())) {
            return false;
        }
    }

    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    for (uint64_t i = 0; i < num_lines; i++) {
        file << "This is test line " << i << " for dataset converter testing." << std::endl;
    }

    file.close();

    // Set proper permissions
    fix_file_permissions_core(path);

    return true;
}

bool create_corrupted_text_test_file(const char* path) {
    if (!path) {
        return false;
    }

    // Ensure directory exists
    std::string path_str(path);
    size_t pos = path_str.find_last_of('/');
    if (pos != std::string::npos) {
        std::string dir = path_str.substr(0, pos);
        if (!create_directory_if_missing(dir.c_str())) {
            return false;
        }
    }

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Write binary data that's not valid text
    for (int i = 0; i < 100; i++) {
        char byte = static_cast<char>(i % 256);
        file.write(&byte, 1);
    }

    file.close();

    // Set proper permissions
    fix_file_permissions_core(path);

    return true;
}