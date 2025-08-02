#include "test-data-validator-parquet.h"
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
// Parquet-specific validation functions
//

static bool is_parquet_file_valid(const char* path) {
    // Basic check for Parquet file format
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Check file size (Parquet files should be at least a few hundred bytes)
    file.seekg(0, std::ios::end);
    uint64_t size = file.tellg();
    if (size < 100) {
        return false;
    }

    // Check for Parquet magic at the end of file
    file.seekg(-4, std::ios::end);
    char magic[4];
    file.read(magic, 4);
    if (file.gcount() != 4) {
        return false;
    }

    return memcmp(magic, "PAR1", 4) == 0;
}

enum test_data_validation_result validate_parquet_test_file(const char* path) {
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
    if (size < 100) {  // Parquet files should be at least 100 bytes
        return TEST_DATA_SIZE_INVALID;
    }

    if (!is_parquet_file_valid(path)) {
        return TEST_DATA_CORRUPTED;
    }

    return TEST_DATA_VALID;
}

bool create_minimal_parquet_dataset(const char* path, uint64_t num_sequences, int32_t sequence_length) {
    if (!path || num_sequences == 0 || sequence_length <= 0) {
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

    // Create a minimal Parquet file (this is a simplified implementation)
    // In practice, you'd use the Arrow/Parquet library
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Write minimal Parquet header
    file.write("PAR1", 4);  // Magic at beginning

    // Write some dummy data (this is not a real Parquet file, just for testing)
    for (uint64_t i = 0; i < num_sequences; i++) {
        for (int32_t j = 0; j < sequence_length; j++) {
            int32_t token = static_cast<int32_t>(i * 1000 + j);
            file.write(reinterpret_cast<const char*>(&token), sizeof(token));
        }
    }

    // Write Parquet footer magic
    file.write("PAR1", 4);  // Magic at end

    file.close();

    // Set proper permissions
    fix_file_permissions_core(path);

    return true;
}

bool create_corrupted_parquet_test_file(const char* path) {
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

    // Write invalid Parquet data
    file.write("XXXX", 4);
    file.write("corrupted parquet data", 22);

    file.close();

    // Set proper permissions
    fix_file_permissions_core(path);

    return true;
}