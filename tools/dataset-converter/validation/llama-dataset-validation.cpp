#include "llama-dataset-validation.h"
#include "llama-dataset-utils.h"
#include "../../common/log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

/**
 * @brief Validate GGUF file format and detect corruption.
 */
bool validate_gguf_file(const char* path, struct validation_result* result) {
    if (!path || !result) {
        if (result) {
            result->is_valid = false;
            result->error_code = DATASET_ERROR_INVALID_PARAMETER;
            strncpy(result->error_message, "Invalid parameters for GGUF validation", sizeof(result->error_message) - 1);
            result->error_message[sizeof(result->error_message) - 1] = '\0';
        }
        return false;
    }

    // Initialize result
    memset(result, 0, sizeof(struct validation_result));
    result->is_valid = false;
    result->error_code = DATASET_SUCCESS;

    FILE* file = fopen(path, "rb");
    if (!file) {
        result->error_code = DATASET_ERROR_FILE_NOT_FOUND;
        strncpy(result->error_message, "GGUF file not found", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return false;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    result->file_size = (uint64_t)file_size;

    // Check minimum file size
    if (file_size < 32) {
        fclose(file);
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        strncpy(result->error_message, "GGUF file too small to be valid", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return false;
    }

    // Read and validate GGUF magic number
    char magic[4];
    if (fread(magic, 1, 4, file) != 4) {
        fclose(file);
        result->error_code = DATASET_ERROR_IO_ERROR;
        strncpy(result->error_message, "Failed to read GGUF magic bytes", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return false;
    }

    // Check GGUF magic bytes
    if (memcmp(magic, "GGUF", 4) != 0) {
        fclose(file);
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        snprintf(result->error_message, sizeof(result->error_message), 
                "Invalid GGUF magic bytes: expected 'GGUF', got '%.4s'", magic);
        return false;
    }

    // Read version
    uint32_t version;
    if (fread(&version, sizeof(uint32_t), 1, file) != 1) {
        fclose(file);
        result->error_code = DATASET_ERROR_IO_ERROR;
        strncpy(result->error_message, "Failed to read GGUF version", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return false;
    }

    result->format_version = version;

    // Validate version (currently support versions 1, 2, 3)
    if (version < 1 || version > 3) {
        fclose(file);
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        snprintf(result->error_message, sizeof(result->error_message), 
                "Unsupported GGUF version: %u (supported: 1-3)", version);
        return false;
    }

    // Read tensor count and kv count
    uint64_t tensor_count, kv_count;
    if (fread(&tensor_count, sizeof(uint64_t), 1, file) != 1 ||
        fread(&kv_count, sizeof(uint64_t), 1, file) != 1) {
        fclose(file);
        result->error_code = DATASET_ERROR_IO_ERROR;
        strncpy(result->error_message, "Failed to read GGUF tensor/kv counts", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return false;
    }

    // Basic sanity checks
    if (tensor_count > 1000000) {
        fclose(file);
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        snprintf(result->error_message, sizeof(result->error_message), 
                "Suspicious tensor count: %llu (max: 1000000)", (unsigned long long)tensor_count);
        return false;
    }

    if (kv_count > 1000000) {
        fclose(file);
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        snprintf(result->error_message, sizeof(result->error_message), 
                "Suspicious key-value count: %llu (max: 1000000)", (unsigned long long)kv_count);
        return false;
    }

    // Additional corruption checks
    // Check if file size is reasonable for the claimed tensor count
    size_t min_expected_size = 32 + (tensor_count * 4) + (kv_count * 8); // Very rough estimate
    if ((uint64_t)file_size < min_expected_size) {
        fclose(file);
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        snprintf(result->error_message, sizeof(result->error_message), 
                "File size (%llu) too small for claimed content (%zu tensors, %zu kvs)", 
                (unsigned long long)file_size, (size_t)tensor_count, (size_t)kv_count);
        return false;
    }

    fclose(file);

    // If we get here, the file appears to be valid
    result->is_valid = true;
    strncpy(result->error_message, "GGUF file validation passed", sizeof(result->error_message) - 1);
    result->error_message[sizeof(result->error_message) - 1] = '\0';
    return true;
}

/**
 * @brief Validate Parquet file format and detect corruption.
 */
bool validate_parquet_file(const char* path, struct validation_result* result) {
    if (!path || !result) {
        if (result) {
            result->is_valid = false;
            result->error_code = DATASET_ERROR_INVALID_PARAMETER;
            strncpy(result->error_message, "Invalid parameters for Parquet validation", sizeof(result->error_message) - 1);
            result->error_message[sizeof(result->error_message) - 1] = '\0';
        }
        return false;
    }

    // Initialize result
    memset(result, 0, sizeof(struct validation_result));
    result->is_valid = false;
    result->error_code = DATASET_SUCCESS;

    FILE* file = fopen(path, "rb");
    if (!file) {
        result->error_code = DATASET_ERROR_FILE_NOT_FOUND;
        strncpy(result->error_message, "Parquet file not found", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return false;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    result->file_size = (uint64_t)file_size;

    // Check minimum file size for Parquet
    if (file_size < 12) {
        fclose(file);
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        strncpy(result->error_message, "Parquet file too small to be valid", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return false;
    }

    // Check Parquet magic bytes at the end of the file
    fseek(file, -4, SEEK_END);
    char magic[4];
    if (fread(magic, 1, 4, file) != 4) {
        fclose(file);
        result->error_code = DATASET_ERROR_IO_ERROR;
        strncpy(result->error_message, "Failed to read Parquet magic bytes", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return false;
    }

    // Check for Parquet magic bytes "PAR1"
    if (memcmp(magic, "PAR1", 4) != 0) {
        fclose(file);
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        snprintf(result->error_message, sizeof(result->error_message), 
                "Invalid Parquet magic bytes: expected 'PAR1', got '%.4s'", magic);
        return false;
    }

    fclose(file);

    // If we get here, the file appears to be valid
    result->is_valid = true;
    strncpy(result->error_message, "Parquet file validation passed", sizeof(result->error_message) - 1);
    result->error_message[sizeof(result->error_message) - 1] = '\0';
    return true;
}

/**
 * @brief Validate text file format and detect issues.
 */
bool validate_text_file(const char* path, struct validation_result* result) {
    if (!path || !result) {
        if (result) {
            result->is_valid = false;
            result->error_code = DATASET_ERROR_INVALID_PARAMETER;
            strncpy(result->error_message, "Invalid parameters for text validation", sizeof(result->error_message) - 1);
            result->error_message[sizeof(result->error_message) - 1] = '\0';
        }
        return false;
    }

    // Initialize result
    memset(result, 0, sizeof(struct validation_result));
    result->is_valid = false;
    result->error_code = DATASET_SUCCESS;

    FILE* file = fopen(path, "rb");
    if (!file) {
        result->error_code = DATASET_ERROR_FILE_NOT_FOUND;
        strncpy(result->error_message, "Text file not found", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return false;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    result->file_size = (uint64_t)file_size;

    // Check if file is empty
    if (file_size == 0) {
        fclose(file);
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        strncpy(result->error_message, "Text file is empty", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return false;
    }

    // Check for binary content (basic heuristic)
    fseek(file, 0, SEEK_SET);
    char buffer[1024];
    size_t bytes_read = fread(buffer, 1, sizeof(buffer), file);
    
    // Count null bytes and non-printable characters
    size_t null_count = 0;
    size_t non_printable_count = 0;
    
    for (size_t i = 0; i < bytes_read; i++) {
        if (buffer[i] == 0) {
            null_count++;
        } else if (buffer[i] < 32 && buffer[i] != '\n' && buffer[i] != '\r' && buffer[i] != '\t') {
            non_printable_count++;
        }
    }

    fclose(file);

    // If more than 10% of the sample contains null bytes, it's likely binary
    if (null_count > bytes_read / 10) {
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        snprintf(result->error_message, sizeof(result->error_message), 
                "File appears to be binary (contains %zu null bytes in %zu byte sample)", 
                null_count, bytes_read);
        return false;
    }

    // If more than 20% contains non-printable characters, it might be corrupted
    if (non_printable_count > bytes_read / 5) {
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        snprintf(result->error_message, sizeof(result->error_message), 
                "File contains suspicious non-printable characters (%zu in %zu byte sample)", 
                non_printable_count, bytes_read);
        return false;
    }

    // If we get here, the file appears to be valid text
    result->is_valid = true;
    strncpy(result->error_message, "Text file validation passed", sizeof(result->error_message) - 1);
    result->error_message[sizeof(result->error_message) - 1] = '\0';
    return true;
}

/**
 * @brief Validate dataset file based on extension or content.
 */
bool validate_dataset_file(const char* path, struct validation_result* result) {
    if (!path || !result) {
        if (result) {
            result->is_valid = false;
            result->error_code = DATASET_ERROR_INVALID_PARAMETER;
            strncpy(result->error_message, "Invalid parameters for dataset validation", sizeof(result->error_message) - 1);
            result->error_message[sizeof(result->error_message) - 1] = '\0';
        }
        return false;
    }

    // Determine file type based on extension
    const char* ext = strrchr(path, '.');
    if (!ext) {
        // No extension, try to detect format by content
        return validate_gguf_file(path, result) || 
               validate_parquet_file(path, result) || 
               validate_text_file(path, result);
    }

    // Check specific formats based on extension
    if (strcmp(ext, ".gguf") == 0) {
        return validate_gguf_file(path, result);
    } else if (strcmp(ext, ".parquet") == 0) {
        return validate_parquet_file(path, result);
    } else if (strcmp(ext, ".txt") == 0 || strcmp(ext, ".text") == 0) {
        return validate_text_file(path, result);
    } else {
        // Unknown extension, try all formats
        return validate_gguf_file(path, result) || 
               validate_parquet_file(path, result) || 
               validate_text_file(path, result);
    }
}

/**
 * @brief Check if a file appears to be corrupted based on basic checks.
 */
bool is_file_corrupted(const char* path) {
    if (!path) {
        return true; // Null path is considered "corrupted"
    }

    struct validation_result result;
    return !validate_dataset_file(path, &result);
}

/**
 * @brief Get a human-readable description of validation result.
 */
const char* validation_result_description(const struct validation_result* result) {
    if (!result) {
        return "Invalid validation result";
    }

    return result->error_message;
}