/**
 * @file llama-dataset-parquet-schema.cpp
 * @brief Parquet schema analysis functionality for dataset converter.
 *
 * This module handles schema analysis for Parquet files, including:
 * - Column type detection (text vs token columns)
 * - Mixed content analysis
 * - Schema validation and metadata extraction
 * - Primary column identification
 *
 * The functions in this module are used to analyze Parquet file schemas
 * and determine the best approach for loading and processing the data.
 */

#ifdef LLAMA_PARQUET
#    include "common/common.h"
#    include "llama-dataset-internal.h"
#    include "llama-dataset-parquet-internal.h"
#    include "llama-dataset-parquet.h"
#    include "llama-dataset-utils.h"
#    include "llama-impl.h"

#    include <arrow/api.h>
#    include <arrow/io/api.h>
#    include <parquet/arrow/reader.h>

#    include <algorithm>
#    include <cstdio>
#    include <cstring>
#    include <memory>
#    include <string>
#    include <vector>

//
// Schema analysis helper functions
//

/**
 * @brief Check if a column contains text data based on its type.
 *
 * This function examines an Arrow data type to determine if it represents
 * text data that needs to be tokenized. Text columns typically contain
 * string or binary data that represents natural language text.
 *
 * @param field_type Arrow data type to check
 * @return true if the column contains text data, false otherwise
 */
bool is_text_column_type(const std::shared_ptr<arrow::DataType> & field_type) {
    if (!field_type) {
        return false;
    }

    switch (field_type->id()) {
        case arrow::Type::STRING:
        case arrow::Type::LARGE_STRING:
        case arrow::Type::BINARY:
        case arrow::Type::LARGE_BINARY:
            return true;
        default:
            return false;
    }
}

/**
 * @brief Check if a column contains token data based on its type.
 *
 * This function examines an Arrow data type to determine if it represents
 * pre-tokenized data. Token columns typically contain integer arrays or
 * lists of integers representing token IDs.
 *
 * @param field_type Arrow data type to check
 * @return true if the column contains token data, false otherwise
 */
bool is_token_column_type(const std::shared_ptr<arrow::DataType> & field_type) {
    if (!field_type) {
        return false;
    }

    switch (field_type->id()) {
        case arrow::Type::INT32:
            return true;
        case arrow::Type::LIST:
            {
                auto list_type = std::static_pointer_cast<arrow::ListType>(field_type);
                return list_type->value_type()->id() == arrow::Type::INT32;
            }
        default:
            return false;
    }
}

/**
 * @brief Helper function to allocate and copy string to C-style string array.
 *
 * This utility function creates a C-style string copy of a C++ string,
 * which is needed for the C API compatibility in the schema info structure.
 *
 * @param str C++ string to copy
 * @return Allocated C-style string, or NULL on allocation failure
 */
static char * copy_string_to_c(const std::string & str) {
    char * c_str = static_cast<char *>(malloc(str.length() + 1));
    if (c_str) {
        strcpy(c_str, str.c_str());
    }
    return c_str;
}

/**
 * @brief Free resources associated with parquet_schema_info.
 *
 * This function properly deallocates all memory associated with a
 * parquet_schema_info structure, including the arrays of column names.
 * It's safe to call this function multiple times or with NULL pointers.
 *
 * @param info Schema info structure to free
 */
void parquet_schema_info_free(struct parquet_schema_info * info) {
    if (!info) {
        return;
    }

    // Free text column names
    if (info->text_columns) {
        for (size_t i = 0; i < info->n_text_columns; i++) {
            free(info->text_columns[i]);
        }
        free(info->text_columns);
        info->text_columns = nullptr;
    }

    // Free token column names
    if (info->token_columns) {
        for (size_t i = 0; i < info->n_token_columns; i++) {
            free(info->token_columns[i]);
        }
        free(info->token_columns);
        info->token_columns = nullptr;
    }

    info->n_text_columns = 0;
    info->n_token_columns = 0;
    info->has_mixed_content = false;
    info->primary_text_column_index = -1;
    info->primary_token_column_index = -1;
}

/**
 * @brief Analyze Parquet table schema for mixed content support (C++ version).
 *
 * This function performs a comprehensive analysis of a Parquet table's schema
 * to identify text and token columns, determine primary columns for processing,
 * and detect mixed content scenarios where both text and pre-tokenized data
 * are present in the same file.
 *
 * The function uses heuristics to identify the most appropriate columns:
 * - Preferred columns specified by the user take priority
 * - Common column names are used as fallbacks (text, content, tokens, etc.)
 * - First available column of the appropriate type is used as last resort
 *
 * @param table Arrow table to analyze
 * @param preferred_text_column Preferred text column name (empty for auto-detection)
 * @param preferred_token_column Preferred token column name (empty for auto-detection)
 * @param info Output structure for schema information
 * @return true on success, false on error
 */
bool analyze_parquet_table_schema(
    const std::shared_ptr<arrow::Table> & table,
    const std::string & preferred_text_column,
    const std::string & preferred_token_column,
    struct parquet_schema_info * info
) {
    if (!table || !info) {
        llama_dataset_set_error("Invalid parameters for schema analysis");
        return false;
    }

    // Initialize the info structure
    memset(info, 0, sizeof(struct parquet_schema_info));
    info->primary_text_column_index = -1;
    info->primary_token_column_index = -1;

    auto schema = table->schema();
    if (!schema) {
        llama_dataset_set_error("Failed to get table schema");
        return false;
    }

    std::vector<std::string> text_column_names;
    std::vector<std::string> token_column_names;

    // Analyze each column in the schema
    for (int i = 0; i < schema->num_fields(); i++) {
        auto field = schema->field(i);
        std::string column_name = field->name();
        auto field_type = field->type();

        if (is_text_column_type(field_type)) {
            text_column_names.push_back(column_name);

            // Check if this is the preferred text column
            if (!preferred_text_column.empty() && column_name == preferred_text_column) {
                info->primary_text_column_index = i;
            }
            // If no preferred column specified, use common text column names
            else if (preferred_text_column.empty() && info->primary_text_column_index == -1) {
                if (column_name == "text" || column_name == "content" ||
                    column_name == "data" || column_name == "input") {
                    info->primary_text_column_index = i;
                }
            }
        }
        else if (is_token_column_type(field_type)) {
            token_column_names.push_back(column_name);

            // Check if this is the preferred token column
            if (!preferred_token_column.empty() && column_name == preferred_token_column) {
                info->primary_token_column_index = i;
            }
            // If no preferred column specified, use common token column names
            else if (preferred_token_column.empty() && info->primary_token_column_index == -1) {
                if (column_name == "tokens" || column_name == "token_ids" ||
                    column_name == "input_ids" || column_name == "data") {
                    info->primary_token_column_index = i;
                }
            }
        }
    }

    // Set fallback primary columns if none found by name matching
    if (info->primary_text_column_index == -1 && !text_column_names.empty()) {
        info->primary_text_column_index = 0; // Use first text column
        for (int i = 0; i < schema->num_fields(); i++) {
            if (is_text_column_type(schema->field(i)->type())) {
                info->primary_text_column_index = i;
                break;
            }
        }
    }

    if (info->primary_token_column_index == -1 && !token_column_names.empty()) {
        info->primary_token_column_index = 0; // Use first token column
        for (int i = 0; i < schema->num_fields(); i++) {
            if (is_token_column_type(schema->field(i)->type())) {
                info->primary_token_column_index = i;
                break;
            }
        }
    }

    // Allocate and copy text column names
    info->n_text_columns = text_column_names.size();
    if (info->n_text_columns > 0) {
        info->text_columns = static_cast<char **>(malloc(info->n_text_columns * sizeof(char *)));
        if (!info->text_columns) {
            llama_dataset_set_error("Failed to allocate memory for text column names");
            return false;
        }

        for (size_t i = 0; i < info->n_text_columns; i++) {
            info->text_columns[i] = copy_string_to_c(text_column_names[i]);
            if (!info->text_columns[i]) {
                llama_dataset_set_error("Failed to allocate memory for text column name");
                parquet_schema_info_free(info);
                return false;
            }
        }
    }

    // Allocate and copy token column names
    info->n_token_columns = token_column_names.size();
    if (info->n_token_columns > 0) {
        info->token_columns = static_cast<char **>(malloc(info->n_token_columns * sizeof(char *)));
        if (!info->token_columns) {
            llama_dataset_set_error("Failed to allocate memory for token column names");
            parquet_schema_info_free(info);
            return false;
        }

        for (size_t i = 0; i < info->n_token_columns; i++) {
            info->token_columns[i] = copy_string_to_c(token_column_names[i]);
            if (!info->token_columns[i]) {
                llama_dataset_set_error("Failed to allocate memory for token column name");
                parquet_schema_info_free(info);
                return false;
            }
        }
    }

    // Determine if we have mixed content
    info->has_mixed_content = (info->n_text_columns > 0 && info->n_token_columns > 0);

    LLAMA_LOG_INFO("Schema analysis complete: %zu text columns, %zu token columns, mixed=%s\n",
                   info->n_text_columns, info->n_token_columns,
                   info->has_mixed_content ? "true" : "false");

    return true;
}

/**
 * @brief Analyze Parquet file schema for mixed content support.
 *
 * This is the main entry point for schema analysis. It opens a Parquet file,
 * reads its schema, and performs comprehensive analysis to determine the
 * structure and content types of the columns.
 *
 * This function handles all the low-level Arrow/Parquet operations and
 * delegates the actual analysis to analyze_parquet_table_schema().
 *
 * @param path Path to the Parquet file
 * @param preferred_text_column Preferred text column name (can be NULL)
 * @param preferred_token_column Preferred token column name (can be NULL)
 * @param info Pointer to store schema analysis results
 * @return true if analysis successful, false otherwise
 */
bool analyze_parquet_schema(
    const char * path,
    const char * preferred_text_column,
    const char * preferred_token_column,
    struct parquet_schema_info * info
) {
    if (!path || !info) {
        llama_dataset_set_error("Invalid parameters for schema analysis");
        return false;
    }

    try {
        // Open Parquet file
        auto result = arrow::io::ReadableFile::Open(path);
        if (!result.ok()) {
            const auto msg = "Failed to open Parquet file for schema analysis: " + result.status().ToString();
            llama_dataset_set_error_with_code(DATASET_ERROR_IO_ERROR, msg.c_str());
            return false;
        }

        // Create Parquet reader
        auto reader = parquet::arrow::OpenFile(result.ValueOrDie(), arrow::default_memory_pool());
        if (!reader.ok()) {
            llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Failed to create Parquet reader for schema analysis");
            return false;
        }

        // Read table
        std::shared_ptr<arrow::Table> table;
        auto status = reader->get()->ReadTable(&table);
        if (!status.ok()) {
            llama_dataset_set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to read Parquet table for schema analysis");
            return false;
        }

        // Convert C strings to C++ strings
        std::string text_col = preferred_text_column ? std::string(preferred_text_column) : std::string();
        std::string token_col = preferred_token_column ? std::string(preferred_token_column) : std::string();

        // Analyze the table schema
        return analyze_parquet_table_schema(table, text_col, token_col, info);

    } catch (const std::exception & e) {
        const auto msg = "Schema analysis failed: " + std::string(e.what());
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, msg.c_str());
        return false;
    }
}

#endif // LLAMA_PARQUET
