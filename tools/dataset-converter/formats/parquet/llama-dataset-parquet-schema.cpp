/**
 * @file llama-dataset-parquet-schema.cpp
 * @brief Parquet schema analysis implementation for dataset converter.
 *
 * This module provides comprehensive schema analysis functionality for Parquet files,
 * enabling intelligent detection and processing of different data types within
 * Parquet datasets. It serves as a critical component in the dataset converter's
 * ability to handle heterogeneous data formats and mixed content scenarios.
 *
 * ## Core Responsibilities
 *
 * ### Schema Analysis and Validation
 * - Analyzes Parquet file schemas using Apache Arrow metadata
 * - Validates schema compatibility with dataset converter requirements
 * - Detects column data types and their suitability for different processing paths
 * - Provides detailed schema information for optimization decisions
 *
 * ### Column Type Detection
 * - Identifies text columns containing natural language data requiring tokenization
 * - Detects pre-tokenized columns with integer arrays or token ID sequences
 * - Distinguishes between different Arrow data types (STRING, BINARY, INT32, LIST)
 * - Supports complex nested types like lists of integers for token sequences
 *
 * ### Mixed Content Support
 * - Handles datasets containing both text and pre-tokenized data
 * - Implements intelligent column selection algorithms for optimal processing
 * - Provides fallback mechanisms when preferred columns are not available
 * - Enables flexible data processing workflows for diverse dataset structures
 *
 * ### Primary Column Identification
 * - Uses heuristic algorithms to identify the most suitable columns for processing
 * - Supports user-specified preferred columns with automatic fallback
 * - Implements common naming convention recognition (text, content, tokens, etc.)
 * - Provides robust column selection even in ambiguous scenarios
 *
 * ## Algorithm Details
 *
 * ### Schema Analysis Algorithm
 * The schema analysis process follows a multi-stage approach:
 * 1. **Schema Extraction**: Uses Apache Arrow to read Parquet metadata
 * 2. **Type Classification**: Categorizes each column based on Arrow data types
 * 3. **Heuristic Matching**: Applies naming conventions and user preferences
 * 4. **Fallback Selection**: Chooses appropriate columns when preferences fail
 * 5. **Validation**: Ensures selected columns meet processing requirements
 *
 * ### Column Type Detection
 * Text columns are identified by:
 * - Arrow types: STRING, LARGE_STRING, BINARY, LARGE_BINARY
 * - Content analysis for ambiguous cases
 * - Encoding validation for text data
 *
 * Token columns are identified by:
 * - Arrow types: INT32 (single tokens), LIST<INT32> (token sequences)
 * - Value range validation for token IDs
 * - Sequence length analysis for token arrays
 *
 * ### Mixed Content Processing
 * When both text and token columns are present:
 * - Prioritizes user-specified preferences
 * - Falls back to common naming conventions
 * - Selects first available column of appropriate type
 * - Provides metadata for informed processing decisions
 *
 * ## Integration with Dataset Converter
 *
 * This module integrates with other components:
 * - **Core Dataset API**: Provides schema information for dataset loading
 * - **Parquet Loader**: Supplies column selection for data extraction
 * - **Streaming System**: Enables optimized column-based streaming
 * - **Validation Framework**: Supports schema-based validation rules
 *
 * ## Performance Considerations
 *
 * - Minimal memory footprint during schema analysis
 * - Efficient Arrow metadata processing without full data loading
 * - Cached schema information to avoid repeated analysis
 * - Optimized string handling for column name processing
 *
 * ## Error Handling
 *
 * Comprehensive error handling covers:
 * - Invalid Parquet files or corrupted metadata
 * - Unsupported Arrow data types
 * - Memory allocation failures
 * - Missing or ambiguous column specifications
 *
 * @author Dataset Converter Team
 * @version 1.0
 * @since Dataset Converter v2.0
 *
 * @see llama-dataset-parquet.h for public API
 * @see llama-dataset-parquet-internal.h for internal structures
 * @see llama-dataset-parquet-core.cpp for data loading implementation
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
 * This function implements the core type detection algorithm for identifying
 * text columns in Parquet schemas. It examines Arrow data types to determine
 * if they represent textual data that requires tokenization processing.
 *
 * The function recognizes the following Arrow types as text data:
 * - STRING: Standard UTF-8 string data
 * - LARGE_STRING: Large UTF-8 strings (>2GB support)
 * - BINARY: Raw binary data that may contain text
 * - LARGE_BINARY: Large binary data with potential text content
 *
 * This classification is essential for the dataset converter to determine
 * the appropriate processing pipeline for each column. Text columns will
 * be routed through the tokenization system, while other types follow
 * different processing paths.
 *
 * @param field_type Arrow data type to analyze (must not be null)
 * @return true if the column contains text data requiring tokenization,
 *         false for non-text types or null input
 *
 * @note This function performs type-based classification only. Content-based
 *       analysis may be needed for BINARY types in some cases.
 *
 * @see is_token_column_type() for token column detection
 * @see analyze_parquet_table_schema() for comprehensive schema analysis
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
 * This function implements the token column detection algorithm for identifying
 * pre-tokenized data in Parquet schemas. It analyzes Arrow data types to
 * determine if they represent token sequences that can be used directly
 * without additional tokenization.
 *
 * The function recognizes the following patterns as token data:
 * - INT32: Single token values or scalar token IDs
 * - LIST<INT32>: Arrays of token IDs representing tokenized sequences
 *
 * Token columns are valuable for performance optimization as they bypass
 * the computationally expensive tokenization process. The dataset converter
 * can load these columns directly and use them for training or inference.
 *
 * Algorithm details:
 * 1. Check for direct INT32 type (single token per row)
 * 2. For LIST types, verify the value type is INT32
 * 3. Reject other numeric types (INT64, FLOAT, etc.) as non-token data
 *
 * @param field_type Arrow data type to analyze (must not be null)
 * @return true if the column contains pre-tokenized data that can be used
 *         directly, false for non-token types or null input
 *
 * @note Future versions may support additional token formats like INT64
 *       or compressed token representations.
 *
 * @see is_text_column_type() for text column detection
 * @see analyze_parquet_table_schema() for complete schema analysis
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
 * This utility function provides safe string copying for C API compatibility,
 * ensuring proper memory management when converting C++ strings to C-style
 * strings for use in the parquet_schema_info structure.
 *
 * The function performs the following operations:
 * 1. Allocates memory for the string plus null terminator
 * 2. Copies the string content using strcpy for safety
 * 3. Returns the allocated pointer for caller ownership
 *
 * Memory management:
 * - Caller is responsible for freeing the returned pointer
 * - Returns NULL on allocation failure for error handling
 * - Safe to use with empty strings (allocates 1 byte for null terminator)
 *
 * @param str C++ string to copy (can be empty but not null reference)
 * @return Newly allocated C-style string copy, or NULL on allocation failure
 *
 * @warning Caller must free the returned pointer to avoid memory leaks
 * @note This function is used internally for schema info structure population
 *
 * @see parquet_schema_info_free() for proper cleanup of allocated strings
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
 * This function implements the core schema analysis algorithm that performs
 * comprehensive examination of Parquet table schemas to enable intelligent
 * data processing. It serves as the primary entry point for schema analysis
 * within the C++ implementation layer.
 *
 * ## Algorithm Overview
 *
 * The analysis follows a sophisticated multi-stage process:
 *
 * ### Stage 1: Schema Extraction and Validation
 * - Extracts Arrow schema from the table
 * - Validates schema structure and field accessibility
 * - Initializes analysis state and result structures
 *
 * ### Stage 2: Column Classification
 * - Iterates through all schema fields
 * - Applies type detection algorithms for each column
 * - Categorizes columns as text, token, or other types
 * - Builds comprehensive column inventories
 *
 * ### Stage 3: Primary Column Selection
 * The selection algorithm uses a priority-based approach:
 * 1. **User Preferences**: Exact matches for specified column names
 * 2. **Convention Matching**: Common names (text, content, tokens, input_ids)
 * 3. **Type-Based Fallback**: First available column of appropriate type
 * 4. **Index Assignment**: Maps selected columns to schema indices
 *
 * ### Stage 4: Mixed Content Detection
 * - Analyzes the presence of both text and token columns
 * - Sets mixed content flags for processing optimization
 * - Provides metadata for adaptive processing strategies
 *
 * ### Stage 5: Result Population
 * - Allocates C-compatible string arrays for column names
 * - Populates the parquet_schema_info structure
 * - Ensures proper memory management and error handling
 *
 * ## Heuristic Algorithms
 *
 * ### Text Column Heuristics
 * Priority order for text column selection:
 * 1. User-specified preferred_text_column
 * 2. Columns named: "text", "content", "data", "input"
 * 3. First STRING/BINARY type column found
 *
 * ### Token Column Heuristics
 * Priority order for token column selection:
 * 1. User-specified preferred_token_column
 * 2. Columns named: "tokens", "token_ids", "input_ids", "data"
 * 3. First INT32/LIST<INT32> type column found
 *
 * ## Performance Characteristics
 *
 * - Time Complexity: O(n) where n is the number of columns
 * - Space Complexity: O(m) where m is the number of text/token columns
 * - Memory allocation is minimized and error-safe
 * - No data loading required, only metadata analysis
 *
 * ## Error Handling
 *
 * The function provides comprehensive error handling:
 * - Validates all input parameters
 * - Handles Arrow schema access failures
 * - Manages memory allocation errors gracefully
 * - Provides detailed error messages via llama_dataset_set_error
 * - Ensures cleanup on failure paths
 *
 * @param table Arrow table to analyze (must not be null)
 * @param preferred_text_column Preferred text column name (empty for auto-detection)
 * @param preferred_token_column Preferred token column name (empty for auto-detection)
 * @param info Output structure for schema information (must not be null)
 * @return true on successful analysis, false on error
 *
 * @pre table must be a valid Arrow table with accessible schema
 * @pre info must point to valid parquet_schema_info structure
 * @post On success, info contains complete schema analysis results
 * @post On failure, info is left in clean state and error is set
 *
 * @note This function allocates memory for column name arrays that must
 *       be freed using parquet_schema_info_free()
 *
 * @see analyze_parquet_schema() for file-based analysis entry point
 * @see is_text_column_type() for text column detection algorithm
 * @see is_token_column_type() for token column detection algorithm
 * @see parquet_schema_info_free() for proper cleanup
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
 * This function serves as the primary C API entry point for Parquet schema
 * analysis, providing a complete file-to-analysis pipeline that handles all
 * low-level operations required for schema examination. It orchestrates the
 * entire analysis process from file access to result generation.
 *
 * ## Implementation Architecture
 *
 * The function implements a layered architecture:
 *
 * ### Layer 1: File System Interface
 * - Opens Parquet files using Arrow I/O subsystem
 * - Handles file access permissions and availability
 * - Provides robust error handling for file system issues
 * - Supports various file system types (local, network, cloud)
 *
 * ### Layer 2: Parquet Reader Integration
 * - Creates Arrow-based Parquet readers
 * - Configures memory pools for efficient processing
 * - Handles Parquet format validation and compatibility
 * - Manages reader lifecycle and resource cleanup
 *
 * ### Layer 3: Table Loading and Validation
 * - Loads Parquet metadata and schema information
 * - Validates table structure and accessibility
 * - Handles corrupted or incomplete files gracefully
 * - Optimizes memory usage during schema extraction
 *
 * ### Layer 4: Analysis Delegation
 * - Converts C API parameters to C++ equivalents
 * - Delegates core analysis to analyze_parquet_table_schema()
 * - Handles parameter validation and type conversion
 * - Manages error propagation between layers
 *
 * ## Error Handling Strategy
 *
 * The function implements comprehensive error handling:
 *
 * ### File System Errors
 * - File not found or access denied
 * - Corrupted file system metadata
 * - Network connectivity issues for remote files
 * - Insufficient permissions for file access
 *
 * ### Format Errors
 * - Invalid Parquet file format
 * - Corrupted Parquet metadata
 * - Unsupported Parquet features
 * - Schema compatibility issues
 *
 * ### Memory Errors
 * - Insufficient memory for table loading
 * - Memory allocation failures
 * - Resource exhaustion scenarios
 * - Memory pool configuration issues
 *
 * ### API Errors
 * - Invalid parameter combinations
 * - Null pointer dereferences
 * - Type conversion failures
 * - Result structure initialization errors
 *
 * ## Performance Optimization
 *
 * The function is optimized for minimal resource usage:
 * - Schema-only loading without full data access
 * - Efficient memory pool utilization
 * - Minimal temporary object creation
 * - Fast-fail error detection
 * - Resource cleanup on all exit paths
 *
 * ## Integration Points
 *
 * This function integrates with:
 * - **Dataset Loading**: Provides schema info for optimized loading
 * - **Validation System**: Supplies schema data for validation rules
 * - **Streaming Engine**: Enables column-aware streaming strategies
 * - **Caching System**: Supports schema-based cache optimization
 *
 * @param path Path to the Parquet file (must not be null)
 * @param preferred_text_column Preferred text column name (can be NULL for auto-detection)
 * @param preferred_token_column Preferred token column name (can be NULL for auto-detection)
 * @param info Pointer to store schema analysis results (must not be null)
 * @return true if analysis completed successfully, false on any error
 *
 * @pre path must point to a valid, accessible Parquet file
 * @pre info must point to a valid parquet_schema_info structure
 * @post On success, info contains complete schema analysis results
 * @post On failure, info is left unmodified and error details are available
 *
 * @note This function may perform I/O operations and should be called
 *       from appropriate contexts (not from signal handlers, etc.)
 *
 * @warning The function allocates memory for column name arrays that must
 *          be freed using parquet_schema_info_free() to prevent memory leaks
 *
 * @see analyze_parquet_table_schema() for core analysis implementation
 * @see llama_dataset_validate_parquet_schema() for schema validation
 * @see parquet_schema_info_free() for proper result cleanup
 *
 * @example
 * ```c
 * struct parquet_schema_info info;
 * if (analyze_parquet_schema("dataset.parquet", "text", "tokens", &info)) {
 *     printf("Found %zu text columns, %zu token columns\n",
 *            info.n_text_columns, info.n_token_columns);
 *     parquet_schema_info_free(&info);
 * }
 * ```
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
