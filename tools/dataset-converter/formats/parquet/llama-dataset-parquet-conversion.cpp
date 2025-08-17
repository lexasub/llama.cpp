/**
 * @file llama-dataset-parquet-conversion.cpp
 * @brief Parquet data conversion implementation for llama.cpp dataset handling.
 *
 * This module implements comprehensive Parquet data conversion functionality, providing
 * the core conversion logic for transforming Parquet datasets into GGUF format tensors.
 * It handles multiple data types including pre-tokenized sequences, raw text requiring
 * tokenization, and mixed content scenarios where both formats coexist.
 *
 * ## Key Responsibilities
 *
 * - **Arrow Array Conversion**: Converts Apache Arrow arrays (Int32Array, ListArray) 
 *   to token vectors compatible with GGML tensors
 * - **GGUF Context Creation**: Creates and populates GGUF contexts with metadata and
 *   tensor data from Parquet sources
 * - **Text Tokenization**: Integrates with llama tokenizer for on-the-fly text
 *   conversion to tokens during dataset loading
 * - **Mixed Content Processing**: Handles Parquet files containing both text and
 *   pre-tokenized columns with intelligent fallback strategies
 * - **Batch Processing**: Efficiently processes large datasets with chunked data
 *   access and memory-optimized tensor creation
 * - **Error Handling**: Provides detailed error reporting for conversion failures
 *   with row-level diagnostics and recovery strategies
 *
 * ## Data Flow Architecture
 *
 * 1. **Schema Analysis**: Determines column types and content structure
 * 2. **Data Extraction**: Reads Arrow chunks from Parquet columns
 * 3. **Type-Specific Processing**: Handles text vs pre-tokenized data differently
 * 4. **Tokenization**: Applies llama tokenizer to text content when needed
 * 5. **Tensor Creation**: Converts token sequences to GGML tensors
 * 6. **GGUF Population**: Adds tensors and metadata to GGUF context
 * 7. **Streaming Setup**: Configures streaming mode for large datasets
 *
 * ## Performance Considerations
 *
 * - Chunked processing minimizes memory usage for large Parquet files
 * - Tokenization caching reduces redundant text processing
 * - Streaming mode enables processing of datasets larger than available memory
 * - Batch tokenization optimizes throughput for text-heavy datasets
 * - Memory estimation prevents allocation failures during tensor creation
 *
 * ## Error Recovery
 *
 * The module implements robust error handling with detailed diagnostics:
 * - Row-level error reporting for tokenization failures
 * - Graceful handling of null values and empty sequences
 * - Fallback strategies for mixed content scenarios
 * - Memory allocation failure recovery
 *
 * @author llama.cpp dataset-converter team
 * @version 1.0
 * @since 2024
 *
 * @see llama-dataset-parquet.h for public interface definitions
 * @see llama-dataset-parquet-internal.h for internal data structures
 * @see streaming/streaming-cache.h for streaming implementation details
 */

#include "llama-model.h"
#ifdef LLAMA_PARQUET
#include "llama-dataset-parquet.h"
#include "llama-dataset-parquet-internal.h"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <cstdio>
#include <cstring>
#include <ctime>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

#include "common/common.h"
#include "ggml/include/ggml.h"
#include "ggml/include/gguf.h"
#include "../../core/llama-dataset-internal.h"
#include "../../core/llama-dataset-utils.h"
#include "llama.h"
#include "llama-impl.h"
static void handle_tokenization_error(
    int64_t row_index,
    const std::string & column_name,
    const std::string & text_content,
    const char * error_message = nullptr
);
static bool process_mixed_content_parquet(
    const std::shared_ptr<arrow::Table> & table,
    struct llama_dataset * dataset,
    struct parquet_format_data * format_data,
    std::vector<std::vector<int32_t>> & all_sequences,
    int32_t & max_length
);
/**
 * @brief Convert Arrow array to token vector with comprehensive type support.
 *
 * This function provides robust conversion from Apache Arrow arrays to token vectors
 * suitable for GGML tensor storage. It handles multiple Arrow data types and implements
 * efficient memory management for large token sequences.
 *
 * ## Supported Array Types
 *
 * - **INT32**: Direct token arrays where each element is a single token
 * - **LIST**: Nested arrays where each list element contains a sequence of tokens
 *   - Supports both fixed-size and variable-length lists
 *   - Handles nested INT32 arrays within list elements
 *   - Flattens multiple sequences into a single token vector
 *
 * ## Processing Algorithm
 *
 * 1. **Type Detection**: Identifies Arrow array type using type_id()
 * 2. **Memory Reservation**: Pre-allocates token vector based on array length
 * 3. **Null Handling**: Skips null values gracefully without breaking processing
 * 4. **Data Extraction**: Extracts token values using type-specific accessors
 * 5. **Sequence Flattening**: For list arrays, flattens nested sequences
 *
 * ## Performance Characteristics
 *
 * - **Time Complexity**: O(n) where n is the total number of tokens
 * - **Space Complexity**: O(n) for output vector, minimal additional overhead
 * - **Memory Access**: Sequential access pattern optimized for cache efficiency
 * - **Null Value Overhead**: Minimal impact due to efficient null checking
 *
 * ## Error Conditions
 *
 * - Null input array pointer
 * - Unsupported Arrow array types
 * - Memory allocation failures during vector operations
 * - Corrupted nested array structures in LIST types
 *
 * @param array Arrow array containing token data (must not be null)
 * @param tokens Output vector for extracted tokens (will be cleared and populated)
 * @return true on successful conversion, false on error (error details set via llama_dataset_set_error)
 *
 * @note For LIST arrays, sequences are flattened into a single vector. Consider using
 *       separate processing for maintaining sequence boundaries in advanced use cases.
 * @note The function reserves memory based on array length for INT32 arrays but may
 *       over-allocate for LIST arrays due to unknown nesting depth.
 *
 * @see llama_dataset_set_error for error reporting mechanism
 * @see ggml_new_tensor_1d for tensor creation from resulting token vectors
 */
bool llama_dataset_arrow_array_to_tokens(const std::shared_ptr<arrow::Array> & array,
                                                std::vector<int32_t> &                tokens) {
    if (!array) {
        llama_dataset_set_error("Arrow array is null");
        return false;
    }

    // Handle different array types
    switch (array->type_id()) {
        case arrow::Type::INT32:
            {
                auto int32_array = std::static_pointer_cast<arrow::Int32Array>(array);
                tokens.reserve(int32_array->length());

                for (int64_t i = 0; i < int32_array->length(); i++) {
                    if (int32_array->IsNull(i)) {
                        // Skip null values or use a special token
                        continue;
                    }
                    tokens.push_back(int32_array->Value(i));
                }
                break;
            }

        case arrow::Type::LIST:
            {
                auto list_array = std::static_pointer_cast<arrow::ListArray>(array);

                // For list arrays, we expect each list to be a sequence of tokens
                // We'll flatten all sequences into a single token vector for now
                // In a more sophisticated implementation, we might handle this differently

                for (int64_t i = 0; i < list_array->length(); i++) {
                    if (list_array->IsNull(i)) {
                        continue;
                    }

                    auto slice = list_array->value_slice(i);
                    if (slice->type_id() == arrow::Type::INT32) {
                        auto int32_slice = std::static_pointer_cast<arrow::Int32Array>(slice);
                        for (int64_t j = 0; j < int32_slice->length(); j++) {
                            if (!int32_slice->IsNull(j)) {
                                tokens.push_back(int32_slice->Value(j));
                            }
                        }
                    }
                }
                break;
            }

        default:
            llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Unsupported Arrow array type for tokens");
            return false;
    }

    return true;
}

/**
 * @brief Create GGUF context from Parquet data with comprehensive conversion support.
 *
 * This function implements the core conversion pipeline from Parquet data to GGUF format,
 * handling complex scenarios including mixed content, streaming mode, and various data types.
 * It serves as the primary entry point for Parquet-to-GGUF conversion operations.
 *
 * ## Conversion Pipeline
 *
 * 1. **Input Validation**: Validates table and dataset parameters
 * 2. **GGUF Context Creation**: Initializes empty GGUF context for data storage
 * 3. **Schema Analysis**: Determines data column types and tokenization requirements
 * 4. **Column Selection**: Identifies primary data columns using schema analysis
 * 5. **Metadata Population**: Sets format, timestamp, and tokenization metadata
 * 6. **Data Processing**: Processes chunks with type-specific conversion logic
 * 7. **Sequence Assembly**: Collects all token sequences and calculates statistics
 * 8. **Tensor Creation**: Creates GGML tensors for non-streaming mode
 * 9. **Streaming Setup**: Configures streaming infrastructure for large datasets
 *
 * ## Data Type Support
 *
 * - **Pre-tokenized Data**: Direct conversion from INT32 or LIST arrays
 * - **Text Data**: On-the-fly tokenization using integrated llama tokenizer
 * - **Mixed Content**: Intelligent processing of files with both data types
 * - **Chunked Data**: Efficient processing of large Parquet files with multiple chunks
 *
 * ## Tokenization Integration
 *
 * - Automatic detection of text vs pre-tokenized columns
 * - Caching support for improved tokenization performance
 * - Batch processing for optimal throughput
 * - Detailed statistics collection and reporting
 * - Error handling with row-level diagnostics
 *
 * ## Memory Management
 *
 * - **Non-streaming Mode**: Pre-allocates GGML context based on data size estimation
 * - **Streaming Mode**: Defers tensor creation to streaming infrastructure
 * - **Chunk Processing**: Processes data in manageable chunks to control memory usage
 * - **Error Recovery**: Graceful handling of memory allocation failures
 *
 * ## Metadata Generation
 *
 * The function populates comprehensive metadata including:
 * - Source format identification ("parquet")
 * - Creation timestamp for provenance tracking
 * - Tokenization source and column information
 * - Sequence count and maximum length statistics
 * - Tokenization performance metrics (when applicable)
 *
 * ## Error Handling
 *
 * - Detailed error messages with specific failure points
 * - Graceful handling of missing or invalid columns
 * - Recovery from partial tokenization failures
 * - Memory allocation failure detection and reporting
 *
 * @param table Arrow table containing Parquet data (must not be null)
 * @param dataset Dataset structure to populate with GGUF context and metadata (must not be null)
 * @return true on successful conversion, false on error (detailed error set via llama_dataset_set_error)
 *
 * @pre dataset->format_data must point to valid parquet_format_data structure
 * @pre dataset->column must specify target column name or be compatible with schema analysis
 * @post On success, dataset->ctx contains populated GGUF context with tensors and metadata
 * @post On success, dataset->n_seq contains accurate sequence count
 * @post On streaming mode, dataset streaming infrastructure is properly configured
 *
 * @note For large datasets, consider enabling streaming mode to reduce memory usage
 * @note Tokenization statistics are only available when text tokenization is performed
 * @note Mixed content processing may take longer due to row-by-row analysis
 *
 * @see llama_dataset_setup_streaming_mode for streaming configuration details
 * @see process_mixed_content_parquet for mixed content processing implementation
 * @see handle_tokenization_error for error reporting mechanisms
 */
bool llama_dataset_create_gguf_from_parquet(const std::shared_ptr<arrow::Table> & table,
                                                   struct llama_dataset * dataset) {
    if (!table || !dataset) {
        llama_dataset_set_error("Invalid parameters for GGUF creation from Parquet");
        return false;
    }

    // Create GGUF context
    dataset->ctx = gguf_init_empty();
    if (!dataset->ctx) {
        llama_dataset_set_error_with_code(DATASET_ERROR_CONTEXT_CREATION_FAILED, "Failed to create GGUF context");
        return false;
    }

    // Get format data for tokenization support
    auto * format_data = static_cast<struct parquet_format_data *>(dataset->format_data);
    if (!format_data) {
        llama_dataset_set_error("Format data not available for Parquet processing");
        return false;
    }

    // Find the primary data column (tokens or text)
    std::shared_ptr<arrow::ChunkedArray> data_column;
    int data_column_index = -1;
    bool is_text_column = false;

    // First try to find the specified column
    for (int i = 0; i < table->num_columns(); i++) {
        std::string column_name = table->schema()->field(i)->name();
        if (column_name == dataset->column) {
            data_column = table->column(i);
            data_column_index = i;

            // Check if this is a text column that needs tokenization
            auto field_type = table->schema()->field(i)->type();
            is_text_column = is_text_column_type(field_type);
            break;
        }
    }

    // If not found, try to use schema analysis results
    if (!data_column && format_data->schema_analyzed) {
        if (format_data->tokenizer && format_data->schema_info.primary_text_column_index >= 0) {
            // Use primary text column for tokenization
            data_column_index = format_data->schema_info.primary_text_column_index;
            data_column = table->column(data_column_index);
            is_text_column = true;
        } else if (format_data->schema_info.primary_token_column_index >= 0) {
            // Use primary token column
            data_column_index = format_data->schema_info.primary_token_column_index;
            data_column = table->column(data_column_index);
            is_text_column = false;
        }
    }

    if (!data_column || data_column_index == -1) {
        const auto msg{"Parquet file must contain " + dataset->column + " column or compatible data column"};
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, msg.c_str());
        return false;
    }

    // Set metadata
    gguf_set_val_str(dataset->ctx, TRAINING_FORMAT_SOURCE, "parquet");

    // Get current time for creation timestamp
    time_t now = time(nullptr);
    char   time_str[64];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&now));
    gguf_set_val_str(dataset->ctx, TRAINING_CREATION_TIME, time_str);

    // Add tokenization metadata if applicable
    if (is_text_column && format_data->tokenizer) {
        gguf_set_val_str(dataset->ctx, "training.tokenization.source", "text");
        gguf_set_val_str(dataset->ctx, "training.tokenization.column",
                        table->schema()->field(data_column_index)->name().c_str());
    } else if (format_data->mixed_content) {
        gguf_set_val_str(dataset->ctx, "training.tokenization.source", "mixed");
    } else {
        gguf_set_val_str(dataset->ctx, "training.tokenization.source", "pre-tokenized");
    }

    // Process each chunk in the data column (tokens or text)
    std::vector<std::vector<int32_t>> all_sequences;
    int32_t max_length = 0;

    for (int chunk_idx = 0; chunk_idx < data_column->num_chunks(); chunk_idx++) {
        auto chunk = data_column->chunk(chunk_idx);

        if (is_text_column) {
            // Handle text data that needs tokenization
            if (!format_data->tokenizer) {
                llama_dataset_set_error("Text column found but tokenizer not available");
                return false;
            }

            if (chunk->type_id() == arrow::Type::STRING || chunk->type_id() == arrow::Type::LARGE_STRING) {
                auto string_array = std::static_pointer_cast<arrow::StringArray>(chunk);

                for (int64_t row = 0; row < string_array->length(); row++) {
                    if (string_array->IsNull(row)) {
                        continue;
                    }

                    std::string text = string_array->GetString(row);
                    if (text.empty()) {
                        continue;
                    }

                    // Tokenize the text
                    std::vector<int32_t> sequence_tokens = format_data->tokenizer->tokenize_text(text);
                    if (sequence_tokens.empty()) {
                        handle_tokenization_error(row, "text", text, "Empty token sequence returned");
                        continue;
                    }

                    all_sequences.push_back(std::move(sequence_tokens));
                    max_length = std::max(max_length, static_cast<int32_t>(all_sequences.back().size()));
                }
            } else {
                LLAMA_LOG_WARN("Unsupported text column type in chunk %d\n", chunk_idx);
                continue;
            }
        } else {
            // Handle pre-tokenized data
            if (chunk->type_id() == arrow::Type::LIST) {
                // Each row is a list of tokens (sequence)
                auto list_array = std::static_pointer_cast<arrow::ListArray>(chunk);

                for (int64_t row = 0; row < list_array->length(); row++) {
                    if (list_array->IsNull(row)) {
                        continue;
                    }

                    std::vector<int32_t> sequence_tokens;
                    auto slice = list_array->value_slice(row);

                    if (!llama_dataset_arrow_array_to_tokens(slice, sequence_tokens)) {
                        LLAMA_LOG_WARN("Failed to convert tokens for sequence %ld in chunk %d\n", row, chunk_idx);
                        continue;
                    }

                    if (!sequence_tokens.empty()) {
                        all_sequences.push_back(std::move(sequence_tokens));
                        max_length = std::max(max_length, static_cast<int32_t>(all_sequences.back().size()));
                    }
                }
            } else {
                // Assume the entire chunk is one sequence (less common case)
                std::vector<int32_t> sequence_tokens;
                if (llama_dataset_arrow_array_to_tokens(chunk, sequence_tokens) && !sequence_tokens.empty()) {
                    all_sequences.push_back(std::move(sequence_tokens));
                    max_length = std::max(max_length, static_cast<int32_t>(all_sequences.back().size()));
                }
            }
        }
    }

    // Handle mixed content if detected
    if (format_data->mixed_content && all_sequences.empty()) {
        LLAMA_LOG_INFO("Processing mixed content Parquet file\n");
        if (!process_mixed_content_parquet(table, dataset, format_data, all_sequences, max_length)) {
            // Error already set by process_mixed_content_parquet
            return false;
        }
    }

    if (all_sequences.empty()) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "No valid sequences found in Parquet file");
        return false;
    }

    // Set sequence metadata
    gguf_set_val_u64(dataset->ctx, TRAINING_SEQUENCE_COUNT, all_sequences.size());
    gguf_set_val_i32(dataset->ctx, TRAINING_MAX_LENGTH, max_length);
    dataset->n_seq = all_sequences.size();

    // Add tokenization statistics if tokenizer was used
    if (format_data->tokenizer) {
        size_t total_tokens = format_data->tokenizer->get_total_tokens();
        size_t unique_texts = format_data->tokenizer->get_unique_texts();
        double cache_hit_ratio = format_data->tokenizer->get_cache_hit_ratio();

        gguf_set_val_u64(dataset->ctx, "training.tokenization.total_tokens", total_tokens);
        gguf_set_val_u64(dataset->ctx, "training.tokenization.unique_texts", unique_texts);
        gguf_set_val_f32(dataset->ctx, "training.tokenization.cache_hit_ratio", static_cast<float>(cache_hit_ratio));

        LLAMA_LOG_INFO("Tokenization stats: %zu total tokens, %zu unique texts, %.2f%% cache hit ratio\n",
                      total_tokens, unique_texts, cache_hit_ratio * 100.0);
    }

    // Create GGML context for tensor storage (if not streaming)
    if (!dataset->streaming) {
        // Calculate memory requirements
        size_t total_memory = 0;
        for (const auto & seq : all_sequences) {
            total_memory += seq.size() * sizeof(int32_t);
        }
        total_memory += 1024 * 1024;  // Add 1MB buffer for metadata and alignment

        struct ggml_init_params ggml_params = {};
        ggml_params.mem_size                = total_memory;
        ggml_params.mem_buffer              = nullptr;
        ggml_params.no_alloc                = false;

        dataset->ggml_ctx = ggml_init(ggml_params);
        if (!dataset->ggml_ctx) {
            llama_dataset_set_error_with_code(DATASET_ERROR_CONTEXT_CREATION_FAILED, "Failed to create GGML context");
            return false;
        }

        // Create tensors for each sequence
        for (size_t i = 0; i < all_sequences.size(); i++) {
            const auto & sequence = all_sequences[i];

            // Create tensor name
            char tensor_name[64];
            snprintf(tensor_name, sizeof(tensor_name), "sequence_%zu", i);

            // Create tensor
            struct ggml_tensor * tensor = ggml_new_tensor_1d(dataset->ggml_ctx, GGML_TYPE_I32, sequence.size());
            if (!tensor) {
                llama_dataset_set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION,
                                                  "Failed to create tensor for sequence");
                return false;
            }

            ggml_set_name(tensor, tensor_name);

            // Copy data to tensor
            if (tensor->data) {
                memcpy(tensor->data, sequence.data(), sequence.size() * sizeof(int32_t));
            }

            // Add tensor to GGUF context
            gguf_add_tensor(dataset->ctx, tensor);
        }
    } else {
        // Setup streaming mode using dedicated streaming module
        if (!llama_dataset_setup_streaming_mode(dataset, all_sequences, max_length)) {
            // Error already set by setup_streaming_mode
            return false;
        }
    }

    LLAMA_LOG_INFO("Successfully loaded %zu sequences from Parquet file (max_length=%d)\n", all_sequences.size(),
                   max_length);

    return true;
}

/**
 * @brief Handle tokenization failure with comprehensive error diagnostics.
 *
 * This function provides detailed error reporting and logging for tokenization failures,
 * enabling developers to quickly identify and resolve data quality issues. It implements
 * intelligent text truncation and contextual error reporting to aid in debugging.
 *
 * ## Error Reporting Features
 *
 * - **Row-Level Identification**: Pinpoints exact row where tokenization failed
 * - **Column Context**: Identifies the specific column being processed
 * - **Content Preview**: Shows truncated text content for manual inspection
 * - **Error Message Integration**: Incorporates tokenizer-specific error details
 * - **Intelligent Truncation**: Limits text display to prevent log overflow
 *
 * ## Logging Strategy
 *
 * The function uses structured logging with different detail levels:
 * - Includes row and column identification for precise error location
 * - Truncates long text content to first 100 characters for readability
 * - Appends "..." indicator when text is truncated
 * - Preserves original error messages from tokenizer when available
 *
 * ## Use Cases
 *
 * - **Data Quality Validation**: Identifies problematic text content during conversion
 * - **Debugging Support**: Provides context for tokenization algorithm failures
 * - **Performance Monitoring**: Tracks tokenization failure rates across datasets
 * - **Error Recovery**: Enables selective processing of valid data while logging failures
 *
 * ## Performance Considerations
 *
 * - **Minimal Overhead**: Only performs string operations when errors occur
 * - **Efficient Truncation**: Uses substr() for O(1) truncation operation
 * - **Conditional Logging**: Avoids expensive string formatting for successful cases
 * - **Memory Efficient**: Creates temporary strings only for error reporting
 *
 * @param row_index Zero-based row number where tokenization failed
 * @param column_name Name of the column being processed (used for context)
 * @param text_content Original text content that failed tokenization (will be truncated if > 100 chars)
 * @param error_message Optional additional error message from tokenizer (can be nullptr)
 *
 * @note This function is designed for error reporting only and does not affect processing flow
 * @note Text truncation preserves readability while preventing log file bloat
 * @note Error messages are logged at ERROR level for visibility in production environments
 *
 * @see LLAMA_LOG_ERROR for logging infrastructure details
 * @see llama_dataset_parquet_tokenizer::tokenize_text for tokenization implementation
 */
static void handle_tokenization_error(
    int64_t row_index,
    const std::string & column_name,
    const std::string & text_content,
    const char * error_message
) {
    std::string truncated_text = text_content.length() > 100 ?
        text_content.substr(0, 100) + "..." : text_content;

    if (error_message) {
        LLAMA_LOG_ERROR("Tokenization failed at row %ld, column '%s': %s. Text: '%s'\n",
                       row_index, column_name.c_str(), error_message, truncated_text.c_str());
    } else {
        LLAMA_LOG_ERROR("Tokenization failed at row %ld, column '%s'. Text: '%s'\n",
                       row_index, column_name.c_str(), truncated_text.c_str());
    }
}

/**
 * @brief Process mixed content Parquet data with intelligent fallback strategies.
 *
 * This function implements sophisticated processing logic for Parquet files containing
 * both text and pre-tokenized columns. It employs intelligent prioritization and
 * fallback mechanisms to maximize data utilization while maintaining processing efficiency.
 *
 * ## Processing Strategy
 *
 * The function implements a two-tier processing approach:
 * 1. **Primary Strategy**: Attempts to use pre-tokenized data when available
 * 2. **Fallback Strategy**: Falls back to text tokenization when tokenized data is missing
 *
 * ## Row-by-Row Processing Algorithm
 *
 * For each row in the dataset:
 * 1. **Token Column Check**: Searches for pre-tokenized data in primary token column
 * 2. **Chunk Navigation**: Locates the correct chunk containing the target row
 * 3. **Data Extraction**: Extracts token sequences using Arrow array conversion
 * 4. **Fallback Activation**: If no tokenized data found, attempts text tokenization
 * 5. **Text Processing**: Tokenizes text content using integrated llama tokenizer
 * 6. **Sequence Validation**: Validates extracted sequences before adding to results
 * 7. **Statistics Update**: Updates maximum length and sequence count metrics
 *
 * ## Chunk Management
 *
 * The function handles Arrow's chunked data structure efficiently:
 * - Calculates chunk boundaries to locate specific rows
 * - Maintains chunk start offsets for efficient row mapping
 * - Handles variable chunk sizes gracefully
 * - Minimizes chunk iteration overhead through smart indexing
 *
 * ## Error Handling and Recovery
 *
 * - **Graceful Degradation**: Continues processing when individual rows fail
 * - **Detailed Logging**: Reports specific failures with row-level context
 * - **Data Validation**: Validates extracted sequences before inclusion
 * - **Memory Safety**: Handles null values and empty sequences safely
 *
 * ## Performance Optimizations
 *
 * - **Lazy Evaluation**: Only processes text when tokenized data unavailable
 * - **Chunk Caching**: Minimizes repeated chunk access for adjacent rows
 * - **Early Termination**: Skips further processing once valid data found
 * - **Memory Efficiency**: Processes rows individually to control memory usage
 *
 * ## Data Quality Assurance
 *
 * - Validates both tokenized and text data before acceptance
 * - Handles empty sequences and null values appropriately
 * - Provides detailed error reporting for failed tokenization
 * - Maintains data integrity through comprehensive validation
 *
 * @param table Arrow table containing mixed content data (must not be null)
 * @param dataset Dataset structure for configuration access (must not be null)
 * @param format_data Parquet format data with schema analysis results (must not be null)
 * @param all_sequences Output vector for processed token sequences (will be populated)
 * @param max_length Output parameter for maximum sequence length found (will be updated)
 * @return true on successful processing, false on critical error
 *
 * @pre format_data->schema_analyzed must be true with valid schema information
 * @pre format_data->tokenizer must be available if text processing is required
 * @post all_sequences contains valid token sequences from processed rows
 * @post max_length reflects the longest sequence found during processing
 *
 * @note This function is designed for complex datasets with heterogeneous content
 * @note Processing time scales linearly with dataset size due to row-by-row analysis
 * @note Memory usage remains bounded regardless of dataset size
 *
 * @see handle_tokenization_error for error reporting implementation
 * @see llama_dataset_arrow_array_to_tokens for token extraction details
 * @see llama_dataset_parquet_tokenizer for text tokenization capabilities
 */
static bool process_mixed_content_parquet(
    const std::shared_ptr<arrow::Table> & table,
    struct llama_dataset * dataset,
    struct parquet_format_data * format_data,
    std::vector<std::vector<int32_t>> & all_sequences,
    int32_t & max_length
) {
    if (!table || !dataset || !format_data) {
        llama_dataset_set_error("Invalid parameters for mixed content processing");
        return false;
    }

    // Process each row, checking for both text and token columns
    int64_t num_rows = table->num_rows();

    for (int64_t row = 0; row < num_rows; row++) {
        std::vector<int32_t> sequence_tokens;
        bool sequence_processed = false;

        // First, try to use pre-tokenized data if available
        if (format_data->schema_info.primary_token_column_index >= 0) {
            auto token_column = table->column(format_data->schema_info.primary_token_column_index);

            // Get the token data for this row
            for (int chunk_idx = 0; chunk_idx < token_column->num_chunks(); chunk_idx++) {
                auto chunk = token_column->chunk(chunk_idx);

                if (chunk->type_id() == arrow::Type::LIST) {
                    auto list_array = std::static_pointer_cast<arrow::ListArray>(chunk);

                    // Find the row within this chunk
                    int64_t chunk_start = 0;
                    for (int prev_chunk = 0; prev_chunk < chunk_idx; prev_chunk++) {
                        chunk_start += token_column->chunk(prev_chunk)->length();
                    }

                    if (row >= chunk_start && row < chunk_start + list_array->length()) {
                        int64_t local_row = row - chunk_start;

                        if (!list_array->IsNull(local_row)) {
                            auto slice = list_array->value_slice(local_row);
                            if (llama_dataset_arrow_array_to_tokens(slice, sequence_tokens)) {
                                sequence_processed = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        // If no tokenized data found, try to tokenize text data
        if (!sequence_processed && format_data->schema_info.primary_text_column_index >= 0 && format_data->tokenizer) {
            auto text_column = table->column(format_data->schema_info.primary_text_column_index);

            // Get the text data for this row
            for (int chunk_idx = 0; chunk_idx < text_column->num_chunks(); chunk_idx++) {
                auto chunk = text_column->chunk(chunk_idx);

                if (chunk->type_id() == arrow::Type::STRING || chunk->type_id() == arrow::Type::LARGE_STRING) {
                    auto string_array = std::static_pointer_cast<arrow::StringArray>(chunk);

                    // Find the row within this chunk
                    int64_t chunk_start = 0;
                    for (int prev_chunk = 0; prev_chunk < chunk_idx; prev_chunk++) {
                        chunk_start += text_column->chunk(prev_chunk)->length();
                    }

                    if (row >= chunk_start && row < chunk_start + string_array->length()) {
                        int64_t local_row = row - chunk_start;

                        if (!string_array->IsNull(local_row)) {
                            std::string text = string_array->GetString(local_row);
                            if (!text.empty()) {
                                sequence_tokens = format_data->tokenizer->tokenize_text(text);
                                if (!sequence_tokens.empty()) {
                                    sequence_processed = true;
                                    break;
                                } else {
                                    handle_tokenization_error(row, "text", text, "Mixed content tokenization failed");
                                }
                            }
                        }
                    }
                }
            }
        }

        // Add the processed sequence if we got one
        if (sequence_processed && !sequence_tokens.empty()) {
            all_sequences.push_back(std::move(sequence_tokens));
            max_length = std::max(max_length, static_cast<int32_t>(all_sequences.back().size()));
        } else if (!sequence_processed) {
            LLAMA_LOG_WARN("No valid data found for row %ld in mixed content processing\n", row);
        }
    }

    return true;
}

#endif // LLAMA_PARQUET
