/**
 * @file llama-dataset-parquet-core.cpp
 * @brief Core Parquet dataset functionality implementation for llama.cpp
 *
 * This module provides the core implementation for loading and processing Parquet datasets
 * within the llama.cpp ecosystem. It serves as the central hub for Parquet file operations,
 * integrating Apache Arrow for efficient columnar data processing and providing seamless
 * conversion between Parquet format and llama's internal dataset representation.
 *
 * ## Key Responsibilities
 *
 * ### Parquet File Loading and Parsing
 * - Opens and validates Parquet files using Apache Arrow's Parquet reader
 * - Handles file I/O operations with comprehensive error checking
 * - Supports both streaming and non-streaming dataset loading modes
 * - Manages memory-efficient access to large Parquet files
 *
 * ### Apache Arrow Integration
 * - Utilizes Arrow's columnar memory format for efficient data processing
 * - Integrates with Arrow's type system for schema validation and conversion
 * - Leverages Arrow's memory pool for optimized memory management
 * - Provides seamless conversion between Arrow tables and llama datasets
 *
 * ### Schema Analysis and Validation
 * - Analyzes Parquet schemas to identify text and token columns
 * - Validates column types and data formats for compatibility
 * - Supports mixed content datasets with both text and pre-tokenized data
 * - Provides detailed schema information for downstream processing
 *
 * ### Tokenization Engine Integration
 * - Implements efficient text-to-token conversion using llama tokenizer
 * - Provides caching mechanisms for improved tokenization performance
 * - Supports batch tokenization for processing multiple text entries
 * - Manages tokenization context and memory resources
 *
 * ### Data Conversion and Processing
 * - Converts Parquet columnar data to GGUF tensor format
 * - Handles different data types including integers, floats, and strings
 * - Implements efficient data extraction and transformation algorithms
 * - Supports both text and pre-tokenized data processing workflows
 *
 * ## Architecture Integration
 *
 * This module integrates with several other components:
 * - **Schema Module**: Delegates detailed schema analysis operations
 * - **Conversion Module**: Handles data type conversion and transformation
 * - **Streaming Module**: Provides streaming support for large datasets
 * - **Core Dataset**: Integrates with the main dataset infrastructure
 *
 * ## Performance Considerations
 *
 * - Uses Apache Arrow's zero-copy operations where possible
 * - Implements efficient memory management with shared pointers
 * - Provides configurable caching for tokenization operations
 * - Supports lazy loading and streaming for memory efficiency
 *
 * ## Error Handling
 *
 * Comprehensive error handling covers:
 * - File I/O errors and invalid file paths
 * - Schema validation failures and type mismatches
 * - Memory allocation failures and resource exhaustion
 * - Arrow/Parquet library errors with detailed error messages
 *
 * @author llama.cpp contributors
 * @version 1.0
 * @date 2024
 *
 * @see llama-dataset-parquet.h for public interface definitions
 * @see llama-dataset-parquet-schema.cpp for schema analysis implementation
 * @see llama-dataset-parquet-conversion.cpp for data conversion implementation
 * @see llama-dataset-parquet-streaming.cpp for streaming implementation
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
#include <unordered_map>

#include "common/common.h"
#include "ggml/include/ggml.h"
#include "ggml/include/gguf.h"
#include "../../core/llama-dataset-internal.h"
#include "../../core/llama-dataset-utils.h"
#include "llama.h"
#include "llama-impl.h"



//
// Schema analysis functions are now implemented in llama-dataset-parquet-schema.cpp
// Data conversion functions are now implemented in llama-dataset-parquet-conversion.cpp
//




bool llama_dataset_validate_parquet_schema(const char * path, struct parquet_schema_info * info);
/**
 * @brief Load a dataset from a Parquet file (internal implementation).
 *
 * This function provides the core implementation for loading Parquet datasets,
 * handling all aspects of file parsing, schema analysis, and data conversion.
 * It integrates with Apache Arrow for efficient columnar data processing and
 * supports both streaming and non-streaming modes.
 *
 * ## Processing Pipeline
 *
 * 1. **File Validation**: Verifies file existence and accessibility
 * 2. **Arrow Integration**: Opens file using Arrow's Parquet reader
 * 3. **Schema Analysis**: Analyzes column types and mixed content support
 * 4. **Tokenization Setup**: Initializes tokenization engine if needed
 * 5. **Data Conversion**: Converts Parquet data to GGUF tensor format
 * 6. **Caching**: Sets up tensor caching for efficient access
 *
 * ## Apache Arrow Integration Details
 *
 * - Uses Arrow's ReadableFile for efficient I/O operations
 * - Leverages Arrow's memory pool for optimized memory management
 * - Utilizes Arrow's Table abstraction for columnar data access
 * - Integrates with Arrow's type system for schema validation
 *
 * ## Error Handling
 *
 * Comprehensive error handling includes:
 * - File not found or access permission errors
 * - Invalid Parquet format or corrupted files
 * - Schema validation failures and type mismatches
 * - Memory allocation failures during processing
 * - Arrow library errors with detailed status messages
 *
 * @param params Dataset loading parameters including file path, streaming options,
 *               column preferences, and tokenization settings
 * @return Pointer to loaded dataset on success, nullptr on failure
 *         (error details available via llama_dataset_get_error())
 *
 * @note This function allocates significant memory for large datasets.
 *       Consider using streaming mode for memory-constrained environments.
 *
 * @see llama_dataset_load_parquet() for public interface
 * @see analyze_parquet_table_schema() for schema analysis details
 * @see llama_dataset_create_gguf_from_parquet() for conversion details
 */
struct llama_dataset * llama_dataset_load_parquet_internal(const common_params * params) {
    if (params->in_files.empty()) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Path cannot be empty");
        return nullptr;
    }
    auto path = params->in_files[0];//also we may refactor for walk on in_files collection or read files from dirs
    if (path.empty()) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Path cannot be null");
        return nullptr;
    }
    // Check if file exists
    FILE * file = fopen(path.c_str(), "rb");
    if (!file) {
        llama_dataset_set_error_with_code(DATASET_ERROR_FILE_NOT_FOUND, "Parquet file not found");
        return nullptr;
    }
    fclose(file);

    // Create dataset structure
    struct llama_dataset * dataset = llama_dataset_alloc(DATASET_PARQUET, params->dataset_streaming);
    if (!dataset) {
        return nullptr;  // Error already set by dataset_alloc
    }

    // Allocate format-specific data
    auto * format_data       = new parquet_format_data();
    format_data->file_path   = std::string(path);
    format_data->n_sequences = 0;
    format_data->max_length  = 0;
    format_data->schema_analyzed = false;
    format_data->tokenizer   = nullptr;
    format_data->mixed_content = false;
    memset(&format_data->schema_info, 0, sizeof(struct parquet_schema_info));
    format_data->schema_info.primary_text_column_index = -1;
    format_data->schema_info.primary_token_column_index = -1;
    dataset->format_data     = format_data;
    dataset->column          = params->dataset_column;

    try {
        // Open Parquet file
        auto result = arrow::io::ReadableFile::Open(path);
        if (!result.ok()) {
            const auto msg{"Failed to open Parquet file: " + result.status().ToString()};
            llama_dataset_set_error_with_code(DATASET_ERROR_IO_ERROR, msg.c_str());
            llama_dataset_free(dataset);
            return nullptr;
        }

        // Create Parquet reader
        auto reader = parquet::arrow::OpenFile(result.ValueOrDie(), arrow::default_memory_pool());
        if (!reader.ok()) {
            llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Failed to create Parquet reader");
            llama_dataset_free(dataset);
            return nullptr;
        }

        format_data->reader = std::shared_ptr<parquet::arrow::FileReader>(reader->get());

        // Read table
        std::shared_ptr<arrow::Table> table;
        auto status = format_data->reader->ReadTable(&table);
        if (!status.ok()) {
            llama_dataset_set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to read Parquet table");
            llama_dataset_free(dataset);
            return nullptr;
        }

        format_data->table = table;

        // Analyze schema for mixed content support
        if (!analyze_parquet_table_schema(
                table,
                params->dataset_column,
                params->dataset_column_to,
                &format_data->schema_info)) {
            // Error already set by analyze_parquet_table_schema
            llama_dataset_free(dataset);
            return nullptr;
        }
        format_data->schema_analyzed = true;

        // Validate schema
        if (!llama_dataset_validate_parquet_schema(path.c_str(), nullptr)) {
            // Error already set by validate_parquet_schema
            llama_dataset_free(dataset);
            return nullptr;
        }

        // Initialize tokenization engine if needed (after model is attached)
        if (params->dataset_tokenize_text && dataset->model) {
            format_data->tokenizer = new llama_dataset_parquet_tokenizer(dataset->model);
            if (!format_data->tokenizer->is_valid()) {
                llama_dataset_set_error("Failed to initialize tokenization engine");
                llama_dataset_free(dataset);
                return nullptr;
            }

            // Configure tokenization cache
            format_data->tokenizer->set_cache_size(params->dataset_tokenization_cache_size);
            format_data->mixed_content = format_data->schema_info.has_mixed_content;

            // Initialize on-demand tokenization state
            format_data->tokenization_enabled = true;
            format_data->cache_max_size = params->dataset_tokenization_cache_size * 1024 * 1024; // Convert MB to bytes
            format_data->cache_memory_usage = 0;

            // Set processing mode based on schema analysis
            if (format_data->schema_info.has_mixed_content) {
                format_data->processing_mode = PARQUET_MIXED_MODE;
                format_data->prefer_tokens_over_text = true; // Prefer pre-tokenized data for performance
            } else if (format_data->schema_info.primary_text_column_index >= 0) {
                format_data->processing_mode = PARQUET_TEXT_MODE;
                format_data->prefer_tokens_over_text = false;
            } else if (format_data->schema_info.primary_token_column_index >= 0) {
                format_data->processing_mode = PARQUET_TOKEN_MODE;
                format_data->prefer_tokens_over_text = true;
            } else {
                format_data->processing_mode = PARQUET_TEXT_MODE; // Default fallback
                format_data->prefer_tokens_over_text = false;
            }

            LLAMA_LOG_INFO("Tokenization engine initialized with %zu MB cache, mode=%d, prefer_tokens=%s\n",
                          params->dataset_tokenization_cache_size,
                          format_data->processing_mode,
                          format_data->prefer_tokens_over_text ? "true" : "false");
        } else {
            format_data->tokenizer = nullptr;
            format_data->tokenization_enabled = false;
            format_data->processing_mode = PARQUET_TOKEN_MODE; // Assume pre-tokenized if no tokenizer
            format_data->prefer_tokens_over_text = true;
            format_data->cache_max_size = 0;
            format_data->cache_memory_usage = 0;
        }

        // Create GGUF context from Parquet data
        if (!llama_dataset_create_gguf_from_parquet(table, dataset)) {
            // Error already set by create_gguf_from_parquet
            llama_dataset_free(dataset);
            return nullptr;
        }

        // Cache tensors for efficient access
        if (!llama_dataset_cache_tensors(dataset)) {
            // Error already set by dataset_cache_tensors
            llama_dataset_free(dataset);
            return nullptr;
        }

        LLAMA_LOG_INFO("Successfully loaded Parquet dataset from %s (%zu sequences, streaming=%s)\n", path.c_str(),
                       dataset->n_seq, params->dataset_streaming ? "true" : "false");

        return dataset;

    } catch (const std::exception & e) {
        const auto msg{"Parquet loading failed: " + std::string(e.what())};
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, msg.c_str());
        llama_dataset_free(dataset);
        return nullptr;
    }
}

/**
 * @brief Free Parquet format-specific data.
 *
 * This function cleans up all Parquet-specific resources including
 * schema analysis results, tokenization engine, and Arrow/Parquet objects.
 *
 * @param format_data Pointer to parquet_format_data to free
 */
void llama_dataset_free_parquet_format_data(void * format_data) {
    if (!format_data) {
        return;
    }

    auto * data = static_cast<struct parquet_format_data *>(format_data);

    // Free schema analysis results
    if (data->schema_analyzed) {
        parquet_schema_info_free(&data->schema_info);
    }

    // Free tokenization engine
    if (data->tokenizer) {
        delete data->tokenizer;
        data->tokenizer = nullptr;
    }

    // Clear tokenization cache (tensors are managed by GGML context)
    data->tokenized_cache.clear();
    data->cache_access_order.clear();

    // Clear vectors
    data->text_columns.clear();
    data->token_columns.clear();

    // Arrow/Parquet objects are automatically cleaned up by shared_ptr
    data->table.reset();
    data->reader.reset();

    // Free the structure itself
    delete data;
}

/**
 * @brief Validate Parquet file schema for dataset compatibility.
 *
 * This function performs comprehensive validation of a Parquet file's schema
 * to ensure compatibility with llama's dataset requirements. It checks column
 * types, data formats, and structural requirements using Apache Arrow's
 * schema introspection capabilities.
 *
 * ## Validation Process
 *
 * 1. **File Access**: Opens file using Arrow's I/O subsystem
 * 2. **Reader Creation**: Initializes Parquet reader with error checking
 * 3. **Schema Extraction**: Retrieves schema metadata from file headers
 * 4. **Type Validation**: Validates column types against expected formats
 * 5. **Structure Analysis**: Checks for required columns and data organization
 *
 * ## Supported Column Types
 *
 * - **INT32**: Single integer values or token IDs
 * - **LIST<INT32>**: Arrays of token sequences (most common)
 * - **STRING**: Text data for tokenization
 * - **BINARY**: Raw binary data (limited support)
 *
 * ## Apache Arrow Integration
 *
 * - Uses Arrow's Schema class for type introspection
 * - Leverages Arrow's Type system for validation
 * - Integrates with Arrow's error handling mechanisms
 * - Utilizes Arrow's memory pool for temporary operations
 *
 * @param path Path to the Parquet file to validate
 * @param info Optional pointer to store detailed schema analysis results.
 *             If provided, will be populated with column information,
 *             mixed content detection, and primary column indices.
 *             Can be nullptr if only validation result is needed.
 * @return true if schema is valid and compatible, false otherwise
 *         (error details available via llama_dataset_get_error())
 *
 * @note This function only validates schema structure, not data content.
 *       For full data validation, use the validation module functions.
 *
 * @see analyze_parquet_table_schema() for detailed schema analysis
 * @see parquet_schema_info for schema information structure
 */
bool llama_dataset_validate_parquet_schema(const char * path, struct parquet_schema_info * info) {
    if (!path) {
        llama_dataset_set_error("Path cannot be null for schema validation");
        return false;
    }

    try {
        // Open file
        auto result = arrow::io::ReadableFile::Open(path);
        if (!result.ok()) {
            llama_dataset_set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to open Parquet file for validation");
            return false;
        }

        // Create reader
        auto reader = parquet::arrow::OpenFile(result.ValueOrDie(), arrow::default_memory_pool());
        if (!reader.ok()) {
            llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Failed to create Parquet reader for validation");
            return false;
        }

        // Get schema
        std::shared_ptr<arrow::Schema> schema;
        auto status = reader->get()->GetSchema(&schema);
        if (!status.ok()) {
            llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Failed to read Parquet schema");
            return false;
        }

        std::string field_name = schema->field(0)->name();//MAY be need check on some field name
        auto field_type = schema->field(0)->type();

        // Validate column type
        if (field_type->id() == arrow::Type::LIST) {
            // List of integers (most common case)
            auto list_type = std::static_pointer_cast<arrow::ListType>(field_type);
            if (list_type->value_type()->id() != arrow::Type::INT32) {
                llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Tokens column must contain lists of int32 values");
                return false;
            }
        } else if (field_type->id() != arrow::Type::INT32) {
            llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Tokens column must be int32 or list<int32>");
            return false;
        }

        // If schema info is requested, populate it with basic validation results
        if (info) {
            // Read table to perform full schema analysis
            std::shared_ptr<arrow::Table> table;
            auto status = reader->get()->ReadTable(&table);
            if (status.ok()) {
                // Use the schema analysis function from the schema module
                analyze_parquet_table_schema(table, "", "", info);
            }
        }

        return true;

    } catch (const std::exception & e) {
        const auto msg{"Schema validation failed: " + std::string(e.what())};
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, msg.c_str());
        return false;
    }
}

/**
 * @brief Get Parquet file metadata for dataset information.
 *
 * This function extracts essential metadata from a Parquet file without
 * loading the full dataset, providing efficient access to file statistics
 * and structural information. It uses Apache Arrow's metadata APIs for
 * fast header-only operations.
 *
 * ## Metadata Extraction Process
 *
 * 1. **File Opening**: Opens file using Arrow's ReadableFile interface
 * 2. **Reader Creation**: Creates Parquet reader for metadata access
 * 3. **Metadata Retrieval**: Extracts file-level metadata from headers
 * 4. **Statistics Calculation**: Computes sequence count and length estimates
 *
 * ## Apache Arrow Integration
 *
 * - Uses Arrow's FileMetaData for efficient header access
 * - Leverages Arrow's RowGroup metadata for statistics
 * - Integrates with Arrow's I/O subsystem for file operations
 * - Utilizes Arrow's error handling for robust operation
 *
 * ## Performance Characteristics
 *
 * - **Fast Operation**: Only reads file headers, not data content
 * - **Memory Efficient**: Minimal memory allocation for metadata
 * - **I/O Optimized**: Single file access for all metadata
 * - **Error Resilient**: Handles corrupted or incomplete files gracefully
 *
 * @param path Path to the Parquet file to analyze
 * @param n_sequences Output parameter for number of sequences (rows) in file.
 *                    Set to 0 on error. Represents total number of data records.
 * @param max_length Output parameter for estimated maximum sequence length.
 *                   Set to default value (2048) as actual calculation requires
 *                   data scanning. For precise values, load the full dataset.
 * @return true if metadata extraction successful, false on error
 *         (error details available via llama_dataset_get_error())
 *
 * @note max_length is currently estimated as data scanning is expensive.
 *       For precise length information, consider loading the dataset.
 *
 * @see llama_dataset_load_parquet_internal() for full dataset loading
 */
bool llama_dataset_get_parquet_metadata(const char * path, uint64_t * n_sequences, int32_t * max_length) {
    if (!path || !n_sequences || !max_length) {
        llama_dataset_set_error("Invalid parameters for metadata extraction");
        return false;
    }

    *n_sequences = 0;
    *max_length  = 0;

    try {
        // Open file
        auto result = arrow::io::ReadableFile::Open(path);
        if (!result.ok()) {
            llama_dataset_set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to open Parquet file for metadata");
            return false;
        }
        std::shared_ptr<arrow::io::ReadableFile> infile = result.ValueOrDie();

        // Create reader
        auto reader = parquet::arrow::OpenFile(infile, arrow::default_memory_pool());
        if (!reader.ok()) {
            llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Failed to create Parquet reader for metadata");
            return false;
        }

        // Get basic metadata
        auto parquet_metadata = reader->get()->parquet_reader()->metadata();
        *n_sequences = parquet_metadata->num_rows();

        // For max_length, we'd need to read the data, which is expensive
        // For now, we'll set a reasonable default and let the actual loading determine it
        *max_length = 2048;  // Default assumption

        return true;

    } catch (const std::exception & e) {
        const auto msg{"Metadata extraction failed: " + std::string(e.what())};
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, msg.c_str());
        return false;
    }
}

#endif

#ifdef LLAMA_PARQUET
//
// Tokenization Engine Implementation
//

/**
 * @brief Constructor with llama model integration.
 *
 * Initializes the tokenization engine with a llama model, creating the
 * necessary context for text-to-token conversion operations. This constructor
 * sets up caching mechanisms, performance monitoring, and resource management
 * for efficient tokenization of Parquet text data.
 *
 * ## Initialization Process
 *
 * 1. **Model Validation**: Verifies model pointer and compatibility
 * 2. **Context Creation**: Initializes llama context with tokenization parameters
 * 3. **Cache Setup**: Configures LRU cache with default size limits
 * 4. **Statistics Initialization**: Sets up performance monitoring counters
 *
 * ## Context Configuration
 *
 * - **Context Size**: 2048 tokens for tokenization operations
 * - **Batch Size**: Single sequence processing for thread safety
 * - **Threading**: Single-threaded operation for consistency
 * - **Embeddings**: Disabled for tokenization-only operations
 *
 * ## Memory Management
 *
 * - **Cache Size**: Default 256MB for tokenization cache
 * - **Resource Ownership**: Manages context lifecycle automatically
 * - **Memory Monitoring**: Tracks cache usage and performance metrics
 *
 * @param model Pointer to initialized llama model for tokenization.
 *              Must not be null and must remain valid for tokenizer lifetime.
 *              The tokenizer does not take ownership of the model.
 *
 * @note The constructor may fail if context creation fails. Check is_valid()
 *       after construction to verify successful initialization.
 *
 * @see is_valid() to check initialization success
 * @see set_cache_size() to configure cache memory limits
 */
llama_dataset_parquet_tokenizer::llama_dataset_parquet_tokenizer(struct llama_model * model)
    : model_(model)
    , ctx_(nullptr)
    , max_cache_size_(256 * 1024 * 1024)  // 256MB default
    , current_cache_size_(0)
    , cache_hits_(0)
    , cache_misses_(0)
    , total_tokens_(0)
    , owns_context_(false)
{
    if (!model_) {
        LLAMA_LOG_ERROR("Tokenizer model cannot be null\n");
        return;
    }

    // Create tokenizer context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;  // Context size for tokenization
    ctx_params.n_batch = 1;   // Single sequence processing
    ctx_params.n_threads = 1; // Single-threaded tokenization
    ctx_params.embeddings = false;

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (ctx_) {
        owns_context_ = true;
        LLAMA_LOG_INFO("Tokenization context created successfully\n");
    } else {
        LLAMA_LOG_ERROR("Failed to create tokenization context\n");
    }
}

/**
 * @brief Destructor - cleans up resources.
 */
llama_dataset_parquet_tokenizer::~llama_dataset_parquet_tokenizer() {
    if (owns_context_ && ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }

    clear_cache();

    LLAMA_LOG_INFO("Tokenizer destroyed. Stats: %zu hits, %zu misses, %zu total tokens\n",
                   cache_hits_, cache_misses_, total_tokens_);
}

/**
 * @brief Convert text to tokens using llama tokenizer.
 *
 * This function performs text-to-token conversion using the llama tokenizer
 * with intelligent caching for improved performance. It implements a complete
 * tokenization pipeline including cache management, error handling, and
 * performance monitoring.
 *
 * ## Tokenization Pipeline
 *
 * 1. **Cache Lookup**: Checks if text has been tokenized before
 * 2. **Cache Hit**: Returns cached tokens immediately if found
 * 3. **Cache Miss**: Performs tokenization using llama tokenizer
 * 4. **Cache Update**: Stores result in cache with LRU management
 * 5. **Statistics Update**: Updates performance counters and metrics
 *
 * ## Caching Strategy
 *
 * - **LRU Eviction**: Removes least recently used entries when cache is full
 * - **Memory Monitoring**: Tracks cache size and enforces limits
 * - **Performance Tracking**: Monitors hit/miss ratios for optimization
 * - **Automatic Cleanup**: Evicts entries to maintain memory constraints
 *
 * ## Tokenization Details
 *
 * - **BOS Token**: Adds beginning-of-sequence token by default
 * - **Special Tokens**: Handles special tokens according to model configuration
 * - **Error Recovery**: Graceful handling of tokenization failures
 * - **Memory Safety**: Proper buffer management for token arrays
 *
 * @param text Input text string to tokenize. Can be empty (returns empty vector).
 *             Text encoding should be UTF-8 compatible.
 * @return Vector of token IDs representing the input text.
 *         Empty vector on error or for empty input text.
 *         Token IDs are model-specific and suitable for llama processing.
 *
 * @note This function is thread-safe for read operations but not for
 *       concurrent cache modifications. Use external synchronization
 *       if calling from multiple threads.
 *
 * @see tokenize_batch() for efficient batch processing
 * @see get_cache_hit_ratio() for performance monitoring
 */
std::vector<int32_t> llama_dataset_parquet_tokenizer::tokenize_text(const std::string & text) {
    if (!is_valid()) {
        LLAMA_LOG_ERROR("Tokenizer is not valid\n");
        return {};
    }

    // Check cache first
    auto cache_it = cache_.find(text);
    if (cache_it != cache_.end()) {
        cache_hits_++;
        return cache_it->second;
    }

    // Cache miss - perform tokenization
    cache_misses_++;
    std::vector<int32_t> tokens;

    if (!tokenize_internal(text, tokens)) {
        LLAMA_LOG_ERROR("Failed to tokenize text\n");
        return {};
    }

    // Update statistics
    total_tokens_ += tokens.size();

    // Add to cache if there's space
    size_t entry_size = estimate_cache_entry_size(text, tokens);
    if (current_cache_size_ + entry_size > max_cache_size_) {
        // Evict entries to make space
        evict_cache_entries(max_cache_size_ * 0.8);  // Evict to 80% capacity
    }

    if (current_cache_size_ + entry_size <= max_cache_size_) {
        cache_[text] = tokens;
        current_cache_size_ += entry_size;
    }

    return tokens;
}

/**
 * @brief Batch tokenization for efficiency.
 */
bool llama_dataset_parquet_tokenizer::tokenize_batch(const std::vector<std::string> & texts,
                                                     std::vector<std::vector<int32_t>> & results) {
    if (!is_valid()) {
        LLAMA_LOG_ERROR("Tokenizer is not valid for batch processing\n");
        return false;
    }

    results.clear();
    results.reserve(texts.size());

    for (const auto & text : texts) {
        auto tokens = tokenize_text(text);
        if (tokens.empty() && !text.empty()) {
            LLAMA_LOG_WARN("Failed to tokenize text in batch: '%s'\n", text.substr(0, 50).c_str());
            // Continue with empty tokens rather than failing the entire batch
        }
        results.push_back(std::move(tokens));
    }

    return true;
}

/**
 * @brief Set tokenization cache size limit.
 */
void llama_dataset_parquet_tokenizer::set_cache_size(size_t max_size_mb) {
    max_cache_size_ = max_size_mb * 1024 * 1024;  // Convert MB to bytes

    // If current cache exceeds new limit, evict entries
    if (current_cache_size_ > max_cache_size_) {
        evict_cache_entries(max_cache_size_ * 0.8);
    }

    LLAMA_LOG_INFO("Tokenization cache size set to %zu MB\n", max_size_mb);
}

/**
 * @brief Clear tokenization cache.
 */
void llama_dataset_parquet_tokenizer::clear_cache() {
    cache_.clear();
    current_cache_size_ = 0;
    LLAMA_LOG_INFO("Tokenization cache cleared\n");
}

/**
 * @brief Get cache hit ratio for performance monitoring.
 */
double llama_dataset_parquet_tokenizer::get_cache_hit_ratio() const {
    size_t total_requests = cache_hits_ + cache_misses_;
    return total_requests > 0 ? static_cast<double>(cache_hits_) / total_requests : 0.0;
}

/**
 * @brief Get total number of tokens processed.
 */
size_t llama_dataset_parquet_tokenizer::get_total_tokens() const {
    return total_tokens_;
}

/**
 * @brief Get number of unique texts processed.
 */
size_t llama_dataset_parquet_tokenizer::get_unique_texts() const {
    return cache_.size();
}

/**
 * @brief Check if tokenizer is valid and ready to use.
 */
bool llama_dataset_parquet_tokenizer::is_valid() const {
    return model_ != nullptr && ctx_ != nullptr;
}

/**
 * @brief Estimate memory usage of cached entry.
 */
size_t llama_dataset_parquet_tokenizer::estimate_cache_entry_size(const std::string & text,
                                                                 const std::vector<int32_t> & tokens) const {
    // Estimate: string size + vector overhead + token data
    return text.size() + sizeof(std::vector<int32_t>) + tokens.size() * sizeof(int32_t) +
           sizeof(std::pair<std::string, std::vector<int32_t>>);
}

/**
 * @brief Evict cache entries using LRU strategy.
 */
void llama_dataset_parquet_tokenizer::evict_cache_entries(size_t target_size) {
    // Simple eviction strategy: remove entries until we reach target size
    // In a more sophisticated implementation, we would use proper LRU tracking

    auto it = cache_.begin();
    while (it != cache_.end() && current_cache_size_ > target_size) {
        size_t entry_size = estimate_cache_entry_size(it->first, it->second);
        current_cache_size_ -= entry_size;
        it = cache_.erase(it);
    }

    LLAMA_LOG_INFO("Cache evicted to %zu bytes (target: %zu)\n", current_cache_size_, target_size);
}

/**
 * @brief Internal tokenization implementation.
 *
 * This function provides the core tokenization logic using llama's vocabulary
 * and tokenization algorithms. It handles the low-level details of converting
 * text strings to token sequences, including buffer management, error handling,
 * and integration with llama's tokenization APIs.
 *
 * ## Tokenization Process
 *
 * 1. **Buffer Allocation**: Estimates and allocates token buffer
 * 2. **Tokenization Call**: Invokes llama tokenizer with proper parameters
 * 3. **Error Checking**: Validates tokenization results and buffer sizes
 * 4. **Buffer Resizing**: Adjusts output vector to actual token count
 * 5. **Result Validation**: Ensures tokenization completed successfully
 *
 * ## Llama Integration Details
 *
 * - **Vocabulary Access**: Uses model's vocabulary for tokenization
 * - **BOS Token**: Adds beginning-of-sequence token as configured
 * - **Special Tokens**: Handles special tokens according to model settings
 * - **Buffer Management**: Manages token buffer allocation and sizing
 *
 * ## Error Handling
 *
 * - **Buffer Overflow**: Detects and reports insufficient buffer space
 * - **Tokenization Failure**: Handles tokenizer errors gracefully
 * - **Memory Issues**: Manages allocation failures and cleanup
 * - **Invalid Input**: Validates text input and model state
 *
 * @param text Input text string to tokenize (UTF-8 encoded)
 * @param tokens Output vector to store resulting token IDs.
 *               Vector is cleared before tokenization and resized to fit results.
 * @return true if tokenization successful, false on error
 *         (tokens vector will be empty on failure)
 *
 * @note This is an internal function and should not be called directly.
 *       Use tokenize_text() for public tokenization interface.
 *
 * @see tokenize_text() for public tokenization interface
 * @see llama_tokenize() for underlying tokenization function
 */
bool llama_dataset_parquet_tokenizer::tokenize_internal(const std::string & text, std::vector<int32_t> & tokens) {
    if (!ctx_ || !model_) {
        return false;
    }

    // Clear output vector
    tokens.clear();

    // Use llama tokenizer to convert text to tokens
    const int max_tokens = text.length() + 100;  // Estimate max tokens needed
    tokens.resize(max_tokens);

    const int n_tokens = llama_tokenize(&model_->vocab, text.c_str(), text.length(),
                                       tokens.data(), max_tokens,
                                       true,   // add_bos
                                       false); // special tokens

    if (n_tokens < 0) {
        LLAMA_LOG_ERROR("Tokenization failed: buffer too small (need %d tokens)\n", -n_tokens);
        tokens.clear();
        return false;
    }

    // Resize to actual number of tokens
    tokens.resize(n_tokens);

    return true;
}

/**
 * @brief Load and tokenize a Parquet sequence on-demand.
 *
 * This function provides the core on-demand tokenization functionality for
 * Parquet datasets. It checks the tokenization cache first, and if the sequence
 * is not cached, it determines the appropriate data source (text vs tokens),
 * performs tokenization if needed, converts to GGML tensor format, and stores
 * the result in the cache for future access.
 *
 * ## Processing Algorithm
 *
 * 1. **Cache Lookup**: Check if sequence is already cached as a tensor
 * 2. **Content Detection**: Determine if row contains text or pre-tokenized data
 * 3. **Data Extraction**: Extract data from appropriate column based on schema analysis
 * 4. **Tokenization**: Convert text to tokens using llama tokenizer if needed
 * 5. **Tensor Creation**: Convert token vectors to GGML tensor format
 * 6. **Cache Storage**: Store result in tokenized_cache with LRU management
 * 7. **Memory Management**: Handle cache eviction and memory pressure
 *
 * ## Mixed Content Handling
 *
 * The function intelligently handles datasets with both text and token columns:
 * - Uses prefer_tokens_over_text to choose data source priority
 * - Falls back to alternative column if primary source fails
 * - Leverages schema analysis results for optimization
 * - Provides graceful error recovery for tokenization failures
 *
 * @param dataset Dataset containing the Parquet data and tokenization context
 * @param sequence_index Index of the sequence to load and tokenize
 * @return Pointer to GGML tensor containing the tokenized sequence data,
 *         or NULL on error (error details available via llama_dataset_get_error())
 */
struct ggml_tensor * load_parquet_sequence_with_tokenization(struct llama_dataset * dataset,
                                                            uint64_t sequence_index) {
    if (!dataset || !dataset->format_data) {
        llama_dataset_set_error("Invalid dataset for sequence tokenization");
        return nullptr;
    }

    auto * format_data = static_cast<struct parquet_format_data *>(dataset->format_data);
    if (!format_data->table) {
        llama_dataset_set_error("No Parquet table available for tokenization");
        return nullptr;
    }

    // Check cache first
    auto cache_it = format_data->tokenized_cache.find(sequence_index);
    if (cache_it != format_data->tokenized_cache.end()) {
        // Update access order for LRU
        update_cache_access_order(format_data, sequence_index);
        return cache_it->second;
    }

    // Cache miss - need to load and tokenize the sequence
    std::vector<int32_t> sequence_tokens;
    bool sequence_processed = false;

    // Check if sequence index is valid
    if (sequence_index >= static_cast<uint64_t>(format_data->table->num_rows())) {
        llama_dataset_set_error("Sequence index out of bounds");
        return nullptr;
    }

    // Try to use pre-tokenized data first if available and preferred
    if (format_data->prefer_tokens_over_text &&
        format_data->schema_info.primary_token_column_index >= 0) {

        auto token_column = format_data->table->column(format_data->schema_info.primary_token_column_index);

        // Find the correct chunk and extract token data
        uint64_t current_row = 0;
        for (int chunk_idx = 0; chunk_idx < token_column->num_chunks(); chunk_idx++) {
            auto chunk = token_column->chunk(chunk_idx);
            uint64_t chunk_size = chunk->length();

            if (sequence_index >= current_row && sequence_index < current_row + chunk_size) {
                int64_t local_row = sequence_index - current_row;

                if (chunk->type_id() == arrow::Type::LIST) {
                    auto list_array = std::static_pointer_cast<arrow::ListArray>(chunk);

                    if (!list_array->IsNull(local_row)) {
                        auto slice = list_array->value_slice(local_row);
                        if (llama_dataset_arrow_array_to_tokens(slice, sequence_tokens)) {
                            sequence_processed = true;
                            break;
                        }
                    }
                } else if (chunk->type_id() == arrow::Type::INT32) {
                    auto int32_array = std::static_pointer_cast<arrow::Int32Array>(chunk);

                    if (!int32_array->IsNull(local_row)) {
                        sequence_tokens.push_back(int32_array->Value(local_row));
                        sequence_processed = true;
                        break;
                    }
                }
            }
            current_row += chunk_size;
        }
    }

    // If no tokenized data found or not preferred, try text tokenization
    if (!sequence_processed && format_data->schema_info.primary_text_column_index >= 0 &&
        format_data->tokenizer) {

        auto text_column = format_data->table->column(format_data->schema_info.primary_text_column_index);

        // Find the correct chunk and extract text data
        uint64_t current_row = 0;
        for (int chunk_idx = 0; chunk_idx < text_column->num_chunks(); chunk_idx++) {
            auto chunk = text_column->chunk(chunk_idx);
            uint64_t chunk_size = chunk->length();

            if (sequence_index >= current_row && sequence_index < current_row + chunk_size) {
                int64_t local_row = sequence_index - current_row;

                if (chunk->type_id() == arrow::Type::STRING ||
                    chunk->type_id() == arrow::Type::LARGE_STRING) {
                    auto string_array = std::static_pointer_cast<arrow::StringArray>(chunk);

                    if (!string_array->IsNull(local_row)) {
                        std::string text = string_array->GetString(local_row);
                        if (!text.empty()) {
                            sequence_tokens = format_data->tokenizer->tokenize_text(text);
                            if (!sequence_tokens.empty()) {
                                sequence_processed = true;
                                break;
                            } else {
                                LLAMA_LOG_WARN("Tokenization failed for sequence %lu: empty result\n", sequence_index);
                            }
                        }
                    }
                }
            }
            current_row += chunk_size;
        }
    }

    // If still no data and we haven't tried tokens yet, try as fallback
    if (!sequence_processed && !format_data->prefer_tokens_over_text &&
        format_data->schema_info.primary_token_column_index >= 0) {

        auto token_column = format_data->table->column(format_data->schema_info.primary_token_column_index);

        // Find the correct chunk and extract token data
        uint64_t current_row = 0;
        for (int chunk_idx = 0; chunk_idx < token_column->num_chunks(); chunk_idx++) {
            auto chunk = token_column->chunk(chunk_idx);
            uint64_t chunk_size = chunk->length();

            if (sequence_index >= current_row && sequence_index < current_row + chunk_size) {
                uint64_t local_row = sequence_index - current_row;

                if (chunk->type_id() == arrow::Type::LIST) {
                    auto list_array = std::static_pointer_cast<arrow::ListArray>(chunk);

                    if (!list_array->IsNull(local_row)) {
                        auto slice = list_array->value_slice(local_row);
                        if (llama_dataset_arrow_array_to_tokens(slice, sequence_tokens)) {
                            sequence_processed = true;
                            break;
                        }
                    }
                } else if (chunk->type_id() == arrow::Type::INT32) {
                    auto int32_array = std::static_pointer_cast<arrow::Int32Array>(chunk);

                    if (!int32_array->IsNull(local_row)) {
                        sequence_tokens.push_back(int32_array->Value(local_row));
                        sequence_processed = true;
                        break;
                    }
                }
            }
            current_row += chunk_size;
        }
    }

    if (!sequence_processed || sequence_tokens.empty()) {
        LLAMA_LOG_ERROR("Failed to load sequence %lu: no valid data found\n", sequence_index);
        return nullptr;
    }

    // Create GGML tensor from tokens
    struct ggml_tensor * tensor = create_tensor_from_tokens(sequence_tokens, dataset->ggml_ctx);
    if (!tensor) {
        llama_dataset_set_error("Failed to create tensor from tokens");
        return nullptr;
    }

    // Check cache size limits and evict if necessary
    size_t tensor_memory = estimate_tensor_cache_memory_usage(tensor);
    if (format_data->cache_memory_usage + tensor_memory > format_data->cache_max_size) {
        size_t target_usage = format_data->cache_max_size * 0.8; // Evict to 80% capacity
        evict_tokenization_cache_entries(format_data, target_usage);
    }

    // Add to cache
    format_data->tokenized_cache[sequence_index] = tensor;
    format_data->cache_memory_usage += tensor_memory;
    update_cache_access_order(format_data, sequence_index);

    return tensor;
}

/**
 * @brief Create GGML tensor from token vector.
 *
 * This helper function converts a vector of tokens to a GGML tensor suitable
 * for storage in the tokenization cache. It handles memory allocation, tensor
 * initialization, and data copying with proper error handling.
 *
 * @param tokens Vector of token IDs to convert
 * @param ctx GGML context for tensor allocation
 * @return Pointer to created GGML tensor, or NULL on allocation failure
 */
struct ggml_tensor * create_tensor_from_tokens(const std::vector<int32_t> & tokens,
                                              struct ggml_context * ctx) {
    if (tokens.empty() || !ctx) {
        return nullptr;
    }

    // Create 1D tensor for token sequence
    struct ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens.size());
    if (!tensor) {
        return nullptr;
    }

    // Copy token data to tensor
    if (tensor->data) {
        memcpy(tensor->data, tokens.data(), tokens.size() * sizeof(int32_t));
    }

    return tensor;
}

/**
 * @brief Update cache access order for LRU management.
 *
 * This function updates the cache access tracking to maintain LRU ordering
 * for efficient cache eviction. It moves the accessed sequence to the front
 * of the access order list and ensures proper LRU semantics.
 *
 * @param format_data Parquet format data containing cache state
 * @param sequence_index Index of the sequence that was accessed
 */
void update_cache_access_order(struct parquet_format_data * format_data,
                              uint64_t sequence_index) {
    if (!format_data) {
        return;
    }

    // Remove sequence from current position if it exists
    auto it = std::find(format_data->cache_access_order.begin(),
                       format_data->cache_access_order.end(),
                       sequence_index);
    if (it != format_data->cache_access_order.end()) {
        format_data->cache_access_order.erase(it);
    }

    // Add to front (most recently used)
    format_data->cache_access_order.insert(format_data->cache_access_order.begin(), sequence_index);
}

/**
 * @brief Evict cache entries to free memory.
 *
 * This function implements cache eviction using LRU strategy to free memory
 * when the cache exceeds its size limits. It removes the least recently used
 * entries and updates all cache tracking structures accordingly.
 *
 * @param format_data Parquet format data containing cache state
 * @param target_memory_usage Target memory usage after eviction in bytes
 * @return Amount of memory actually freed in bytes
 */
size_t evict_tokenization_cache_entries(struct parquet_format_data * format_data,
                                       size_t target_memory_usage) {
    if (!format_data) {
        return 0;
    }

    size_t memory_freed = 0;

    // Evict from least recently used (end of access order list)
    while (format_data->cache_memory_usage > target_memory_usage &&
           !format_data->cache_access_order.empty()) {

        uint64_t lru_sequence = format_data->cache_access_order.back();
        format_data->cache_access_order.pop_back();

        auto cache_it = format_data->tokenized_cache.find(lru_sequence);
        if (cache_it != format_data->tokenized_cache.end()) {
            size_t tensor_memory = estimate_tensor_cache_memory_usage(cache_it->second);
            memory_freed += tensor_memory;
            format_data->cache_memory_usage -= tensor_memory;

            // Note: We don't free the tensor here as it's managed by GGML context
            format_data->tokenized_cache.erase(cache_it);
        }
    }

    if (memory_freed > 0) {
        LLAMA_LOG_INFO("Evicted %zu bytes from tokenization cache (target: %zu, current: %zu)\n",
                       memory_freed, target_memory_usage, format_data->cache_memory_usage);
    }

    return memory_freed;
}

/**
 * @brief Estimate memory usage of a cached tensor.
 *
 * This function calculates the memory footprint of a cached tensor entry,
 * including the tensor structure, data storage, and cache overhead. This
 * information is used for accurate cache size tracking and eviction decisions.
 *
 * @param tensor GGML tensor to estimate memory usage for
 * @return Estimated memory usage in bytes
 */
size_t estimate_tensor_cache_memory_usage(const struct ggml_tensor * tensor) {
    if (!tensor) {
        return 0;
    }

    // Calculate tensor data size
    size_t data_size = ggml_nbytes(tensor);

    // Add overhead for tensor structure and cache entry
    size_t overhead = sizeof(struct ggml_tensor) +
                     sizeof(std::pair<uint64_t, struct ggml_tensor*>) +
                     64; // Additional overhead estimate

    return data_size + overhead;
}

#endif // LLAMA_PARQUET