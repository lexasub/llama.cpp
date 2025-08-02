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
#include "llama-dataset-internal.h"
#include "llama-dataset-utils.h"
#include "llama.h"
#include "llama-impl.h"



//
// Schema analysis functions are now implemented in llama-dataset-parquet-schema.cpp
// Data conversion functions are now implemented in llama-dataset-parquet-conversion.cpp
//




bool llama_dataset_validate_parquet_schema(const char * path, struct parquet_schema_info * info);
/**
 * @brief Load a dataset from a Parquet file (internal implementation).
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
                params->dataset_text_column,
                params->dataset_token_column,
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

        // Initialize tokenization engine if needed
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

            LLAMA_LOG_INFO("Tokenization engine initialized with %zu MB cache\n",
                          params->dataset_tokenization_cache_size);
        } else {
            format_data->tokenizer = nullptr;
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

    // Clear vectors
    data->text_columns.clear();
    data->token_columns.clear();
    data->tokenized_cache.clear();

    // Arrow/Parquet objects are automatically cleaned up by shared_ptr
    data->table.reset();
    data->reader.reset();

    // Free the structure itself
    delete data;
}

/**
 * @brief Validate Parquet file schema for dataset compatibility.
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

#endif // LLAMA_PARQUET
