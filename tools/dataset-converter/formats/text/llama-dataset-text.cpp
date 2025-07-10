#include "llama-dataset-text.h"
#include "llama-dataset-internal.h"
#include "llama-dataset-utils.h"
#include "../../common/log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <vector>

#include "llama-dataset-text.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

#include "../../common/log.h"
#include "../../ggml/include/ggml.h"
#include "../../ggml/include/gguf.h"
#include "../../include/llama.h"
#include "llama-dataset-internal.h"
#include "llama-dataset-utils.h"
#include "llama-impl.h"

/**
 * @brief Tokenize a line of text using the provided model.
 *
 * This function tokenizes a line of text using the provided model and returns the tokens.
 *
 * @param model Model to use for tokenization
 * @param line Line of text to tokenize
 * @param line_len Length of the line
 * @param tokens Output buffer for tokens
 * @param n_tokens_max Maximum number of tokens to output
 * @return Number of tokens, or negative value on error
 */
int32_t tokenize_line(struct llama_model * model, const char * line, int32_t line_len, llama_token * tokens, int32_t n_tokens_max) {
    if (!model || !line || !tokens) {
        return -1;
    }

    // Get the vocabulary from the model
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    if (!vocab) {
        return -1;
    }

    // Tokenize the line
    // add_special=false: don't add BOS/EOS tokens for individual lines
    // parse_special=false: treat special tokens as regular text
    return llama_tokenize(vocab, line, line_len, tokens, n_tokens_max, false, false);
}
bool process_and_cache_text_sequences(struct llama_dataset * dataset,
                                     const std::vector<std::vector<llama_token>> & tokenized_lines,
                                     int32_t max_seq_len,
                                     bool apply_padding);
/**
 * @brief Load a dataset from a text file with tokenization.
 *
 * This function loads a dataset from a text file, tokenizes each line using the provided model,
 * and creates a GGUF context with the tokenized sequences.
 *
 * @param path Path to the text file
 * @param model Model to use for tokenization
 * @param streaming Whether to use streaming mode (currently not supported for text)
 * @return Pointer to the dataset, or NULL on error
 */
struct llama_dataset * llama_dataset_load_text_internal(const char * path, struct llama_model * model, bool streaming) {
    if (!path) {
        set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Path cannot be null");
        return nullptr;
    }

    if (!model) {
        set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Model cannot be null for text tokenization");
        return nullptr;
    }

    // Check if file exists
    std::ifstream file(path);
    if (!file.is_open()) {
        set_error_with_code(DATASET_ERROR_FILE_NOT_FOUND, "Text file not found");
        return nullptr;
    }

    // If streaming mode is requested, warn that it's not supported for text files
    if (streaming) {
        LLAMA_LOG_WARN("Streaming mode not supported for text files, falling back to full loading");
        streaming = false;
    }

    // Create dataset structure
    struct llama_dataset * dataset = dataset_alloc(DATASET_TEXT, streaming);
    if (!dataset) {
        return nullptr; // Error already set by dataset_alloc
    }

    // Read the file line by line and tokenize each line
    std::vector<std::vector<llama_token>> tokenized_lines;
    std::string line;
    const int32_t max_tokens_per_line = 2048; // Reasonable limit for a line
    std::vector<llama_token> tokens(max_tokens_per_line);

    // First pass: tokenize all lines and collect statistics
    int32_t max_seq_len = 0;
    uint64_t total_lines = 0;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue; // Skip empty lines
        }

        // Tokenize the line
        int32_t n_tokens = tokenize_line(model, line.c_str(), line.length(), tokens.data(), max_tokens_per_line);

        if (n_tokens <= 0) {
            // Error or empty line after tokenization
            if (n_tokens < 0) {
                LLAMA_LOG_WARN("Failed to tokenize line %zu: %s", total_lines + 1, line.c_str());
            }
            continue;
        }

        // Store the tokenized line
        tokenized_lines.push_back(std::vector<llama_token>(tokens.data(), tokens.data() + n_tokens));

        // Update statistics
        max_seq_len = std::max(max_seq_len, n_tokens);
        total_lines++;
    }

    // Check if we have any valid lines
    if (tokenized_lines.empty()) {
        set_error_with_code(DATASET_ERROR_TOKENIZATION_FAILED, "No valid lines found in the text file");
        llama_dataset_free(dataset);
        return nullptr;
    }

    // Create GGUF context
    struct gguf_context * ctx = gguf_init_empty();
    if (!ctx) {
        set_error_with_code(DATASET_ERROR_CONTEXT_CREATION_FAILED, "Failed to create GGUF context");
        llama_dataset_free(dataset);
        return nullptr;
    }

    // Add metadata
    gguf_set_val_str(ctx, DATASET_SOURCE_FORMAT, "text");
    gguf_set_val_u32(ctx, DATASET_SEQUENCE_COUNT, tokenized_lines.size());
    gguf_set_val_u32(ctx, DATASET_MAX_LENGTH, max_seq_len);

    // Add tokenizer information
    char model_desc[256];
    int32_t desc_len = llama_model_desc(model, model_desc, sizeof(model_desc));
    if (desc_len > 0) {
        gguf_set_val_str(ctx, DATASET_TOKENIZER, model_desc);
    } else {
        gguf_set_val_str(ctx, DATASET_TOKENIZER, "unknown");
    }

    // Add creation time
    time_t now = time(nullptr);
    char time_str[32];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&now));
    gguf_set_val_str(ctx, DATASET_CREATION_TIME, time_str);

    // Create GGML context for tensors
    struct ggml_init_params params;
    params.mem_size = 16 * 1024 * 1024; // Start with 16MB, will be resized as needed
    params.mem_buffer = NULL;
    params.no_alloc = false;

    struct ggml_context * ggml_ctx = ggml_init(params);
    if (!ggml_ctx) {
        set_error_with_code(DATASET_ERROR_CONTEXT_CREATION_FAILED, "Failed to create GGML context");
        gguf_free(ctx);
        llama_dataset_free(dataset);
        return nullptr;
    }

    // Set dataset fields first so process_and_cache_text_sequences can use them
    dataset->ctx = ctx;
    dataset->ggml_ctx = ggml_ctx;
    dataset->n_seq = tokenized_lines.size();

    // Use enhanced sequence processing with variable length handling and caching
    // This handles tensor creation, padding (if needed), and caching optimization
    if (!process_and_cache_text_sequences(dataset, tokenized_lines, max_seq_len, false)) {
        // Error already set by process_and_cache_text_sequences
        llama_dataset_free(dataset);
        return nullptr;
    }

    llama_dataset_clear_error(); // Clear any previous errors
    return dataset;
}

/**
 * @brief Create a tensor for a tokenized sequence with proper handling of variable lengths.
 *
 * This function creates a tensor for a tokenized sequence, handling variable sequence lengths
 * efficiently. It can optionally apply padding if needed for batch processing.
 *
 * @param ggml_ctx GGML context to create the tensor in
 * @param tokens Token data
 * @param n_tokens Number of tokens
 * @param tensor_name Name for the tensor
 * @param pad_to_length Optional padding length (0 = no padding)
 * @return Pointer to the created tensor, or NULL on error
 */
struct ggml_tensor * create_text_sequence_tensor(struct ggml_context * ggml_ctx,
                                               const llama_token * tokens,
                                               int32_t n_tokens,
                                               const char * tensor_name,
                                               int32_t pad_to_length) {
    if (!ggml_ctx || !tokens || n_tokens <= 0) {
        set_error("Invalid parameters for tensor creation");
        return nullptr;
    }

    // Determine final tensor length (with or without padding)
    int32_t tensor_length = (pad_to_length > 0 && pad_to_length > n_tokens) ? pad_to_length : n_tokens;

    // Create tensor with appropriate length
    struct ggml_tensor * tensor = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I32, tensor_length);
    if (!tensor) {
        set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to allocate tensor for sequence");
        return nullptr;
    }

    // Set tensor name if provided
    if (tensor_name) {
        ggml_set_name(tensor, tensor_name);
    }

    // Verify tensor data allocation
    if (!tensor->data) {
        set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION, "Tensor data allocation failed");
        return nullptr;
    }

    // Copy token data to tensor
    memcpy(tensor->data, tokens, n_tokens * sizeof(llama_token));

    // Apply padding if needed
    if (pad_to_length > 0 && pad_to_length > n_tokens) {
        // Fill remaining space with padding token (typically 0 or a special padding token)
        llama_token * tensor_data = (llama_token *)tensor->data;
        for (int32_t i = n_tokens; i < pad_to_length; i++) {
            tensor_data[i] = 0; // Use 0 as padding token
        }
    }

    return tensor;
}

/**
 * @brief Process and cache tokenized sequences with variable length handling.
 *
 * This function processes tokenized sequences and creates optimized tensor cache
 * for efficient repeated access. It handles variable sequence lengths properly
 * and can apply padding if needed for batch processing.
 *
 * @param dataset Dataset to process
 * @param tokenized_lines Vector of tokenized sequences
 * @param max_seq_len Maximum sequence length in the dataset
 * @param apply_padding Whether to apply padding to sequences
 * @return true on success, false on error
 */
bool process_and_cache_text_sequences(struct llama_dataset * dataset,
                                     const std::vector<std::vector<llama_token>> & tokenized_lines,
                                     int32_t max_seq_len,
                                     bool apply_padding) {
    if (!dataset || !dataset->ctx || !dataset->ggml_ctx) {
        set_error("Invalid dataset for sequence processing");
        return false;
    }

    if (tokenized_lines.empty()) {
        set_error("No tokenized sequences to process");
        return false;
    }

    // Clear existing tensors from GGUF context if any
    // Note: In a real implementation, we might want to check if tensors already exist

    // Process each tokenized sequence
    for (size_t i = 0; i < tokenized_lines.size(); i++) {
        const std::vector<llama_token> & tokens = tokenized_lines[i];

        if (tokens.empty()) {
            LLAMA_LOG_WARN("Skipping empty sequence at index %zu", i);
            continue;
        }

        // Create tensor name
        char tensor_name[32];
        snprintf(tensor_name, sizeof(tensor_name), "sequence.%zu", i);

        // Determine padding length if needed
        int32_t pad_length = apply_padding ? max_seq_len : 0;

        // Create tensor for this sequence
        struct ggml_tensor * tensor = create_text_sequence_tensor(
            dataset->ggml_ctx,
            tokens.data(),
            tokens.size(),
            tensor_name,
            pad_length
        );

        if (!tensor) {
            // Error already set by create_text_sequence_tensor
            return false;
        }

        // Add tensor to GGUF context
        gguf_add_tensor(dataset->ctx, tensor);
    }

    // Update dataset sequence count
    dataset->n_seq = tokenized_lines.size();

    // Cache tensor pointers for efficient repeated access with variable length optimization
    if (!dataset_cache_tensors(dataset)) {
        return false; // Error already set by dataset_cache_tensors
    }

    // Validate and optimize the tensor cache for variable sequence lengths
    if (!validate_and_optimize_tensor_cache(dataset)) {
        LLAMA_LOG_WARN("Tensor cache validation failed, but continuing");
        // Don't fail completely, just log the warning
    }

    return true;
}
/**
 * @brief Optimize tensor cache for variable sequence length access patterns.
 *
 * This function optimizes the tensor cache specifically for text sequences with
 * variable lengths, improving access patterns and memory efficiency.
 *
 * @param dataset Dataset to optimize
 * @return true on success, false on error
 */
bool optimize_text_sequence_cache(struct llama_dataset * dataset) {
    if (!dataset || !dataset->cached_tensors || dataset->n_seq == 0) {
        set_error("Invalid dataset for cache optimization");
        return false;
    }

    // Collect sequence length statistics for optimization
    std::vector<int32_t> sequence_lengths;
    sequence_lengths.reserve(dataset->n_seq);

    uint64_t total_tokens = 0;
    int32_t min_length = INT32_MAX;
    int32_t max_length = 0;

    for (uint64_t i = 0; i < dataset->n_seq; i++) {
        struct ggml_tensor * tensor = dataset->cached_tensors[i];
        if (!tensor) continue;

        int32_t length = (int32_t)tensor->ne[0];
        sequence_lengths.push_back(length);

        total_tokens += length;
        min_length = std::min(min_length, length);
        max_length = std::max(max_length, length);
    }

    if (sequence_lengths.empty()) {
        set_error("No valid sequences found for optimization");
        return false;
    }

    // Calculate statistics
    double avg_length = (double)total_tokens / sequence_lengths.size();

    // Sort lengths to find median and percentiles
    std::sort(sequence_lengths.begin(), sequence_lengths.end());
    int32_t median_length = sequence_lengths[sequence_lengths.size() / 2];
    int32_t p75_length = sequence_lengths[(sequence_lengths.size() * 3) / 4];
    int32_t p90_length = sequence_lengths[(sequence_lengths.size() * 9) / 10];

    // Update metadata with optimization statistics
    if (dataset->ctx) {
        gguf_set_val_u32(dataset->ctx, "dataset.min_length", min_length);
        gguf_set_val_u32(dataset->ctx, "dataset.max_length", max_length);
        gguf_set_val_f32(dataset->ctx, "dataset.avg_length", (float)avg_length);
        gguf_set_val_u32(dataset->ctx, "dataset.median_length", median_length);
        gguf_set_val_u32(dataset->ctx, "dataset.p75_length", p75_length);
        gguf_set_val_u32(dataset->ctx, "dataset.p90_length", p90_length);

        // Add optimization recommendations
        if (max_length > min_length * 3) {
            gguf_set_val_str(dataset->ctx, "dataset.optimization_note",
                           "High length variation detected - consider bucketing for batch processing");
        } else {
            gguf_set_val_str(dataset->ctx, "dataset.optimization_note",
                           "Sequence lengths are relatively uniform");
        }
    }

    LLAMA_LOG_INFO("Text sequence cache optimized: %zu sequences, lengths [%d-%d], avg=%.1f, median=%d",
                   sequence_lengths.size(), min_length, max_length, avg_length, median_length);

    return true;
}
