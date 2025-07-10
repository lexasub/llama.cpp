#pragma once

#include <string> // For std::string

#include "llama.h" // For struct llama_model

// Structure for passing conversion parameters.
struct llama_convert_params {
    std::string input_path;
    std::string output_path;
    int32_t max_seq_len;
    bool pre_tokenized;
    std::string input_type;
    const struct llama_model* model; // Pointer to the loaded model
    std::string parquet_text_column;
    std::string parquet_tokens_column;
};

// Class encapsulating the high-level logic for converting
// input data to the GGUF format.
struct llama_gguf_converter {
    // Default constructor.
    llama_gguf_converter() = default;

    // Method to execute the conversion process.
    // params: A structure containing all necessary parameters for conversion.
    // Returns true on successful conversion, false on error.
    bool llama_gguf_converter_convert(const struct llama_convert_params& params);
};
