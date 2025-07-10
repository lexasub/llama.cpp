// Main utility for converting a text dataset to the GGUF format for training models in llama.cpp.
//
// Logic:
// 1. Parses command line arguments.
// 2. Loads the tokenizer model.
// 3. Uses the llama_gguf_converter class to perform the entire conversion process:
//    - First pass over the input data to collect metadata (sequence lengths).
//    - Creation of the GGUF file and writing all collected metadata to it.
//    - Second pass over the input data to add each sequence as a separate tensor to the GGUF file.
// 4. After successful conversion, uses llama_gguf_reader to read and print
//    some meta-information and the first record from the created GGUF file.
//
// This two-pass approach allows processing datasets significantly larger than
// available RAM.

#include <algorithm>  // For std::min
#include <array>      // For std::array
#include <cinttypes>  // For PRIu64
#include <iostream>
#include <limits>     // For std::numeric_limits
#include <memory>     // For std::unique_ptr
#include <string>
#include <vector>

#include "dataset-to-gguf/llama-gguf-converter.h"  // Include our new llama_gguf_converter class
#include "dataset-to-gguf/llama-gguf-reader.h"
#include "llama.h"  // For llama_backend_init, llama_backend_free, llama_model_load_from_file, llama_model_free

// Structure for storing command line parameters
struct llama_training_data_params {
    std::string model_path          = "models/7B/ggml-model-f16.gguf"; // Path to the model for the tokenizer
    std::string input_path          = "input.txt";                     // Path to the input text file
    std::string output_path         = "output.gguf";                   // Path to save the GGUF file
    int32_t     max_seq_len         = 2048;                            // Maximum sequence length
    bool        pre_tokenized       = false;                           // Flag: if true, input data is already tokenized (token IDs as numbers)
    std::string input_type          = "text";                          // Type of input data (e.g., "text", "parquet")
    bool        do_preview          = false;                           // Flag: if true, perform a preview
    int32_t     preview_count       = 1;                               // Number of sequences for preview
    bool        detokenize_preview  = false;                           // Flag: if true, detokenize preview
    std::string parquet_text_column = "text";                          // Column name for raw text in Parquet files
    std::string parquet_tokens_column = "tokens";                      // Column name for pre-tokenized data (list<int32>) in Parquet files
};

// Forward declaration of the parameter parsing function
void llama_training_data_params_parse(int argc, char ** argv, llama_training_data_params & params);

// Function for parsing command line arguments
void llama_training_data_params_parse(int argc, char ** argv, llama_training_data_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--vocab-model" || arg == "-m") {
            params.model_path = argv[++i];
        } else if (arg == "--input" || arg == "-i") {
            params.input_path = argv[++i];
        } else if (arg == "--output" || arg == "-o") {
            params.output_path = argv[++i];
        } else if (arg == "--max-seq-len" || arg == "-l") {
            params.max_seq_len = std::stoi(argv[++i]);
        } else if (arg == "--pre-tokenized" || arg == "-p") {
            params.pre_tokenized = true;
        } else if (arg == "--input-type" || arg == "-t") {
            params.input_type = argv[++i];
        } else if (arg == "--preview") {
            params.do_preview = true; // Enable preview
        } else if (arg == "--preview-count") {
            params.preview_count = std::stoi(argv[++i]);
            if (params.preview_count <= 0) {
                fprintf(stderr, "error: --preview-count must be a positive integer.\n");
                exit(1);
            }
            params.do_preview = true; // Enable preview if count is specified
        } else if (arg == "--detokenize-preview") {
            params.detokenize_preview = true;
            params.do_preview = true; // Enable preview if detokenization is specified
        } else if (arg == "--parquet-text-column") {
            params.parquet_text_column = argv[++i];
        } else if (arg == "--parquet-tokens-column") {
            params.parquet_tokens_column = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -h, --help            show this help message and exit\n");
            printf("  -m, --vocab-model     path to model for tokenizer (default: %s)\n", params.model_path.c_str());
            printf("  -i, --input           path to input text file (default: %s)\n", params.input_path.c_str());
            printf("  -o, --output          path to output gguf file (default: %s)\n", params.output_path.c_str());
            printf("  -l, --max-seq-len     max sequence length (default: %d)\n", params.max_seq_len);
            printf("  -p, --pre-tokenized   input file contains pre-tokenized data (space-separated token IDs)\n");
            printf("  -t, --input-type      type of input data (e.g., 'text', 'parquet') (default: %s)\n", params.input_type.c_str());
            printf("  --preview             read and print metadata and first sequence from the output GGUF file (enables preview)\n");
            printf("  --preview-count <N>   number of sequences to preview (default: 1, implies --preview)\n");
            printf("  --detokenize-preview  detokenize previewed sequences (implies --preview)\n");
            printf("  --parquet-text-column <name>  column name for raw text in Parquet files (default: 'text')\n");
            printf("  --parquet-tokens-column <name> column name for pre-tokenized data (list<int32>) in Parquet files (default: 'tokens')\n");
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            exit(1);
        }
    }
}

int main(int argc, char ** argv) {
    llama_training_data_params params_raw;
    llama_training_data_params_parse(argc, argv, params_raw);

    // Print parameters for verification
    printf("Parameters:\n");
    printf("  Model for tokenizer: %s\n", params_raw.model_path.c_str());
    printf("  Input file: %s\n", params_raw.input_path.c_str());
    printf("  Output file: %s\n", params_raw.output_path.c_str());
    printf("  Max sequence length: %d\n", params_raw.max_seq_len);
    printf("  Pre-tokenized input: %s\n", params_raw.pre_tokenized ? "Yes" : "No");
    printf("  Input type: %s\n", params_raw.input_type.c_str());
    printf("  Do preview: %s\n", params_raw.do_preview ? "Yes" : "No");
    if (params_raw.do_preview) {
        printf("  Preview count: %d\n", params_raw.preview_count);
        printf("  Detokenize preview: %s\n", params_raw.detokenize_preview ? "Yes" : "No");
    }
    if (params_raw.input_type == "parquet") {
        printf("  Parquet text column: %s\n", params_raw.parquet_text_column.c_str());
        printf("  Parquet tokens column: %s\n", params_raw.parquet_tokens_column.c_str());
    }
    printf("\n");

    // Initialize llama.cpp
    llama_backend_init();

    // Load the model for its tokenizer
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(params_raw.model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr, "error: failed to load model from %s\n", params_raw.model_path.c_str());
        llama_backend_free();
        return 1;
    }

    // --- Diagnostic Test: Reading tokenizer model GGUF file ---
    printf("--- Diagnostic Test: Reading tokenizer model GGUF file ---\n");
    try {
        llama_gguf_reader tokenizer_model_reader(params_raw.model_path);
        if (tokenizer_model_reader.llama_gguf_reader_is_initialized()) {
            printf("  Tokenizer Model GGUF file opened successfully.\n");
            printf("  Tokenizer Model Name: %s\n", tokenizer_model_reader.llama_gguf_reader_get_metadata_str("general.name", "N/A").c_str());
            printf("  Tokenizer Model Architecture: %s\n", tokenizer_model_reader.llama_gguf_reader_get_metadata_str("general.architecture", "N/A").c_str());
            printf("  Tokenizer Model Tensor Count: %ld\n", tokenizer_model_reader.llama_gguf_reader_get_tensor_count());
            printf("  Diagnostic Test: Tokenizer Model GGUF read successful.\n");
        } else {
            fprintf(stderr, "error: Diagnostic Test: Tokenizer Model GGUF read failed to initialize.\n");
            llama_model_free(model); // Free model before exiting
            llama_backend_free();
            return 1;
        }
    } catch (const std::runtime_error & e) {
        fprintf(stderr, "error: Diagnostic Test: Tokenizer Model GGUF read failed: %s\n", e.what());
        llama_model_free(model); // Free model before exiting
        llama_backend_free();
        return 1;
    }
    printf("--- End of Diagnostic Test ---\n\n");


    // Prepare parameters for llama_gguf_converter
    llama_convert_params convert_params;
    convert_params.input_path = params_raw.input_path;
    convert_params.output_path = params_raw.output_path;
    convert_params.max_seq_len = params_raw.max_seq_len;
    convert_params.pre_tokenized = params_raw.pre_tokenized;
    convert_params.input_type = params_raw.input_type;
    convert_params.model = model; // Pass pointer to the loaded model
    convert_params.parquet_text_column = params_raw.parquet_text_column; // Pass Parquet text column name
    convert_params.parquet_tokens_column = params_raw.parquet_tokens_column; // Pass Parquet tokens column name

    // Create and run the converter
    llama_gguf_converter converter;
    bool success = converter.llama_gguf_converter_convert(convert_params);

    // Clean up llama model
    llama_model_free(model);
    llama_backend_free();

    if (!success) {
        fprintf(stderr, "error: GGUF conversion failed.\n");
        return 1;
    }

    printf("Conversion successful!\n");
    printf("Output file: %s\n", params_raw.output_path.c_str());

    // --- Preview generated GGUF file (if requested) ---
    if (params_raw.do_preview) {
        printf("\n--- Previewing generated GGUF file ---\n");
        try {
            llama_gguf_reader reader(params_raw.output_path);

            if (!reader.llama_gguf_reader_is_initialized()) {
                fprintf(stderr, "error: llama_gguf_reader failed to initialize for preview.\n");
                return 1;
            }

            printf("  Dataset Name: %s\n", reader.llama_gguf_reader_get_metadata_str("training.dataset.name", "N/A").c_str());
            printf("  Sequence Count: %lu\n", reader.llama_gguf_reader_get_metadata_u64("training.sequence.count", 0));
            printf("  Tokenizer Model: %s\n", reader.llama_gguf_reader_get_metadata_str("training.tokenizer.gguf.model", "N/A").c_str());

            int64_t tensor_count = reader.llama_gguf_reader_get_tensor_count();
            if (tensor_count > 0) {
                // Print N first sequences
                for (int64_t i = 0; i < std::min((int64_t)params_raw.preview_count, tensor_count); ++i) {
                    printf("  Sequence (training.tensor.%" PRId64 "):\n", i);
                    std::vector<llama_token> sequence_tokens;
                    if (reader.llama_gguf_reader_read_tensor_data(i, sequence_tokens)) {
                        printf("    Length: %zu tokens\n", sequence_tokens.size());
                        printf("    Tokens: [");
                        for (size_t j = 0; j < std::min((size_t)10, sequence_tokens.size()); ++j) { // Print up to 10 tokens
                            printf("%d%s", sequence_tokens[j], (j == std::min((size_t)10, sequence_tokens.size()) - 1) ? "" : ", ");
                        }
                        if (sequence_tokens.size() > 10) {
                            printf("...");
                        }
                        printf("]\n");

                        if (params_raw.detokenize_preview) {
                            // Detokenization
                            std::string detokenized_text = "";
                            // Buffer for a single token
                            std::array<char, 256> piece_buf; // Large enough buffer for a single token
                            for (llama_token token : sequence_tokens) {
                                int n_chars = llama_token_to_piece(llama_model_get_vocab(model), token, piece_buf.data(), piece_buf.size(), 1, false);
                                if (n_chars > 0) {
                                    detokenized_text.append(piece_buf.data(), n_chars);
                                }
                            }
                            printf("    Detokenized: \"%s\"\n", detokenized_text.c_str());
                        }

                    } else {
                        fprintf(stderr, "    Error: Could not read data for sequence %" PRId64 ".\n", i);
                    }
                }
            } else {
                printf("  No sequences found in the GGUF file.\n");
            }

        } catch (const std::runtime_error & e) {
            fprintf(stderr, "error: GGUF preview failed: %s\n", e.what());
            return 1;
        }
        printf("--- End of GGUF file preview ---\n");
    }

    return 0;
}
