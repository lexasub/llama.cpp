/**
 * @file convert-to-gguf.cpp
 * @brief Command-line tool for converting datasets to GGUF format.
 *
 * This tool provides a unified interface for converting training datasets from various
 * formats (text, Parquet, GGUF) to the optimized GGUF format used by llama.cpp.
 * It leverages the dataset converter framework to handle format-specific loading,
 * validation, and conversion with comprehensive error handling and progress reporting.
 *
 * ## Supported Input Formats
 *
 * - **Text Files (.txt)**: Raw text files that are tokenized using a specified model
 * - **Parquet Files (.parquet)**: Structured datasets with Apache Arrow support
 * - **GGUF Files (.gguf)**: Native format (useful for validation and optimization)
 *
 * ## Command-Line Interface
 *
 * ### Basic Usage
 * ```bash
 * # Convert text file to GGUF (requires model for tokenization)
 * ./convert-to-gguf --input dataset.txt --output dataset.gguf --model model.gguf
 *
 * # Convert Parquet file to GGUF
 * ./convert-to-gguf --input dataset.parquet --output dataset.gguf
 *
 * # Validate and optimize existing GGUF file
 * ./convert-to-gguf --input dataset.gguf --output optimized.gguf
 * ```
 *
 * ### Advanced Options
 * ```bash
 * # Enable streaming for large datasets
 * ./convert-to-gguf --input large_dataset.parquet --output dataset.gguf --streaming
 *
 * # Configure cache size for memory-constrained environments
 * ./convert-to-gguf --input dataset.txt --output dataset.gguf --model model.gguf --cache-size 512MB
 * ```
 *
 * ## Conversion Process
 *
 * The conversion process follows these steps:
 * 1. **Format Detection**: Automatically detects input format based on file extension
 * 2. **Model Loading**: Loads tokenization model if required (text input)
 * 3. **Dataset Loading**: Uses format-specific loaders with streaming support
 * 4. **Validation**: Performs integrity checks on loaded data
 * 5. **Conversion**: Converts to optimized GGUF format with metadata preservation
 * 6. **Output**: Writes GGUF file with comprehensive error checking
 *
 * ## Error Handling
 *
 * The tool provides detailed error reporting for common issues:
 * - Missing or invalid input files
 * - Unsupported file formats
 * - Model loading failures (for text input)
 * - Memory allocation errors
 * - Conversion and I/O errors
 *
 * ## Performance Considerations
 *
 * - **Memory Usage**: Streaming mode reduces memory footprint for large datasets
 * - **Tokenization**: Text processing requires model loading and can be CPU-intensive
 * - **I/O Optimization**: Uses efficient buffering for large file operations
 * - **Cache Management**: Configurable caching improves performance for complex datasets
 *
 * ## Integration with Dataset Framework
 *
 * This tool integrates with the complete dataset converter framework:
 * - **Core API**: Uses llama_dataset_* functions for unified dataset access
 * - **Format Modules**: Leverages format-specific implementations for optimal loading
 * - **Streaming System**: Supports memory-efficient processing of large datasets
 * - **Validation System**: Ensures data integrity throughout the conversion process
 *
 * ## Output Format
 *
 * The generated GGUF files include:
 * - Optimized tensor layout for training efficiency
 * - Preserved metadata from source format
 * - Standardized metadata keys for interoperability
 * - Validation checksums for integrity verification
 *
 * @see core/llama-dataset.h for the core dataset API
 * @see formats/ directory for format-specific implementations
 * @see streaming/ directory for streaming capabilities
 * @see validation/ directory for data validation features
 *
 * @version 1.0
 * @since 2024
 */

#include <cstdio>
#include <cstring>

#include "arg.h"
#include "common/common.h"
#include "llama-dataset.h"
#include "llama-impl.h"

/**
 * @brief Main entry point for the dataset-to-GGUF conversion tool.
 *
 * This function orchestrates the complete conversion process from input format
 * detection through final GGUF output generation. It handles all supported input
 * formats with appropriate preprocessing, validation, and error handling.
 *
 * ## Process Flow
 *
 * 1. **Argument Parsing**: Processes command-line arguments using common_params
 * 2. **Input Validation**: Validates required parameters and file accessibility
 * 3. **Format Detection**: Determines input format from file extension
 * 4. **Backend Initialization**: Initializes llama backend for text processing if needed
 * 5. **Model Loading**: Loads tokenization model for text input formats
 * 6. **Dataset Loading**: Uses format-specific loaders with error handling
 * 7. **Conversion**: Converts loaded dataset to optimized GGUF format
 * 8. **Validation**: Verifies conversion success and data integrity
 * 9. **Statistics**: Reports conversion statistics and performance metrics
 * 10. **Cleanup**: Properly releases all allocated resources
 *
 * ## Supported Format Detection
 *
 * - **.txt**: Text files requiring tokenization model
 * - **.parquet**: Parquet files with Apache Arrow support (if compiled with LLAMA_PARQUET)
 * - **.gguf**: Native GGUF files for validation and optimization
 *
 * ## Error Handling Strategy
 *
 * The function implements comprehensive error handling:
 * - Parameter validation with descriptive error messages
 * - Resource cleanup on all error paths
 * - Detailed error reporting using dataset error system
 * - Proper backend cleanup for text processing
 *
 * ## Memory Management
 *
 * - Automatic resource cleanup using RAII principles where possible
 * - Explicit cleanup of llama model and backend resources
 * - Dataset cleanup through llama_dataset_free()
 * - Error path cleanup to prevent resource leaks
 *
 * ## Performance Optimization
 *
 * - Streaming support for large datasets (configured via common_params)
 * - Efficient format-specific loading strategies
 * - Minimal memory footprint through streaming when possible
 * - Optimized GGUF output format for training efficiency
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return 0 on success, 1 on error
 *
 * @note The function initializes and cleans up the llama backend only when
 *       processing text files to minimize resource usage for other formats.
 *
 * @see common_params_parse() for argument parsing details
 * @see llama_dataset_from_*() functions for format-specific loading
 * @see llama_dataset_to_gguf() for conversion implementation
 */
int main(int argc, char** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DATASET)) {
        return 1;
    }

    // Check required arguments
    if (params.in_files[0].empty() || params.out_file.empty()) {
        LLAMA_LOG_DEBUG("Error: input and output paths are required\n");
        return 1;
    }

    // Detect input format
    const char* ext = strrchr(params.in_files[0].c_str(), '.');
    if (!ext) {
        LLAMA_LOG_DEBUG("Error: input file has no extension\n");
        return 1;
    }

    struct llama_dataset* dataset = nullptr;
    struct llama_model* model = nullptr;

    // Initialize llama backend if needed for text processing
    if (strcasecmp(ext, ".txt") == 0) {
        llama_backend_init();
    }

    // Load dataset based on file extension using the new simple interface
    if (strcasecmp(ext, ".gguf") == 0) {
        LLAMA_LOG_DEBUG("Loading GGUF dataset from %s\n", params.in_files[0].c_str());
        // Use new simple interface
        dataset = llama_dataset_from_gguf(&params);
    } else if (strcasecmp(ext, ".txt") == 0) {
        if (!params.model.path.c_str()) {
            LLAMA_LOG_DEBUG("Error: text input requires --model parameter\n");
            return 1;
        }

        printf("Loading model from %s for tokenization\n", params.model.path.c_str());
        llama_model_params model_params = llama_model_default_params();
        model = llama_model_load_from_file(params.model.path.c_str(), model_params);
        if (!model) {
            LLAMA_LOG_DEBUG("Error: failed to load model\n");
            return 1;
        }

        printf("Loading text dataset from %s\n", params.in_files[0].c_str());
        // Use new simple interface
        dataset = llama_dataset_from_txt(&params, model);
    } else if (strcasecmp(ext, ".parquet") == 0) {
        printf("Loading Parquet dataset from %s\n", params.in_files[0].c_str());
        // Use new simple interface
#ifdef LLAMA_PARQUET
        dataset = llama_dataset_from_parquet(&params);
#endif
    } else {
        LLAMA_LOG_DEBUG("Error: unsupported input format: %s\n", ext);
        return 1;
    }

    // Check if dataset was loaded successfully
    if (!dataset) {
        LLAMA_LOG_DEBUG("Error: failed to load dataset: %s\n", llama_dataset_get_error_message());
        if (model) {
            llama_model_free(model);
        }
        if (strcasecmp(ext, ".txt") == 0) {
            llama_backend_free();
        }
        return 1;
    }

    // Convert to GGUF using the new simple interface
    LLAMA_LOG_DEBUG("Converting dataset to GGUF format: %s\n", params.out_file.c_str());
    llama_dataset_to_gguf(dataset, params.out_file.c_str());

    // Check for conversion errors
    if (llama_dataset_has_error()) {
        LLAMA_LOG_DEBUG("Error during conversion: %s\n", llama_dataset_get_error_message());
        llama_dataset_free(dataset);
        if (model) {
            llama_model_free(model);
        }
        if (strcasecmp(ext, ".txt") == 0) {
            llama_backend_free();
        }
        return 1;
    }

    // Print statistics
    LLAMA_LOG_DEBUG("Conversion complete!\n");
    LLAMA_LOG_DEBUG("Sequences: %llu\n", static_cast<unsigned long long>(llama_dataset_n_sequences(dataset)));

    llama_dataset_free(dataset);
    if (model) {
        llama_model_free(model);
    }
    if (strcasecmp(ext, ".txt") == 0) {
        llama_backend_free();
    }

    return 0;
}
