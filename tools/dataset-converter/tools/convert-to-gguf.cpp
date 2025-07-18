#include <cstdio>
#include <cstring>

#include "arg.h"
#include "common/common.h"
#include "llama-dataset.h"
#include "llama-impl.h"

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
