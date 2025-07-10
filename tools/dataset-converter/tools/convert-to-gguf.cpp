#include "llama-dataset.h"
#include "../../common/common.h"
#include "../../common/log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

void print_usage(const char* program_name) {
    fprintf(stderr, "Usage: %s [options] <input_file> <output_file>\n\n", program_name);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -h, --help            Show this help message and exit\n");
    fprintf(stderr, "  --model MODEL         Path to the model for tokenization (required for text input)\n");
    fprintf(stderr, "  --streaming           Use streaming mode for large datasets\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Input formats supported: GGUF, text, Parquet\n");
    fprintf(stderr, "Output format: GGUF\n");
}

int main(int argc, char** argv) {
    const char* model_path = NULL;
    const char* input_path = NULL;
    const char* output_path = NULL;
    bool streaming = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--model") == 0) {
            if (i + 1 < argc) {
                model_path = argv[++i];
            } else {
                fprintf(stderr, "Error: --model requires a path argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--streaming") == 0) {
            streaming = true;
        } else if (input_path == NULL) {
            input_path = argv[i];
        } else if (output_path == NULL) {
            output_path = argv[i];
        } else {
            fprintf(stderr, "Error: too many arguments\n");
            print_usage(argv[0]);
            return 1;
        }
    }

    // Check required arguments
    if (input_path == NULL || output_path == NULL) {
        fprintf(stderr, "Error: input and output paths are required\n");
        print_usage(argv[0]);
        return 1;
    }

    // Detect input format
    const char* ext = strrchr(input_path, '.');
    if (!ext) {
        fprintf(stderr, "Error: input file has no extension\n");
        return 1;
    }

    struct llama_dataset* dataset = NULL;
    struct llama_model* model = NULL;

    // Initialize llama backend if needed for text processing
    if (strcasecmp(ext, ".txt") == 0) {
        llama_backend_init();
    }

    // Load dataset based on file extension using the new simple interface
    if (strcasecmp(ext, ".gguf") == 0) {
        printf("Loading GGUF dataset from %s\n", input_path);
        // Use new simple interface
        dataset = from_gguf(input_path);
    } else if (strcasecmp(ext, ".txt") == 0) {
        if (!model_path) {
            fprintf(stderr, "Error: text input requires --model parameter\n");
            return 1;
        }

        printf("Loading model from %s for tokenization\n", model_path);
        llama_model_params model_params = llama_model_default_params();
        model = llama_load_model_from_file(model_path, model_params);
        if (!model) {
            fprintf(stderr, "Error: failed to load model\n");
            return 1;
        }

        printf("Loading text dataset from %s\n", input_path);
        // Use new simple interface
        dataset = from_txt(input_path, model);
    } else if (strcasecmp(ext, ".parquet") == 0) {
        printf("Loading Parquet dataset from %s\n", input_path);
        // Use new simple interface
        dataset = from_parquet(input_path);
    } else {
        fprintf(stderr, "Error: unsupported input format: %s\n", ext);
        return 1;
    }

    // Check if dataset was loaded successfully
    if (!dataset) {
        fprintf(stderr, "Error: failed to load dataset: %s\n", llama_dataset_get_error_message());
        if (model) {
            llama_free_model(model);
        }
        if (strcasecmp(ext, ".txt") == 0) {
            llama_backend_free();
        }
        return 1;
    }

    // Convert to GGUF using the new simple interface
    printf("Converting dataset to GGUF format: %s\n", output_path);
    to_gguf(dataset, output_path);

    // Check for conversion errors
    if (llama_dataset_has_error()) {
        fprintf(stderr, "Error during conversion: %s\n", llama_dataset_get_error_message());
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
    printf("Conversion complete!\n");
    printf("Sequences: %llu\n", (unsigned long long)n_sequences(dataset));

    // Cleanup
    llama_dataset_free(dataset);
    if (model) {
        llama_free_model(model);
    }
    if (strcasecmp(ext, ".txt") == 0) {
        llama_backend_free();
    }

    return 0;
}
