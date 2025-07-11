#include "arg.h"
#include "common.h" // Still needed for common_init, common_params_parse for model loading, and logging
#include "log.h"
#include "llama.h"
#include "ggml-opt.h"   // Explicitly include for ggml_opt_result definition

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>
#include <memory>    // For std::unique_ptr
#include <stdexcept> // For std::runtime_error

// Include our GGUF dataset reader
#include <cinttypes>

#include "dataset-to-gguf/llama-gguf-reader.h"

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

int main(int argc, char ** argv) {
    common_params params;
    // Use LLAMA_EXAMPLE_FINETUNE to ensure relevant common parameters are parsed.
    // common_params_parse also handles --help and basic validation.
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    // Additional checks for parameters specific to this finetune example
    if (params.in_files.empty()) {
        LOG_ERR("error: --input (dataset) is required.\n");
        return 1;
    }
    if (params.lr.epochs <= 0) {
        LOG_ERR("error: --epochs must be a positive integer.\n");
        return 1;
    }
    if (params.val_split < 0.0f || params.val_split >= 1.0f) {
        LOG_ERR("error: --val-split must be between 0.0 and 1.0 (exclusive of 1.0).\n");
        return 1;
    }

    // Print parameters for verification
    printf("Parameters:\n");
    printf("  Base Model: %s\n", params.model.path.c_str());
    printf("  Dataset: %s\n", params.in_files[0].c_str()); // Assuming only one dataset file for simplicity
    printf("  GPU Layers: %d\n", params.n_gpu_layers);
    printf("  Use mmap: %s\n", params.use_mmap ? "Yes" : "No");
    if (!params.lora_adapter.empty()) {
        printf("  LoRA Adapter: %s\n", params.lora_adapter.c_str());
        printf("  LoRA Base: %s\n", params.lora_base.c_str());
    }
    printf("  Learning Rate at first epoch: %f\n", params.lr.lr0);
    printf("  Epochs: %d\n", params.lr.epochs);
    printf("  Validation Split: %f\n", params.val_split);
    printf("  Training Context: %d (0 = auto)\n", params.n_ctx_train);
    printf("  Parameter Filter: %s\n", params.param_filter.c_str());
    printf("  Save Model: %s\n", params.do_save ? "Yes" : "No");
    if (params.do_save) {
        printf("  Save Path: %s\n", params.out_file.c_str());
    }
    printf("  Threads: %d\n", params.n_threads);
    printf("\n");

    // Initialize llama.cpp backend
    llama_backend_init();
    common_init(); // Handles NUMA and other common initializations

    // Load the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;
    model_params.use_mmap     = params.use_mmap;

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);

    if (model == nullptr) {
        LOG_ERR("%s: unable to load model from %s\n", __func__, params.model.path.c_str());
        llama_backend_free();
        return 1;
    }

    // Apply LoRA adapter if specified
    if (!params.lora_adapter.empty()) {
        if (params.lora_base.empty()) {
            LOG_ERR("%s: --lora-base is required when --lora-adapter is used\n", __func__);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        int err = llama_model_apply_lora_from_file(model, params.lora_adapter.c_str(), params.lora_base.c_str(), params.n_threads);
        if (err != 0) {
            LOG_ERR("%s: failed to apply lora adapter\n", __func__);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        LOG_INF("%s: applied LoRA adapter from '%s' with base '%s'\n", __func__, params.lora_adapter.c_str(), params.lora_base.c_str());
    }

    // Create llama_context
    llama_context_params ctx_params = llama_context_default_params();
    // Use a large enough context for training, or let it be determined by the dataset
    // Use llama_model_n_ctx to get the model's default context size
    ctx_params.n_ctx = params.n_ctx_train > 0 ? (uint32_t)params.n_ctx_train : llama_model_n_ctx(model);

    // Use llama_init_from_model (deprecated llama_new_context_with_model)
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        LOG_ERR("%s: failed to create llama context\n", __func__);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    // --- Load GGUF Dataset ---
    LOG_INF("%s: Loading GGUF dataset from '%s'\n", __func__, params.in_files[0].c_str());
    std::unique_ptr<llama_gguf_reader> dataset_reader;
    try {
        dataset_reader = std::make_unique<llama_gguf_reader>(params.in_files[0]);
    } catch (const std::runtime_error& e) {
        LOG_ERR("%s: Failed to load GGUF dataset: %s\n", __func__, e.what());
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    if (!dataset_reader->llama_gguf_reader_is_initialized()) {
        LOG_ERR("%s: GGUF dataset reader not initialized.\n", __func__);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    int64_t total_sequences = dataset_reader->llama_gguf_reader_get_tensor_count();
    if (total_sequences == 0) {
        LOG_ERR("%s: GGUF dataset contains no sequences.\n", __func__);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    LOG_INF("%s: Dataset loaded. Total sequences: %" PRId64 "\n", __func__, total_sequences);

    // Determine training context size from dataset if not specified
    if (params.n_ctx_train == 0) {
        // Find the maximum sequence length in the dataset to set n_ctx_train
        uint32_t max_seq_len_in_dataset = 0; // Use uint32_t for consistency with n_ctx
        for (int64_t i = 0; i < total_sequences; ++i) {
            max_seq_len_in_dataset = std::max(max_seq_len_in_dataset, (uint32_t)dataset_reader->llama_gguf_reader_get_tensor_size(i) / (uint32_t)sizeof(llama_token));
        }
        params.n_ctx_train = max_seq_len_in_dataset;
        LOG_INF("%s: Auto-determined training context size (n_ctx_train): %d\n", __func__, params.n_ctx_train);
        // Warn if the determined context size is larger than the model's context size
        if (params.n_ctx_train > llama_model_n_ctx(model)) {
            LOG_DBG("%s: Auto-determined training context size (%d) is larger than model's native context size (%d). Sequences will be truncated by llama_opt_dataset_add_data.\n", __func__, params.n_ctx_train, llama_model_n_ctx(model));
        }
    }


    // Create llama_opt_dataset_t and populate it with sequences from the GGUF dataset
    // llama_opt_dataset_init takes the context and the training context size
    llama_opt_dataset_t dataset = llama_opt_dataset_init(ctx, params.n_ctx_train);

    for (int64_t i = 0; i < total_sequences; ++i) {
        std::vector<llama_token> sequence_tokens;
        if (dataset_reader->llama_gguf_reader_read_tensor_data(i, sequence_tokens)) {
            if (sequence_tokens.empty()) {
                LOG_DBG("%s: Skipping empty sequence at index %" PRId64 ".\n", __func__, i);
                continue;
            }

            // Ensure sequence_tokens has at least 2 tokens for input/target pair
            if (sequence_tokens.size() < 2) {
                LOG_DBG("%s: Skipping sequence %" PRId64 " with less than 2 tokens (%zu).\n", __func__, i, sequence_tokens.size());
                continue;
            }

            // Add the sequence to the dataset.
            // llama_opt_dataset_add_data handles splitting into input/target and truncation based on n_ctx_train.
            llama_opt_dataset_add_data(dataset, sequence_tokens.data(), sequence_tokens.size());

        } else {
            LOG_ERR("%s: Failed to read sequence at index %" PRId64 " from GGUF dataset. Skipping.\n", __func__, i);
            // Continue to next sequence
        }
    }

    LOG_INF("%s: Total data points in dataset: %" PRId64 "\n", __func__, ggml_opt_dataset_ndata(dataset));

    // Optimizer parameters
    struct ggml_opt_optimizer_params optimizer_params = ggml_opt_get_default_optimizer_params(nullptr);
    optimizer_params.adamw.alpha = params.lr.lr0; // Learning rate

    struct llama_opt_params lopt_params {
        /*n_ctx_train     =*/ params.n_ctx_train,
        /*param_filter    =*/ llama_opt_param_filter_parse(params.param_filter.c_str()), // Parse filter string
        /*param_filter_ud =*/ nullptr,
        /*get_opt_pars    =*/ ggml_opt_get_constant_optimizer_params,
        /*get_opt_pars_ud =*/ &optimizer_params,
    };
    llama_opt_init(ctx, model, lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    for (int epoch = 0; epoch < params.lr.epochs; ++epoch) { // Corrected: use params.lr.epochs
        LOG_INF("%s: Epoch %d/%d\n", __func__, epoch + 1, params.lr.epochs);
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
            ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        fprintf(stderr, "\n"); // Newline after progress bar

        // Accessing members of ggml_opt_result_t (struct ggml_opt_result)
        LOG_INF("%s: Epoch %d results: Train Loss = %f, Eval Loss = %f\n", __func__, epoch + 1,
            (double)result_train->loss_sum / result_train->n_iter,
            (double)result_eval->loss_sum / result_eval->n_iter);

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    if (params.do_save) {
        LOG_INF("%s: Saving finetuned model to '%s'\n", __func__, params.out_file.c_str());
        llama_model_save_to_file(model, params.out_file.c_str());
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
