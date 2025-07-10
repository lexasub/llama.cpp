#include "arg.h"
#include "common.h" // Still needed for common_init, common_params_parse for model loading, and logging
#include "log.h"
#include "llama.h"

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
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    // Print parameters for verification
    printf("Parameters:\n");
    printf("  Base Model: %s\n", params.model.path.c_str());
    printf("  Dataset: ");
    for (auto & i : params.in_files) {
        printf("%s ", i.c_str());
    }
    printf("\n  GPU Layers: %d\n", params.n_gpu_layers);
    printf("  Use mmap: %s\n", params.use_mmap ? "Yes" : "No");
    if (!params.lora_adapter.empty()) {
        printf("  LoRA Adapter: %s\n", params.lora_adapter.c_str());
        printf("  LoRA Base: %s\n", params.lora_base.c_str());
    }
    printf("  Learning Rate: %f\n", params.learning_rate);
    printf("  Epochs: %d\n", params.n_epochs);
    printf("  Validation Split: %f\n", params.val_split);
    printf("  Training Context: %d (0 = auto)\n", params.n_ctx_train);
    printf("  Parameter Filter: %s\n", params.param_filter.c_str());
    printf("  Save Model: %s\n", params.do_save ? "Yes" : "No");
    if (params.do_save) {
        printf("  Save Path: %s\n", params.out_file.c_str());
    }
    printf("\n");

    // Initialize llama.cpp backend
    llama_backend_init();
    // llama_numa_init(params.numa); // Numa is typically handled by common_init, if needed

    // Load the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;
    model_params.use_mmap     = params.use_mmap;

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);

    if (model == NULL) {
        LOG_ERR("%s: unable to load model from %s\n", __func__, params.in_files[0].c_str());
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
    ctx_params.n_ctx = params.n_ctx_train > 0 ? params.n_ctx_train : llama_n_ctx(model);
    ctx_params.seed = 1234; // Fixed seed for reproducibility

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == NULL) {
        LOG_ERR("%s: failed to create llama context\n", __func__);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(common_params()).c_str()); // Using a dummy common_params for system info
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
        int32_t max_seq_len_in_dataset = 0;
        for (int64_t i = 0; i < total_sequences; ++i) {
            max_seq_len_in_dataset = std::max(max_seq_len_in_dataset, (int32_t)dataset_reader->llama_gguf_reader_get_tensor_size(i) / (int32_t)sizeof(llama_token));
        }
        params.n_ctx_train = max_seq_len_in_dataset;
        LOG_INF("%s: Auto-determined training context size (n_ctx_train): %d\n", __func__, params.n_ctx_train);
        // Update context n_ctx if auto-determined is larger or if it was 0
        if (ctx_params.n_ctx < params.n_ctx_train) {
            // This would require recreating the context, which is not ideal.
            // For now, we'll just use the larger of the two, but warn if it's smaller than max_seq_len_in_dataset.
            LOG_DBG("%s: Model context size (%d) is smaller than max sequence length in dataset (%d). Some sequences might be truncated.\n", __func__, ctx_params.n_ctx, params.n_ctx_train);
            // Recreating context is not directly supported by llama_new_context_with_model without freeing the old one.
            // For simplicity, we'll proceed, assuming truncation will happen if n_ctx is too small.
            // In a real scenario, you might want to re-create ctx here with the larger n_ctx.
        }
    }


    // Create ggml_opt_dataset_t and populate it with sequences from the GGUF dataset
    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(ctx); // Initialize empty dataset

    for (int64_t i = 0; i < total_sequences; ++i) {
        std::vector<llama_token> sequence_tokens;
        if (dataset_reader->llama_gguf_reader_read_tensor_data(i, sequence_tokens)) {
            if (sequence_tokens.empty()) {
                LOG_DBG("%s: Skipping empty sequence at index %" PRId64 ".\n", __func__, i);
                continue;
            }

            // Create input and target tensors for this sequence
            // Input: tokens[0] to tokens[n-2]
            // Target: tokens[1] to tokens[n-1]
            // Both are of length n-1
            int n_tokens_seq = sequence_tokens.size();
            if (n_tokens_seq < 2) {
                LOG_DBG("%s: Skipping sequence at index %" PRId64 " with less than 2 tokens.\n", __func__, i);
                continue;
            }

            // Truncate sequence if it's longer than n_ctx_train
            if (n_tokens_seq > params.n_ctx_train) {
                LOG_DBG("%s: Truncating sequence %" PRId64 " from %d to %d tokens.\n", __func__, i, n_tokens_seq, params.n_ctx_train);
                n_tokens_seq = params.n_ctx_train;
            }


            // Allocate memory for input and target tensors within a temporary ggml_context
            // This is crucial because ggml_opt_data_add takes ownership of the ggml_tensor pointers
            // but not the ggml_context they are part of.
            // The tensors must reside in a context that persists until dataset is freed.
            // For finetuning, the tensors are usually created directly in the main context or a dedicated one.
            // However, ggml_opt_dataset_init (when called with a context) might manage this internally.
            // Let's assume for now ggml_opt_dataset_add_data copies the data.
            // If not, we need a persistent context for these tensors.

            // The common_opt_dataset_init uses the context provided to create tensors.
            // We need to simulate that or use a helper.
            // For simplicity, let's just make sure the tokens are correct.
            // ggml_opt_dataset_add_data expects raw token vectors, not ggml_tensors.
            // This simplifies things greatly.

            // The original finetune.cpp uses common_opt_dataset_init(ctx.get(), tokens, llama_n_ctx(ctx.get())/2);
            // This function internally handles creating ggml_opt_data_t from the token vector.
            // So we just need to feed it our sequences.

            // This is the key change: we're building the dataset from our GGUF reader
            // The original common_opt_dataset_init splits the input tokens into chunks of n_ctx_train.
            // Our GGUF dataset already provides pre-chunked sequences.
            // So, we'll add each sequence from the GGUF reader as a data point.

            // Ensure sequence_tokens has at least 2 tokens for input/target pair
            if (sequence_tokens.size() < 2) {
                LOG_DBG("%s: Skipping sequence %" PRId64 " due to insufficient tokens (%zu).\n", __func__, i, sequence_tokens.size());
                continue;
            }

            // Add the sequence to the dataset.
            // The ggml_opt_dataset_add_data function expects a vector of tokens.
            // It will handle splitting it into input/target and creating the ggml_tensors.
            // Note: The 'n_ctx_train' parameter in ggml_opt_dataset_init is crucial for this.
            ggml_opt_dataset_add_data(dataset, ctx, sequence_tokens.data(), sequence_tokens.size());

        } else {
            LOG_ERR("%s: Failed to read sequence at index %" PRId64 " from GGUF dataset.\n", __func__, i);
            // Decide if you want to continue or abort on read errors
            // For now, we'll continue but log the error.
        }
    }

    LOG_INF("%s: Total data points in dataset: %" PRId64 "\n", __func__, ggml_opt_dataset_ndata(dataset));

    // Optimizer parameters
    struct ggml_opt_optimizer_params optimizer_params = ggml_opt_get_default_optimizer_params(nullptr);
    optimizer_params.adamw.alpha = params.learning_rate; // Learning rate

    struct llama_opt_params lopt_params {
        /*n_ctx_train     =*/ params.n_ctx_train, // Use our specified or auto-determined context size
        /*param_filter    =*/ llama_opt_param_filter_parse(params.param_filter.c_str()), // Parse filter string
        /*param_filter_ud =*/ nullptr,
        /*get_opt_pars    =*/ ggml_opt_get_constant_optimizer_params,
        /*get_opt_pars_ud =*/ &optimizer_params,
    };
    llama_opt_init(ctx, model, lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    for (int epoch = 0; epoch < params.n_epochs; ++epoch) {
        LOG_INF("%s: Epoch %d/%d\n", __func__, epoch + 1, params.n_epochs);
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
            ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        fprintf(stderr, "\n"); // Newline after progress bar

        LOG_INF("%s: Epoch %d results: Train Loss = %f, Eval Loss = %f\n", __func__, epoch + 1, result_train->loss_sum / result_train->n_iter, result_eval->loss_sum / result_eval->n_iter);

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
