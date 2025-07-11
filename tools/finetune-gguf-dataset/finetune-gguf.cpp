// Этот файл содержит исправленную версию кода для дообучения (finetuning)
// модели с использованием датасета в формате GGUF.
//
// Основные исправления:
// 1. Заменены устаревшие вызовы API `llama_opt_*` на современные `ggml_opt_*`.
// 2. Исправлена работа со структурой `ggml_opt_result_t`:
//    - Структуры теперь создаются на стеке.
//    - Удалены вызовы несуществующих функций `ggml_opt_result_init` и `ggml_opt_result_free`.
//    - В функцию `llama_opt_epoch` передаются указатели на структуры.
//    - Доступ к полям структуры осуществляется через оператор ".".
// 3. Заменена устаревшая функция `llama_model_n_ctx` на `llama_n_ctx`.
// 4. Остальные вызовы, такие как `llama_model_apply_lora_from_file`, оставлены без изменений,
//    так как они корректны для современных версий llama.cpp. Убедитесь, что ваш репозиторий
//    полностью обновлен.

#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-opt.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>
#include <memory>
#include <stdexcept>
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

    printf("Parameters:\n");
    printf("  Base Model: %s\n", params.model.path.c_str());
    printf("  Dataset: %s\n", params.in_files[0].c_str());
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

    common_init();

    common_init_result llama_init = common_init_from_params(params);
    llama_model_ptr   & model_ptr = llama_init.model;
    llama_context_ptr & ctx_ptr   = llama_init.context;

    llama_model * model = model_ptr.get();
    llama_context * ctx = ctx_ptr.get();

    if (model == nullptr) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    if (!params.lora_adapter.empty()) {
        if (params.lora_base.empty()) {
            LOG_ERR("%s: --lora-base is required when --lora-adapter is used\n", __func__);
            return 1;
        }
        int err = llama_model_apply_lora_from_file(model, params.lora_adapter.c_str(), params.lora_base.c_str(), params.n_threads);
        if (err != 0) {
            LOG_ERR("%s: failed to apply lora adapter\n", __func__);
            return 1;
        }
        LOG_INF("%s: applied LoRA adapter from '%s' with base '%s'\n", __func__, params.lora_adapter.c_str(), params.lora_base.c_str());
    }

    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    LOG_INF("%s: Loading GGUF dataset from '%s'\n", __func__, params.in_files[0].c_str());
    std::unique_ptr<llama_gguf_reader> dataset_reader;
    try {
        dataset_reader = std::make_unique<llama_gguf_reader>(params.in_files[0]);
    } catch (const std::runtime_error& e) {
        LOG_ERR("%s: Failed to load GGUF dataset: %s\n", __func__, e.what());
        return 1;
    }

    if (!dataset_reader->llama_gguf_reader_is_initialized()) {
        LOG_ERR("%s: GGUF dataset reader not initialized.\n", __func__);
        return 1;
    }

    int64_t total_sequences = dataset_reader->llama_gguf_reader_get_tensor_count();
    if (total_sequences == 0) {
        LOG_ERR("%s: GGUF dataset contains no sequences.\n", __func__);
        return 1;
    }

    LOG_INF("%s: Dataset loaded. Total sequences: %" PRId64 "\n", __func__, total_sequences);

    if (params.n_ctx_train == 0) {
        uint32_t max_seq_len_in_dataset = 0;
        for (int64_t i = 0; i < total_sequences; ++i) {
            max_seq_len_in_dataset = std::max(max_seq_len_in_dataset, (uint32_t)dataset_reader->llama_gguf_reader_get_tensor_size(i) / (uint32_t)sizeof(llama_token));
        }
        params.n_ctx_train = max_seq_len_in_dataset;
        LOG_INF("%s: Auto-determined training context size (n_ctx_train): %d\n", __func__, params.n_ctx_train);
        // FIX: Use llama_n_ctx(ctx) to get the context size of the loaded model/context.
        if ((uint32_t)params.n_ctx_train > llama_n_ctx(ctx)) {
            LOG_DBG("%s: Auto-determined training context size (%d) is larger than model's context size (%d). Sequences will be truncated.\n", __func__, params.n_ctx_train, llama_n_ctx(ctx));
        }
    }

    // FIX: Use ggml_opt_dataset_* functions and types.
    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(ctx, params.n_ctx_train);

    for (int64_t i = 0; i < total_sequences; ++i) {
        std::vector<llama_token> sequence_tokens;
        if (dataset_reader->llama_gguf_reader_read_tensor_data(i, sequence_tokens)) {
            if (sequence_tokens.size() < 2) {
                LOG_DBG("%s: Skipping sequence %" PRId64 " with less than 2 tokens (%zu).\n", __func__, i, sequence_tokens.size());
                continue;
            }
            // FIX: Use ggml_opt_dataset_add_data.
            ggml_opt_dataset_add_data(dataset, sequence_tokens.data(), sequence_tokens.size());
        } else {
            LOG_ERR("%s: Failed to read sequence at index %" PRId64 " from GGUF dataset. Skipping.\n", __func__, i);
        }
    }

    LOG_INF("%s: Total data points in dataset: %" PRId64 "\n", __func__, ggml_opt_dataset_ndata(dataset));

    struct llama_opt_params lopt_params {
        /*n_ctx_train     =*/ (uint32_t)params.n_ctx_train,
        // FIX: Use ggml_opt_param_filter_parse.
        /*param_filter    =*/ ggml_opt_param_filter_parse(params.param_filter.c_str()),
        /*param_filter_ud =*/ nullptr,
        /*get_opt_pars    =*/ common_opt_lr_pars,
        /*get_opt_pars_ud =*/ &params.lr,
        /*optimizer_type  =*/ params.optimizer,
    };
    llama_opt_init(ctx, model, lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    // FIX: Correct handling of ggml_opt_result_t.
    // Declare on stack, no init/free needed.
    ggml_opt_result_t result_train;
    ggml_opt_result_t result_eval;

    for (params.lr.epoch = 0; params.lr.epoch < params.lr.epochs; ++params.lr.epoch) {
        LOG_INF("%s: Epoch %d/%d\n", __func__, params.lr.epoch + 1, params.lr.epochs);
        
        // Reset results for each epoch.
        result_train = {0};
        result_eval = {0};

        // FIX: Pass pointers to the result structs.
        llama_opt_epoch(ctx, dataset, &result_train, &result_eval, idata_split,
            ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        fprintf(stderr, "\n");

        // FIX: Access struct members with '.' instead of '->'.
        LOG_INF("%s: Epoch %d results: Train Loss = %f, Eval Loss = %f\n", __func__, params.lr.epoch + 1,
            (double)result_train.loss_sum / result_train.n_iter,
            (double)result_eval.loss_sum / result_eval.n_iter);
    }

    if (params.do_save) {
        LOG_INF("%s: Saving finetuned model to '%s'\n", __func__, params.out_file.c_str());
        llama_model_save_to_file(model, params.out_file.c_str());
    }

    llama_backend_free();

    return 0;
}

