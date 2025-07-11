// Этот файл содержит исправленную версию кода для дообучения (finetuning)
// модели с использованием датасета в формате GGUF, адаптированную для
// более старых версий API llama.cpp.
//
// Основные исправления:
// 1. Полностью переработана логика загрузки данных для совместимости со старым
//    API ggml_opt_dataset_init, который требует предварительного выделения памяти.
// 2. Удалена несуществующая функция ggml_opt_dataset_add_data. Данные теперь
//    копируются вручную в предварительно выделенные тензоры.
// 3. Заменена несуществующая функция ggml_opt_param_filter_parse на
//    llama_opt_param_filter_lora.
// 4. Закомментирован блок применения LoRA, так как функция
//    llama_model_apply_lora_from_file отсутствует в старых версиях.

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "arg.h"
#include "common.h"
#include "dataset-to-gguf/llama-gguf-reader.h"
#include "ggml-opt.h"
#include "ggml.h"  // Нужно для доступа к dataset->data и dataset->label
#include "llama-context.h"
#include "llama.h"
#include "log.h"

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

    /*
    // FIX: Закомментировано из-за отсутствия функции в старых версиях API.
    // Для использования этой функциональности, пожалуйста, обновите вашу версию llama.cpp.
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
    */

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
        if ((uint32_t)params.n_ctx_train > llama_n_ctx(ctx)) {
            LOG_DBG("%s: Auto-determined training context size (%d) is larger than model's context size (%d). Sequences will be truncated.\n", __func__, params.n_ctx_train, llama_n_ctx(ctx));
        }
    }

    // --- НОВАЯ ЛОГИКА ЗАГРУЗКИ ДАННЫХ ДЛЯ СТАРОГО API ---
    LOG_INF("%s: Reading all sequences into memory...\n", __func__);
    std::vector<llama_token> all_tokens;
    for (int64_t i = 0; i < total_sequences; ++i) {
        std::vector<llama_token> sequence_tokens;
        if (dataset_reader->llama_gguf_reader_read_tensor_data(i, sequence_tokens)) {
            if (sequence_tokens.size() < 2) {
                continue;
            }
            all_tokens.insert(all_tokens.end(), sequence_tokens.begin(), sequence_tokens.end());
        }
    }
    LOG_INF("%s: Total tokens in memory: %zu\n", __func__, all_tokens.size());

    const int64_t n_ctx_train = params.n_ctx_train;
    const int64_t n_datapoint = n_ctx_train - 1;
    const int64_t n_label     = n_ctx_train - 1;
    const int64_t ndata       = (all_tokens.size() - 1) / n_datapoint;

    if (ndata == 0) {
        LOG_ERR("%s: Not enough tokens to create even one training example.\n", __func__);
        return 1;
    }

    LOG_INF("%s: Creating dataset with %" PRId64 " examples...\n", __func__, ndata);
    struct ggml_opt_dataset * dataset = ggml_opt_dataset_init(GGML_TYPE_I32, GGML_TYPE_I32, n_datapoint, n_label, ndata, ndata);
    LOG_INF("%s: Populating dataset...\n", __func__);
    for (int64_t i = 0; i < ndata; ++i) {
        const int64_t token_start_index = i * n_datapoint;

        // Получаем указатели на данные и метки для текущего примера
        llama_token* data_ptr  = reinterpret_cast<llama_token *>(
            static_cast<char *>(ggml_opt_dataset_data(dataset)->data) + i * ggml_opt_dataset_data(dataset)->nb[1]);
        llama_token* label_ptr = reinterpret_cast<llama_token *>(
            static_cast<char *>(ggml_opt_dataset_labels(dataset)->data) + i * ggml_opt_dataset_labels(dataset)->nb[1]);

        // Копируем данные (входные токены)
        memcpy(data_ptr, all_tokens.data() + token_start_index, n_datapoint * sizeof(llama_token));
        // Копируем метки (целевые токены, со сдвигом на 1)
        memcpy(label_ptr, all_tokens.data() + token_start_index + 1, n_label * sizeof(llama_token));
    }
    LOG_INF("%s: Dataset populated.\n", __func__);


    struct llama_opt_params lopt_params {
        /*n_ctx_train     =*/ (uint32_t)params.n_ctx_train,
        // FIX: Использование llama_opt_param_filter_lora, так как parse-функция отсутствует.
        /*param_filter    =*/ llama_opt_param_filter_lora,
        /*param_filter_ud =*/ nullptr,
        /*get_opt_pars    =*/ common_opt_lr_pars,
        /*get_opt_pars_ud =*/ &params.lr,
        /*optimizer_type  =*/ params.optimizer,
    };
    llama_opt_init(ctx, model, lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train;
    ggml_opt_result_t result_eval;

    for (params.lr.epoch = 0; params.lr.epoch < params.lr.epochs; ++params.lr.epoch) {
        LOG_INF("%s: Epoch %d/%d\n", __func__, params.lr.epoch + 1, params.lr.epochs);

        result_train = {nullptr};
        result_eval = {nullptr};

        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
            ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        fprintf(stderr, "\n");
        double loss;
        double unc;
        ggml_opt_result_loss(result_train, &loss, &unc);
        //ggml_opt_result_eval(result_train, &loss, &unc);
        LOG_INF("%s: Epoch %d results: Train Loss = %f\n", __func__, params.lr.epoch + 1, loss); //, Eval Loss = %f
    }

    if (params.do_save) {
        LOG_INF("%s: Saving finetuned model to '%s'\n", __func__, params.out_file.c_str());
        llama_model_save_to_file(model, params.out_file.c_str());
    }

    // ggml_opt_dataset_free(dataset); // Если есть такая функция в вашей версии
    llama_backend_free();

    return 0;
}

