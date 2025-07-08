// Утилита для конвертации текстового датасета в формат GGUF для обучения моделей в llama.cpp.
//
// Логика работы:
// 1. Загружает модель-токенизатор.
// 2. Проходит по входному текстовому файлу (первый проход) для сбора метаданных:
//    - Токенизирует каждую строку.
//    - Сохраняет длину каждой последовательности токенов (с учетом обрезки по --max-seq-len).
// 3. Создает GGUF-файл и записывает в него все собранные метаданные.
// 4. Проходит по входному файлу второй раз для записи данных:
//    - Токенизирует каждую строку.
//    - Записывает каждую последовательность токенов как отдельный тензор в GGUF-файл.
//
// Такой двухпроходный подход позволяет обрабатывать датасеты, значительно превышающие
// объем доступной оперативной памяти.

#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../../src/llama-model.h"
#include "common.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

// Структура для хранения параметров командной строки
struct training_data_params {
    std::string model_path    = "models/7B/ggml-model-f16.gguf"; // Путь к модели для токенизатора
    std::string input_path    = "input.txt";                     // Путь к входному текстовому файлу
    std::string output_path   = "output.gguf";                   // Путь для сохранения GGUF файла
    int32_t     max_seq_len   = 2048;                            // Максимальная длина последовательности
};

// Функция для парсинга аргументов командной строки
void training_data_params_parse(int argc, char **argv, training_data_params &params) {
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
        } else if (arg == "-h" || arg == "--help") {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -h, --help            show this help message and exit\n");
            printf("  -m, --vocab-model     path to model for tokenizer (default: %s)\n", params.model_path.c_str());
            printf("  -i, --input           path to input text file (default: %s)\n", params.input_path.c_str());
            printf("  -o, --output          path to output gguf file (default: %s)\n", params.output_path.c_str());
            printf("  -l, --max-seq-len     max sequence length (default: %d)\n", params.max_seq_len);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            exit(1);
        }
    }
}

int main(int argc, char **argv) {
    training_data_params params;
    training_data_params_parse(argc, argv, params);

    // Выводим параметры для проверки
    printf("Parameters:\n");
    printf("  Model for tokenizer: %s\n", params.model_path.c_str());
    printf("  Input file: %s\n", params.input_path.c_str());
    printf("  Output file: %s\n", params.output_path.c_str());
    printf("  Max sequence length: %d\n\n", params.max_seq_len);

    // Инициализация llama.cpp
    llama_backend_init(true); // NUMA-aware init

    // Загрузка модели для использования ее токенизатора
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr, "error: failed to load model from %s\n", params.model_path.c_str());
        return 1;
    }

    // --- ПЕРВЫЙ ПРОХОД: Сбор длин последовательностей ---
    printf("First pass: Reading input file and tokenizing to get sequence lengths...\n");
    std::vector<uint32_t> sequence_lengths;
    std::ifstream input_file(params.input_path);
    if (!input_file.is_open()) {
        fprintf(stderr, "error: failed to open input file %s\n", params.input_path.c_str());
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::string line;
    std::vector<llama_token> tokens_buffer(params.max_seq_len);
    while (std::getline(input_file, line)) {
        if (line.empty()) {
            continue;
        }
        int n_tokens = llama_tokenize(&model->vocab, line.c_str(), line.length(), tokens_buffer.data(), params.max_seq_len, false, true);
        if (n_tokens < 0) {
            fprintf(stderr, "error: tokenization failed for line: %s\n", line.c_str());
            // Пропускаем строку, но не прерываем весь процесс
            continue;
        }
        sequence_lengths.push_back(n_tokens);
    }
    input_file.close();
    printf("First pass complete. Found %zu sequences.\n\n", sequence_lengths.size());

    // --- ЗАПИСЬ GGUF ФАЙЛА ---
    printf("Creating GGUF file...\n");
    struct gguf_context * ctx = gguf_init_for_write(false);
    if (!ctx) {
        fprintf(stderr, "error: failed to initialize gguf context\n");
        return 1;
    }

    // Запись метаданных
    const uint64_t sequence_count = sequence_lengths.size();
    const int vocab_size = llama_vocab_n_tokens(&model->vocab);

    gguf_set_val_str(ctx, "training.format.version", "1.0");
    gguf_set_val_str(ctx, "training.dataset.name", params.input_path.c_str());
    time_t now = time(0);
    char buf[sizeof "2011-10-08T07:07:09Z"];
    strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
    gguf_set_val_str(ctx, "training.file.creation_date", buf);

    // Запись информации о токенизаторе
    char model_name_buffer[128];
    llama_model_meta_val_str(model, "general.architecture", model_name_buffer, sizeof(model_name_buffer));
    gguf_set_val_str(ctx, "training.tokenizer.gguf.model", model_name_buffer);

    // Запись словаря
    std::vector<const char *> vocab_list;
    for (int i = 0; i < vocab_size; ++i) {
        vocab_list.push_back(llama_vocab_get_text(&model->vocab, i));
    }
    gguf_set_arr_str(ctx, "training.tokenizer.gguf.vocab", vocab_list.data(), vocab_size);

    gguf_set_val_u64(ctx, "training.sequence.count", sequence_count);
    gguf_set_arr_data(ctx, "training.sequence.lengths", GGUF_TYPE_UINT32, sequence_lengths.data(), sequence_count);
    printf("Metadata written.\n");

    // --- ВТОРОЙ ПРОХОД: Запись тензоров ---
    printf("Second pass: Writing tensors to GGUF file...\n");
    input_file.open(params.input_path); // Открываем файл заново
    for (uint64_t i = 0; i < sequence_count; ++i) {
        if (!std::getline(input_file, line)) {
            fprintf(stderr, "error: file ended prematurely on second pass. Expected %zu sequences, got %llu.\n", sequence_lengths.size(), i);
            break;
        }
        if (line.empty() && sequence_lengths[i] == 0) {
            // Если пустая строка была обработана в первом проходе, пропускаем
            continue;
        }

        int n_tokens = llama_tokenize(&model->vocab, line.c_str(), line.length(), tokens_buffer.data(), params.max_seq_len, false, true);
        if (n_tokens != (int)sequence_lengths[i]) {
            fprintf(stderr, "warning: tokenization mismatch on second pass for sequence %llu. Expected %u tokens, got %d.\n", i, sequence_lengths[i], n_tokens);
            if (n_tokens < 0) continue;
        }

        char tensor_name[128];
        snprintf(tensor_name, sizeof(tensor_name), "training.tensor.%llu", i);

        gguf_add_tensor(ctx, ggml_new_tensor_1d(ggml_init({0, 0, 0, 0}), GGML_TYPE_I32, n_tokens));
        gguf_set_tensor_name(ctx, tensor_name);
        gguf_set_tensor_data(ctx, tensor_name, tokens_buffer.data(), n_tokens * sizeof(int32_t));
    }
    input_file.close();
    printf("Second pass complete.\n\n");

    // Сохранение файла на диск
    printf("Writing GGUF data to %s...\n", params.output_path.c_str());
    FILE * fout = fopen(params.output_path.c_str(), "wb");
    if (!fout) {
        fprintf(stderr, "error: failed to open output file %s\n", params.output_path.c_str());
        gguf_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    fwrite(gguf_get_meta_data(ctx), 1, gguf_get_meta_size(ctx), fout);
    for (uint64_t i = 0; i < gguf_get_n_tensors(ctx); ++i) {
        struct ggml_tensor * tensor = gguf_get_tensor_by_index(ctx, i);
        fwrite(tensor->data, 1, ggml_nbytes(tensor), fout);
    }
    fclose(fout);

    printf("Conversion successful!\n");
    printf("Output file: %s\n", params.output_path.c_str());

    // Очистка
    gguf_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
