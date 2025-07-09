// Главная утилита для конвертации текстового датасета в формат GGUF для обучения моделей в llama.cpp.
//
// Логика работы:
// 1. Парсит аргументы командной строки.
// 2. Загружает модель-токенизатор.
// 3. Использует DataReader для первого прохода по входным данным, чтобы собрать метаданные (длины последовательностей).
// 4. Использует класс GGUFWriter для создания GGUF-файла и записи в него всех собранных метаданных.
// 5. Использует DataReader для второго прохода по входным данным, чтобы добавить каждую последовательность
//    как отдельный тензор в GGUF-файл через GGUFWriter.
//
// Такой двухпроходный подход позволяет обрабатывать датасеты, значительно превышающие
// объем доступной оперативной памяти.

#include <inttypes.h>  // Для PRIu64

#include <cstdio>      // Для snprintf
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>  // Для std::unique_ptr
#include <string>
#include <vector>

#include "common.h"                            // Для общих утилит, если требуются (например, common_params)
#include "dataset-to-gguf/dataset-reader.h"
#include "dataset-to-gguf/gguf-file.h"         // Включаем класс GGUFFile
#include "dataset-to-gguf/gguf-writer.h"       // Включаем наш класс для записи GGUF
#include "dataset-to-gguf/text-reader.h"

// Структура для хранения параметров командной строки
struct training_data_params {
    std::string model_path    = "models/7B/ggml-model-f16.gguf"; // Путь к модели для токенизатора
    std::string input_path    = "input.txt";                     // Путь к входному текстовому файлу
    std::string output_path   = "output.gguf";                   // Путь для сохранения GGUF файла
    int32_t     max_seq_len   = 2048;                            // Максимальная длина последовательности
    bool        pre_tokenized = false;                           // Флаг: если true, входные данные уже токенизированы (токены в виде чисел)
    std::string input_type    = "text";                          // Тип входных данных (например, "text", "parquet")
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
        } else if (arg == "--pre-tokenized" || arg == "-p") {
            params.pre_tokenized = true; // Устанавливаем флаг, если входные данные уже токенизированы
        } else if (arg == "--input-type" || arg == "-t") {
            params.input_type = argv[++i]; // Указываем тип входных данных
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
    printf("  Max sequence length: %d\n", params.max_seq_len);
    printf("  Pre-tokenized input: %s\n", params.pre_tokenized ? "Yes" : "No");
    printf("  Input type: %s\n\n", params.input_type.c_str());

    // Инициализация llama.cpp
    llama_backend_init();

    // Загрузка модели для использования ее токенизатора
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr, "error: failed to load model from %s\n", params.model_path.c_str());
        return 1;
    }

    // --- Создание DataReader на основе input_type ---
    std::unique_ptr<DatasetReader> reader;
    if (params.input_type == "text") {
        reader = std::make_unique<TextDatasetReader>(model, params.max_seq_len, params.pre_tokenized);
    } else {
        fprintf(stderr, "error: Unsupported input type: %s\n", params.input_type.c_str());
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Открытие источника данных
    if (!reader->open(params.input_path)) {
        fprintf(stderr, "error: Failed to open data source %s\n", params.input_path.c_str());
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // --- ПЕРВЫЙ ПРОХОД: Сбор длин последовательностей ---
    printf("First pass: Reading input data and getting sequence lengths...\n");
    std::vector<uint32_t> sequence_lengths; // Будет хранить длины последовательностей
    std::vector<llama_token> tokens;

    while (reader->read_next_sequence(tokens)) {
        sequence_lengths.push_back(tokens.size());
    }
    printf("First pass complete. Found %zu sequences.\n\n", sequence_lengths.size());

    // --- ЗАПИСЬ GGUF ФАЙЛА ---
    printf("Creating GGUF file...\n");
    // Создаем экземпляр GGUFFile, который будет управлять GGUF контекстом
    std::unique_ptr<GGUFFile> gguf_file;
    try {
        gguf_file = std::make_unique<GGUFFile>();
    } catch (const std::runtime_error& e) {
        fprintf(stderr, "error: Failed to initialize GGUFFile: %s\n", e.what());
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Передаем указатель на gguf_file в GGUFWriter
    GGUFWriter writer(gguf_file.get());

    // Инициализируем метаданные GGUF файла
    writer.init_metadata(model, params.input_path, sequence_lengths.size());
    printf("Metadata written.\n");

    // --- ВТОРОЙ ПРОХОД: Запись тензоров ---
    printf("Second pass: Writing tensors to GGUF file...\n");
    if (!reader->reset()) {
        fprintf(stderr, "error: Failed to reset data reader for second pass.\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    uint64_t current_sequence_idx = 0;
    while (reader->read_next_sequence(tokens)) {
        if (current_sequence_idx >= sequence_lengths.size()) {
            fprintf(stderr, "error: file ended prematurely on second pass. Expected %zu sequences, but reached end of file at %lu.\n", sequence_lengths.size(), current_sequence_idx);
            break;
        }

        uint32_t expected_n_tokens = sequence_lengths[current_sequence_idx];
        uint32_t actual_n_tokens = tokens.size();

        if (actual_n_tokens != expected_n_tokens) {
            fprintf(stderr, "warning: tokenization mismatch on second pass for sequence %lu. Expected %u tokens, got %u.\n", current_sequence_idx, expected_n_tokens, actual_n_tokens);
            // Если количество токенов не совпадает, используем то, что получили на втором проходе
            // Это может быть неидеально, но позволяет продолжить
        }

        // Добавляем тензор только если есть токены
        if (actual_n_tokens > 0) {
            writer.add_sequence_tensor(current_sequence_idx, tokens);
        } else {
            // Если ожидалось 0 токенов, но строка не была пустой, выводим предупреждение
            if (expected_n_tokens != 0) {
                fprintf(stderr, "warning: sequence %lu resulted in 0 tokens on second pass, but expected %u.\n", current_sequence_idx, expected_n_tokens);
            }
        }
        current_sequence_idx++;
    }
    reader->close(); // Закрываем DataReader после использования
    printf("Second pass complete.\n\n");

    // Сохранение файла на диск
    printf("Writing GGUF data to %s...\n", params.output_path.c_str());
    if (!writer.write_to_file(params.output_path)) {
        fprintf(stderr, "error: failed to write GGUF file %s\n", params.output_path.c_str());
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    printf("Conversion successful!\n");
    printf("Output file: %s\n", params.output_path.c_str());

    // Очистка
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
