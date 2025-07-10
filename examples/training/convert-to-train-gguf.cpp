// Главная утилита для конвертации текстового датасета в формат GGUF для обучения моделей в llama.cpp.
//
// Логика работы:
// 1. Парсит аргументы командной строки.
// 2. Загружает модель-токенизатор.
// 3. Использует класс GGUFConverter для выполнения всего процесса конвертации:
//    - Первый проход по входным данным для сбора метаданных (длины последовательностей).
//    - Создание GGUF-файла и запись в него всех собранных метаданных.
//    - Второй проход по входным данным для добавления каждой последовательности
//      как отдельного тензора в GGUF-файл.
// 4. После успешной конвертации, использует GGUFReader для чтения и вывода
//    некоторой метаинформации и первой записи из созданного GGUF файла.
//
// Такой двухпроходный подход позволяет обрабатывать датасеты, значительно превышающие
// объем доступной оперативной памяти.

#include <algorithm>  // Для std::min
#include <array>      // Для std::array
#include <cinttypes>
#include <iostream>
#include <limits>  // Для std::numeric_limits
#include <memory>  // Для std::unique_ptr
#include <string>
#include <vector>

#include "common.h"                               // Для общих утилит, если требуются (например, common_params)
#include "dataset-to-gguf/gguf-converter.h"       // Включаем наш новый класс GGUFConverter
#include "dataset-to-gguf/gguf-reader.h"          // Включаем наш новый класс GGUFReader
#include "llama.h"  // Для llama_backend_init, llama_backend_free, llama_model_load_from_file, llama_model_free

// Структура для хранения параметров командной строки
struct training_data_params {
    std::string model_path    = "models/7B/ggml-model-f16.gguf"; // Путь к модели для токенизатора
    std::string input_path    = "input.txt";                     // Путь к входному текстовому файлу
    std::string output_path   = "output.gguf";                   // Путь для сохранения GGUF файла
    int32_t     max_seq_len   = 2048;                            // Максимальная длина последовательности
    bool        pre_tokenized = false;                           // Флаг: если true, входные данные уже токенизированы (токены в виде чисел)
    std::string input_type    = "text";                          // Тип входных данных (например, "text", "parquet")
    bool        do_preview    = false;                           // Флаг: если true, выполнить предварительный просмотр
    int32_t     preview_count = 1;                               // Количество последовательностей для предварительного просмотра
    bool        detokenize_preview = false;                      // Флаг: если true, детокенизировать предварительный просмотр
    std::string parquet_text_column = "text";                    // Имя столбца с текстом в Parquet файле
    std::string parquet_tokens_column = "tokens";                // Имя столбца с токенами в Parquet файле
};

// Предварительная декларация функции парсинга параметров
void training_data_params_parse(int argc, char **argv, training_data_params &params);

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
            params.pre_tokenized = true;
        } else if (arg == "--input-type" || arg == "-t") {
            params.input_type = argv[++i];
        } else if (arg == "--preview") {
            params.do_preview = true; // Включаем предварительный просмотр
        } else if (arg == "--preview-count") {
            params.preview_count = std::stoi(argv[++i]);
            if (params.preview_count <= 0) {
                fprintf(stderr, "error: --preview-count must be a positive integer.\n");
                exit(1);
            }
            params.do_preview = true; // Включаем предварительный просмотр, если указан count
        } else if (arg == "--detokenize-preview") {
            params.detokenize_preview = true;
            params.do_preview = true; // Включаем предварительный просмотр, если указана детокенизация
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

int main(int argc, char **argv) {
    training_data_params params_raw;
    training_data_params_parse(argc, argv, params_raw);

    // Выводим параметры для проверки
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

    // Инициализация llama.cpp
    llama_backend_init();

    // Загрузка модели для использования ее токенизатора
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(params_raw.model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr, "error: failed to load model from %s\n", params_raw.model_path.c_str());
        llama_backend_free();
        return 1;
    }

    // --- Диагностический тест: Чтение файла модели токенизатора с помощью GGUFReader ---
    printf("--- Diagnostic Test: Reading tokenizer model GGUF file ---\n");
    try {
        GGUFReader tokenizer_model_reader(params_raw.model_path);
        if (tokenizer_model_reader.is_initialized()) {
            printf("  Tokenizer Model GGUF file opened successfully.\n");
            printf("  Tokenizer Model Name: %s\n", tokenizer_model_reader.get_metadata_str("general.name", "N/A").c_str());
            printf("  Tokenizer Model Architecture: %s\n", tokenizer_model_reader.get_metadata_str("general.architecture", "N/A").c_str());
            printf("  Tokenizer Model Tensor Count: %ld\n", tokenizer_model_reader.get_tensor_count());
            printf("  Diagnostic Test: Tokenizer Model GGUF read successful.\n");
        } else {
            fprintf(stderr, "error: Diagnostic Test: Tokenizer Model GGUF read failed to initialize.\n");
            llama_model_free(model); // Освобождаем модель перед выходом
            llama_backend_free();
            return 1;
        }
    } catch (const std::runtime_error& e) {
        fprintf(stderr, "error: Diagnostic Test: Tokenizer Model GGUF read failed: %s\n", e.what());
        llama_model_free(model); // Освобождаем модель перед выходом
        llama_backend_free();
        return 1;
    }
    printf("--- End of Diagnostic Test ---\n\n");


    // Подготовка параметров для GGUFConverter
    ConvertParams convert_params;
    convert_params.input_path = params_raw.input_path;
    convert_params.output_path = params_raw.output_path;
    convert_params.max_seq_len = params_raw.max_seq_len;
    convert_params.pre_tokenized = params_raw.pre_tokenized;
    convert_params.input_type = params_raw.input_type;
    convert_params.model = model; // Передаем указатель на загруженную модель
    convert_params.parquet_text_column = params_raw.parquet_text_column; // Передаем имя текстового столбца Parquet
    convert_params.parquet_tokens_column = params_raw.parquet_tokens_column; // Передаем имя столбца токенов Parquet

    // Создаем и запускаем конвертер
    GGUFConverter converter;
    bool success = converter.convert(convert_params);

    // Очистка модели llama
    llama_model_free(model);
    llama_backend_free();

    if (!success) {
        fprintf(stderr, "error: GGUF conversion failed.\n");
        return 1;
    }

    printf("Conversion successful!\n");
    printf("Output file: %s\n", params_raw.output_path.c_str());

    // --- Предварительный просмотр созданного GGUF файла (если запрошено) ---
    if (params_raw.do_preview) {
        printf("\n--- Previewing generated GGUF file ---\n");
        try {
            GGUFReader reader(params_raw.output_path);

            if (!reader.is_initialized()) {
                fprintf(stderr, "error: GGUFReader failed to initialize for preview.\n");
                return 1;
            }

            printf("  Dataset Name: %s\n", reader.get_metadata_str("training.dataset.name", "N/A").c_str());
            printf("  Sequence Count: %lu\n", reader.get_metadata_u64("training.sequence.count", 0));
            printf("  Tokenizer Model: %s\n", reader.get_metadata_str("training.tokenizer.gguf.model", "N/A").c_str());

            int64_t tensor_count = reader.get_tensor_count();
            if (tensor_count > 0) {
                // Выводим N первых последовательностей
                for (int64_t i = 0; i < std::min((int64_t)params_raw.preview_count, tensor_count); ++i) {
                    printf("  Sequence (training.tensor.%" PRId64 "):\n", i);
                    std::vector<llama_token> sequence_tokens;
                    if (reader.read_tensor_data(i, sequence_tokens)) {
                        printf("    Length: %zu tokens\n", sequence_tokens.size());
                        printf("    Tokens: [");
                        for (size_t j = 0; j < std::min((size_t)10, sequence_tokens.size()); ++j) { // Выводим до 10 токенов
                            printf("%d%s", sequence_tokens[j], (j == std::min((size_t)10, sequence_tokens.size()) - 1) ? "" : ", ");
                        }
                        if (sequence_tokens.size() > 10) {
                            printf("...");
                        }
                        printf("]\n");

                        if (params_raw.detokenize_preview) {
                            // Детокенизация
                            std::string detokenized_text = "";
                            // Буфер для одного токена
                            std::array<char, 256> piece_buf; // Достаточно большой буфер для одного токена
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

        } catch (const std::runtime_error& e) {
            fprintf(stderr, "error: GGUF preview failed: %s\n", e.what());
            return 1;
        }
        printf("--- End of GGUF file preview ---\n");
    }

    return 0;
}
