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

#include "dataset-to-gguf/gguf-converter.h" // Включаем наш новый класс GGUFConverter
#include "dataset-to-gguf/gguf-reader.h"    // Включаем наш новый класс GGUFReader
#include "common.h"                         // Для общих утилит, если требуются (например, common_params)
#include "llama.h"                          // Для llama_backend_init, llama_backend_free, llama_model_load_from_file, llama_model_free

#include <iostream>
#include <string>
#include <vector>
#include <memory> // Для std::unique_ptr
#include <limits> // Для std::numeric_limits

// Структура для хранения параметров командной строки
struct training_data_params {
    std::string model_path    = "models/7B/ggml-model-f16.gguf"; // Путь к модели для токенизатора
    std::string input_path    = "input.txt";                     // Путь к входному текстовому файлу
    std::string output_path   = "output.gguf";                   // Путь для сохранения GGUF файла
    int32_t     max_seq_len   = 2048;                            // Максимальная длина последовательности
    bool        pre_tokenized = false;                           // Флаг: если true, входные данные уже токенизированы (токены в виде чисел)
    std::string input_type    = "text";                          // Тип входных данных (например, "text", "parquet")
    bool        do_preview    = false;                           // Флаг: если true, выполнить предварительный просмотр
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
            printf("  --preview             read and print metadata and first sequence from the output GGUF file\n");
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
    printf("  Do preview: %s\n\n", params_raw.do_preview ? "Yes" : "No");

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

    // Подготовка параметров для GGUFConverter
    ConvertParams convert_params;
    convert_params.input_path = params_raw.input_path;
    convert_params.output_path = params_raw.output_path;
    convert_params.max_seq_len = params_raw.max_seq_len;
    convert_params.pre_tokenized = params_raw.pre_tokenized;
    convert_params.input_type = params_raw.input_type;
    convert_params.model = model; // Передаем указатель на загруженную модель

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
                printf("  First Sequence (training.tensor.0):\n");
                std::vector<llama_token> first_sequence_tokens;
                if (reader.read_tensor_data(0, first_sequence_tokens)) {
                    printf("    Length: %zu tokens\n", first_sequence_tokens.size());
                    printf("    Tokens: [");
                    for (size_t i = 0; i < std::min((size_t)10, first_sequence_tokens.size()); ++i) { // Выводим до 10 токенов
                        printf("%d%s", first_sequence_tokens[i], (i == std::min((size_t)10, first_sequence_tokens.size()) - 1) ? "" : ", ");
                    }
                    if (first_sequence_tokens.size() > 10) {
                        printf("...");
                    }
                    printf("]\n");
                } else {
                    fprintf(stderr, "    Error: Could not read data for first sequence.\n");
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
