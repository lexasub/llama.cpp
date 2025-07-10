#include "gguf-converter.h"

#include <cstdio>     // Для fprintf, snprintf
#include <iostream>
#include <memory>     // Для std::unique_ptr
#include <stdexcept>  // Для std::runtime_error
#include <vector>

#include "dataset-reader.h"
#include "gguf-file.h"    // Для GGUFFile
#include "gguf-writer.h"  // Для GGUFWriter
#include "llama.h"        // Для llama_model_free, llama_backend_free
#include "parquet-reader.h"
#include "text-reader.h"

// Метод для выполнения процесса конвертации.
bool GGUFConverter::convert(const ConvertParams& params) {
    // --- Создание DataReader на основе input_type ---
    std::unique_ptr<DatasetReader> reader;
    if (params.input_type == "text") {
        reader = std::make_unique<TextDatasetReader>(params.model, params.max_seq_len, params.pre_tokenized);
    } else if (params.input_type == "parquet") {
        reader = std::make_unique<ParquetDatasetReader>(params.model, params.max_seq_len, params.pre_tokenized, params.parquet_text_column, params.parquet_tokens_column);
    } else {
        fprintf(stderr, "error: Unsupported input type: %s\n", params.input_type.c_str());
        return false;
    }

    // Открытие источника данных
    if (!reader->open(params.input_path)) {
        fprintf(stderr, "error: Failed to open data source %s\n", params.input_path.c_str());
        return false;
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
        return false;
    }

    // Передаем указатель на gguf_file в GGUFWriter
    GGUFWriter writer(gguf_file.get());

    // Инициализируем метаданные GGUF файла
    writer.init_metadata(params.model, params.input_path, sequence_lengths.size());
    printf("Metadata written.\n");

    // --- ВТОРОЙ ПРОХОД: Запись тензоров ---
    printf("Second pass: Writing tensors to GGUF file...\n");
    if (!reader->reset()) {
        fprintf(stderr, "error: Failed to reset data reader for second pass.\n");
        return false;
    }

    uint64_t current_sequence_idx = 0;
    while (reader->read_next_sequence(tokens)) {
        if (current_sequence_idx >= sequence_lengths.size()) {
            fprintf(stderr, "error: file ended prematurely on second pass. Expected %zu sequences, but reached end of file at %lu.\n", sequence_lengths.size(), current_sequence_idx);
            break;
        }

        uint32_t expected_n_tokens = sequence_lengths[current_sequence_idx];
        uint32_t actual_n_tokens = tokens.size();

        // Если количество токенов не совпадает, это критическая ошибка, так как
        // метаданные, собранные на первом проходе, будут неверными для этого тензора.
        // Прерываем конвертацию, чтобы избежать создания поврежденного GGUF файла.
        if (actual_n_tokens != expected_n_tokens) {
            fprintf(stderr, "error: Tokenization mismatch on second pass for sequence %lu. Expected %u tokens, got %u.\n", current_sequence_idx, expected_n_tokens, actual_n_tokens);
            fprintf(stderr, "This indicates a non-deterministic tokenizer or an issue with input reading. Aborting conversion.\n");
            return false; // Прерываем конвертацию
        }

        // Добавляем тензор только если есть токены
        if (actual_n_tokens > 0) {
            writer.add_sequence_tensor(current_sequence_idx, tokens);
        } else {
            // Если ожидалось 0 токенов, но строка не была пустой, выводим предупреждение
            if (expected_n_tokens != 0) {
                fprintf(stderr, "warning: sequence %lu resulted in 0 tokens on second pass, but expected %u.\n", current_sequence_idx, expected_n_tokens);
                fprintf(stderr, "This indicates a non-deterministic tokenizer or an issue with input reading. Aborting conversion.\n");
                return false; // Прерываем конвертацию
            }
        }
        current_sequence_idx++;
    }
    reader->close(); // Закрываем DataReader после использования
    printf("Second pass complete.\n\n");

    // Сохранение файла на диск
    printf("Writing GGUF data to %s...\n", params.output_path.c_str());
    if (!writer.write_to_file(params.output_path)) {
        fprintf(stderr, "error: Failed to write GGUF file %s\n", params.output_path.c_str());
        return false;
    }

    printf("Conversion successful!\n");
    printf("Output file: %s\n", params.output_path.c_str());

    return true;
}
