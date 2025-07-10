#include "gguf-converter.h"

#include <cinttypes>
#include <cstdio>     // For fprintf, snprintf
#include <iostream>
#include <memory>     // For std::unique_ptr
#include <stdexcept>  // For std::runtime_error
#include <vector>

#include "dataset-reader.h"
#include "gguf-file.h"    // For GGUFFile
#include "gguf-writer.h"  // For GGUFWriter
#include "llama.h"        // For llama_model_free, llama_backend_free
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

    uint64_t total_sequence_count = 0;
    std::vector<uint32_t> sequence_lengths; // Будет хранить длины последовательностей для текстовых файлов

    // --- ПЕРВЫЙ ПРОХОД: Сбор длин последовательностей или получение общего количества ---
    printf("First pass: Reading input data and getting sequence lengths...\n");

    if (params.input_type == "parquet") {
        // Для Parquet, получаем общее количество последовательностей из метаданных
        total_sequence_count = reader->get_total_sequences();
        printf("First pass complete. Found %" PRIu64 " sequences (from Parquet metadata).\n\n", total_sequence_count);
    } else { // Для текстовых файлов
        // Для текстовых файлов, выполняем полный первый проход для подсчета последовательностей
        // и их длин (так как это единственный способ узнать точное количество токенов).
        std::vector<llama_token> tokens;
        while (reader->read_next_sequence(tokens)) {
            sequence_lengths.push_back(tokens.size());
        }
        total_sequence_count = sequence_lengths.size();
        printf("First pass complete. Found %" PRIu64 " sequences.\n\n", total_sequence_count);
    }

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
    writer.init_metadata(params.model, params.input_path, total_sequence_count);
    printf("Metadata written.\n");

    // --- ВТОРОЙ ПРОХОД: Запись тензоров ---
    printf("Second pass: Writing tensors to GGUF file...\n");
    if (!reader->reset()) {
        fprintf(stderr, "error: Failed to reset data reader for second pass.\n");
        return false;
    }

    uint64_t current_sequence_idx = 0;
    std::vector<llama_token> tokens; // Переиспользуем вектор токенов
    while (reader->read_next_sequence(tokens)) {
        if (current_sequence_idx >= total_sequence_count) {
            fprintf(stderr, "error: file ended prematurely on second pass. Expected %" PRIu64 " sequences, but reached end of file at %" PRIu64 ".\n", total_sequence_count, current_sequence_idx);
            break;
        }

        uint32_t expected_n_tokens;
        if (params.input_type == "text") {
            // Для текстовых файлов используем длины, собранные на первом проходе
            expected_n_tokens = sequence_lengths[current_sequence_idx];
        } else {
            // Для Parquet, мы не знаем ожидаемую длину заранее,
            // поэтому просто используем фактическую длину прочитанной последовательности.
            // Если Parquet-файл содержит пустые последовательности, они будут обработаны.
            expected_n_tokens = tokens.size();
        }

        uint32_t actual_n_tokens = tokens.size();

        // Если количество токенов не совпадает (только для текстовых, где мы это знаем заранее),
        // это критическая ошибка, так как метаданные, собранные на первом проходе, будут неверными для этого тензора.
        // Прерываем конвертацию, чтобы избежать создания поврежденного GGUF файла.
        if (params.input_type == "text" && actual_n_tokens != expected_n_tokens) {
            fprintf(stderr, "error: Tokenization mismatch on second pass for sequence %" PRIu64 ". Expected %u tokens, got %u.\n", current_sequence_idx, expected_n_tokens, actual_n_tokens);
            fprintf(stderr, "This indicates a non-deterministic tokenizer or an issue with input reading. Aborting conversion.\n");
            return false; // Прерываем конвертацию
        }

        // Добавляем тензор только если есть токены
        if (actual_n_tokens > 0) {
            writer.add_sequence_tensor(current_sequence_idx, tokens);
        } else {
            // Если ожидалось 0 токенов, но строка не была пустой, выводим предупреждение
            // (Это условие `expected_n_tokens != 0` актуально только для текстовых файлов,
            // где мы могли получить 0 токенов на первом проходе для непустой строки.)
            if (params.input_type == "text" && expected_n_tokens != 0) {
                fprintf(stderr, "warning: sequence %" PRIu64 " resulted in 0 tokens on second pass, but expected %u.\n", current_sequence_idx, expected_n_tokens);
                // Продолжаем, так как это может быть допустимо для некоторых датасетов,
                // но предупреждаем о возможном несоответствии.
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
