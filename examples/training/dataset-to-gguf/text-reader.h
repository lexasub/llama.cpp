#pragma once

#include <fstream>  // Для std::ifstream
#include <sstream>  // Для std::istringstream
#include <string>   // Для std::string
#include <vector>   // Для std::vector

#include "dataset-reader.h"
#include "llama.h"  // Для llama_tokenize и llama_model

// Реализация DataReader для чтения текстовых файлов.
// Поддерживает как обычный текст, так и предварительно токенизированные данные.
class TextDatasetReader : public DatasetReader {
public:
    // Конструктор.
    // model: указатель на модель llama для токенизации (может быть nullptr, если pre_tokenized = true).
    // max_seq_len: максимальная длина последовательности для обрезки.
    // pre_tokenized: если true, входные данные уже токенизированы (токены в виде чисел).
    TextDatasetReader(const struct llama_model* model, int32_t max_seq_len, bool pre_tokenized);

    // Деструктор.
    ~TextDatasetReader();

    // Открывает текстовый файл для чтения.
    bool open(const std::string& path) override;

    // Читает следующую последовательность токенов из файла.
    // Если pre_tokenized = true, парсит числа из строки.
    // Если pre_tokenized = false, токенизирует строку с помощью llama_model.
    bool read_next_sequence(std::vector<llama_token>& tokens) override;

    // Закрывает файл.
    void close() override;

    // Сбрасывает файловый указатель к началу файла.
    bool reset() override;

    // Метод для получения общего количества последовательностей в датасете.
    // Для текстовых файлов это будет количество строк.
    uint64_t get_total_sequences() const override;

private:
    const struct llama_model* model; // Модель для токенизации
    int32_t max_seq_len;             // Максимальная длина последовательности
    bool pre_tokenized;              // Флаг предварительной токенизации
    std::ifstream input_file;        // Объект файлового потока
    std::string file_path_;          // Путь к файлу для reset и get_total_sequences
    std::vector<llama_token> tokens_buffer; // Внутренний буфер для токенов
};

