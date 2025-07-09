#pragma once

#include "gguf-writer.h"
#include "llama.h" // Для struct llama_model

#include <string>
#include <vector>
#include <memory> // Для std::unique_ptr

// Структура для передачи параметров конвертации
struct ConvertParams {
    std::string input_path;
    std::string output_path;
    int32_t max_seq_len;
    bool pre_tokenized;
    std::string input_type;
    const struct llama_model* model; // Указатель на загруженную модель
};

// Класс GGUFConverter инкапсулирует высокоуровневую логику конвертации
// входных данных в формат GGUF.
class GGUFConverter {
public:
    // Конструктор по умолчанию.
    GGUFConverter() = default;

    // Метод для выполнения процесса конвертации.
    // params: структура, содержащая все необходимые параметры для конвертации.
    // Возвращает true в случае успешной конвертации, false в случае ошибки.
    bool convert(const ConvertParams& params);
};

