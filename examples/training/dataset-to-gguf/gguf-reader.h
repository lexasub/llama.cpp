#pragma once

#include "gguf-file.h" // Для GGUFFile
#include "llama.h"     // Для llama_token
#include "gguf.h"      // Для gguf_type (пока оставляем, так как используется в других местах)
#include "ggml.h"      // Для ggml_type

#include <string>
#include <vector>
#include <memory> // Для std::unique_ptr
#include <map>    // Для хранения метаданных

// Класс GGUFReader предназначен для чтения GGUF файлов,
// предоставляя доступ к метаданным и данным тензоров.
class GGUFReader {
public:
    // Конструктор: инициализирует ридер для чтения из указанного GGUF файла.
    // path: путь к GGUF файлу.
    // throws std::runtime_error if file cannot be opened or context cannot be initialized.
    GGUFReader(const std::string& path);

    // Деструктор.
    ~GGUFReader() = default;

    // Проверяет, был ли ридер успешно инициализирован.
    bool is_initialized() const { return gguf_file_ptr != nullptr && gguf_file_ptr->is_initialized(); }

    // Получает строковое значение метаданных по ключу.
    std::string get_metadata_str(const std::string& key, const std::string& defaultValue = "") const;

    // Получает uint64_t значение метаданных по ключу.
    uint64_t get_metadata_u64(const std::string& key, uint64_t defaultValue = 0) const;

    // Получает количество тензоров в файле.
    int64_t get_tensor_count() const;

    // Получает имя тензора по индексу.
    std::string get_tensor_name(int64_t index) const;

    // Получает тип тензора по индексу.
    // Возвращает ggml_type, так как gguf_get_tensor_type возвращает ggml_type.
    ggml_type get_tensor_type(int64_t index) const;

    // Получает размер тензора в байтах по индексу.
    size_t get_tensor_size(int64_t index) const;

    // Читает данные тензора по индексу в вектор токенов.
    // index: индекс тензора.
    // tokens: вектор, в который будут прочитаны токены.
    // Возвращает true в случае успеха, false в случае ошибки (например, тензор не найден,
    // или его тип не GGML_TYPE_I32, или размер не соответствует).
    bool read_tensor_data(int64_t index, std::vector<llama_token>& tokens) const;

private:
    std::unique_ptr<GGUFFile> gguf_file_ptr; // Указатель на объект GGUFFile
    std::string file_path_;                  // Путь к файлу, из которого читается GGUF
};

