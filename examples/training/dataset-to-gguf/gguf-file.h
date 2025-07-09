#pragma once

#include "gguf.h" // Для struct gguf_context, enum gguf_type
#include "ggml.h" // Для struct ggml_tensor

#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept> // Для std::runtime_error

// Класс GGUFFile инкапсулирует gguf_context и предоставляет
// высокоуровневый API для работы с GGUF файлами (чтение/запись метаданных и тензоров).
class GGUFFile {
public:
    // Конструктор: инициализирует пустой GGUF контекст для записи.
    GGUFFile();

    // Конструктор: инициализирует GGUF контекст из существующего файла для чтения.
    // path: путь к GGUF файлу для открытия.
    // throws std::runtime_error if file cannot be opened or context cannot be initialized.
    GGUFFile(const std::string& path);

    // Деструктор: освобождает GGUF контекст.
    ~GGUFFile();

    // Проверяет, был ли GGUF контекст успешно инициализирован.
    bool is_initialized() const { return ctx != nullptr; }

    // --- Методы для работы с метаданными (KV-парами) ---

    // Устанавливает строковое значение для ключа.
    void set_val_str(const std::string& key, const std::string& value);

    // Устанавливает uint64_t значение для ключа.
    void set_val_u64(const std::string& key, uint64_t value);

    // Устанавливает массив строк для ключа.
    void set_arr_str(const std::string& key, const std::vector<const char*>& values);

    // Устанавливает массив данных заданного типа.
    // data: указатель на данные.
    // n: количество элементов.
    void set_arr_data(const std::string& key, gguf_type type, const void* data, size_t n);

    // Получает строковое значение по ключу.
    // key: ключ метаданных.
    // defaultValue: значение, возвращаемое, если ключ не найден или имеет неправильный тип.
    std::string get_val_str(const std::string& key, const std::string& defaultValue = "") const;

    // Получает uint64_t значение по ключу.
    uint64_t get_val_u64(const std::string& key, uint64_t defaultValue = 0) const;

    // --- Методы для работы с тензорами ---

    // Добавляет тензор в GGUF контекст.
    // tensor: указатель на ggml_tensor, который будет добавлен.
    // Важно: ggml_tensor должен быть создан во временном ggml_context,
    // который будет освобожден после добавления.
    void add_tensor(struct ggml_tensor* tensor);

    // Устанавливает данные для тензора по его имени.
    // name: имя тензора.
    // data: указатель на сырые данные тензора.
    void set_tensor_data(const std::string& name, const void* data);

    // Получает количество тензоров в GGUF файле.
    int64_t get_n_tensors() const;

    // Получает ggml_tensor по индексу.
    // Возвращает nullptr, если индекс некорректен.
    struct ggml_tensor* get_tensor_by_idx(int64_t idx) const;

    // --- Методы для сохранения/загрузки файла ---

    // Записывает весь GGUF контекст в файл.
    // output_path: путь к файлу для сохранения.
    // only_meta: если true, записывает только метаданные (без данных тензоров).
    // Возвращает true в случае успеха, false в случае ошибки.
    bool write_to_file(const std::string& output_path, bool only_meta = false);

private:
    struct gguf_context* ctx; // Внутренний GGUF контекст

    // Приватная вспомогательная функция для поиска ключа
    int64_t find_key(const std::string& key) const;
};

