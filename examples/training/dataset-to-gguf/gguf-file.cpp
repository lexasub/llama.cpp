#include "gguf-file.h"

#include <iostream>
#include <vector>
#include <ctime>
#include <cstring> // Для memcpy
#include <cstdio>  // Для snprintf
#include <stdexcept> // Для std::runtime_error

// Конструктор: инициализирует пустой GGUF контекст для записи.
GGUFFile::GGUFFile() : ctx(nullptr) {
    ctx = gguf_init_empty();
    if (!ctx) {
        throw std::runtime_error("Failed to initialize empty GGUF context.");
    }
}

// Конструктор: инициализирует GGUF контекст из существующего файла для чтения.
GGUFFile::GGUFFile(const std::string& path) : ctx(nullptr) {
    struct gguf_init_params params = {};
    params.no_alloc = true; // Мы не хотим, чтобы gguf_init_from_file выделял ggml_context
                            // для тензоров, так как мы будем управлять этим отдельно.
    ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) {
        throw std::runtime_error("Failed to initialize GGUF context from file: " + path);
    }
}

// Деструктор: освобождает GGUF контекст.
GGUFFile::~GGUFFile() {
    if (ctx) {
        gguf_free(ctx);
        ctx = nullptr;
    }
}

// --- Методы для работы с метаданными (KV-парами) ---

// Устанавливает строковое значение для ключа.
void GGUFFile::set_val_str(const std::string& key, const std::string& value) {
    if (!ctx) throw std::runtime_error("GGUF context not initialized.");
    gguf_set_val_str(ctx, key.c_str(), value.c_str());
}

// Устанавливает uint64_t значение для ключа.
void GGUFFile::set_val_u64(const std::string& key, uint64_t value) {
    if (!ctx) throw std::runtime_error("GGUF context not initialized.");
    gguf_set_val_u64(ctx, key.c_str(), value);
}

// Устанавливает массив строк для ключа.
void GGUFFile::set_arr_str(const std::string& key, const std::vector<const char*>& values) {
    if (!ctx) throw std::runtime_error("GGUF context not initialized.");
    // Исправление: Приведение типа для соответствия сигнатуре gguf_set_arr_str
    // values.data() возвращает const char* const*, а gguf_set_arr_str ожидает const char**.
    // const_cast используется для снятия const с верхнего уровня указателя,
    // так как gguf_set_arr_str не изменяет сами строки.
    gguf_set_arr_str(ctx, key.c_str(), const_cast<const char**>(values.data()), values.size());
}

// Устанавливает массив данных заданного типа.
void GGUFFile::set_arr_data(const std::string& key, gguf_type type, const void* data, size_t n) {
    if (!ctx) throw std::runtime_error("GGUF context not initialized.");
    gguf_set_arr_data(ctx, key.c_str(), type, data, n);
}

// Получает строковое значение по ключу.
std::string GGUFFile::get_val_str(const std::string& key, const std::string& defaultValue) const {
    if (!ctx) return defaultValue;
    int64_t key_id = find_key(key);
    if (key_id == -1 || gguf_get_kv_type(ctx, key_id) != GGUF_TYPE_STRING) {
        return defaultValue;
    }
    return gguf_get_val_str(ctx, key_id);
}

// Получает uint64_t значение по ключу.
uint64_t GGUFFile::get_val_u64(const std::string& key, uint64_t defaultValue) const {
    if (!ctx) return defaultValue;
    int64_t key_id = find_key(key);
    if (key_id == -1 || gguf_get_kv_type(ctx, key_id) != GGUF_TYPE_UINT64) {
        return defaultValue;
    }
    return gguf_get_val_u64(ctx, key_id);
}

// --- Методы для работы с тензорами ---

// Добавляет тензор в GGUF контекст.
void GGUFFile::add_tensor(struct ggml_tensor* tensor) {
    if (!ctx) throw std::runtime_error("GGUF context not initialized.");
    gguf_add_tensor(ctx, tensor);
}

// Устанавливает данные для тензора по его имени.
void GGUFFile::set_tensor_data(const std::string& name, const void* data) {
    if (!ctx) throw std::runtime_error("GGUF context not initialized.");
    gguf_set_tensor_data(ctx, name.c_str(), data);
}

// Получает количество тензоров в GGUF файле.
int64_t GGUFFile::get_n_tensors() const {
    if (!ctx) return 0;
    return gguf_get_n_tensors(ctx);
}

// Получает ggml_tensor по индексу.
struct ggml_tensor* GGUFFile::get_tensor_by_idx(int64_t idx) const {
    if (!ctx || idx < 0 || idx >= gguf_get_n_tensors(ctx)) {
        return nullptr;
    }
    // Внимание: gguf_get_tensor_by_idx возвращает ggml_tensor*,
    // но его данные могут быть не загружены в память, если gguf_init_from_file
    // был вызван с no_alloc = true (как в нашем случае для чтения).
    // Для получения данных тензора из файла потребуется дополнительная логика
    // чтения из файла по смещению gguf_get_tensor_offset.
    return get_tensor_by_idx(idx);
}

// --- Методы для сохранения/загрузки файла ---

// Записывает весь GGUF контекст в файл.
bool GGUFFile::write_to_file(const std::string& output_path, bool only_meta) {
    if (!ctx) {
        std::cerr << "Error: GGUF context is not initialized. Cannot write to file." << std::endl;
        return false;
    }
    if (!gguf_write_to_file(ctx, output_path.c_str(), only_meta)) {
        std::cerr << "Error: Failed to write GGUF file to " << output_path << std::endl;
        return false;
    }
    return true;
}

// Приватная вспомогательная функция для поиска ключа
int64_t GGUFFile::find_key(const std::string& key) const {
    if (!ctx) return -1;
    return gguf_find_key(ctx, key.c_str());
}
