#include "gguf-reader.h"
#include <iostream>
#include <fstream> // Для чтения данных тензора напрямую
#include <stdexcept> // Для std::runtime_error
#include <cstring> // Для memcpy
#include <algorithm> // Для std::min

// Конструктор: инициализирует ридер для чтения из указанного GGUF файла.
GGUFReader::GGUFReader(const std::string& path) : gguf_file_ptr(nullptr), file_path_(path) {
    try {
        // Инициализируем GGUFFile в режиме чтения (с no_alloc = true для ggml_context)
        gguf_file_ptr = std::make_unique<GGUFFile>(path);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: GGUFReader constructor failed to initialize GGUFFile from path '" << path << "': " << e.what() << std::endl;
        // Перебрасываем исключение, так как инициализация не удалась
        throw;
    }
}

// Получает строковое значение метаданных по ключу.
std::string GGUFReader::get_metadata_str(const std::string& key, const std::string& defaultValue) const {
    if (!is_initialized()) {
        std::cerr << "Error (GGUFReader::get_metadata_str): GGUFReader is not initialized. Cannot retrieve metadata for key '" << key << "'." << std::endl;
        return defaultValue;
    }
    return gguf_file_ptr->get_val_str(key, defaultValue);
}

// Получает uint64_t значение метаданных по ключу.
uint64_t GGUFReader::get_metadata_u64(const std::string& key, uint64_t defaultValue) const {
    if (!is_initialized()) {
        std::cerr << "Error (GGUFReader::get_metadata_u64): GGUFReader is not initialized. Cannot retrieve metadata for key '" << key << "'." << std::endl;
        return defaultValue;
    }
    return gguf_file_ptr->get_val_u64(key, defaultValue);
}

// Получает количество тензоров в файле.
int64_t GGUFReader::get_tensor_count() const {
    if (!is_initialized()) {
        std::cerr << "Error (GGUFReader::get_tensor_count): GGUFReader is not initialized. Cannot get tensor count." << std::endl;
        return 0;
    }
    return gguf_file_ptr->get_n_tensors();
}

// Получает имя тензора по индексу.
std::string GGUFReader::get_tensor_name(int64_t index) const {
    if (!is_initialized()) {
        std::cerr << "Error (GGUFReader::get_tensor_name): GGUFReader is not initialized. Cannot get tensor name for index " << index << "." << std::endl;
        return "";
    }
    struct gguf_context* ctx_internal = gguf_file_ptr->get_gguf_context();
    if (!ctx_internal) {
        std::cerr << "Error (GGUFReader::get_tensor_name): Internal GGUF context is null for index " << index << "." << std::endl;
        return "";
    }
    if (index < 0 || index >= gguf_get_n_tensors(ctx_internal)) {
        std::cerr << "Error (GGUFReader::get_tensor_name): Tensor index " << index << " is out of bounds (total tensors: " << gguf_get_n_tensors(ctx_internal) << ")." << std::endl;
        return "";
    }
    return gguf_get_tensor_name(ctx_internal, index);
}

// Получает тип тензора по индексу.
ggml_type GGUFReader::get_tensor_type(int64_t index) const {
    if (!is_initialized()) {
        std::cerr << "Error (GGUFReader::get_tensor_type): GGUFReader is not initialized. Cannot get tensor type for index " << index << "." << std::endl;
        return GGML_TYPE_COUNT; // Неизвестный тип
    }
    struct gguf_context* ctx_internal = gguf_file_ptr->get_gguf_context();
    if (!ctx_internal) {
        std::cerr << "Error (GGUFReader::get_tensor_type): Internal GGUF context is null for index " << index << "." << std::endl;
        return GGML_TYPE_COUNT;
    }
    if (index < 0 || index >= gguf_get_n_tensors(ctx_internal)) {
        std::cerr << "Error (GGUFReader::get_tensor_type): Tensor index " << index << " is out of bounds (total tensors: " << gguf_get_n_tensors(ctx_internal) << ")." << std::endl;
        return GGML_TYPE_COUNT;
    }
    // Возвращаем ggml_type напрямую, без приведения к gguf_type
    return gguf_get_tensor_type(ctx_internal, index);
}

// Получает размер тензора в байтах по индексу.
size_t GGUFReader::get_tensor_size(int64_t index) const {
    if (!is_initialized()) {
        std::cerr << "Error (GGUFReader::get_tensor_size): GGUFReader is not initialized. Cannot get tensor size for index " << index << "." << std::endl;
        return 0;
    }
    struct gguf_context* ctx_internal = gguf_file_ptr->get_gguf_context();
    if (!ctx_internal) {
        std::cerr << "Error (GGUFReader::get_tensor_size): Internal GGUF context is null for index " << index << "." << std::endl;
        return 0;
    }
    if (index < 0 || index >= gguf_get_n_tensors(ctx_internal)) {
        std::cerr << "Error (GGUFReader::get_tensor_size): Tensor index " << index << " is out of bounds (total tensors: " << gguf_get_n_tensors(ctx_internal) << ")." << std::endl;
        return 0;
    }
    return gguf_get_tensor_size(ctx_internal, index);
}

// Читает данные тензора по индексу в вектор токенов.
bool GGUFReader::read_tensor_data(int64_t index, std::vector<llama_token>& tokens) const {
    if (!is_initialized()) {
        std::cerr << "Error (GGUFReader::read_tensor_data): GGUFReader is not initialized. Cannot read tensor data." << std::endl;
        return false;
    }

    struct gguf_context* ctx_internal = gguf_file_ptr->get_gguf_context();
    if (!ctx_internal) {
        std::cerr << "Error (GGUFReader::read_tensor_data): Internal GGUF context is null in GGUFFile." << std::endl;
        return false;
    }

    if (index < 0 || index >= gguf_get_n_tensors(ctx_internal)) {
        std::cerr << "Error (GGUFReader::read_tensor_data): Tensor with index " << index << " not found or out of bounds." << std::endl;
        return false;
    }

    // Получаем ggml_type напрямую и сравниваем с GGML_TYPE_I32
    ggml_type tensor_ggml_type = gguf_get_tensor_type(ctx_internal, index);
    if (tensor_ggml_type != GGML_TYPE_I32) {
        std::cerr << "Error (GGUFReader::read_tensor_data): Tensor type for '" << gguf_get_tensor_name(ctx_internal, index)
                  << "' is not GGML_TYPE_I32. Actual type: " << ggml_type_name(tensor_ggml_type) << std::endl;
        return false;
    }

    size_t expected_bytes = gguf_get_tensor_size(ctx_internal, index);
    if (expected_bytes == 0) {
        // Если тензор пустой, просто возвращаем пустой вектор токенов
        tokens.clear();
        return true;
    }

    size_t num_tokens = expected_bytes / sizeof(llama_token);
    if (expected_bytes % sizeof(llama_token) != 0) {
        std::cerr << "Warning (GGUFReader::read_tensor_data): Tensor size " << expected_bytes
                  << " bytes is not a multiple of llama_token size (" << sizeof(llama_token)
                  << " bytes) for tensor '" << gguf_get_tensor_name(ctx_internal, index) << "'. Data might be corrupted." << std::endl;
    }

    tokens.resize(num_tokens);

    size_t data_offset_in_file = gguf_get_data_offset(ctx_internal) + gguf_get_tensor_offset(ctx_internal, index);

    // Открываем файл для чтения данных, используя сохраненный путь
    std::ifstream file(file_path_, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error (GGUFReader::read_tensor_data): Could not open GGUF file '" << file_path_ << "' for reading tensor data." << std::endl;
        return false;
    }

    // Seek to the calculated offset
    file.seekg(data_offset_in_file, std::ios::beg);
    if (file.fail()) {
        std::cerr << "Error (GGUFReader::read_tensor_data): Failed to seek to offset " << data_offset_in_file
                  << " in file '" << file_path_ << "'. Stream state: good=" << file.good() << " eof=" << file.eof()
                  << " fail=" << file.fail() << " bad=" << file.bad() << std::endl;
        file.close();
        return false;
    }

    // Read the tensor data into the tokens vector
    file.read(reinterpret_cast<char*>(tokens.data()), expected_bytes);

    if (!file) { // Check if the read operation failed or reached EOF before reading all bytes
        std::cerr << "Error (GGUFReader::read_tensor_data): Failed to read " << expected_bytes << " bytes for tensor '"
                  << gguf_get_tensor_name(ctx_internal, index) << "' from file '" << file_path_ << "'." << std::endl;
        std::cerr << "  Stream state after read: good=" << file.good() << " eof=" << file.eof()
                  << " fail=" << file.fail() << " bad=" << file.bad() << std::endl;
        std::cerr << "  Bytes actually read: " << file.gcount() << std::endl;
        file.close();
        return false;
    }
    // Verify that the number of bytes read matches the expected bytes
    if (file.gcount() != (std::streamsize)expected_bytes) {
        std::cerr << "Error (GGUFReader::read_tensor_data): Mismatch in bytes read for tensor '"
                  << gguf_get_tensor_name(ctx_internal, index) << "'. Expected " << expected_bytes
                  << ", but read " << file.gcount() << "." << std::endl;
        file.close();
        return false;
    }

    file.close();
    return true;
}
