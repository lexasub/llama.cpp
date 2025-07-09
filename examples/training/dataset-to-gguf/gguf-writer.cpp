#include "gguf-writer.h"

#include <cinttypes>
#include <cstdio>   // Для snprintf
#include <cstring>  // Для memcpy
#include <ctime>
#include <iostream>
#include <stdexcept>  // Для std::runtime_error
#include <vector>

#include "gguf-file.h"  // Включаем GGUFFile
#include "llama.h"  // Для llama_model_get_vocab, llama_vocab_n_tokens, llama_vocab_get_text, llama_model_meta_val_str

// Конструктор: принимает указатель на объект GGUFFile
GGUFWriter::GGUFWriter(GGUFFile* gguf_file_ptr) : gguf_file(gguf_file_ptr) {
    if (!gguf_file) {
        throw std::runtime_error("GGUFFile pointer provided to GGUFWriter is null.");
    }
    if (!gguf_file->is_initialized()) {
        throw std::runtime_error("GGUFFile provided to GGUFWriter is not initialized.");
    }
}

// Инициализирует метаданные GGUF файла
void GGUFWriter::init_metadata(const struct llama_model* model, const std::string& input_path, uint64_t sequence_count) {
    if (!gguf_file) {
        std::cerr << "Error: GGUFFile is not set. Cannot set metadata." << std::endl;
        return;
    }

    gguf_file->set_val_str("training.format.version", "1.0");
    gguf_file->set_val_str("training.dataset.name", input_path);

    // Установка даты создания файла
    time_t now = time(0);
    char buf[sizeof "2011-10-08T07:07:09Z"];
    strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
    gguf_file->set_val_str("training.file.creation_date", buf);

    // Установка информации о токенизаторе
    char arch_name_buffer[128];
    int res = llama_model_meta_val_str(model, "general.architecture", arch_name_buffer, sizeof(arch_name_buffer));
    if (res >= 0) {
        gguf_file->set_val_str("training.tokenizer.gguf.model", arch_name_buffer);
    } else {
        gguf_file->set_val_str("training.tokenizer.gguf.model", "unknown");
    }

    // Установка словаря токенизатора
    const struct llama_vocab* vocab = llama_model_get_vocab(model);
    int vocab_size = llama_vocab_n_tokens(vocab);
    std::vector<const char *> vocab_list;
    vocab_list.reserve(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        vocab_list.push_back(llama_vocab_get_text(vocab, i));
    }
    gguf_file->set_arr_str("training.tokenizer.gguf.vocab", vocab_list);

    // Установка общего количества последовательностей
    gguf_file->set_val_u64("training.sequence.count", sequence_count);
}

// Добавляет последовательность токенов в GGUF файл как тензор
void GGUFWriter::add_sequence_tensor(uint64_t index, const std::vector<llama_token>& tokens) {
    if (!gguf_file) {
        std::cerr << "Error: GGUFFile is not set. Cannot add sequence tensor." << std::endl;
        return;
    }

    if (tokens.empty()) {
        return;
    }

    char tensor_name[128];
    snprintf(tensor_name, sizeof(tensor_name), "training.tensor.%" PRIu64, index);

    // Выделяем достаточно памяти для временного ggml_context, чтобы вместить тензор
    size_t n_tokens = tokens.size();
    size_t tensor_mem_size = ggml_tensor_overhead() + n_tokens * sizeof(int32_t);

    struct ggml_init_params ggml_params = {};
    ggml_params.mem_size   = tensor_mem_size;
    ggml_params.mem_buffer = NULL;
    ggml_params.no_alloc   = false;

    struct ggml_context * ggml_ctx = ggml_init(ggml_params);
    if (!ggml_ctx) {
        std::cerr << "Error: Failed to initialize ggml context for tensor " << index << std::endl;
        return;
    }

    // Создаем одномерный тензор типа GGML_TYPE_I32
    struct ggml_tensor * tensor = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(tensor, tensor_name);

    // Копируем данные токенов в буфер тензора
    memcpy(tensor->data, tokens.data(), n_tokens * sizeof(int32_t));

    // Добавляем тензор в GGUF контекст через GGUFFile
    gguf_file->add_tensor(tensor);

    // Устанавливаем данные тензора в GGUF контексте через GGUFFile
    gguf_file->set_tensor_data(tensor_name, tokens.data());

    ggml_free(ggml_ctx); // Освобождаем временный ggml контекст
}

// Записывает весь GGUF контекст (метаданды и тензоры) в указанный файл
bool GGUFWriter::write_to_file(const std::string& output_path) {
    if (!gguf_file) {
        std::cerr << "Error: GGUFFile is not set. Cannot write to file." << std::endl;
        return false;
    }
    return gguf_file->write_to_file(output_path, false);
}
