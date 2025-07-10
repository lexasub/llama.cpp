
#include "text-reader.h"

#include <algorithm>  // Для std::min
#include <iostream>

#include "llama.h"  // Для llama_tokenize, llama_model_get_vocab, llama_vocab_n_tokens, llama_vocab_get_text

// Конструктор
TextDatasetReader::TextDatasetReader(const struct llama_model* model, int32_t max_seq_len, bool pre_tokenized)
    : model(model), max_seq_len(max_seq_len), pre_tokenized(pre_tokenized), tokens_buffer(max_seq_len) {}

// Деструктор
TextDatasetReader::~TextDatasetReader() {
    close();
}

// Открывает текстовый файл для чтения
bool TextDatasetReader::open(const std::string& path) {
    file_path_ = path; // Store the file path
    input_file.open(path);
    if (!input_file.is_open()) {
        std::cerr << "Error: Failed to open input file " << path << std::endl;
        return false;
    }
    return true;
}

// Читает следующую последовательность токенов из файла
bool TextDatasetReader::read_next_sequence(std::vector<llama_token>& tokens) {
    std::string line;
    if (!std::getline(input_file, line)) {
        return false; // Конец файла или ошибка чтения
    }

    tokens.clear(); // Очищаем вектор для новой последовательности
    int n_tokens = 0;

    if (line.empty()) {
        // Пустая строка, возвращаем пустую последовательность
        return true;
    }

    if (pre_tokenized) {
        // Режим предварительно токенизированных данных: парсим токены из строки
        std::istringstream iss(line);
        llama_token token_id;
        while (iss >> token_id) {
            if (n_tokens < max_seq_len) {
                tokens.push_back(token_id);
                n_tokens++;
            } else {
                // Обрезаем, если превышает max_seq_len
                break;
            }
        }
    } else {
        // Режим текстовых данных: токенизируем строку
        // Убедимся, что буфер достаточно большой
        if (tokens_buffer.size() < (size_t)max_seq_len) {
            tokens_buffer.resize(max_seq_len);
        }
        n_tokens = llama_tokenize(llama_model_get_vocab(model), line.c_str(), line.length(), tokens_buffer.data(), max_seq_len, false, true);
        if (n_tokens < 0) {
            std::cerr << "Error: Tokenization failed for line: " << line << std::endl;
            // Возвращаем пустую последовательность в случае ошибки токенизации
            return true;
        }
        tokens.assign(tokens_buffer.begin(), tokens_buffer.begin() + n_tokens);
    }
    return true;
}

// Закрывает файл
void TextDatasetReader::close() {
    if (input_file.is_open()) {
        input_file.close();
    }
}

// Сбрасывает файловый указатель к началу файла
bool TextDatasetReader::reset() {
    if (input_file.is_open()) {
        input_file.clear(); // Очистить все флаги ошибок (например, EOF)
        input_file.seekg(0, std::ios::beg); // Переместить указатель в начало
        return true;
    }
    // If not open, try to open it again using the stored path
    return open(file_path_);
}

// Метод для получения общего количества последовательностей в датасете.
// Для текстовых файлов это будет количество строк.
// Примечание: Этот метод будет медленным для очень больших файлов,
// так как он читает весь файл для подсчета строк.
uint64_t TextDatasetReader::get_total_sequences() const {
    if (file_path_.empty()) {
        std::cerr << "Error (TextDatasetReader::get_total_sequences): File path not set." << std::endl;
        return 0;
    }

    std::ifstream temp_file(file_path_);
    if (!temp_file.is_open()) {
        std::cerr << "Error (TextDatasetReader::get_total_sequences): Failed to open file '" << file_path_ << "' for counting lines." << std::endl;
        return 0;
    }

    uint64_t count = 0;
    std::string line;
    while (std::getline(temp_file, line)) {
        count++;
    }
    temp_file.close();
    return count;
}
