#ifndef GGUF_WRITER_H
#define GGUF_WRITER_H

#include "gguf-file.h" // Включаем наш новый класс GGUFFile
#include "llama.h"     // Для llama_token

#include <string>
#include <vector>
#include <cstdint>

// Класс для инкапсуляции логики записи GGUF файла.
// Теперь он использует GGUFFile для низкоуровневых операций.
class GGUFWriter {
  public:
    // Конструктор, принимающий указатель на объект GGUFFile.
    // gguf_file: указатель на инициализированный объект GGUFFile,
    //            который будет использоваться для записи.
    GGUFWriter(GGUFFile * gguf_file);

    // Деструктор (не освобождает gguf_file, так как он управляется извне).
    ~GGUFWriter() = default;

    // Инициализирует метаданные GGUF файла.
    // model: указатель на загруженную модель llama для получения информации о токенизаторе.
    // input_path: путь к входному файлу, используется для имени датасета.
    // sequence_count: общее количество последовательностей.
    void init_metadata(const struct llama_model * model, const std::string & input_path, uint64_t sequence_count);

    // Добавляет последовательность токенов в GGUF файл как тензор.
    // index: индекс последовательности (используется для имени тензора).
    // tokens: вектор токенов, представляющий последовательность.
    void add_sequence_tensor(uint64_t index, const std::vector<llama_token> & tokens);

    // Записывает весь GGUF контекст (метаданные и тензоры) в указанный файл.
    // output_path: путь к выходному GGUF файлу.
    // Возвращает true в случае успеха, false в случае ошибки.
    bool write_to_file(const std::string & output_path);

  private:
    GGUFFile * gguf_file;  // Указатель на объект GGUFFile
};

#endif
