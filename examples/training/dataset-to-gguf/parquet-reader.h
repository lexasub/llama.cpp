#pragma once

#include "dataset-reader.h" // Include the base DataReader class
#include "llama.h"       // For llama_token

// Include necessary Apache Arrow and Parquet headers
// You will need to link against these libraries (e.g., -larrow -lparquet)
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>

#include <string>
#include <vector>
#include <memory> // For std::unique_ptr

// Implementation of DataReader for reading Parquet files.
// This class will handle reading tokenized sequences from a Parquet file.
class ParquetDatasetReader : public DatasetReader {
public:
    // Constructor.
    // model: Pointer to the llama model for tokenization (can be nullptr if data is pre-tokenized).
    // max_seq_len: Maximum sequence length for truncation.
    // pre_tokenized: If true, input data is already tokenized (token IDs in a numeric column).
    // Note: For Parquet, 'pre_tokenized' implies reading a column of list<int32> or similar.
    ParquetDatasetReader(const struct llama_model* model, int32_t max_seq_len, bool pre_tokenized);

    // Destructor.
    ~ParquetDatasetReader();

    // Opens the Parquet file for reading.
    // path: Path to the Parquet file.
    // Returns true if the source is successfully opened, otherwise false.
    bool open(const std::string& path) override;

    // Reads the next sequence of tokens from the Parquet file.
    // tokens: Vector where the read tokens will be stored.
    // Returns true if a sequence is successfully read, otherwise false (including end of file).
    bool read_next_sequence(std::vector<llama_token>& tokens) override;

    // Closes the Parquet file.
    void close() override;

    // Resets the reader to the beginning of the Parquet file.
    // Returns true if reset is successful, otherwise false.
    bool reset() override;

private:
    const struct llama_model* model_; // Llama model for tokenization (if needed)
    int32_t max_seq_len_;             // Maximum sequence length
    bool pre_tokenized_;              // Flag for pre-tokenized data

    std::shared_ptr<arrow::io::ReadableFile> input_file_; // Arrow file handle
    std::unique_ptr<parquet::arrow::FileReader> parquet_reader_; // Parquet reader
    std::shared_ptr<arrow::Table> current_table_; // Current table batch being processed

    int64_t current_row_in_table_; // Current row index within the current_table_
    int current_column_index_;     // Index of the column containing text/tokens
    std::string m_filePath;

    // Private helper to get the next batch of data (if using batch processing)
    bool get_next_batch();

    // Determine the column index based on whether data is pre-tokenized or raw text
    // This will need to be flexible based on your Parquet schema
    // For simplicity, let's assume a fixed column name for text or tokens
    const std::string text_column_name_ = "text";
    const std::string tokens_column_name_ = "tokens"; // Assuming tokens are stored as list<int32>
};

