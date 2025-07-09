#include "parquet-reader.h"
#include "llama.h"
#include <iostream>
#include <algorithm> // For std::min

// Constructor
ParquetDatasetReader::ParquetDatasetReader(const struct llama_model* model, int32_t max_seq_len, bool pre_tokenized)
    : model_(model),
      max_seq_len_(max_seq_len),
      pre_tokenized_(pre_tokenized),
      current_row_in_table_(0),
      current_column_index_(-1) { // Initialize to -1, will be set in open
}

// Destructor
ParquetDatasetReader::~ParquetDatasetReader() {
    close();
}

// Opens the Parquet file for reading.
bool ParquetDatasetReader::open(const std::string& path) {
    // Close any previously open file
    close();

    m_filePath = path; // Store the file path for reset()

    // Open the Parquet file
    arrow::Status status = arrow::io::ReadableFile::Open(path).Value(&input_file_);
    if (!status.ok()) {
        std::cerr << "Error (ParquetDatasetReader::open): Failed to open Parquet file '" << path << "': " << status.ToString() << std::endl;
        return false;
    }

    // Create a Parquet reader using parquet::arrow::OpenFile
    // PARQUET_ASSIGN_OR_THROW is a macro that handles status checking and assignment
    // For manual error handling, it's like:
    // auto reader_result = parquet::arrow::OpenFile(input_file_, arrow::default_memory_pool());
    // if (!reader_result.ok()) { /* handle error */ }
    // parquet_reader_ = std::move(reader_result.ValueUnsafe());
    arrow::Result<std::unique_ptr<parquet::arrow::FileReader>> reader_raw =
        parquet::arrow::OpenFile(input_file_, arrow::default_memory_pool());

    if (!reader_raw.ok()) {
        std::cerr << "Error (ParquetDatasetReader::open): Failed to create Parquet file reader for '" << path << "': " << reader_raw.status().ToString() << std::endl;
        close();
        return false;
    }
    parquet_reader_ = std::move(reader_raw.ValueUnsafe());


    // Get the schema to determine the correct column index
    std::shared_ptr<arrow::Schema> schema;
    status = parquet_reader_->GetSchema(&schema); // Corrected: Use GetSchema and pass by address
    if (!status.ok() || schema == nullptr) {
        std::cerr << "Error (ParquetDatasetReader::open): Failed to get schema from Parquet file: " << status.ToString() << std::endl;
        close();
        return false;
    }

    // Determine the column index based on pre_tokenized_ flag
    if (pre_tokenized_) {
        current_column_index_ = schema->GetFieldIndex(tokens_column_name_);
        if (current_column_index_ == -1) {
            std::cerr << "Error (ParquetDatasetReader::open): Pre-tokenized mode selected, but column '" << tokens_column_name_ << "' not found in Parquet schema." << std::endl;
            close();
            return false;
        }
        // Validate column type: should be List<Int32>
        if (schema->field(current_column_index_)->type()->id() != arrow::Type::LIST) {
            std::cerr << "Error (ParquetDatasetReader::open): Column '" << tokens_column_name_ << "' is not of LIST type as expected for pre-tokenized data. Actual type: " << schema->field(current_column_index_)->type()->ToString() << std::endl;
            close();
            return false;
        }
        auto list_type = std::static_pointer_cast<arrow::ListType>(schema->field(current_column_index_)->type());
        if (list_type->value_type()->id() != arrow::Type::INT32) {
            std::cerr << "Error (ParquetDatasetReader::open): List items in column '" << tokens_column_name_ << "' are not of INT32 type as expected. Actual value type: " << list_type->value_type()->ToString() << std::endl;
            close();
            return false;
        }

    } else {
        current_column_index_ = schema->GetFieldIndex(text_column_name_);
        if (current_column_index_ == -1) {
            std::cerr << "Error (ParquetDatasetReader::open): Raw text mode selected, but column '" << text_column_name_ << "' not found in Parquet schema." << std::endl;
            close();
            return false;
        }
        // Validate column type: should be String
        if (schema->field(current_column_index_)->type()->id() != arrow::Type::STRING) {
            std::cerr << "Error (ParquetDatasetReader::open): Column '" << text_column_name_ << "' is not of STRING type as expected for raw text. Actual type: " << schema->field(current_column_index_)->type()->ToString() << std::endl;
            close();
            return false;
        }
    }

    // Read the first batch
    return get_next_batch();
}

// Reads the next sequence of tokens from the Parquet file.
bool ParquetDatasetReader::read_next_sequence(std::vector<llama_token>& tokens) {
    tokens.clear();

    // If current_table_ is null or we've processed all rows in the current batch, get the next batch
    if (!current_table_ || current_row_in_table_ >= current_table_->num_rows()) {
        if (!get_next_batch()) {
            return false; // No more batches or error getting next batch
        }
    }

    if (!current_table_ || current_table_->num_rows() == 0) {
        return false; // Should not happen if get_next_batch was successful, but as a safeguard
    }

    // Assuming single chunk for simplicity. For multi-chunk columns, you'd iterate through chunks.
    std::shared_ptr<arrow::Array> column_array = current_table_->column(current_column_index_)->chunk(0);

    if (pre_tokenized_) {
        // Pre-tokenized data: read List<Int32> array
        auto list_array = std::static_pointer_cast<arrow::ListArray>(column_array);
        auto value_array = std::static_pointer_cast<arrow::Int32Array>(list_array->values());

        if (list_array->IsNull(current_row_in_table_)) {
            // Handle null list (empty sequence)
            current_row_in_table_++;
            return true;
        }

        int32_t start_offset = list_array->value_offset(current_row_in_table_);
        int32_t end_offset = list_array->value_offset(current_row_in_table_ + 1);
        int32_t num_tokens_in_row = end_offset - start_offset;

        tokens.reserve(std::min((int32_t)max_seq_len_, num_tokens_in_row));
        for (int32_t i = 0; i < num_tokens_in_row && i < max_seq_len_; ++i) {
            tokens.push_back(static_cast<llama_token>(value_array->Value(start_offset + i)));
        }

    } else {
        // Raw text data: read String array and tokenize
        if (!model_) {
            std::cerr << "Error (ParquetDatasetReader::read_next_sequence): Llama model not provided for tokenization of raw text." << std::endl;
            return false;
        }

        auto string_array = std::static_pointer_cast<arrow::StringArray>(column_array);
        if (string_array->IsNull(current_row_in_table_)) {
            // Handle null string (empty sequence)
            current_row_in_table_++;
            return true;
        }

        std::string text = string_array->GetString(current_row_in_table_);
        std::vector<llama_token> tokens_buffer(max_seq_len_); // Use a temporary buffer for tokenization

        int n_tokens = llama_tokenize(llama_model_get_vocab(model_), text.c_str(), text.length(), tokens_buffer.data(), max_seq_len_, false, true);
        if (n_tokens < 0) {
            std::cerr << "Error (ParquetDatasetReader::read_next_sequence): Tokenization failed for text: '" << text << "'" << std::endl;
            current_row_in_table_++;
            return true; // Return true with empty tokens to continue processing
        }
        tokens.assign(tokens_buffer.begin(), tokens_buffer.begin() + n_tokens);
    }

    current_row_in_table_++;
    return true;
}

// Closes the Parquet file.
void ParquetDatasetReader::close() {
    parquet_reader_.reset();
    current_table_.reset();
    if (input_file_) { // Corrected: Removed IsOpen() as ReadableFile doesn't have it
        arrow::Status status = input_file_->Close();
        if (!status.ok()) {
            std::cerr << "Warning (ParquetDatasetReader::close): Failed to close Arrow file: " << status.ToString() << std::endl;
        }
    }
    input_file_.reset();
    current_row_in_table_ = 0;
    current_column_index_ = -1;
    m_filePath.clear(); // Clear the stored path
}

// Resets the reader to the beginning of the Parquet file.
bool ParquetDatasetReader::reset() {
    if (m_filePath.empty()) { // Corrected: Check if path is stored
        std::cerr << "Error (ParquetDatasetReader::reset): Cannot reset, file path was not stored." << std::endl;
        return false;
    }
    // Re-open the file and re-initialize the reader
    return open(m_filePath); // Corrected: Use the stored path
}

// Private helper to get the next batch of data
bool ParquetDatasetReader::get_next_batch() {
    current_table_.reset(); // Clear previous table
    current_row_in_table_ = 0; // Reset row index for new table

    // Read the entire table. For very large files, consider reading row groups or batches.
    // For simplicity and initial implementation, reading the whole table is fine.
    // For streaming, you would iterate through row groups or record batches.
    arrow::Status status = parquet_reader_->ReadTable(&current_table_); // Corrected: Pass by address
    if (!status.ok()) {
        std::cerr << "Error (ParquetDatasetReader::get_next_batch): Failed to read table from Parquet file: " << status.ToString() << std::endl;
        return false;
    }

    if (!current_table_ || current_table_->num_rows() == 0) {
        return false; // No more data
    }
    return true;
}
