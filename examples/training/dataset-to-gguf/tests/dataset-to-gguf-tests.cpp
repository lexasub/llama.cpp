#include <cassert>     // For assert
#include <filesystem>  // For working with the file system (creating/deleting temporary files)
#include <fstream>
#include <iostream>    // For std::cerr
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dataset-to-gguf/gguf-converter.h"
#include "dataset-to-gguf/gguf-file.h"
#include "dataset-to-gguf/gguf-reader.h"
#include "dataset-to-gguf/gguf-writer.h"
#include "dataset-to-gguf/text-reader.h"
#include "llama.h"  // For llama_backend_init, llama_backend_free, llama_model_load_from_file, llama_model_free

namespace fs = std::filesystem;

// Global variables for tests requiring llama_model
static llama_model* g_llama_model = nullptr;
static std::string g_test_model_path = "../../gte-small.Q2_K.gguf"; // Specify the actual path to your model

// Helper for assertions
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "Assertion failed: " << message << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
            return false; \
        } \
    } while (0)

// Global setup for llama.cpp backend
bool SetUpLlamaBackend() {
    llama_backend_init();
    // Load the model for the tokenizer
    llama_model_params model_params = llama_model_default_params();
    g_llama_model = llama_model_load_from_file(g_test_model_path.c_str(), model_params);
    if (g_llama_model == nullptr) {
        std::cerr << "WARNING: Failed to load llama model for tests from " << g_test_model_path << ". Some tests may be skipped or fail." << std::endl;
        // It's okay to continue if model loading fails, but tests relying on it will skip.
    }
    return true;
}

// Global teardown for llama.cpp backend
void TearDownLlamaBackend() {
    if (g_llama_model) {
        llama_model_free(g_llama_model);
        g_llama_model = nullptr;
    }
    llama_backend_free();
}

// =============================================================================
// Tests for GGUFFile
// =============================================================================

bool TestGGUFFile_DefaultConstructorInitializesContext() {
    printf("  TestGGUFFile_DefaultConstructorInitializesContext\n");
    GGUFFile gguf_file;
    TEST_ASSERT(gguf_file.is_initialized(), "GGUFFile should be initialized by default constructor");
    return true;
}

bool TestGGUFFile_ConstructorFromFileThrowsOnError() {
    printf("  TestGGUFFile_ConstructorFromFileThrowsOnError\n");
    bool threw_exception = false;
    try {
        GGUFFile("non_existent_file.gguf");
    } catch (const std::runtime_error& e) {
        threw_exception = true;
    }
    TEST_ASSERT(threw_exception, "Constructor should throw for non-existent file");
    return true;
}

bool TestGGUFFile_SetAndGetMetadataString() {
    printf("  TestGGUFFile_SetAndGetMetadataString\n");
    GGUFFile gguf_file;
    gguf_file.set_val_str("test.key.string", "test_value");
    TEST_ASSERT(gguf_file.get_val_str("test.key.string") == "test_value", "Failed to get correct string value");
    TEST_ASSERT(gguf_file.get_val_str("non.existent.key", "default_value") == "default_value", "Failed to get default string value");
    return true;
}

bool TestGGUFFile_SetAndGetMetadataU64() {
    printf("  TestGGUFFile_SetAndGetMetadataU64\n");
    GGUFFile gguf_file;
    gguf_file.set_val_u64("test.key.u64", 12345ULL);
    TEST_ASSERT(gguf_file.get_val_u64("test.key.u64") == 12345ULL, "Failed to get correct u64 value");
    TEST_ASSERT(gguf_file.get_val_u64("non.existent.key.u64", 99ULL) == 99ULL, "Failed to get default u64 value");
    return true;
}

bool TestGGUFFile_SetAndGetMetadataStringArray() {
    printf("  TestGGUFFile_SetAndGetMetadataStringArray\n");
    GGUFFile gguf_file;
    std::vector<const char*> arr = {"val1", "val2", "val3"};
    gguf_file.set_arr_str("test.key.array_str", arr);
    // As noted before, verifying array content requires more complex logic to read the GGUF file.
    // For now, we assert that the operation doesn't crash.
    return true;
}

// =============================================================================
// Tests for GGUFReader
// =============================================================================

// Helper to create a temporary GGUF file for GGUFReader tests
bool CreateTestGGUFFile(const std::string& path, llama_model* model_ptr) {
    GGUFFile writer_file;
    GGUFWriter writer(&writer_file);

    writer.init_metadata(model_ptr, "dummy_input.txt", 2); // 2 sequences

    std::vector<llama_token> seq1 = {1, 2, 3, 4, 5};
    std::vector<llama_token> seq2 = {10, 20, 30};
    writer.add_sequence_tensor(0, seq1);
    writer.add_sequence_tensor(1, seq2);

    return writer.write_to_file(path);
}

bool TestGGUFReader_ConstructorInitializesFromFile() {
    printf("  TestGGUFReader_ConstructorInitializesFromFile\n");
    std::string test_gguf_path = "test_output_reader.gguf";
    TEST_ASSERT(CreateTestGGUFFile(test_gguf_path, g_llama_model), "Failed to create test GGUF file for reader test");

    GGUFReader reader(test_gguf_path);
    TEST_ASSERT(reader.is_initialized(), "GGUFReader should be initialized from file");
    fs::remove(test_gguf_path);
    return true;
}

bool TestGGUFReader_GetMetadata() {
    printf("  TestGGUFReader_GetMetadata\n");
    std::string test_gguf_path = "test_output_reader_meta.gguf";
    TEST_ASSERT(CreateTestGGUFFile(test_gguf_path, g_llama_model), "Failed to create test GGUF file for reader meta test");

    GGUFReader reader(test_gguf_path);
    TEST_ASSERT(reader.get_metadata_str("training.dataset.name") == "dummy_input.txt", "Incorrect dataset name");
    TEST_ASSERT(reader.get_metadata_u64("training.sequence.count") == 2ULL, "Incorrect sequence count");
    // The tokenizer model name might vary, so just check it's not empty/default if model was loaded
    if (g_llama_model) {
        TEST_ASSERT(reader.get_metadata_str("training.tokenizer.gguf.model", "default") != "default", "Tokenizer model name should not be default");
    }
    fs::remove(test_gguf_path);
    return true;
}

bool TestGGUFReader_GetTensorCount() {
    printf("  TestGGUFReader_GetTensorCount\n");
    std::string test_gguf_path = "test_output_reader_count.gguf";
    TEST_ASSERT(CreateTestGGUFFile(test_gguf_path, g_llama_model), "Failed to create test GGUF file for reader count test");

    GGUFReader reader(test_gguf_path);
    TEST_ASSERT(reader.get_tensor_count() == 2, "Incorrect tensor count");
    fs::remove(test_gguf_path);
    return true;
}

bool TestGGUFReader_GetTensorNameAndTypeAndSize() {
    printf("  TestGGUFReader_GetTensorNameAndTypeAndSize\n");
    std::string test_gguf_path = "test_output_reader_tensor_info.gguf";
    TEST_ASSERT(CreateTestGGUFFile(test_gguf_path, g_llama_model), "Failed to create test GGUF file for reader tensor info test");

    GGUFReader reader(test_gguf_path);
    TEST_ASSERT(reader.get_tensor_name(0) == "training.tensor.0", "Incorrect tensor name for index 0");
    TEST_ASSERT(reader.get_tensor_type(0) == GGML_TYPE_I32, "Incorrect tensor type for index 0");
    TEST_ASSERT(reader.get_tensor_size(0) == 5 * sizeof(llama_token), "Incorrect tensor size for index 0");

    TEST_ASSERT(reader.get_tensor_name(1) == "training.tensor.1", "Incorrect tensor name for index 1");
    TEST_ASSERT(reader.get_tensor_type(1) == GGML_TYPE_I32, "Incorrect tensor type for index 1");
    TEST_ASSERT(reader.get_tensor_size(1) == 3 * sizeof(llama_token), "Incorrect tensor size for index 1");
    fs::remove(test_gguf_path);
    return true;
}

bool TestGGUFReader_ReadTensorData() {
    printf("  TestGGUFReader_ReadTensorData\n");
    std::string test_gguf_path = "test_output_reader_data.gguf";
    TEST_ASSERT(CreateTestGGUFFile(test_gguf_path, g_llama_model), "Failed to create test GGUF file for reader data test");

    GGUFReader reader(test_gguf_path);
    std::vector<llama_token> tokens;

    // Read first sequence
    TEST_ASSERT(reader.read_tensor_data(0, tokens), "Failed to read tensor data for index 0");
    TEST_ASSERT(tokens.size() == 5, "Incorrect token count for index 0");
    TEST_ASSERT(tokens[0] == 1, "Incorrect token value at index 0, pos 0");
    TEST_ASSERT(tokens[4] == 5, "Incorrect token value at index 0, pos 4");

    // Read second sequence
    TEST_ASSERT(reader.read_tensor_data(1, tokens), "Failed to read tensor data for index 1");
    TEST_ASSERT(tokens.size() == 3, "Incorrect token count for index 1");
    TEST_ASSERT(tokens[0] == 10, "Incorrect token value at index 1, pos 0");
    TEST_ASSERT(tokens[2] == 30, "Incorrect token value at index 1, pos 2");
    fs::remove(test_gguf_path);
    return true;
}

bool TestGGUFReader_ReadTensorDataInvalidIndex() {
    printf("  TestGGUFReader_ReadTensorDataInvalidIndex\n");
    std::string test_gguf_path = "test_output_reader_invalid_idx.gguf";
    TEST_ASSERT(CreateTestGGUFFile(test_gguf_path, g_llama_model), "Failed to create test GGUF file for reader invalid index test");

    GGUFReader reader(test_gguf_path);
    std::vector<llama_token> tokens;
    TEST_ASSERT(!reader.read_tensor_data(99, tokens), "Reading invalid index should fail");
    fs::remove(test_gguf_path);
    return true;
}

// =============================================================================
// Tests for TextDataReader
// =============================================================================

// Helper to set up TextDataReader test files
struct TextDataReaderTestFixture {
    std::string test_text_file = "test_input.txt";
    std::string test_pretokenized_file = "test_pretokenized.txt";
    llama_model* model_for_reader_test = nullptr;

    TextDataReaderTestFixture(llama_model* model) : model_for_reader_test(model) {
        // Create test text file
        std::ofstream ofs(test_text_file);
        ofs << "Hello world\n";
        ofs << "This is a test line.\n";
        ofs << "\n"; // Empty line
        ofs << "Another line";
        ofs.close();

        // Create test pre-tokenized file
        std::ofstream ofs_pretokenized(test_pretokenized_file);
        ofs_pretokenized << "101 200 300 102\n";
        ofs_pretokenized << "500 600\n";
        ofs_pretokenized << "\n"; // Empty line
        ofs_pretokenized << "700";
        ofs_pretokenized.close();
    }

    ~TextDataReaderTestFixture() {
        fs::remove(test_text_file);
        fs::remove(test_pretokenized_file);
    }
};

bool TestTextDataReader_OpenFile() {
    printf("  TestTextDataReader_OpenFile\n");
    TextDataReaderTestFixture fixture(g_llama_model);
    TextDatasetReader reader(fixture.model_for_reader_test, 128, false);
    TEST_ASSERT(reader.open(fixture.test_text_file), "Failed to open valid text file");
    reader.close();
    TEST_ASSERT(!reader.open("non_existent.txt"), "Opened non-existent file unexpectedly");
    return true;
}

bool TestTextDataReader_ReadNextSequenceTextMode() {
    printf("  TestTextDataReader_ReadNextSequenceTextMode\n");
    if (g_llama_model == nullptr) {
        printf("    Skipping: Llama model not loaded.\n");
        return true; // Skip test gracefully
    }

    TextDataReaderTestFixture fixture(g_llama_model);
    TextDatasetReader reader(fixture.model_for_reader_test, 128, false);
    TEST_ASSERT(reader.open(fixture.test_text_file), "Failed to open text file for read test");

    std::vector<llama_token> tokens;

    // Read "Hello world"
    TEST_ASSERT(reader.read_next_sequence(tokens), "Failed to read first sequence");
    TEST_ASSERT(!tokens.empty(), "First sequence should not be empty");

    // Read "This is a test line."
    TEST_ASSERT(reader.read_next_sequence(tokens), "Failed to read second sequence");
    TEST_ASSERT(!tokens.empty(), "Second sequence should not be empty");

    // Read empty line
    TEST_ASSERT(reader.read_next_sequence(tokens), "Failed to read empty line");
    TEST_ASSERT(tokens.empty(), "Empty line should result in 0 tokens");

    // Read "Another line"
    TEST_ASSERT(reader.read_next_sequence(tokens), "Failed to read third sequence");
    TEST_ASSERT(!tokens.empty(), "Third sequence should not be empty");

    // End of file
    TEST_ASSERT(!reader.read_next_sequence(tokens), "Should be end of file");
    reader.close();
    return true;
}

bool TestTextDataReader_ReadNextSequencePreTokenizedMode() {
    printf("  TestTextDataReader_ReadNextSequencePreTokenizedMode\n");
    TextDataReaderTestFixture fixture(g_llama_model);
    TextDatasetReader reader(fixture.model_for_reader_test, 128, true);
    TEST_ASSERT(reader.open(fixture.test_pretokenized_file), "Failed to open pre-tokenized file for read test");

    std::vector<llama_token> tokens;

    // Read "101 200 300 102"
    TEST_ASSERT(reader.read_next_sequence(tokens), "Failed to read first pre-tokenized sequence");
    TEST_ASSERT(tokens.size() == 4, "Incorrect token count for first pre-tokenized sequence");
    TEST_ASSERT(tokens[0] == 101, "Incorrect token value for first pre-tokenized sequence");
    TEST_ASSERT(tokens[1] == 200, "Incorrect token value for first pre-tokenized sequence");

    // Read "500 600"
    TEST_ASSERT(reader.read_next_sequence(tokens), "Failed to read second pre-tokenized sequence");
    TEST_ASSERT(tokens.size() == 2, "Incorrect token count for second pre-tokenized sequence");
    TEST_ASSERT(tokens[0] == 500, "Incorrect token value for second pre-tokenized sequence");

    // Read empty line
    TEST_ASSERT(reader.read_next_sequence(tokens), "Failed to read empty pre-tokenized line");
    TEST_ASSERT(tokens.empty(), "Empty pre-tokenized line should result in 0 tokens");

    // Read "700"
    TEST_ASSERT(reader.read_next_sequence(tokens), "Failed to read third pre-tokenized sequence");
    TEST_ASSERT(tokens.size() == 1, "Incorrect token count for third pre-tokenized sequence");
    TEST_ASSERT(tokens[0] == 700, "Incorrect token value for third pre-tokenized sequence");

    // End of file
    TEST_ASSERT(!reader.read_next_sequence(tokens), "Should be end of pre-tokenized file");
    reader.close();
    return true;
}

bool TestTextDataReader_ResetFunctionality() {
    printf("  TestTextDataReader_ResetFunctionality\n");
    TextDataReaderTestFixture fixture(g_llama_model);
    TextDatasetReader reader(fixture.model_for_reader_test, 128, false);
    TEST_ASSERT(reader.open(fixture.test_text_file), "Failed to open text file for reset test");

    std::vector<llama_token> tokens;
    reader.read_next_sequence(tokens); // Read one line
    reader.read_next_sequence(tokens); // Read another line

    TEST_ASSERT(reader.reset(), "Failed to reset reader"); // Reset to beginning

    // Should read the first line again
    TEST_ASSERT(reader.read_next_sequence(tokens), "Failed to read first sequence after reset");
    // (Add specific token check if you know the expected tokens for "Hello world")
    reader.close();
    return true;
}

bool TestTextDataReader_GetTotalSequences() {
    printf("  TestTextDataReader_GetTotalSequences\n");
    TextDataReaderTestFixture fixture(g_llama_model);

    TextDatasetReader reader_text(fixture.model_for_reader_test, 128, false);
    TEST_ASSERT(reader_text.open(fixture.test_text_file), "Failed to open text file for total sequences test");
    TEST_ASSERT(reader_text.get_total_sequences() == 4, "Incorrect total sequence count for text file"); // 4 lines in test_input.txt
    reader_text.close();

    TextDatasetReader reader_pretokenized(fixture.model_for_reader_test, 128, true);
    TEST_ASSERT(reader_pretokenized.open(fixture.test_pretokenized_file), "Failed to open pre-tokenized file for total sequences test");
    TEST_ASSERT(reader_pretokenized.get_total_sequences() == 4, "Incorrect total sequence count for pre-tokenized file"); // 4 lines in test_pretokenized.txt
    reader_pretokenized.close();
    return true;
}

// =============================================================================
// Tests for GGUFConverter (integration)
// =============================================================================

// Helper to set up GGUFConverter test files
struct GGUFConverterTestFixture {
    std::string input_text_file = "converter_input.txt";
    std::string output_gguf_file = "converter_output.gguf";
    llama_model* model_for_converter_test = nullptr;

    GGUFConverterTestFixture(llama_model* model) : model_for_converter_test(model) {
        // Create test text file
        std::ofstream ofs(input_text_file);
        ofs << "The quick brown fox jumps over the lazy dog.\n";
        ofs << "Hello, GGUF conversion!\n";
        ofs.close();
    }

    ~GGUFConverterTestFixture() {
        fs::remove(input_text_file);
        fs::remove(output_gguf_file);
    }
};

bool TestGGUFConverter_ConvertTextFileSuccess() {
    printf("  TestGGUFConverter_ConvertTextFileSuccess\n");
    if (g_llama_model == nullptr) {
        printf("    Skipping: Llama model not loaded.\n");
        return true; // Skip test gracefully
    }

    GGUFConverterTestFixture fixture(g_llama_model);

    ConvertParams params;
    params.model = fixture.model_for_converter_test;
    params.input_path = fixture.input_text_file;
    params.output_path = fixture.output_gguf_file;
    params.max_seq_len = 128;
    params.pre_tokenized = false;
    params.input_type = "text";
    params.parquet_text_column = "text"; // Not used for text, but for completeness
    params.parquet_tokens_column = "tokens"; // Not used for text, but for completeness

    GGUFConverter converter;
    TEST_ASSERT(converter.convert(params), "GGUF conversion failed");

    // Verify file was created
    TEST_ASSERT(fs::exists(fixture.output_gguf_file), "Output GGUF file was not created");

    // Verify GGUF file content using GGUFReader
    GGUFReader reader(fixture.output_gguf_file);
    TEST_ASSERT(reader.is_initialized(), "GGUFReader failed to initialize for verification");
    TEST_ASSERT(reader.get_metadata_u64("training.sequence.count") == 2ULL, "Incorrect sequence count in GGUF metadata");
    TEST_ASSERT(reader.get_tensor_count() == 2, "Incorrect tensor count in GGUF file");

    std::vector<llama_token> tokens;
    TEST_ASSERT(reader.read_tensor_data(0, tokens), "Failed to read first tensor data");
    TEST_ASSERT(!tokens.empty(), "First sequence should not be empty");

    TEST_ASSERT(reader.read_tensor_data(1, tokens), "Failed to read second tensor data");
    TEST_ASSERT(!tokens.empty(), "Second sequence should not be empty");
    return true;
}

// =============================================================================
// Main function to run all tests
// =============================================================================

int main(int argc, char **argv) {
    printf("Running dataset-to-gguf tests...\n\n");

    // Global setup for llama.cpp backend
    if (!SetUpLlamaBackend()) {
        printf("Global setup failed. Exiting tests.\n");
        return 1;
    }

    int failed_tests = 0;

    // Run GGUFFile tests
    printf("--- GGUFFile Tests ---\n");
    if (!TestGGUFFile_DefaultConstructorInitializesContext()) failed_tests++;
    if (!TestGGUFFile_ConstructorFromFileThrowsOnError()) failed_tests++;
    if (!TestGGUFFile_SetAndGetMetadataString()) failed_tests++;
    if (!TestGGUFFile_SetAndGetMetadataU64()) failed_tests++;
    if (!TestGGUFFile_SetAndGetMetadataStringArray()) failed_tests++;
    printf("\n");

    // Run GGUFReader tests
    printf("--- GGUFReader Tests ---\n");
    if (!TestGGUFReader_ConstructorInitializesFromFile()) failed_tests++;
    if (!TestGGUFReader_GetMetadata()) failed_tests++;
    if (!TestGGUFReader_GetTensorCount()) failed_tests++;
    if (!TestGGUFReader_GetTensorNameAndTypeAndSize()) failed_tests++;
    if (!TestGGUFReader_ReadTensorData()) failed_tests++;
    if (!TestGGUFReader_ReadTensorDataInvalidIndex()) failed_tests++;
    printf("\n");

    // Run TextDataReader tests
    printf("--- TextDataReader Tests ---\n");
    if (!TestTextDataReader_OpenFile()) failed_tests++;
    if (!TestTextDataReader_ReadNextSequenceTextMode()) failed_tests++;
    if (!TestTextDataReader_ReadNextSequencePreTokenizedMode()) failed_tests++;
    if (!TestTextDataReader_ResetFunctionality()) failed_tests++;
    if (!TestTextDataReader_GetTotalSequences()) failed_tests++;
    printf("\n");

    // Run GGUFConverter integration tests
    printf("--- GGUFConverter Tests ---\n");
    if (!TestGGUFConverter_ConvertTextFileSuccess()) failed_tests++;
    printf("\n");

    // Add ParquetDataReader tests here when you have test files and logic
    // printf("--- ParquetDataReader Tests ---\n");
    // if (!TestParquetDataReader_OpenFile()) failed_tests++;
    // ...

    // Global teardown for llama.cpp backend
    TearDownLlamaBackend();

    if (failed_tests == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("%d tests failed.\n", failed_tests);
        return 1;
    }
}
