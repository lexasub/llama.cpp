#ifndef TEST_CORE_FUNCTIONALITY_H
#define TEST_CORE_FUNCTIONALITY_H

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "common/log.h"
#include "llama-dataset.h"
#include "llama-impl.h"

#ifdef LLAMA_PARQUET
#include "llama-dataset-parquet.h"
#endif

// =============================================================================
// SHARED TEST CONSTANTS
// =============================================================================

// Test data file paths
#define TEST_DATA_SMALL_GGUF "test_data/small_dataset.gguf"
#define TEST_DATA_PARQUET "test_data/parquet_dataset.parquet"
#define TEST_DATA_TEXT "test_data/text_dataset.txt"
#define TEST_DATA_CORRUPTED_GGUF "test_data/corrupted_dataset.gguf"
#define TEST_DATA_CORRUPTED_PARQUET "test_data/corrupted_dataset.parquet"
#define TEST_DATA_LARGE_TEXT "test_data/large_text_dataset.txt"
#define TEST_DATA_LARGE_PARQUET "test_data/large_parquet_dataset.parquet"
#define TEST_DATA_BASIC_TEXT "test_data/basic_text_dataset.parquet"
#define TEST_DATA_MIXED_CONTENT "test_data/mixed_content_dataset.parquet"
#define TEST_DATA_MULTILINGUAL "test_data/multilingual_dataset.parquet"
#define TEST_DATA_PRETOKENIZED "test_data/pretokenized_dataset.parquet"
#define TEST_DATA_SPECIAL_CHARS "test_data/special_chars_dataset.parquet"

// Test output file paths
#define TEST_OUTPUT_GGUF "test_gguf_output.gguf"
#define TEST_OUTPUT_INTEGRATION "test_integration_output.gguf"
#define TEST_OUTPUT_RELOADED "test_integration_reloaded.gguf"
#define TEST_OUTPUT_TEXT "test_text_data.txt"
#define TEST_OUTPUT_PARQUET "test_parquet_output.parquet"
#define TEST_OUTPUT_STREAMING "test_streaming_output.gguf"

// Test limits and thresholds
#define TEST_MAX_DISPLAY_TOKENS 5
#define TEST_RANDOM_ACCESS_COUNT 10
#define TEST_MEMORY_TOLERANCE_KB 1024
#define TEST_PERFORMANCE_THRESHOLD_MS 100.0
#define TEST_MAX_SEQUENCE_LENGTH 1024
#define TEST_MIN_SEQUENCE_LENGTH 1

// Test error codes
#define TEST_ERROR_NONE 0
#define TEST_ERROR_NULL_POINTER 1
#define TEST_ERROR_FILE_NOT_FOUND 2
#define TEST_ERROR_INVALID_FORMAT 3
#define TEST_ERROR_MEMORY_ALLOCATION 4
#define TEST_ERROR_VALIDATION_FAILED 5

// =============================================================================
// SHARED TEST MACROS
// =============================================================================

// Assertion macros with logging
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            LLAMA_LOG_ERROR("TEST ASSERTION FAILED: %s\n", message); \
            assert(false && message); \
        } \
    } while(0)

#define TEST_ASSERT_NOT_NULL(ptr, name) \
    TEST_ASSERT((ptr) != nullptr, "Expected " name " to be non-null")

#define TEST_ASSERT_NULL(ptr, name) \
    TEST_ASSERT((ptr) == nullptr, "Expected " name " to be null")

#define TEST_ASSERT_EQUAL(expected, actual, name) \
    TEST_ASSERT((expected) == (actual), "Expected " name " values to be equal")

#define TEST_ASSERT_NOT_EQUAL(expected, actual, name) \
    TEST_ASSERT((expected) != (actual), "Expected " name " values to be different")

#define TEST_ASSERT_GREATER(actual, threshold, name) \
    TEST_ASSERT((actual) > (threshold), "Expected " name " to be greater than threshold")

#define TEST_ASSERT_LESS(actual, threshold, name) \
    TEST_ASSERT((actual) < (threshold), "Expected " name " to be less than threshold")

#define TEST_ASSERT_NO_ERROR() \
    TEST_ASSERT(!llama_dataset_has_error(), "Expected no dataset error")

#define TEST_ASSERT_HAS_ERROR() \
    TEST_ASSERT(llama_dataset_has_error(), "Expected dataset error")

#define TEST_ASSERT_ERROR_CLEARED() \
    do { \
        llama_dataset_clear_error(); \
        TEST_ASSERT_NO_ERROR(); \
    } while(0)

// Test section logging macros
#define TEST_LOG_SECTION(name) \
    LLAMA_LOG_INFO("\n=== %s ===\n", name)

#define TEST_LOG_SUBSECTION(name) \
    LLAMA_LOG_INFO("\n--- %s ---\n", name)

#define TEST_LOG_SUCCESS(name) \
    LLAMA_LOG_INFO("✓ %s\n", name)

#define TEST_LOG_FAILURE(name) \
    LLAMA_LOG_ERROR("✗ %s\n", name)

#define TEST_LOG_INFO(format, ...) \
    LLAMA_LOG_INFO("  " format, ##__VA_ARGS__)

#define TEST_LOG_WARNING(format, ...) \
    LLAMA_LOG_WARN("  WARNING: " format, ##__VA_ARGS__)

#define TEST_LOG_ERROR(format, ...) \
    LLAMA_LOG_ERROR("  ERROR: " format, ##__VA_ARGS__)

// Performance testing macros
#define TEST_PERFORMANCE_START(timer) \
    TestTimer timer; \
    timer.start()

#define TEST_PERFORMANCE_END(timer, operation, threshold_ms) \
    do { \
        double elapsed = timer.elapsed_ms(); \
        timer.log_elapsed(operation); \
        if (elapsed > threshold_ms) { \
            TEST_LOG_WARNING("Performance threshold exceeded: %.2f ms > %.2f ms\n", elapsed, threshold_ms); \
        } \
    } while(0)

// Memory testing macros
#define TEST_MEMORY_START(tracker) \
    TestMemoryTracker tracker

#define TEST_MEMORY_CHECK(tracker, stage, threshold_kb) \
    do { \
        tracker.log_usage(stage); \
        size_t delta_kb = tracker.get_delta_kb(); \
        if (delta_kb > threshold_kb) { \
            TEST_LOG_WARNING("Memory usage threshold exceeded: %zu KB > %zu KB\n", delta_kb, threshold_kb); \
        } \
    } while(0)

// =============================================================================
// SHARED TEST DATA STRUCTURES
// =============================================================================

// Test sequence data for creating known datasets
struct test_sequence_data {
    std::vector<int32_t> tokens;
    int32_t length;
    
    test_sequence_data() : length(0) {}
    test_sequence_data(const std::vector<int32_t>& t) : tokens(t), length(static_cast<int32_t>(t.size())) {}
    test_sequence_data(const std::vector<int32_t>& t, int32_t len) : tokens(t), length(len) {}
};

// Test dataset metadata
struct test_dataset_metadata {
    uint64_t sequence_count;
    int32_t max_sequence_length;
    int32_t min_sequence_length;
    const char* source_format;
    bool has_metadata;
    bool supports_streaming;
    size_t estimated_memory_usage;
    
    test_dataset_metadata() : 
        sequence_count(0), max_sequence_length(0), min_sequence_length(0),
        source_format(nullptr), has_metadata(false), supports_streaming(false),
        estimated_memory_usage(0) {}
};

// Test configuration for different test scenarios
struct test_config {
    bool enable_streaming;
    bool enable_validation;
    bool enable_performance_tests;
    bool enable_memory_tests;
    const char* test_data_directory;
    double performance_threshold_ms;
    size_t memory_threshold_kb;
    
    test_config() :
        enable_streaming(false), enable_validation(true),
        enable_performance_tests(false), enable_memory_tests(false),
        test_data_directory("test_data"), performance_threshold_ms(TEST_PERFORMANCE_THRESHOLD_MS),
        memory_threshold_kb(TEST_MEMORY_TOLERANCE_KB) {}
};

// Test result structure for aggregating test outcomes
struct test_result {
    int tests_run;
    int tests_passed;
    int tests_failed;
    int tests_skipped;
    double total_time_ms;
    size_t peak_memory_kb;
    std::vector<std::string> failure_messages;
    
    test_result() : 
        tests_run(0), tests_passed(0), tests_failed(0), tests_skipped(0),
        total_time_ms(0.0), peak_memory_kb(0) {}
        
    double success_rate() const {
        return tests_run > 0 ? (double)tests_passed / tests_run * 100.0 : 0.0;
    }
    
    void log_summary() const {
        LLAMA_LOG_INFO("\n=== Test Summary ===\n");
        LLAMA_LOG_INFO("Tests run: %d\n", tests_run);
        LLAMA_LOG_INFO("Tests passed: %d\n", tests_passed);
        LLAMA_LOG_INFO("Tests failed: %d\n", tests_failed);
        LLAMA_LOG_INFO("Tests skipped: %d\n", tests_skipped);
        LLAMA_LOG_INFO("Success rate: %.1f%%\n", success_rate());
        LLAMA_LOG_INFO("Total time: %.2f ms\n", total_time_ms);
        LLAMA_LOG_INFO("Peak memory: %zu KB\n", peak_memory_kb);
        
        if (!failure_messages.empty()) {
            LLAMA_LOG_INFO("\nFailure details:\n");
            for (const auto& msg : failure_messages) {
                LLAMA_LOG_ERROR("  - %s\n", msg.c_str());
            }
        }
    }
};

// =============================================================================
// SHARED TEST UTILITY FUNCTIONS
// =============================================================================

/**
 * Compare two datasets for exact equality
 * @param ds1 First dataset to compare
 * @param ds2 Second dataset to compare
 * @return true if datasets are identical, false otherwise
 */
bool compare_datasets_exact(struct llama_dataset* ds1, struct llama_dataset* ds2);

/**
 * Compare two datasets with tolerance for minor differences
 * @param ds1 First dataset to compare
 * @param ds2 Second dataset to compare
 * @param tolerance Maximum allowed difference per token
 * @return true if datasets are equivalent within tolerance
 */
bool compare_datasets_with_tolerance(struct llama_dataset* ds1, struct llama_dataset* ds2, int32_t tolerance = 0);

/**
 * Create a simple text dataset file for testing
 * @param path Output file path
 * @param lines Vector of text lines to write
 * @return true if file was created successfully
 */
bool create_test_text_file(const char* path, const std::vector<std::string>& lines);

/**
 * Create a test GGUF file with known sequences
 * @param path Output file path
 * @param sequences Vector of token sequences to include
 * @return true if file was created successfully
 */
bool create_test_gguf_file(const char* path, const std::vector<test_sequence_data>& sequences);

/**
 * Clean up test output files
 * @param files Vector of file paths to remove
 */
void cleanup_test_files(const std::vector<const char*>& files);

/**
 * Clean up all test output files using standard patterns
 */
void cleanup_all_test_files();

/**
 * Validate basic dataset properties
 * @param dataset Dataset to validate
 * @param expected_min_sequences Minimum expected sequence count
 * @return true if dataset passes basic validation
 */
bool validate_basic_dataset_properties(struct llama_dataset* dataset, uint64_t expected_min_sequences = 1);

/**
 * Validate dataset metadata consistency
 * @param dataset Dataset to validate
 * @param expected_metadata Expected metadata values
 * @return true if metadata is consistent
 */
bool validate_dataset_metadata(struct llama_dataset* dataset, const test_dataset_metadata& expected_metadata);

/**
 * Display dataset summary information
 * @param dataset Dataset to summarize
 * @param name Name/description of the dataset
 */
void display_dataset_summary(struct llama_dataset* dataset, const char* name);

/**
 * Display detailed dataset information including all sequences
 * @param dataset Dataset to display
 * @param name Name/description of the dataset
 * @param max_sequences Maximum number of sequences to display
 */
void display_dataset_detailed(struct llama_dataset* dataset, const char* name, int max_sequences = 5);

/**
 * Test sequence access with bounds checking
 * @param dataset Dataset to test
 * @param sequence_index Index of sequence to access
 * @return true if access was successful and data is valid
 */
bool test_sequence_access(struct llama_dataset* dataset, uint64_t sequence_index);

/**
 * Test all sequences in a dataset for validity
 * @param dataset Dataset to test
 * @return true if all sequences are valid
 */
bool test_all_sequences_valid(struct llama_dataset* dataset);

/**
 * Test out-of-bounds access handling
 * @param dataset Dataset to test
 * @return true if out-of-bounds access is handled correctly
 */
bool test_out_of_bounds_access(struct llama_dataset* dataset);

/**
 * Test null dataset handling for all API functions
 * @return true if null handling works correctly
 */
bool test_null_dataset_handling();

/**
 * Test dataset error handling and recovery
 * @param dataset Dataset to test (may be null)
 * @return true if error handling works correctly
 */
bool test_error_handling_comprehensive(struct llama_dataset* dataset);

/**
 * Verify file exists and is readable
 * @param path File path to check
 * @return true if file exists and is readable
 */
bool verify_test_file_exists(const char* path);

/**
 * Get file size in bytes
 * @param path File path to check
 * @return File size in bytes, or 0 if file doesn't exist
 */
size_t get_file_size(const char* path);

/**
 * Check if dataset format is supported in current build
 * @param format Dataset format to check
 * @return true if format is supported
 */
bool is_format_supported(enum dataset_type format);

// =============================================================================
// SHARED TEST UTILITY CLASSES
// =============================================================================

/**
 * Performance timer for measuring test execution times
 */
class TestTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    bool is_running;

public:
    TestTimer();
    void start();
    void stop();
    void reset();
    double elapsed_ms() const;
    double elapsed_seconds() const;
    void log_elapsed(const char* operation_name) const;
    bool running() const { return is_running; }
};

/**
 * Memory usage tracker for testing memory consumption
 */
class TestMemoryTracker {
private:
    size_t initial_memory;
    size_t peak_memory;
    bool tracking_enabled;
    size_t get_memory_usage() const;

public:
    TestMemoryTracker();
    void start_tracking();
    void stop_tracking();
    void reset();
    size_t get_current_usage() const;
    size_t get_initial_usage() const { return initial_memory; }
    size_t get_peak_usage() const { return peak_memory; }
    size_t get_delta() const;
    size_t get_delta_kb() const;
    void log_usage(const char* stage_name) const;
    void log_peak_usage() const;
    bool is_tracking() const { return tracking_enabled; }
};

/**
 * Test data creator for generating test files with known content
 */
class TestDataCreator {
public:
    // Create a small GGUF test file with known sequences
    static bool create_small_gguf(const char* path);
    
    // Create equivalent Parquet file with same content
    static bool create_equivalent_parquet(const char* path);
    
    // Create equivalent text file with same content
    static bool create_equivalent_text(const char* path);
    
    // Create test sequences with known patterns
    static std::vector<test_sequence_data> create_test_sequences();
    
    // Create test sequences with specific characteristics
    static std::vector<test_sequence_data> create_test_sequences_with_length(int32_t min_len, int32_t max_len, size_t count);
    
    // Create large test dataset for performance testing
    static std::vector<test_sequence_data> create_large_test_sequences(size_t sequence_count, int32_t avg_length);
    
    // Create GGUF file from token sequences
    static bool create_gguf_from_sequences(const char* path, const std::vector<std::vector<int32_t>>& sequences);

private:
    static std::vector<int32_t> generate_random_sequence(int32_t length, int32_t min_token = 1, int32_t max_token = 1000);
};

/**
 * RAII helper for automatic dataset cleanup
 */
class TestDatasetGuard {
private:
    struct llama_dataset* dataset;

public:
    explicit TestDatasetGuard(struct llama_dataset* ds);
    ~TestDatasetGuard();
    
    // Non-copyable
    TestDatasetGuard(const TestDatasetGuard&) = delete;
    TestDatasetGuard& operator=(const TestDatasetGuard&) = delete;
    
    // Movable
    TestDatasetGuard(TestDatasetGuard&& other) noexcept;
    TestDatasetGuard& operator=(TestDatasetGuard&& other) noexcept;
    
    struct llama_dataset* get() const;
    struct llama_dataset* release();
    bool is_valid() const;
    void reset(struct llama_dataset* new_dataset = nullptr);
};

/**
 * Test parameter builder for creating common_params configurations
 */
class TestParamsBuilder {
private:
    common_params params;

public:
    TestParamsBuilder();
    TestParamsBuilder& with_file(const char* file_path);
    TestParamsBuilder& with_streaming(bool enable_streaming = true);
    TestParamsBuilder& with_text_column(const char* column_name);
    TestParamsBuilder& with_token_column(const char* column_name);
    TestParamsBuilder& with_validation(bool enable_validation = true);
    common_params build() const;
    void reset();
};

/**
 * Test suite runner for organizing and executing multiple tests
 */
class TestSuiteRunner {
private:
    std::string suite_name;
    test_result result;
    test_config config;
    TestTimer suite_timer;
    TestMemoryTracker suite_memory;

public:
    explicit TestSuiteRunner(const std::string& name);
    explicit TestSuiteRunner(const std::string& name, const test_config& cfg);
    
    void run_test(const std::string& test_name, std::function<bool()> test_func);
    void skip_test(const std::string& test_name, const std::string& reason);
    void set_config(const test_config& cfg);
    const test_result& get_result() const { return result; }
    void log_final_summary() const;
    
    // Convenience methods for common test patterns
    bool run_dataset_test(const std::string& test_name, const char* file_path, 
                         enum dataset_type format, std::function<bool(struct llama_dataset*)> test_func);
    bool run_comparison_test(const std::string& test_name, struct llama_dataset* ds1, 
                           struct llama_dataset* ds2, bool exact_match = true);
    bool run_performance_test(const std::string& test_name, std::function<bool()> test_func, 
                            double threshold_ms = TEST_PERFORMANCE_THRESHOLD_MS);
};

// =============================================================================
// SHARED TEST HELPER FUNCTIONS
// =============================================================================

/**
 * Load dataset with error handling and logging
 * @param file_path Path to dataset file
 * @param format Dataset format type
 * @param streaming Enable streaming mode
 * @return Dataset pointer or nullptr on failure
 */
struct llama_dataset* load_test_dataset(const char* file_path, 
                                       enum dataset_type format = DATASET_GGUF,
                                       bool streaming = false);

/**
 * Load dataset with custom parameters
 * @param params Common parameters for dataset loading
 * @param format Dataset format type
 * @return Dataset pointer or nullptr on failure
 */
struct llama_dataset* load_test_dataset_with_params(const common_params& params, enum dataset_type format);

/**
 * Test dataset conversion round-trip
 * @param source_dataset Source dataset to convert
 * @param output_path Output file path
 * @return true if round-trip conversion preserves data
 */
bool test_dataset_conversion_roundtrip(struct llama_dataset* source_dataset, const char* output_path);

/**
 * Test streaming vs non-streaming equivalence
 * @param file_path Path to dataset file
 * @param format Dataset format type
 * @return true if streaming and non-streaming produce identical results
 */
bool test_streaming_equivalence(const char* file_path, enum dataset_type format = DATASET_GGUF);

/**
 * Test error handling for various error conditions
 * @param test_name Name of the error test
 * @param test_function Function that should trigger an error
 * @return true if error was handled correctly
 */
bool test_error_handling(const char* test_name, std::function<struct llama_dataset*()> test_function);

/**
 * Validate metadata access for a dataset
 * @param dataset Dataset to test
 * @return true if metadata access works correctly
 */
bool test_metadata_access(struct llama_dataset* dataset);

/**
 * Test random sequence access performance
 * @param dataset Dataset to test
 * @param access_count Number of random accesses to perform
 * @return Average access time in milliseconds
 */
double test_random_access_performance(struct llama_dataset* dataset, int access_count = TEST_RANDOM_ACCESS_COUNT);

/**
 * Test sequential access performance
 * @param dataset Dataset to test
 * @return Average access time in milliseconds
 */
double test_sequential_access_performance(struct llama_dataset* dataset);

/**
 * Test dataset loading performance
 * @param file_path Path to dataset file
 * @param format Dataset format type
 * @param streaming Enable streaming mode
 * @return Loading time in milliseconds
 */
double test_dataset_loading_performance(const char* file_path, enum dataset_type format, bool streaming = false);

/**
 * Test memory usage during dataset operations
 * @param dataset Dataset to test
 * @param operation_name Name of the operation being tested
 * @param operation Function to execute and measure
 * @return Peak memory usage in KB
 */
size_t test_memory_usage(struct llama_dataset* dataset, const char* operation_name, 
                        std::function<void()> operation);

/**
 * Initialize test environment and create necessary directories
 * @return true if initialization was successful
 */
bool initialize_test_environment();

/**
 * Cleanup test environment and remove temporary files
 */
void cleanup_test_environment();

/**
 * Run comprehensive dataset validation
 * @param dataset Dataset to validate
 * @param expected_metadata Expected metadata (optional)
 * @return true if dataset passes all validation checks
 */
bool run_comprehensive_dataset_validation(struct llama_dataset* dataset, 
                                         const test_dataset_metadata* expected_metadata = nullptr);

#endif // TEST_CORE_FUNCTIONALITY_H