#include "test_core_functionality.h"
#include "../../core/llama-dataset.h"
#include "../../core/llama-dataset-error.h"
#include "../../core/llama-dataset-metadata.h"
#include "../../../../common/common.h"
#include "../../../../include/llama.h"
#include "../../../../src/llama-impl.h"
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <chrono>
#include <algorithm>
#include <cstring>

// =============================================================================
// MISSING DEFINITIONS AND CONSTANTS
// =============================================================================

// Test constants
#define TEST_MAX_DISPLAY_TOKENS 10
#define TEST_OUTPUT_GGUF "test_output.gguf"
#define TEST_OUTPUT_INTEGRATION "test_integration.gguf"
#define TEST_OUTPUT_RELOADED "test_reloaded.gguf"
#define TEST_OUTPUT_TEXT "test_output.txt"
#define TEST_OUTPUT_PARQUET "test_output.parquet"
#define TEST_OUTPUT_STREAMING "test_streaming.gguf"

// Dataset type enum is already defined in llama-dataset.h

// Test data structures
struct test_sequence_data {
    std::vector<int32_t> tokens;
    int32_t length;

    test_sequence_data(const std::vector<int32_t>& t, int32_t l) : tokens(t), length(l) {}
};

struct test_dataset_metadata {
    uint64_t sequence_count;
    const char* source_format;

    test_dataset_metadata() : sequence_count(0), source_format(nullptr) {}
};

struct test_config {
    bool enable_streaming = false;
    bool enable_performance_tests = true;
    double performance_threshold_ms = 1000.0;
};

struct test_result {
    int tests_run = 0;
    int tests_passed = 0;
    int tests_failed = 0;
    int tests_skipped = 0;
    double total_time_ms = 0.0;
    std::vector<std::string> failure_messages;

    void log_summary() const {
        printf("Test Summary: %d run, %d passed, %d failed, %d skipped\n",
               tests_run, tests_passed, tests_failed, tests_skipped);
    }
};

// Test utility classes forward declarations
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
};

class TestMemoryTracker {
private:
    size_t initial_memory;
    size_t peak_memory;
    bool tracking_enabled;

public:
    TestMemoryTracker();
    void start_tracking();
    void stop_tracking();
    void reset();
    size_t get_memory_usage() const;
    size_t get_current_usage() const;
    size_t get_peak_usage() const;
    size_t get_delta() const;
    size_t get_delta_kb() const;
    void log_usage(const char* stage_name) const;
    void log_peak_usage() const;
};

class TestDatasetGuard {
private:
    struct llama_dataset* dataset;

public:
    explicit TestDatasetGuard(struct llama_dataset* ds);
    ~TestDatasetGuard();
    TestDatasetGuard(TestDatasetGuard&& other) noexcept;
    TestDatasetGuard& operator=(TestDatasetGuard&& other) noexcept;
    TestDatasetGuard(const TestDatasetGuard&) = delete;
    TestDatasetGuard& operator=(const TestDatasetGuard&) = delete;

    struct llama_dataset* get() const;
    struct llama_dataset* release();
    bool is_valid() const;
    void reset(struct llama_dataset* new_dataset = nullptr);
};

class TestParamsBuilder {
private:
    common_params params;

public:
    TestParamsBuilder();
    TestParamsBuilder& with_file(const char* file_path);
    TestParamsBuilder& with_streaming(bool enable_streaming);
    TestParamsBuilder& with_text_column(const char* column_name);
    TestParamsBuilder& with_token_column(const char* column_name);
    TestParamsBuilder& with_validation(bool enable_validation);
    common_params build() const;
    void reset();
};

class TestDataCreator {
public:
    static std::vector<test_sequence_data> create_test_sequences();
    static bool create_small_gguf(const char* path);
    static bool create_equivalent_parquet(const char* path);
    static bool create_equivalent_text(const char* path);
    static std::vector<test_sequence_data> create_test_sequences_with_length(int32_t min_len, int32_t max_len, size_t count);
    static std::vector<test_sequence_data> create_large_test_sequences(size_t sequence_count, int32_t avg_length);
    static std::vector<int32_t> generate_random_sequence(int32_t length, int32_t min_token = 1, int32_t max_token = 1000);
    static bool create_gguf_from_sequences(const char* path, const std::vector<std::vector<int32_t>>& sequences);
};

class TestSuiteRunner {
private:
    std::string suite_name;
    test_config config;
    test_result result;
    TestTimer suite_timer;
    TestMemoryTracker suite_memory;

public:
    explicit TestSuiteRunner(const std::string& name);
    TestSuiteRunner(const std::string& name, const test_config& cfg);

    void run_test(const std::string& test_name, std::function<bool()> test_func);
    void skip_test(const std::string& test_name, const std::string& reason);
    void set_config(const test_config& cfg);
    void log_final_summary() const;

    bool run_dataset_test(const std::string& test_name, const char* file_path,
                         enum dataset_type format, std::function<bool(struct llama_dataset*)> test_func);
    bool run_comparison_test(const std::string& test_name, struct llama_dataset* ds1,
                           struct llama_dataset* ds2, bool expect_equal = true);
    bool run_performance_test(const std::string& test_name, std::function<bool()> test_func,
                            double threshold_ms = 1000.0);
};

// Logging macros
#define TEST_LOG_SUCCESS(msg) printf("✓ %s\n", msg)
#define TEST_LOG_FAILURE(msg) printf("✗ %s\n", msg)
#define TEST_LOG_WARNING(msg, ...) printf("⚠ " msg, __VA_ARGS__)
#define TEST_LOG_INFO(msg, ...) printf(msg, __VA_ARGS__)
#define TEST_LOG_ERROR(msg, ...) printf("ERROR: " msg, __VA_ARGS__)
// =============================================================================

bool compare_datasets_exact(struct llama_dataset* ds1, struct llama_dataset* ds2) {
    if (!ds1 || !ds2) {
        LLAMA_LOG_ERROR("One or both datasets are null\n");
        return false;
    }

    // Compare sequence counts
    uint64_t count1 = llama_dataset_n_sequences(ds1);
    uint64_t count2 = llama_dataset_n_sequences(ds2);

    if (count1 != count2) {
        LLAMA_LOG_ERROR("Sequence counts differ: %lu vs %lu\n", count1, count2);
        return false;
    }

    // Compare each sequence
    for (uint64_t i = 0; i < count1; i++) {
        int32_t len1 = llama_dataset_sequence_length(ds1, i);
        int32_t len2 = llama_dataset_sequence_length(ds2, i);

        if (len1 != len2) {
            LLAMA_LOG_ERROR("Sequence %lu lengths differ: %d vs %d\n", i, len1, len2);
            return false;
        }

        const int32_t* seq1 = llama_dataset_sequence(ds1, i);
        const int32_t* seq2 = llama_dataset_sequence(ds2, i);

        if (!seq1 || !seq2) {
            LLAMA_LOG_ERROR("Sequence %lu data is null\n", i);
            return false;
        }

        for (int32_t j = 0; j < len1; j++) {
            if (seq1[j] != seq2[j]) {
                LLAMA_LOG_ERROR("Sequence %lu data differs at position %d: %d vs %d\n", i, j, seq1[j], seq2[j]);
                return false;
            }
        }
    }

    return true;
}

bool create_test_text_file(const char* path, const std::vector<std::string>& lines) {
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    for (const auto& line : lines) {
        file << line << std::endl;
    }

    file.close();
    return true;
}

void cleanup_test_files(const std::vector<const char*>& files) {
    for (const char* file : files) {
        if (file) {
            remove(file);
        }
    }
}

bool validate_basic_dataset_properties(struct llama_dataset* dataset, uint64_t expected_min_sequences) {
    if (!dataset) {
        LLAMA_LOG_ERROR("Dataset is null\n");
        return false;
    }

    uint64_t seq_count = llama_dataset_n_sequences(dataset);
    if (seq_count < expected_min_sequences) {
        LLAMA_LOG_ERROR("Sequence count %lu is less than expected minimum %lu\n", seq_count, expected_min_sequences);
        return false;
    }

    // Validate first sequence if any exist
    if (seq_count > 0) {
        int32_t seq_len = llama_dataset_sequence_length(dataset, 0);
        if (seq_len <= 0) {
            LLAMA_LOG_ERROR("First sequence has invalid length: %d\n", seq_len);
            return false;
        }

        const int32_t* seq_data = llama_dataset_sequence(dataset, 0);
        if (!seq_data) {
            LLAMA_LOG_ERROR("First sequence data is null\n");
            return false;
        }

        struct ggml_tensor* tensor = llama_dataset_sequence_tensor(dataset, 0);
        if (!tensor) {
            LLAMA_LOG_ERROR("First sequence tensor is null\n");
            return false;
        }
    }

    return true;
}

void display_dataset_summary(struct llama_dataset* dataset, const char* name) {
    if (!dataset) {
        LLAMA_LOG_INFO("%s: null dataset\n", name);
        return;
    }

    uint64_t seq_count = llama_dataset_n_sequences(dataset);
    LLAMA_LOG_INFO("%s summary:\n", name);
    LLAMA_LOG_INFO("  Sequence count: %lu\n", seq_count);

    if (seq_count > 0) {
        int32_t first_len = llama_dataset_sequence_length(dataset, 0);
        LLAMA_LOG_INFO("  First sequence length: %d\n", first_len);

        const int32_t* first_data = llama_dataset_sequence(dataset, 0);
        if (first_data && first_len > 0) {
            LLAMA_LOG_INFO("  First few tokens: ");
            for (int i = 0; i < std::min(first_len, TEST_MAX_DISPLAY_TOKENS); i++) {
                LLAMA_LOG_INFO("%d ", first_data[i]);
            }
            LLAMA_LOG_INFO("\n");
        }

        // Display metadata if available
        const char* format = llama_dataset_get_metadata_str(dataset, TRAINING_FORMAT_SOURCE);
        if (format) {
            LLAMA_LOG_INFO("  Source format: %s\n", format);
        }

        int64_t metadata_count = llama_dataset_get_metadata_int(dataset, TRAINING_SEQUENCE_COUNT, -1);
        if (metadata_count != -1) {
            LLAMA_LOG_INFO("  Metadata sequence count: %ld\n", metadata_count);
        }
    }
}

bool test_sequence_access(struct llama_dataset* dataset, uint64_t sequence_index) {
    if (!dataset) {
        return false;
    }

    uint64_t seq_count = llama_dataset_n_sequences(dataset);
    if (sequence_index >= seq_count) {
        LLAMA_LOG_ERROR("Sequence index %lu is out of bounds (count: %lu)\n", sequence_index, seq_count);
        return false;
    }

    int32_t seq_len = llama_dataset_sequence_length(dataset, sequence_index);
    if (seq_len <= 0) {
        LLAMA_LOG_ERROR("Sequence %lu has invalid length: %d\n", sequence_index, seq_len);
        return false;
    }

    const int32_t* seq_data = llama_dataset_sequence(dataset, sequence_index);
    if (!seq_data) {
        LLAMA_LOG_ERROR("Sequence %lu data is null\n", sequence_index);
        return false;
    }

    struct ggml_tensor* tensor = llama_dataset_sequence_tensor(dataset, sequence_index);
    if (!tensor) {
        LLAMA_LOG_ERROR("Sequence %lu tensor is null\n", sequence_index);
        return false;
    }

    return true;
}

bool test_out_of_bounds_access(struct llama_dataset* dataset) {
    if (!dataset) {
        return false;
    }

    uint64_t seq_count = llama_dataset_n_sequences(dataset);
    uint64_t invalid_index = seq_count + 1;

    // Test out-of-bounds sequence length
    int32_t invalid_len = llama_dataset_sequence_length(dataset, invalid_index);
    if (invalid_len != 0) {
        LLAMA_LOG_ERROR("Out-of-bounds sequence length should be 0, got %d\n", invalid_len);
        return false;
    }

    // Test out-of-bounds sequence access
    const int32_t* invalid_seq = llama_dataset_sequence(dataset, invalid_index);
    if (invalid_seq != nullptr) {
        LLAMA_LOG_ERROR("Out-of-bounds sequence access should return null\n");
        return false;
    }

    // Test out-of-bounds tensor access
    struct ggml_tensor* invalid_tensor = llama_dataset_sequence_tensor(dataset, invalid_index);
    if (invalid_tensor != nullptr) {
        LLAMA_LOG_ERROR("Out-of-bounds tensor access should return null\n");
        return false;
    }

    return true;
}

bool test_null_dataset_handling() {
    // Test all API functions with null dataset
    uint64_t seq_count = llama_dataset_n_sequences(nullptr);
    if (seq_count != 0) {
        LLAMA_LOG_ERROR("Null dataset sequence count should be 0, got %lu\n", seq_count);
        return false;
    }

    int32_t seq_len = llama_dataset_sequence_length(nullptr, 0);
    if (seq_len != 0) {
        LLAMA_LOG_ERROR("Null dataset sequence length should be 0, got %d\n", seq_len);
        return false;
    }

    const int32_t* seq_data = llama_dataset_sequence(nullptr, 0);
    if (seq_data != nullptr) {
        LLAMA_LOG_ERROR("Null dataset sequence access should return null\n");
        return false;
    }

    struct ggml_tensor* tensor = llama_dataset_sequence_tensor(nullptr, 0);
    if (tensor != nullptr) {
        LLAMA_LOG_ERROR("Null dataset tensor access should return null\n");
        return false;
    }

    // Test metadata functions with null dataset
    const char* str_meta = llama_dataset_get_metadata_str(nullptr, TRAINING_FORMAT_SOURCE);
    if (str_meta != nullptr) {
        LLAMA_LOG_ERROR("Null dataset string metadata should return null\n");
        return false;
    }

    int64_t int_meta = llama_dataset_get_metadata_int(nullptr, TRAINING_SEQUENCE_COUNT, -1);
    if (int_meta != -1) {
        LLAMA_LOG_ERROR("Null dataset int metadata should return default value\n");
        return false;
    }

    float float_meta = llama_dataset_get_metadata_float(nullptr, "test.float", -1.0f);
    if (float_meta != -1.0f) {
        LLAMA_LOG_ERROR("Null dataset float metadata should return default value\n");
        return false;
    }

    return true;
}

bool compare_datasets_with_tolerance(struct llama_dataset* ds1, struct llama_dataset* ds2, int32_t tolerance) {
    if (!ds1 || !ds2) {
        LLAMA_LOG_ERROR("One or both datasets are null\n");
        return false;
    }

    // Compare sequence counts
    uint64_t count1 = llama_dataset_n_sequences(ds1);
    uint64_t count2 = llama_dataset_n_sequences(ds2);

    if (count1 != count2) {
        LLAMA_LOG_ERROR("Sequence counts differ: %lu vs %lu\n", count1, count2);
        return false;
    }

    // Compare each sequence with tolerance
    for (uint64_t i = 0; i < count1; i++) {
        int32_t len1 = llama_dataset_sequence_length(ds1, i);
        int32_t len2 = llama_dataset_sequence_length(ds2, i);

        if (len1 != len2) {
            LLAMA_LOG_ERROR("Sequence %lu lengths differ: %d vs %d\n", i, len1, len2);
            return false;
        }

        const int32_t* seq1 = llama_dataset_sequence(ds1, i);
        const int32_t* seq2 = llama_dataset_sequence(ds2, i);

        if (!seq1 || !seq2) {
            LLAMA_LOG_ERROR("Sequence %lu data is null\n", i);
            return false;
        }

        for (int32_t j = 0; j < len1; j++) {
            int32_t diff = abs(seq1[j] - seq2[j]);
            if (diff > tolerance) {
                LLAMA_LOG_ERROR("Sequence %lu data differs at position %d: %d vs %d (diff: %d > tolerance: %d)\n",
                               i, j, seq1[j], seq2[j], diff, tolerance);
                return false;
            }
        }
    }

    return true;
}

bool create_test_gguf_file(const char* path, const std::vector<test_sequence_data>& sequences) {
    std::vector<std::vector<int32_t>> token_sequences;
    for (const auto& seq : sequences) {
        token_sequences.push_back(seq.tokens);
    }
    return TestDataCreator::create_gguf_from_sequences(path, token_sequences);
}

void cleanup_all_test_files() {
    std::vector<const char*> files = {
        TEST_OUTPUT_GGUF,
        TEST_OUTPUT_INTEGRATION,
        TEST_OUTPUT_RELOADED,
        TEST_OUTPUT_TEXT,
        TEST_OUTPUT_PARQUET,
        TEST_OUTPUT_STREAMING
    };
    cleanup_test_files(files);
}

bool validate_dataset_metadata(struct llama_dataset* dataset, const test_dataset_metadata& expected_metadata) {
    if (!dataset) {
        LLAMA_LOG_ERROR("Dataset is null\n");
        return false;
    }

    // Check sequence count
    uint64_t actual_count = llama_dataset_n_sequences(dataset);
    if (expected_metadata.sequence_count > 0 && actual_count != expected_metadata.sequence_count) {
        LLAMA_LOG_ERROR("Sequence count mismatch: expected %lu, got %lu\n",
                       expected_metadata.sequence_count, actual_count);
        return false;
    }

    // Check source format if specified
    if (expected_metadata.source_format) {
        const char* actual_format = llama_dataset_get_metadata_str(dataset, TRAINING_FORMAT_SOURCE);
        if (!actual_format || strcmp(actual_format, expected_metadata.source_format) != 0) {
            LLAMA_LOG_ERROR("Source format mismatch: expected %s, got %s\n",
                           expected_metadata.source_format, actual_format ? actual_format : "null");
            return false;
        }
    }

    return true;
}

void display_dataset_detailed(struct llama_dataset* dataset, const char* name, int max_sequences) {
    if (!dataset) {
        LLAMA_LOG_INFO("%s: null dataset\n", name);
        return;
    }

    uint64_t seq_count = llama_dataset_n_sequences(dataset);
    LLAMA_LOG_INFO("%s detailed information:\n", name);
    LLAMA_LOG_INFO("  Total sequences: %lu\n", seq_count);

    int sequences_to_show = std::min((int)seq_count, max_sequences);
    for (int i = 0; i < sequences_to_show; i++) {
        int32_t seq_len = llama_dataset_sequence_length(dataset, i);
        const int32_t* seq_data = llama_dataset_sequence(dataset, i);

        LLAMA_LOG_INFO("  Sequence %d (length %d): ", i, seq_len);
        if (seq_data && seq_len > 0) {
            for (int j = 0; j < std::min(seq_len, TEST_MAX_DISPLAY_TOKENS); j++) {
                LLAMA_LOG_INFO("%d ", seq_data[j]);
            }
            if (seq_len > TEST_MAX_DISPLAY_TOKENS) {
                LLAMA_LOG_INFO("... ");
            }
        }
        LLAMA_LOG_INFO("\n");
    }

    if ((int)seq_count > max_sequences) {
        LLAMA_LOG_INFO("  ... and %lu more sequences\n", seq_count - max_sequences);
    }
}

bool test_all_sequences_valid(struct llama_dataset* dataset) {
    if (!dataset) {
        return false;
    }

    uint64_t seq_count = llama_dataset_n_sequences(dataset);
    for (uint64_t i = 0; i < seq_count; i++) {
        if (!test_sequence_access(dataset, i)) {
            LLAMA_LOG_ERROR("Sequence %lu failed validation\n", i);
            return false;
        }
    }

    return true;
}

bool test_error_handling_comprehensive(struct llama_dataset* dataset) {
    // Test null dataset handling
    if (!test_null_dataset_handling()) {
        return false;
    }

    // Test out-of-bounds access if dataset is valid
    if (dataset && !test_out_of_bounds_access(dataset)) {
        return false;
    }

    // Test error state management
    llama_dataset_clear_error();
    if (llama_dataset_has_error()) {
        LLAMA_LOG_ERROR("Error should be cleared\n");
        return false;
    }

    return true;
}

bool verify_test_file_exists(const char* path) {
    if (!path) {
        return false;
    }

    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

size_t get_file_size(const char* path) {
    if (!path) {
        return 0;
    }

    struct stat st;
    if (stat(path, &st) != 0) {
        return 0;
    }

    return st.st_size;
}

bool is_format_supported(enum dataset_type format) {
    switch (format) {
        case DATASET_GGUF:
            return true;
        case DATASET_PARQUET:
#ifdef LLAMA_PARQUET
            return true;
#else
            return false;
#endif
        case DATASET_TEXT:
            return true;
        default:
            return false;
    }
}

// =============================================================================
// TEST UTILITY CLASS IMPLEMENTATIONS
// =============================================================================

TestTimer::TestTimer() : is_running(false) {
    start();
}

void TestTimer::start() {
    start_time = std::chrono::high_resolution_clock::now();
    is_running = true;
}

void TestTimer::stop() {
    is_running = false;
}

void TestTimer::reset() {
    start_time = std::chrono::high_resolution_clock::now();
    is_running = false;
}

double TestTimer::elapsed_ms() const {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    return duration.count() / 1000.0;
}

double TestTimer::elapsed_seconds() const {
    return elapsed_ms() / 1000.0;
}

void TestTimer::log_elapsed(const char* operation_name) const {
    LLAMA_LOG_INFO("  %s time: %.2f ms\n", operation_name, elapsed_ms());
}

TestMemoryTracker::TestMemoryTracker() : tracking_enabled(false) {
    initial_memory = get_memory_usage();
    peak_memory = initial_memory;
    tracking_enabled = true;
}

void TestMemoryTracker::start_tracking() {
    initial_memory = get_memory_usage();
    peak_memory = initial_memory;
    tracking_enabled = true;
}

void TestMemoryTracker::stop_tracking() {
    tracking_enabled = false;
}

void TestMemoryTracker::reset() {
    initial_memory = get_memory_usage();
    peak_memory = initial_memory;
}

size_t TestMemoryTracker::get_memory_usage() const {
    // Simple memory usage estimation - platform specific
#ifdef __linux__
    FILE* file = fopen("/proc/self/status", "r");
    if (!file) return 0;

    char line[128];
    size_t vm_rss = 0;

    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line, "VmRSS: %zu kB", &vm_rss);
            break;
        }
    }

    fclose(file);
    return vm_rss * 1024; // Convert to bytes
#else
    // Fallback for non-Linux systems
    return 0;
#endif
}

size_t TestMemoryTracker::get_current_usage() const {
    size_t current = get_memory_usage();
    if (tracking_enabled && current > peak_memory) {
        const_cast<TestMemoryTracker*>(this)->peak_memory = current;
    }
    return current;
}

size_t TestMemoryTracker::get_delta() const {
    return get_current_usage() - initial_memory;
}

size_t TestMemoryTracker::get_peak_usage() const {
    return peak_memory;
}

size_t TestMemoryTracker::get_delta_kb() const {
    return get_delta() / 1024;
}

void TestMemoryTracker::log_usage(const char* stage_name) const {
    LLAMA_LOG_INFO("  %s memory usage: %zu KB (delta: %zu KB)\n",
                   stage_name, get_current_usage() / 1024, get_delta_kb());
}

void TestMemoryTracker::log_peak_usage() const {
    LLAMA_LOG_INFO("  Peak memory usage: %zu KB (delta: %zu KB)\n",
                   peak_memory / 1024, (peak_memory - initial_memory) / 1024);
}

TestDatasetGuard::TestDatasetGuard(struct llama_dataset* ds) : dataset(ds) {}

TestDatasetGuard::~TestDatasetGuard() {
    if (dataset) {
        llama_dataset_free(dataset);
    }
}

TestDatasetGuard::TestDatasetGuard(TestDatasetGuard&& other) noexcept : dataset(other.dataset) {
    other.dataset = nullptr;
}

TestDatasetGuard& TestDatasetGuard::operator=(TestDatasetGuard&& other) noexcept {
    if (this != &other) {
        if (dataset) {
            llama_dataset_free(dataset);
        }
        dataset = other.dataset;
        other.dataset = nullptr;
    }
    return *this;
}

struct llama_dataset* TestDatasetGuard::get() const {
    return dataset;
}

struct llama_dataset* TestDatasetGuard::release() {
    struct llama_dataset* result = dataset;
    dataset = nullptr;
    return result;
}

bool TestDatasetGuard::is_valid() const {
    return dataset != nullptr;
}

void TestDatasetGuard::reset(struct llama_dataset* new_dataset) {
    if (dataset) {
        llama_dataset_free(dataset);
    }
    dataset = new_dataset;
}

TestParamsBuilder::TestParamsBuilder() {
    // Initialize with default values
}

TestParamsBuilder& TestParamsBuilder::with_file(const char* file_path) {
    params.in_files.clear();
    params.in_files.push_back(file_path);
    return *this;
}

TestParamsBuilder& TestParamsBuilder::with_streaming(bool enable_streaming) {
    params.dataset_streaming = enable_streaming;
    return *this;
}

TestParamsBuilder& TestParamsBuilder::with_text_column(const char* column_name) {
    params.dataset_column = column_name;
    return *this;
}

TestParamsBuilder& TestParamsBuilder::with_token_column(const char* column_name) {
    params.dataset_column_to = column_name;
    return *this;
}

TestParamsBuilder& TestParamsBuilder::with_validation(bool enable_validation) {
    // Note: validation flag would be added to common_params if it exists
    (void)enable_validation; // Suppress unused parameter warning
    return *this;
}

common_params TestParamsBuilder::build() const {
    return params;
}

void TestParamsBuilder::reset() {
    params = common_params();
}

// =============================================================================
// SHARED TEST HELPER FUNCTION IMPLEMENTATIONS
// =============================================================================

struct llama_dataset* load_test_dataset(const char* file_path, enum dataset_type format, bool streaming) {
    TestParamsBuilder builder;
    common_params params = builder.with_file(file_path).with_streaming(streaming).build();

    struct llama_dataset* dataset = nullptr;

    switch (format) {
        case DATASET_GGUF:
            dataset = llama_dataset_from_gguf(&params);
            break;
        case DATASET_PARQUET:
#ifdef LLAMA_PARQUET
            dataset = llama_dataset_from_parquet(&params);
#else
            LLAMA_LOG_ERROR("Parquet support not compiled in\n");
            return nullptr;
#endif
            break;
        case DATASET_TEXT:
            dataset = llama_dataset_from_txt(&params, nullptr);
            break;
        default:
            LLAMA_LOG_ERROR("Unsupported dataset format: %d\n", format);
            return nullptr;
    }

    if (!dataset) {
        LLAMA_LOG_ERROR("Failed to load dataset from %s: %s\n", file_path, llama_dataset_get_error());
    }

    return dataset;
}

bool test_dataset_conversion_roundtrip(struct llama_dataset* source_dataset, const char* output_path) {
    if (!source_dataset) {
        return false;
    }

    // Convert to GGUF
    llama_dataset_to_gguf(source_dataset, output_path);
    if (llama_dataset_has_error()) {
        LLAMA_LOG_ERROR("Failed to convert dataset: %s\n", llama_dataset_get_error());
        return false;
    }

    // Load converted dataset
    struct llama_dataset* converted = load_test_dataset(output_path, DATASET_GGUF, false);
    if (!converted) {
        return false;
    }

    // Compare datasets
    bool equal = compare_datasets_exact(source_dataset, converted);
    llama_dataset_free(converted);

    // Clean up output file
    remove(output_path);

    return equal;
}

bool test_streaming_equivalence(const char* file_path, enum dataset_type format) {
    // Load in non-streaming mode
    struct llama_dataset* non_streaming = load_test_dataset(file_path, format, false);
    if (!non_streaming) {
        return false;
    }

    // Load in streaming mode
    struct llama_dataset* streaming = load_test_dataset(file_path, format, true);
    if (!streaming) {
        llama_dataset_free(non_streaming);
        return false;
    }

    // Compare datasets
    bool equal = compare_datasets_exact(non_streaming, streaming);

    llama_dataset_free(non_streaming);
    llama_dataset_free(streaming);

    return equal;
}

bool test_error_handling(const char* test_name, std::function<struct llama_dataset*()> test_function) {
    LLAMA_LOG_INFO("Testing %s error handling...\n", test_name);

    // Clear any existing errors
    llama_dataset_clear_error();

    // Execute test function
    struct llama_dataset* result = test_function();

    // Check that error was properly set
    bool has_error = llama_dataset_has_error();
    bool result_is_null = (result == nullptr);

    if (result) {
        llama_dataset_free(result);
    }

    llama_dataset_clear_error();

    if (!result_is_null) {
        LLAMA_LOG_ERROR("Expected null result for %s\n", test_name);
        return false;
    }

    if (!has_error) {
        LLAMA_LOG_ERROR("Expected error to be set for %s\n", test_name);
        return false;
    }

    LLAMA_LOG_INFO("✓ %s error handling works correctly\n", test_name);
    return true;
}

bool test_metadata_access(struct llama_dataset* dataset) {
    if (!dataset) {
        return false;
    }

    // Test string metadata
    const char* format = llama_dataset_get_metadata_str(dataset, TRAINING_FORMAT_SOURCE);
    LLAMA_LOG_INFO("  Source format metadata: %s\n", format ? format : "not found");

    // Test integer metadata
    int64_t count = llama_dataset_get_metadata_int(dataset, TRAINING_SEQUENCE_COUNT, -1);
    if (count != -1) {
        LLAMA_LOG_INFO("  Sequence count from metadata: %ld\n", count);
        uint64_t actual_count = llama_dataset_n_sequences(dataset);
        if (count != (int64_t)actual_count) {
            LLAMA_LOG_ERROR("  Metadata sequence count mismatch: %ld vs %lu\n", count, actual_count);
            return false;
        }
    } else {
        LLAMA_LOG_INFO("  Sequence count metadata not found\n");
    }

    // Test float metadata with default value
    float test_float = llama_dataset_get_metadata_float(dataset, "test.float", -1.0f);
    if (test_float != -1.0f) {
        LLAMA_LOG_INFO("  Test float metadata: %f\n", test_float);
    }

    return true;
}

double test_random_access_performance(struct llama_dataset* dataset, int access_count) {
    if (!dataset) {
        return -1.0;
    }

    uint64_t seq_count = llama_dataset_n_sequences(dataset);
    if (seq_count == 0) {
        return -1.0;
    }

    TestTimer timer;
    timer.start();

    for (int i = 0; i < access_count; i++) {
        uint64_t idx = rand() % seq_count;
        int32_t len = llama_dataset_sequence_length(dataset, idx);
        const int32_t* data = llama_dataset_sequence(dataset, idx);

        // Validate access
        if (len <= 0 || !data) {
            LLAMA_LOG_ERROR("Random access failed at index %lu\n", idx);
            return -1.0;
        }
    }

    double total_time = timer.elapsed_ms();
    double avg_time = total_time / access_count;

    LLAMA_LOG_INFO("  Random access performance: %.3f ms average (%.3f ms total for %d accesses)\n",
                   avg_time, total_time, access_count);

    return avg_time;
}

// =============================================================================
// TEST DATA CREATOR IMPLEMENTATIONS
// =============================================================================

std::vector<test_sequence_data> TestDataCreator::create_test_sequences() {
    std::vector<test_sequence_data> sequences;

    // Sequence 1: Simple ascending
    sequences.push_back({{1, 2, 3, 4, 5}, 5});

    // Sequence 2: Short sequence
    sequences.push_back({{10, 20, 30}, 3});

    // Sequence 3: Medium sequence
    sequences.push_back({{100, 200, 300, 400}, 4});

    return sequences;
}

bool TestDataCreator::create_small_gguf(const char* path) {
    auto sequences = create_test_sequences();
    std::vector<std::vector<int32_t>> token_sequences;

    for (const auto& seq : sequences) {
        token_sequences.push_back(seq.tokens);
    }

    return create_gguf_from_sequences(path, token_sequences);
}

bool TestDataCreator::create_equivalent_parquet(const char* path) {
    // Create a placeholder Parquet file
    // In a real implementation, this would use Arrow to create proper Parquet
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Write minimal Parquet-like header (placeholder)
    const char header[] = "PAR1"; // Parquet magic number
    file.write(header, 4);
    file.close();
    return true;
}

bool TestDataCreator::create_equivalent_text(const char* path) {
    auto sequences = create_test_sequences();
    std::vector<std::string> lines;

    for (const auto& seq : sequences) {
        std::string line;
        for (size_t i = 0; i < seq.tokens.size(); i++) {
            if (i > 0) line += " ";
            line += std::to_string(seq.tokens[i]);
        }
        lines.push_back(line);
    }

    return create_test_text_file(path, lines);
}

std::vector<test_sequence_data> TestDataCreator::create_test_sequences_with_length(int32_t min_len, int32_t max_len, size_t count) {
    std::vector<test_sequence_data> sequences;
    sequences.reserve(count);

    for (size_t i = 0; i < count; i++) {
        int32_t length = min_len + (rand() % (max_len - min_len + 1));
        std::vector<int32_t> tokens = generate_random_sequence(length);
        sequences.emplace_back(tokens, length);
    }

    return sequences;
}

std::vector<test_sequence_data> TestDataCreator::create_large_test_sequences(size_t sequence_count, int32_t avg_length) {
    std::vector<test_sequence_data> sequences;
    sequences.reserve(sequence_count);

    for (size_t i = 0; i < sequence_count; i++) {
        // Vary length around average
        int32_t length = avg_length + (rand() % 21) - 10; // ±10 tokens
        if (length < 1) length = 1;

        std::vector<int32_t> tokens = generate_random_sequence(length);
        sequences.emplace_back(tokens, length);
    }

    return sequences;
}

std::vector<int32_t> TestDataCreator::generate_random_sequence(int32_t length, int32_t min_token, int32_t max_token) {
    std::vector<int32_t> tokens;
    tokens.reserve(length);

    for (int32_t i = 0; i < length; i++) {
        int32_t token = min_token + (rand() % (max_token - min_token + 1));
        tokens.push_back(token);
    }

    return tokens;
}

bool TestDataCreator::create_gguf_from_sequences(const char* path, const std::vector<std::vector<int32_t>>& sequences) {
    // This is a simplified GGUF creation - in practice would use gguf_context
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Write GGUF magic number
    const char magic[] = "GGUF";
    file.write(magic, 4);

    // Write version
    uint32_t version = 3;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Write tensor count
    uint64_t tensor_count = sequences.size();
    file.write(reinterpret_cast<const char*>(&tensor_count), sizeof(tensor_count));

    // Write metadata count (minimal)
    uint64_t metadata_count = 1;
    file.write(reinterpret_cast<const char*>(&metadata_count), sizeof(metadata_count));

    file.close();
    return true;
}

// =============================================================================
// ADDITIONAL SHARED TEST HELPER FUNCTION IMPLEMENTATIONS
// =============================================================================

struct llama_dataset* load_test_dataset_with_params(const common_params& params, enum dataset_type format) {
    struct llama_dataset* dataset = nullptr;

    switch (format) {
        case DATASET_GGUF:
            dataset = llama_dataset_from_gguf(&params);
            break;
        case DATASET_PARQUET:
#ifdef LLAMA_PARQUET
            dataset = llama_dataset_from_parquet(&params);
#else
            LLAMA_LOG_ERROR("Parquet support not compiled in\n");
            return nullptr;
#endif
            break;
        case DATASET_TEXT:
            dataset = llama_dataset_from_txt(&params, nullptr);
            break;
        default:
            LLAMA_LOG_ERROR("Unsupported dataset format: %d\n", format);
            return nullptr;
    }

    if (!dataset) {
        LLAMA_LOG_ERROR("Failed to load dataset: %s\n", llama_dataset_get_error());
    }

    return dataset;
}

double test_sequential_access_performance(struct llama_dataset* dataset) {
    if (!dataset) {
        return -1.0;
    }

    uint64_t seq_count = llama_dataset_n_sequences(dataset);
    if (seq_count == 0) {
        return -1.0;
    }

    TestTimer timer;
    timer.start();

    for (uint64_t i = 0; i < seq_count; i++) {
        int32_t len = llama_dataset_sequence_length(dataset, i);
        const int32_t* data = llama_dataset_sequence(dataset, i);

        // Validate access
        if (len <= 0 || !data) {
            LLAMA_LOG_ERROR("Sequential access failed at index %lu\n", i);
            return -1.0;
        }
    }

    double total_time = timer.elapsed_ms();
    double avg_time = total_time / seq_count;

    LLAMA_LOG_INFO("  Sequential access performance: %.3f ms average (%.3f ms total for %lu sequences)\n",
                   avg_time, total_time, seq_count);

    return avg_time;
}

double test_dataset_loading_performance(const char* file_path, enum dataset_type format, bool streaming) {
    TestTimer timer;
    timer.start();

    struct llama_dataset* dataset = load_test_dataset(file_path, format, streaming);

    double loading_time = timer.elapsed_ms();

    if (dataset) {
        llama_dataset_free(dataset);
        LLAMA_LOG_INFO("  Dataset loading time: %.2f ms (streaming: %s)\n",
                       loading_time, streaming ? "yes" : "no");
    } else {
        LLAMA_LOG_ERROR("  Dataset loading failed\n");
        return -1.0;
    }

    return loading_time;
}

size_t test_memory_usage(struct llama_dataset* dataset, const char* operation_name, std::function<void()> operation) {
    if (!dataset) {
        return 0;
    }

    TestMemoryTracker tracker;
    tracker.start_tracking();

    operation();

    size_t peak_usage = tracker.get_peak_usage();
    LLAMA_LOG_INFO("  %s peak memory usage: %zu KB\n", operation_name, peak_usage / 1024);

    return peak_usage / 1024; // Return in KB
}

bool initialize_test_environment() {
    // Create test data directory if it doesn't exist
    struct stat st;
    if (stat("test_data", &st) != 0) {
        if (mkdir("test_data", 0755) != 0) {
            LLAMA_LOG_ERROR("Failed to create test_data directory\n");
            return false;
        }
    }

    // Initialize random seed for reproducible tests
    srand(42);

    LLAMA_LOG_INFO("Test environment initialized\n");
    return true;
}

void cleanup_test_environment() {
    cleanup_all_test_files();
    LLAMA_LOG_INFO("Test environment cleaned up\n");
}

bool run_comprehensive_dataset_validation(struct llama_dataset* dataset, const test_dataset_metadata* expected_metadata) {
    if (!dataset) {
        LLAMA_LOG_ERROR("Dataset is null\n");
        return false;
    }

    // Basic property validation
    if (!validate_basic_dataset_properties(dataset, 1)) {
        return false;
    }

    // Metadata validation if expected values provided
    if (expected_metadata && !validate_dataset_metadata(dataset, *expected_metadata)) {
        return false;
    }

    // Sequence validation
    if (!test_all_sequences_valid(dataset)) {
        return false;
    }

    // Error handling validation
    if (!test_error_handling_comprehensive(dataset)) {
        return false;
    }

    LLAMA_LOG_INFO("Comprehensive dataset validation passed\n");
    return true;
}

// =============================================================================
// TEST SUITE RUNNER IMPLEMENTATION
// =============================================================================

TestSuiteRunner::TestSuiteRunner(const std::string& name) : suite_name(name) {
    suite_timer.start();
    suite_memory.start_tracking();
}

TestSuiteRunner::TestSuiteRunner(const std::string& name, const test_config& cfg) : suite_name(name), config(cfg) {
    suite_timer.start();
    suite_memory.start_tracking();
}

void TestSuiteRunner::run_test(const std::string& test_name, std::function<bool()> test_func) {
    LLAMA_LOG_INFO("\n--- Running test: %s ---\n", test_name.c_str());

    TestTimer test_timer;
    test_timer.start();

    result.tests_run++;

    try {
        bool success = test_func();

        if (success) {
            result.tests_passed++;
            TEST_LOG_SUCCESS(test_name.c_str());
        } else {
            result.tests_failed++;
            result.failure_messages.push_back(test_name + ": Test function returned false");
            TEST_LOG_FAILURE(test_name.c_str());
        }
    } catch (const std::exception& e) {
        result.tests_failed++;
        result.failure_messages.push_back(test_name + ": Exception - " + e.what());
        TEST_LOG_FAILURE((test_name + " (exception)").c_str());
        LLAMA_LOG_ERROR("Exception: %s\n", e.what());
    }

    double test_time = test_timer.elapsed_ms();
    result.total_time_ms += test_time;

    if (config.enable_performance_tests && test_time > config.performance_threshold_ms) {
        TEST_LOG_WARNING("Test exceeded performance threshold: %.2f ms > %.2f ms\n",
                        test_time, config.performance_threshold_ms);
    }
}

void TestSuiteRunner::skip_test(const std::string& test_name, const std::string& reason) {
    result.tests_skipped++;
    LLAMA_LOG_INFO("Skipping test: %s - %s\n", test_name.c_str(), reason.c_str());
}

void TestSuiteRunner::set_config(const test_config& cfg) {
    config = cfg;
}

void TestSuiteRunner::log_final_summary() const {
    suite_timer.log_elapsed("Total suite execution");
    suite_memory.log_peak_usage();
    result.log_summary();
}

bool TestSuiteRunner::run_dataset_test(const std::string& test_name, const char* file_path,
                                      enum dataset_type format, std::function<bool(struct llama_dataset*)> test_func) {
    bool test_passed = false;
    run_test(test_name, [&]() {
        TestDatasetGuard dataset(load_test_dataset(file_path, format, config.enable_streaming));
        if (!dataset.is_valid()) {
            return false;
        }
        test_passed = test_func(dataset.get());
        return test_passed;
    });
    return test_passed;
}

bool TestSuiteRunner::run_comparison_test(const std::string& test_name, struct llama_dataset* ds1,
                                         struct llama_dataset* ds2, bool exact_match) {
    bool test_passed = false;
    run_test(test_name, [&]() {
        if (exact_match) {
            test_passed = compare_datasets_exact(ds1, ds2);
        } else {
            test_passed = compare_datasets_with_tolerance(ds1, ds2, 1);
        }
        return test_passed;
    });
    return test_passed;
}

bool TestSuiteRunner::run_performance_test(const std::string& test_name, std::function<bool()> test_func,
                                          double threshold_ms) {
    bool test_passed = false;
    run_test(test_name, [&]() {
        TestTimer timer;
        timer.start();

        test_passed = test_func();

        double elapsed = timer.elapsed_ms();
        if (elapsed > threshold_ms) {
            TEST_LOG_WARNING("Performance test exceeded threshold: %.2f ms > %.2f ms\n", elapsed, threshold_ms);
        }

        return test_passed;
    });
    return test_passed;
}
