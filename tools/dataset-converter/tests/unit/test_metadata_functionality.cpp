#include "test_core_functionality.h"
#include "llama-dataset.h"
#include "llama-dataset-metadata.h"
#include "llama-dataset-error.h"
#include <gtest/gtest.h>
#include <string>
#include <cmath>
#include <climits>

/**
 * @file test_metadata_functionality.cpp
 * @brief Unit tests for metadata access functionality
 * 
 * Tests the metadata module's ability to:
 * - Extract metadata from different dataset formats
 * - Convert between data types (string, int, float)
 * - Validate and normalize metadata keys
 * - Handle format-specific key mapping
 * - Provide consistent error handling
 */

class MetadataTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clear any previous errors
        llama_dataset_clear_error();
    }

    void TearDown() override {
        // Clean up any test datasets
        if (test_dataset) {
            llama_dataset_free(test_dataset);
            test_dataset = nullptr;
        }
    }

    // Helper function to create a mock dataset with known metadata
    struct llama_dataset* create_mock_dataset_with_metadata() {
        // This is a placeholder that will be implemented during the GREEN phase
        // For now, return nullptr to make tests fail (RED phase)
        return nullptr;
    }

    struct llama_dataset* test_dataset = nullptr;
};

// Test metadata string access
TEST_F(MetadataTest, GetMetadataString_ValidKey_ReturnsValue) {
    // RED PHASE: Test that should fail initially
    // This test defines the behavior we want for string metadata access
    
    // Create a mock dataset with metadata (this will fail initially)
    test_dataset = create_mock_dataset_with_metadata();
    ASSERT_NE(test_dataset, nullptr);
    
    // Test getting a string metadata value
    const char* result = llama_dataset_get_metadata_str(test_dataset, TRAINING_DATASET_NAME);
    
    EXPECT_NE(result, nullptr);
    EXPECT_STREQ(result, "test_dataset");
    EXPECT_FALSE(llama_dataset_has_error());
}

TEST_F(MetadataTest, GetMetadataString_InvalidDataset_ReturnsNull) {
    // Test error handling for invalid dataset
    const char* result = llama_dataset_get_metadata_str(nullptr, TRAINING_DATASET_NAME);
    
    EXPECT_EQ(result, nullptr);
    EXPECT_TRUE(llama_dataset_has_error());
    EXPECT_STREQ(llama_dataset_get_error_message(), "Invalid parameters");
}

TEST_F(MetadataTest, GetMetadataString_InvalidKey_ReturnsNull) {
    test_dataset = create_mock_dataset_with_metadata();
    ASSERT_NE(test_dataset, nullptr);
    
    // Test with null key
    const char* result = llama_dataset_get_metadata_str(test_dataset, nullptr);
    
    EXPECT_EQ(result, nullptr);
    EXPECT_TRUE(llama_dataset_has_error());
}

TEST_F(MetadataTest, GetMetadataString_NonexistentKey_ReturnsNull) {
    test_dataset = create_mock_dataset_with_metadata();
    ASSERT_NE(test_dataset, nullptr);
    
    const char* result = llama_dataset_get_metadata_str(test_dataset, "nonexistent.key");
    
    EXPECT_EQ(result, nullptr);
    // Should not set error for missing keys - this is expected behavior
    EXPECT_FALSE(llama_dataset_has_error());
}

// Test metadata integer access
TEST_F(MetadataTest, GetMetadataInt_ValidKey_ReturnsValue) {
    test_dataset = create_mock_dataset_with_metadata();
    ASSERT_NE(test_dataset, nullptr);
    
    int64_t result = llama_dataset_get_metadata_int(test_dataset, TRAINING_SEQUENCE_COUNT, -1);
    
    EXPECT_EQ(result, 1000);
    EXPECT_FALSE(llama_dataset_has_error());
}

TEST_F(MetadataTest, GetMetadataInt_InvalidDataset_ReturnsDefault) {
    int64_t result = llama_dataset_get_metadata_int(nullptr, TRAINING_SEQUENCE_COUNT, 42);
    
    EXPECT_EQ(result, 42);
    EXPECT_TRUE(llama_dataset_has_error());
}

TEST_F(MetadataTest, GetMetadataInt_NonexistentKey_ReturnsDefault) {
    test_dataset = create_mock_dataset_with_metadata();
    ASSERT_NE(test_dataset, nullptr);
    
    int64_t result = llama_dataset_get_metadata_int(test_dataset, "nonexistent.key", 99);
    
    EXPECT_EQ(result, 99);
    EXPECT_FALSE(llama_dataset_has_error());
}

TEST_F(MetadataTest, GetMetadataInt_StringToIntConversion_ReturnsConverted) {
    test_dataset = create_mock_dataset_with_metadata();
    ASSERT_NE(test_dataset, nullptr);
    
    // Test conversion from string "123" to int 123
    int64_t result = llama_dataset_get_metadata_int(test_dataset, "test.string.number", -1);
    
    EXPECT_EQ(result, 123);
    EXPECT_FALSE(llama_dataset_has_error());
}

// Test metadata float access
TEST_F(MetadataTest, GetMetadataFloat_ValidKey_ReturnsValue) {
    test_dataset = create_mock_dataset_with_metadata();
    ASSERT_NE(test_dataset, nullptr);
    
    float result = llama_dataset_get_metadata_float(test_dataset, "test.float.value", -1.0f);
    
    EXPECT_FLOAT_EQ(result, 3.14f);
    EXPECT_FALSE(llama_dataset_has_error());
}

TEST_F(MetadataTest, GetMetadataFloat_InvalidDataset_ReturnsDefault) {
    float result = llama_dataset_get_metadata_float(nullptr, "test.key", 42.0f);
    
    EXPECT_FLOAT_EQ(result, 42.0f);
    EXPECT_TRUE(llama_dataset_has_error());
}

TEST_F(MetadataTest, GetMetadataFloat_StringToFloatConversion_ReturnsConverted) {
    test_dataset = create_mock_dataset_with_metadata();
    ASSERT_NE(test_dataset, nullptr);
    
    // Test conversion from string "2.718" to float 2.718
    float result = llama_dataset_get_metadata_float(test_dataset, "test.string.float", -1.0f);
    
    EXPECT_FLOAT_EQ(result, 2.718f);
    EXPECT_FALSE(llama_dataset_has_error());
}

// Test internal metadata validation functions
TEST_F(MetadataTest, ValidateKey_ValidKey_ReturnsTrue) {
    EXPECT_TRUE(llama_dataset_metadata_validate_key_internal("valid.key"));
    EXPECT_TRUE(llama_dataset_metadata_validate_key_internal(TRAINING_DATASET_NAME));
}

TEST_F(MetadataTest, ValidateKey_InvalidKey_ReturnsFalse) {
    EXPECT_FALSE(llama_dataset_metadata_validate_key_internal(nullptr));
    EXPECT_FALSE(llama_dataset_metadata_validate_key_internal(""));
    
    // Test very long key (over 256 characters)
    std::string long_key(300, 'a');
    EXPECT_FALSE(llama_dataset_metadata_validate_key_internal(long_key.c_str()));
}

TEST_F(MetadataTest, NormalizeKey_ValidKey_ReturnsNormalized) {
    const char* result = llama_dataset_metadata_normalize_key_internal("Test.Key");
    EXPECT_NE(result, nullptr);
    // For now, normalization just returns the key as-is
    EXPECT_STREQ(result, "Test.Key");
}

// Test type conversion utilities
TEST_F(MetadataTest, ConvertToInt_ValidString_ReturnsTrue) {
    int64_t result;
    
    EXPECT_TRUE(llama_dataset_metadata_convert_to_int_internal("123", &result));
    EXPECT_EQ(result, 123);
    
    EXPECT_TRUE(llama_dataset_metadata_convert_to_int_internal("-456", &result));
    EXPECT_EQ(result, -456);
    
    EXPECT_TRUE(llama_dataset_metadata_convert_to_int_internal("0", &result));
    EXPECT_EQ(result, 0);
}

TEST_F(MetadataTest, ConvertToInt_InvalidString_ReturnsFalse) {
    int64_t result;
    
    EXPECT_FALSE(llama_dataset_metadata_convert_to_int_internal(nullptr, &result));
    EXPECT_FALSE(llama_dataset_metadata_convert_to_int_internal("", &result));
    EXPECT_FALSE(llama_dataset_metadata_convert_to_int_internal("abc", &result));
    EXPECT_FALSE(llama_dataset_metadata_convert_to_int_internal("123abc", &result));
    EXPECT_FALSE(llama_dataset_metadata_convert_to_int_internal("12.34", &result));
}

TEST_F(MetadataTest, ConvertToFloat_ValidString_ReturnsTrue) {
    float result;
    
    EXPECT_TRUE(llama_dataset_metadata_convert_to_float_internal("3.14", &result));
    EXPECT_FLOAT_EQ(result, 3.14f);
    
    EXPECT_TRUE(llama_dataset_metadata_convert_to_float_internal("-2.718", &result));
    EXPECT_FLOAT_EQ(result, -2.718f);
    
    EXPECT_TRUE(llama_dataset_metadata_convert_to_float_internal("0.0", &result));
    EXPECT_FLOAT_EQ(result, 0.0f);
    
    EXPECT_TRUE(llama_dataset_metadata_convert_to_float_internal("42", &result));
    EXPECT_FLOAT_EQ(result, 42.0f);
}

TEST_F(MetadataTest, ConvertToFloat_InvalidString_ReturnsFalse) {
    float result;
    
    EXPECT_FALSE(llama_dataset_metadata_convert_to_float_internal(nullptr, &result));
    EXPECT_FALSE(llama_dataset_metadata_convert_to_float_internal("", &result));
    EXPECT_FALSE(llama_dataset_metadata_convert_to_float_internal("abc", &result));
    EXPECT_FALSE(llama_dataset_metadata_convert_to_float_internal("12.34abc", &result));
}

// Test enhanced type conversion with overflow protection
TEST_F(MetadataTest, ConvertToIntSafe_OverflowDetection_ReturnsCorrectly) {
    int64_t result;
    bool overflow_detected;
    
    // Test normal values
    EXPECT_TRUE(llama_dataset_metadata_convert_to_int_safe_internal("123", &result, &overflow_detected));
    EXPECT_EQ(result, 123);
    EXPECT_FALSE(overflow_detected);
    
    // Test maximum int64_t value
    EXPECT_TRUE(llama_dataset_metadata_convert_to_int_safe_internal("9223372036854775807", &result, &overflow_detected));
    EXPECT_EQ(result, INT64_MAX);
    EXPECT_FALSE(overflow_detected);
    
    // Test minimum int64_t value
    EXPECT_TRUE(llama_dataset_metadata_convert_to_int_safe_internal("-9223372036854775808", &result, &overflow_detected));
    EXPECT_EQ(result, INT64_MIN);
    EXPECT_FALSE(overflow_detected);
    
    // Test overflow (value too large)
    EXPECT_FALSE(llama_dataset_metadata_convert_to_int_safe_internal("9223372036854775808", &result, &overflow_detected));
    EXPECT_TRUE(overflow_detected);
    
    // Test underflow (value too small)
    EXPECT_FALSE(llama_dataset_metadata_convert_to_int_safe_internal("-9223372036854775809", &result, &overflow_detected));
    EXPECT_TRUE(overflow_detected);
}

TEST_F(MetadataTest, ConvertToFloatSafe_SpecialValueDetection_ReturnsCorrectly) {
    float result;
    bool special_value_detected;
    
    // Test normal values
    EXPECT_TRUE(llama_dataset_metadata_convert_to_float_safe_internal("3.14", &result, &special_value_detected));
    EXPECT_FLOAT_EQ(result, 3.14f);
    EXPECT_FALSE(special_value_detected);
    
    // Test infinity
    EXPECT_TRUE(llama_dataset_metadata_convert_to_float_safe_internal("inf", &result, &special_value_detected));
    EXPECT_TRUE(std::isinf(result));
    EXPECT_TRUE(special_value_detected);
    
    EXPECT_TRUE(llama_dataset_metadata_convert_to_float_safe_internal("-inf", &result, &special_value_detected));
    EXPECT_TRUE(std::isinf(result));
    EXPECT_TRUE(special_value_detected);
    
    // Test NaN
    EXPECT_TRUE(llama_dataset_metadata_convert_to_float_safe_internal("nan", &result, &special_value_detected));
    EXPECT_TRUE(std::isnan(result));
    EXPECT_TRUE(special_value_detected);
    
    // Test very large values that might overflow
    EXPECT_FALSE(llama_dataset_metadata_convert_to_float_safe_internal("1e50", &result, &special_value_detected));
    EXPECT_TRUE(special_value_detected);
}

// Test type checking utilities
TEST_F(MetadataTest, IsValidIntString_ValidatesCorrectly) {
    EXPECT_TRUE(llama_dataset_metadata_is_valid_int_string_internal("123"));
    EXPECT_TRUE(llama_dataset_metadata_is_valid_int_string_internal("-456"));
    EXPECT_TRUE(llama_dataset_metadata_is_valid_int_string_internal("0"));
    EXPECT_TRUE(llama_dataset_metadata_is_valid_int_string_internal("9223372036854775807"));
    
    EXPECT_FALSE(llama_dataset_metadata_is_valid_int_string_internal(nullptr));
    EXPECT_FALSE(llama_dataset_metadata_is_valid_int_string_internal(""));
    EXPECT_FALSE(llama_dataset_metadata_is_valid_int_string_internal("abc"));
    EXPECT_FALSE(llama_dataset_metadata_is_valid_int_string_internal("123abc"));
    EXPECT_FALSE(llama_dataset_metadata_is_valid_int_string_internal("12.34"));
    EXPECT_FALSE(llama_dataset_metadata_is_valid_int_string_internal("9223372036854775808")); // Overflow
}

TEST_F(MetadataTest, IsValidFloatString_ValidatesCorrectly) {
    EXPECT_TRUE(llama_dataset_metadata_is_valid_float_string_internal("3.14"));
    EXPECT_TRUE(llama_dataset_metadata_is_valid_float_string_internal("-2.718"));
    EXPECT_TRUE(llama_dataset_metadata_is_valid_float_string_internal("0.0"));
    EXPECT_TRUE(llama_dataset_metadata_is_valid_float_string_internal("42"));
    EXPECT_TRUE(llama_dataset_metadata_is_valid_float_string_internal("1e-10"));
    EXPECT_TRUE(llama_dataset_metadata_is_valid_float_string_internal("inf"));
    EXPECT_TRUE(llama_dataset_metadata_is_valid_float_string_internal("nan"));
    
    EXPECT_FALSE(llama_dataset_metadata_is_valid_float_string_internal(nullptr));
    EXPECT_FALSE(llama_dataset_metadata_is_valid_float_string_internal(""));
    EXPECT_FALSE(llama_dataset_metadata_is_valid_float_string_internal("abc"));
    EXPECT_FALSE(llama_dataset_metadata_is_valid_float_string_internal("12.34abc"));
}

// Test default value handling with validation
TEST_F(MetadataTest, GetIntWithValidation_ReturnsCorrectly) {
    test_dataset = create_mock_dataset_with_metadata();
    ASSERT_NE(test_dataset, nullptr);
    
    bool found;
    
    // Test existing key
    int64_t result = llama_dataset_metadata_get_int_with_validation_internal(test_dataset, TRAINING_SEQUENCE_COUNT, -1, &found);
    EXPECT_EQ(result, 1000);
    EXPECT_TRUE(found);
    
    // Test non-existent key
    result = llama_dataset_metadata_get_int_with_validation_internal(test_dataset, "nonexistent.key", 42, &found);
    EXPECT_EQ(result, 42);
    EXPECT_FALSE(found);
    
    // Test invalid dataset
    result = llama_dataset_metadata_get_int_with_validation_internal(nullptr, TRAINING_SEQUENCE_COUNT, 99, &found);
    EXPECT_EQ(result, 99);
    EXPECT_FALSE(found);
    EXPECT_TRUE(llama_dataset_has_error());
}

TEST_F(MetadataTest, GetFloatWithValidation_ReturnsCorrectly) {
    test_dataset = create_mock_dataset_with_metadata();
    ASSERT_NE(test_dataset, nullptr);
    
    bool found;
    
    // Test existing key
    float result = llama_dataset_metadata_get_float_with_validation_internal(test_dataset, "test.float.value", -1.0f, &found);
    EXPECT_FLOAT_EQ(result, 3.14f);
    EXPECT_TRUE(found);
    
    // Test non-existent key
    result = llama_dataset_metadata_get_float_with_validation_internal(test_dataset, "nonexistent.key", 42.0f, &found);
    EXPECT_FLOAT_EQ(result, 42.0f);
    EXPECT_FALSE(found);
    
    // Test invalid dataset
    result = llama_dataset_metadata_get_float_with_validation_internal(nullptr, "test.key", 99.0f, &found);
    EXPECT_FLOAT_EQ(result, 99.0f);
    EXPECT_FALSE(found);
    EXPECT_TRUE(llama_dataset_has_error());
}

// Test format-specific key mapping
TEST_F(MetadataTest, MapKeyForFormat_ReturnsAppropriateKey) {
    // For now, this just returns the key as-is
    const char* result = llama_dataset_metadata_map_key_for_format_internal("test.key", 0);
    EXPECT_STREQ(result, "test.key");
}