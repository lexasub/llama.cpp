# Dataset Converter Integration Tests

This directory contains comprehensive integration tests for the simple dataset loader interface.

## Test Files

### Unit Tests
- `test-dataset.cpp` - Basic interface tests and error handling
- `test_core_functionality.cpp` - Core functionality tests for all factory functions
- `test_streaming.cpp` - Streaming vs non-streaming equivalence tests

### Integration Tests
- `test_integration.cpp` - Comprehensive end-to-end integration tests
- `create_test_data.cpp` - Utility to create test data files

### Test Data
- `test_data/` - Directory containing test datasets in various formats
  - `small_dataset.gguf` - Small GGUF file with known sequences
  - `text_dataset.txt` - Equivalent text file with same sequences
  - `parquet_dataset.parquet` - Equivalent Parquet file with same sequences
  - `large_text_dataset.txt` - Large text file for performance testing
  - `large_parquet_dataset.parquet` - Large Parquet file for performance testing
  - `corrupted_dataset.gguf` - Corrupted GGUF file for error testing
  - `corrupted_dataset.parquet` - Corrupted Parquet file for error testing

## Integration Test Coverage

The integration tests (`test_integration.cpp`) cover the following scenarios:

### 1. End-to-End Workflow Testing
- **Load → Access → Convert → Save → Reload** pipeline
- Data integrity verification through multiple conversions
- Performance measurement for each step
- Double conversion testing (convert → save → reload → convert again)

**Requirements tested:** 1.3, 4.4, 7.1, 7.2

### 2. Memory Usage Validation
- Memory usage comparison between streaming and non-streaming modes
- Memory usage tracking during dataset loading and access
- Validation that streaming uses less memory when available
- Fallback behavior when streaming is not supported

**Requirements tested:** 5.1, 5.4

### 3. Performance Comparison
- Load time comparison across formats (GGUF, Parquet, Text)
- Access time comparison for sequence data retrieval
- Memory usage comparison across formats and loading modes
- Performance metrics reporting

**Requirements tested:** 5.4, 6.2

### 4. Cross-Format Data Consistency
- Creation of equivalent test datasets in multiple formats
- Data consistency verification across GGUF, Parquet, and Text formats
- Sequence-by-sequence comparison to ensure identical data
- Metadata consistency validation

**Requirements tested:** 1.1, 1.2, 1.3, 7.1, 7.2

### 5. Large Dataset Handling
- Performance testing with large datasets (1000+ sequences)
- Memory usage validation for large files
- Random access performance testing
- Streaming behavior validation for large datasets

**Requirements tested:** 5.1, 5.2, 5.4

### 6. Error Recovery and Robustness
- Corrupted file handling
- Multiple error condition testing
- Error state recovery validation
- Invalid path and parameter handling

**Requirements tested:** 6.2, 8.1

## Running the Tests

### Prerequisites
1. Build the dataset converter with all dependencies (ggml, gguf, Arrow/Parquet)
2. Ensure test data files exist (run `create_test_data` if needed)

### Building Tests
```bash
# From the project root
mkdir -p build && cd build
cmake ..
make test_integration test_core_functionality test_streaming create_test_data
```

### Creating Test Data
```bash
# Create test data files
./create_test_data
```

### Running Individual Tests
```bash
# Basic interface tests
./test-dataset

# Core functionality tests
./test_core_functionality

# Streaming tests
./test_streaming

# Comprehensive integration tests
./test_integration
```

### Running All Tests
```bash
# Run all tests in sequence
./test-dataset && ./test_core_functionality && ./test_streaming && ./test_integration
```

## Expected Output

### Successful Test Run
```
=== Running dataset integration tests ===

=== Testing end-to-end workflow ===
Step 1: Loading original dataset...
  Load time: 2.5 ms
Step 2: Accessing and validating data...
  Sequence count: 4
  First sequence tokens: 1 2 3...
Step 3: Converting to new GGUF file...
  Convert time: 1.8 ms
Step 4: Reloading converted file...
  Reload time: 2.1 ms
Step 5: Comparing original and converted datasets...
  Comparing 4 sequences...
  Datasets are identical: yes
Step 6: Testing double conversion...
  Double conversion preserves data: yes
✓ End-to-end workflow test passed

=== Testing memory usage validation ===
Testing non-streaming mode memory usage...
  Non-streaming memory usage: 4096 bytes
  Non-streaming memory after access: 4096 bytes
Testing streaming mode memory usage...
  Streaming memory usage: 2048 bytes
  Streaming memory after access: 3072 bytes
  Streaming uses less initial memory: yes
  Data consistency between modes: yes
✓ Memory usage validation test passed

[... additional test output ...]

=== All integration tests completed successfully! ===
```

### Test Failure Indicators
- Assertion failures with detailed error messages
- Memory usage inconsistencies
- Data corruption detection
- Performance regression warnings

## Test Data Specifications

### Small Test Dataset
- **Sequences:** 4 sequences with known token patterns
- **Lengths:** Variable (3-5 tokens per sequence)
- **Content:** Integer tokens: [1,2,3,4,5], [10,20,30], [100,200,300,400], [1000,2000]
- **Purpose:** Data consistency validation across formats

### Large Test Dataset
- **Sequences:** 1000 sequences
- **Lengths:** Variable (10-29 tokens per sequence)
- **Content:** Generated integer patterns for performance testing
- **Purpose:** Performance and memory usage validation

### Corrupted Test Files
- **GGUF:** Invalid magic number and random data
- **Parquet:** Invalid magic number and random data
- **Purpose:** Error handling and recovery testing

## Troubleshooting

### Common Issues

1. **Test data files missing**
   - Run `create_test_data` to generate required files
   - Check file permissions in test_data directory

2. **Memory usage tests failing**
   - May fail on systems with different memory management
   - Check if streaming is actually supported for the format

3. **Performance tests showing unexpected results**
   - Results vary by system performance
   - Focus on relative performance rather than absolute values

4. **Cross-format consistency failures**
   - May indicate incomplete format loader implementation
   - Check if Parquet/Text loaders are fully implemented

### Debug Mode
Set environment variable for verbose output:
```bash
export LLAMA_LOG_LEVEL=DEBUG
./test_integration
```

## Contributing

When adding new tests:
1. Follow the existing test structure and naming conventions
2. Add appropriate assertions and error checking
3. Include performance measurements where relevant
4. Update this README with new test descriptions
5. Ensure tests clean up temporary files

## Requirements Mapping

| Requirement | Test Coverage |
|-------------|---------------|
| 1.1 - GGUF internal representation | Cross-format consistency tests |
| 1.2 - Consistent GGUF inspection | Metadata validation tests |
| 1.3 - GGUF validation standard | End-to-end workflow tests |
| 5.1 - GGUF streaming support | Memory usage validation |
| 5.4 - Streaming fallback | Performance comparison tests |
| 6.2 - Clean implementation | Error recovery tests |
| 7.1 - Consistent sequence access | Cross-format consistency tests |
| 7.2 - Consistent indexing | Data integrity validation |