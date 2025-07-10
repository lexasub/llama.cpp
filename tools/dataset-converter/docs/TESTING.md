# Testing Framework for Dataset Converter

## Overview

The dataset converter includes a comprehensive testing framework to ensure reliability, performance, and correctness. This document describes the testing structure, available tests, and how to run them.

## Test Structure

The tests are organized into several categories:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test end-to-end workflows
- **Streaming Tests**: Test streaming functionality and optimizations
- **Format Tests**: Test specific format handling (GGUF, Text, Parquet)
- **Performance Tests**: Test performance characteristics

## Available Tests

### Core Functionality Tests

- `test-dataset`: Basic dataset functionality tests
- `test-core-functionality`: Comprehensive core functionality tests
- `test-format-fixes`: Tests for format-specific fixes

### Streaming Tests

- `test-streaming`: Basic streaming functionality tests
- `test-streaming-optimizations`: Tests for streaming optimizations

### Integration Tests

- `test-integration`: End-to-end workflow tests
- `test-validation-integration`: Data validation integration tests

### Utility Tests

- `test-data-validation-tool`: Tests for data validation tools
- `test-monitor-demo`: Tests for test monitoring functionality

## Running Tests

### Running All Tests

To run all tests:

```bash
ctest
```

### Running Specific Test Categories

To run specific test categories:

```bash
# Run all dataset tests
ctest -R "test-dataset"

# Run all streaming tests
ctest -R "test-streaming"

# Run all integration tests
ctest -R "test-integration"

# Run multiple categories
ctest -R "test-dataset|test-streaming|test-integration"
```

### Running Tests with Verbose Output

To run tests with verbose output:

```bash
ctest -V -R "test-streaming"
```

## Test Monitoring

The testing framework includes a monitoring system to track test execution:

- **Memory usage monitoring**: Tracks memory usage during test execution
- **Execution time monitoring**: Measures test execution time
- **Error detection**: Captures and reports test failures
- **Stack trace capture**: Captures stack traces for crashes

To run tests with monitoring:

```bash
./run-monitored-tests.sh test-streaming
```

## Test Data

The tests use various test datasets:

- **Small test datasets**: For basic functionality testing
- **Large test datasets**: For performance and memory usage testing
- **Corrupted test datasets**: For error handling testing

Test data is automatically generated or copied to the build directory during the build process.

## Adding New Tests

To add a new test:

1. Create a new test file in the appropriate directory:
   - Unit tests: `tests/unit/`
   - Integration tests: `tests/integration/`
   - Streaming tests: `tests/streaming/`

2. Add the test to `CMakeLists.txt`:

```cmake
add_executable(test-new-feature tests/unit/test_new_feature.cpp)
add_dependencies(test-new-feature llama-dataset)
target_link_libraries(test-new-feature PRIVATE llama-dataset)

add_test(
    NAME test-new-feature
    COMMAND $<TARGET_FILE:test-new-feature>
)
set_tests_properties(test-new-feature PROPERTIES LABELS "training")
```

3. Run the test:

```bash
ctest -R "test-new-feature"
```

## Test Reports

Test reports are generated in the `reports/` directory:

- **BUILD_VALIDATION_REPORT.md**: Report on build validation
- **CORE_FUNCTIONALITY_TEST_REPORT.md**: Report on core functionality tests
- **INTEGRATION_WORKFLOW_TEST_REPORT.md**: Report on integration workflow tests
- **STREAMING_FUNCTIONALITY_REPORT.md**: Report on streaming functionality tests

## Troubleshooting Tests

### Missing Test Executables

If test executables are missing:

1. Check if the test is properly defined in `CMakeLists.txt`
2. Rebuild the project with `ninja -C build`
3. Check for compilation errors

### Test Failures

If tests fail:

1. Run the test with verbose output: `ctest -V -R "test-name"`
2. Check the test output for error messages
3. Run the test with monitoring: `./run-monitored-tests.sh test-name`
4. Check the test report in the `reports/` directory

### Memory Issues

If tests have memory issues:

1. Run the test with memory monitoring: `./run-monitored-tests.sh test-name`
2. Check for memory leaks and usage patterns
3. Verify that resources are properly cleaned up

## Conclusion

The testing framework provides comprehensive validation of the dataset converter functionality, ensuring reliability, performance, and correctness across different formats and usage patterns.