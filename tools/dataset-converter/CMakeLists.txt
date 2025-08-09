# Dataset Converter CMake Configuration

# Add format subdirectories
add_subdirectory(formats/gguf)
add_subdirectory(formats/text)

# Add tests subdirectory
add_subdirectory(tests)

# Add dataset converter core library
add_library(llama-dataset-core
    core/llama-dataset-registry.cpp
    core/llama-dataset-error.cpp
    core/llama-dataset-metadata.cpp
    core/llama-dataset-validation.cpp
)

target_include_directories(llama-dataset-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(llama-dataset-core PUBLIC
    common
    dataset_gguf_format
    dataset_text_format
)

# Add format registry test
add_executable(test-format-registry
    tests/test_format_registry.cpp
)

target_include_directories(test-format-registry PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(test-format-registry PRIVATE
    llama-dataset-core
    common
)

# Add validation test executable
add_executable(test-validation-enhanced
    tests/test_validation_enhanced.cpp
)

target_include_directories(test-validation-enhanced PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(test-validation-enhanced PRIVATE
    llama-dataset-core
    common
    gtest
    gtest_main
    gmock
)

# Add validation command-line tool
add_executable(llama-dataset-validate
    tools/llama-dataset-validate.cpp
)

target_include_directories(llama-dataset-validate PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(llama-dataset-validate PRIVATE
    llama-dataset-core
    common
)

# Add test to CTest
enable_testing()
add_test(NAME format-registry COMMAND test-format-registry)
add_test(NAME validation-enhanced COMMAND test-validation-enhanced)