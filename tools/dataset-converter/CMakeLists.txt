cmake_minimum_required(VERSION 3.15)
project(dataset-converter)

# Find required packages
find_package(Threads REQUIRED)

# Option to control the build of the dataset-converter tool
option(PROJECT_BUILD_DATASET_CONVERTER "Build the dataset-converter tool" ON)

if(NOT PROJECT_BUILD_DATASET_CONVERTER)
    message(STATUS "Skipping build of dataset-converter tool.")
    return()
endif()

# Set the C++ standard to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include common directories
include_directories(.
        ../../common
        ../../
        ../../src
        ../../ggml/include
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/formats/gguf
        ${CMAKE_CURRENT_SOURCE_DIR}/formats/parquet
        ${CMAKE_CURRENT_SOURCE_DIR}/formats/text
        ${CMAKE_CURRENT_SOURCE_DIR}/core
        ${CMAKE_CURRENT_SOURCE_DIR}/streaming
        ${CMAKE_CURRENT_SOURCE_DIR}/safety
        ${CMAKE_CURRENT_SOURCE_DIR}/validation
        ${CMAKE_CURRENT_SOURCE_DIR}/platform)

# Include modular CMake files for each component
add_subdirectory(core)
add_subdirectory(formats/gguf)
add_subdirectory(formats/text)
add_subdirectory(formats/parquet)
add_subdirectory(streaming)
add_subdirectory(validation)
add_subdirectory(platform)

# Create the main dataset converter library as an interface library
add_library(dataset_convert_lib INTERFACE)

# Link all module libraries to the main library
target_link_libraries(dataset_convert_lib INTERFACE
    dataset_core
    dataset_gguf_format
    dataset_text_format
    dataset_streaming
    dataset_validation
    dataset_platform
    llama
    common
    Threads::Threads
    ggml
)

# Link Parquet format library if available
if(LLAMA_PARQUET)
    target_link_libraries(dataset_convert_lib INTERFACE dataset_parquet_format)
endif()

# Define the executable target
add_executable(dataset_converter "${CMAKE_CURRENT_SOURCE_DIR}/tools/convert-to-gguf.cpp")
target_link_libraries(dataset_converter PRIVATE dataset_convert_lib)

add_executable(streaming_analyzer "${CMAKE_CURRENT_SOURCE_DIR}/tools/streaming-optimization-analysis.cpp")
target_link_libraries(streaming_analyzer PRIVATE dataset_convert_lib)

add_executable(test_data_validator "${CMAKE_CURRENT_SOURCE_DIR}/tools/test-data-validation-tool.cpp")
target_link_libraries(test_data_validator PRIVATE dataset_convert_lib)

install(TARGETS dataset_converter streaming_analyzer test_data_validator
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

# Include tests subdirectory
add_subdirectory(tests)
