cmake_minimum_required(VERSION 3.15)
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
        ${CMAKE_CURRENT_SOURCE_DIR}/validation)

# Check for Arrow/Parquet dependencies
if(LLAMA_PARQUET)
    find_package(Arrow REQUIRED)
    find_package(Parquet REQUIRED)
    add_definitions(-DLLAMA_DATASET_PARQUET_SUPPORT)
endif()

# Option to control the build of the dataset-converter tool
option(PROJECT_BUILD_DATASET_CONVERTER "Build the dataset-converter tool" ON)

if(NOT PROJECT_BUILD_DATASET_CONVERTER)
    message(STATUS "Skipping build of dataset-converter tool.")
    return()
endif()

# Set the C++ standard to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Gather all .cpp source files for the executable
file(GLOB_RECURSE DATASET_CONVERTER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/formats/gguf/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/formats/parquet/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/formats/text/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/streaming/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/validation/*.cpp"
)

add_library(dataset_convert_lib STATIC ${DATASET_CONVERTER_SOURCES})

# Link dependencies to the target
target_link_libraries(dataset_convert_lib PRIVATE
        llama
        common
        Threads::Threads
        ggml
)

# Link Arrow/Parquet if enabled
if(LLAMA_PARQUET)
    target_link_libraries(dataset_convert_lib PRIVATE
            Arrow::arrow_shared
            Parquet::parquet_shared
    )
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

enable_testing()
add_executable(create-test-data ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/create_test_data.cpp)
target_link_libraries(create-test-data PRIVATE dataset_convert_lib)
add_executable(create-test-data-gguf ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/create_test_gguf.cpp)
target_link_libraries(create-test-data-gguf PRIVATE dataset_convert_lib)

set_target_properties(create-test-data PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(create-test-data-gguf PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
function(add_test_target TEST_SRC)
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)

    file(RELATIVE_PATH REL_TEST_NAME ${CMAKE_CURRENT_SOURCE_DIR} ${TEST_SRC})
    string(REPLACE "/" "_" REL_TEST_NAME ${REL_TEST_NAME})
    string(REPLACE "\\" "_" REL_TEST_NAME ${REL_TEST_NAME})
    string(REPLACE "." "_" REL_TEST_NAME ${REL_TEST_NAME})
    string(REPLACE " " "_" REL_TEST_NAME ${REL_TEST_NAME})

    add_executable(${REL_TEST_NAME} ${TEST_SRC})
    target_link_libraries(${REL_TEST_NAME} PRIVATE dataset_convert_lib)

    set_target_properties(${REL_TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    add_test(
            NAME ${REL_TEST_NAME}
            COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_BINARY_DIR}
            sh -c "./create-test-data && ./create-test-data-gguf && ./${REL_TEST_NAME}"
    )
endfunction()

file(GLOB_RECURSE TEST_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/streaming/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/*.cpp"
)

foreach(TEST_SRC ${TEST_SOURCES})
    add_test_target(${TEST_SRC})
endforeach()
