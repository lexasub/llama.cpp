# CMakeLists.txt for dataset converter tests
# Comprehensive test infrastructure for the dataset converter framework

cmake_minimum_required(VERSION 3.14)

# Test configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../core
    ${CMAKE_CURRENT_SOURCE_DIR}/../formats
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Mock format loader library
add_library(mock_format_loader STATIC
    mock_format_loader.cpp
    mock_format_loader.h
)

target_link_libraries(mock_format_loader
    llama-dataset-core
    llama-dataset-error
    llama-dataset-registry
)

# Core module tests
add_executable(test_core_module
    test_core_module.cpp
)

target_link_libraries(test_core_module
    mock_format_loader
    llama-dataset-core
    llama-dataset-error
    llama-dataset-registry
    llama-dataset-metadata
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)

# Mock loader validation tests
add_executable(test_mock_loader
    test_mock_loader.cpp
)

target_link_libraries(test_mock_loader
    mock_format_loader
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)

# Error handling integration tests
add_executable(test_error_integration
    test_error_integration.cpp
)

target_link_libraries(test_error_integration
    mock_format_loader
    llama-dataset-core
    llama-dataset-error
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)

# Registry system tests
add_executable(test_registry_integration
    test_registry_integration.cpp
)

target_link_libraries(test_registry_integration
    mock_format_loader
    llama-dataset-registry
    llama-dataset-error
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)

# Add tests to CTest
enable_testing()

add_test(NAME CoreModuleTests COMMAND test_core_module)
add_test(NAME MockLoaderTests COMMAND test_mock_loader)
add_test(NAME ErrorIntegrationTests COMMAND test_error_integration)
add_test(NAME RegistryIntegrationTests COMMAND test_registry_integration)

# Test properties
set_tests_properties(CoreModuleTests PROPERTIES
    TIMEOUT 30
    LABELS "core;unit"
)

set_tests_properties(MockLoaderTests PROPERTIES
    TIMEOUT 10
    LABELS "mock;unit"
)

set_tests_properties(ErrorIntegrationTests PROPERTIES
    TIMEOUT 15
    LABELS "error;integration"
)

set_tests_properties(RegistryIntegrationTests PROPERTIES
    TIMEOUT 15
    LABELS "registry;integration"
)

# Coverage configuration (if enabled)
if(ENABLE_COVERAGE)
    include(CodeCoverage)
    append_coverage_compiler_flags()
    
    setup_target_for_coverage_gcovr_html(
        NAME coverage_tests
        EXECUTABLE ctest
        DEPENDENCIES test_core_module test_mock_loader test_error_integration test_registry_integration
        BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../
        EXCLUDE "tests/*" "build/*" "external/*"
    )
endif()

# Valgrind configuration (if available)
find_program(VALGRIND_PROGRAM valgrind)
if(VALGRIND_PROGRAM)
    add_custom_target(valgrind_tests
        COMMAND ${VALGRIND_PROGRAM} --tool=memcheck --leak-check=full --show-leak-kinds=all 
                --track-origins=yes --verbose --error-exitcode=1 
                $<TARGET_FILE:test_core_module>
        DEPENDS test_core_module
        COMMENT "Running core module tests under Valgrind"
    )
endif()

# Documentation generation for tests (disabled - no Doxyfile.in available)
# find_program(DOXYGEN_PROGRAM doxygen)
# if(DOXYGEN_PROGRAM)
#     configure_file(
#         ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
#         ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
#         @ONLY
#     )
#     
#     add_custom_target(test_docs
#         COMMAND ${DOXYGEN_PROGRAM} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
#         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#         COMMENT "Generating test documentation with Doxygen"
#     )
# endif()

# Performance validation tests
add_executable(test_performance_validation
    test_performance_validation.cpp
)

target_link_libraries(test_performance_validation
    mock_format_loader
    llama-dataset-core
    llama-dataset-error
    llama-dataset-registry
    Threads::Threads
)

# Performance benchmark suite
add_executable(benchmark_performance
    benchmark_performance.cpp
)

target_link_libraries(benchmark_performance
    mock_format_loader
    llama-dataset-core
    llama-dataset-error
    llama-dataset-registry
    Threads::Threads
)

# Add performance tests to CTest
add_test(NAME PerformanceValidationTests COMMAND test_performance_validation)
set_tests_properties(PerformanceValidationTests PROPERTIES
    TIMEOUT 120
    LABELS "performance;validation"
)

# Performance benchmark (not run by default in CTest)
add_test(NAME PerformanceBenchmark COMMAND benchmark_performance --iterations 50)
set_tests_properties(PerformanceBenchmark PROPERTIES
    TIMEOUT 300
    LABELS "benchmark;performance"
)

# Performance benchmarking (optional)
if(BUILD_BENCHMARKS)
    find_package(benchmark REQUIRED)
    
    add_executable(benchmark_core_module
        benchmark_core_module.cpp
    )
    
    target_link_libraries(benchmark_core_module
        mock_format_loader
        llama-dataset-core
        benchmark::benchmark
        benchmark::benchmark_main
    )
    
    add_test(NAME CoreModuleBenchmarks COMMAND benchmark_core_module)
endif()

# Simple performance test (no external dependencies)
add_executable(test_performance_simple
    test_performance_simple.cpp
)

target_link_libraries(test_performance_simple
    Threads::Threads
)

# Add performance test to CTest
add_test(NAME PerformanceSimple COMMAND test_performance_simple)
set_tests_properties(PerformanceSimple PROPERTIES
    TIMEOUT 60
    LABELS "performance;simple"
)

# Install test executables (optional)
if(INSTALL_TESTS)
    install(TARGETS test_core_module test_mock_loader test_error_integration test_registry_integration test_performance_simple
        RUNTIME DESTINATION bin/tests
    )
    
    install(FILES
        mock_format_loader.h
        DESTINATION include/tests
    )
endif()
# Simple architecture integration test for D3 task (no external dependencies)
add_executable(test-architecture-integration-simple test_architecture_integration_simple.cpp)
# No external libraries needed - this is a self-contained validation test
install(TARGETS test-architecture-integration-simple RUNTIME)
add_test(NAME architecture-integration-simple COMMAND test-architecture-integration-simple)

# Minimal streaming tokenization test for E1 task completion
add_executable(test_streaming_tokenization_minimal
    test_streaming_tokenization_minimal.cpp
)

target_link_libraries(test_streaming_tokenization_minimal
    Threads::Threads
)

# Add minimal streaming tokenization test to CTest
add_test(NAME StreamingTokenizationMinimal COMMAND test_streaming_tokenization_minimal)
set_tests_properties(StreamingTokenizationMinimal PROPERTIES
    TIMEOUT 120
    LABELS "streaming;tokenization;minimal;e1"
)