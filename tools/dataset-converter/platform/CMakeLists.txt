# Platform Module CMake Configuration
# This file defines the build configuration for the platform compatibility modules

# Define platform module source files
set(PLATFORM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/platform-compat.cpp
)

# Define platform module header files
set(PLATFORM_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/platform-compat.h
)

# Create platform library
add_library(dataset_platform STATIC ${PLATFORM_SOURCES} ${PLATFORM_HEADERS})

# Set C++ standard
target_compile_features(dataset_platform PRIVATE cxx_std_17)

# Include directories for platform modules using standardized paths
if(DEFINED DATASET_CONVERTER_INCLUDE_DIRS)
    target_include_directories(dataset_platform PRIVATE ${DATASET_CONVERTER_INCLUDE_DIRS})
else()
    # Fallback to manual include directories if parent hasn't defined them
    target_include_directories(dataset_platform PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../core
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../src
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    )
endif()

# Link core dependencies
target_link_libraries(dataset_platform PUBLIC
    llama
    ggml
)

# Set target properties
set_target_properties(dataset_platform PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Export variables for parent CMakeLists.txt
set(PLATFORM_LIBRARY dataset_platform PARENT_SCOPE)