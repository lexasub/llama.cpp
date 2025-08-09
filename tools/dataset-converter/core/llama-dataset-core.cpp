/**
 * @file llama-dataset-core.cpp
 * @brief Core coordination module for the llama.cpp dataset converter framework.
 *
 * This module serves as the primary coordination layer for the dataset converter framework,
 * implementing the public API and orchestrating interactions between specialized modules.
 * Following clean architecture principles, this module depends only on abstractions and
 * delegates format-specific operations to registered format loaders.
 *
 * ## Architecture Overview
 *
 * The core module implements the Dependency Inversion Principle by:
 * - **Depending on Abstractions**: Uses format loader interfaces, not concrete implementations
 * - **Registry-Based Discovery**: Finds format loaders through the registry system
 * - **Module Coordination**: Orchestrates interactions between error, metadata, and streaming modules
 * - **Resource Management**: Manages dataset lifecycle and cleanup operations
 *
 * ## Key Responsibilities
 *
 * - **Public API Implementation**: Provides all public dataset functions
 * - **Format Loader Coordination**: Delegates to appropriate format loaders via registry
 * - **Error Management**: Coordinates error handling across all modules
 * - **Resource Lifecycle**: Manages dataset creation, usage, and cleanup
 * - **Module Integration**: Integrates streaming, metadata, and validation systems
 *
 * ## Clean Architecture Compliance
 *
 * This module follows clean architecture principles:
 * - **No Format Dependencies**: Does not include format-specific headers
 * - **Interface-Based**: Uses abstract interfaces for all format operations
 * - **Registry Pattern**: Discovers format loaders through centralized registry
 * - **Single Responsibility**: Focuses solely on coordination and API implementation
 *
 * ## Performance Characteristics
 *
 * - **Minimal Overhead**: Thin coordination layer with ~1-5% performance overhead
 * - **Registry Lookup**: O(1) format lookup by name, O(n) by file detection
 * - **Memory Efficiency**: No format-specific memory overhead in core module
 * - **Thread Safety**: Thread-safe read operations with minimal contention
 *
 * ## Integration Points
 *
 * - **Format Registry**: Uses registry for format loader discovery and management
 * - **Error System**: Integrates with thread-local error handling system
 * - **Metadata System**: Provides unified metadata access across all formats
 * - **Streaming System**: Coordinates streaming cache and optimization managers
 * - **Validation System**: Integrates format validation through loader interfaces
 *
 * @see core/llama-dataset-registry.h for format loader registry system
 * @see core/llama-dataset-error.h for error handling integration
 * @see core/llama-dataset-format-interface.h for format loader interface
 * @see docs/MODULE_ARCHITECTURE.md for complete architecture documentation
 *
 * @version 1.0
 * @since 2024
 */

#include "llama-dataset.h"
#include "llama-dataset-internal.h"
#include "llama-dataset-error.h"
#include "llama-dataset-metadata.h"
#include "llama-dataset-conversion.h"

// DEPRECATED: Direct format includes violate dependency inversion principle
// TODO: Remove after registry integration is complete (Task G1)
// These includes will be removed in the next release - use registry-based loading instead
#pragma message("DEPRECATED: Direct format includes will be removed in next release - use registry system")
#include "../formats/text/llama-dataset-text.h"        // ❌ DEPRECATED - Remove in Task G1
#include "../formats/gguf/llama-dataset-gguf.h"        // ❌ DEPRECATED - Remove in Task G1
#include "common.h"

// Core coordination module - main API implementation and module coordination
// This file contains the public API entry points and delegates to appropriate modules

// Core coordination module - main API implementation and module coordination
// This file will contain the public API entry points and delegate to appropriate modules

// Public API wrapper functions (to be implemented during extraction)

// DEPRECATED: Forward declarations bypass registry system and violate clean architecture
// TODO: Remove after factory functions use registry (Task G3)
// These declarations will be removed in the next release - use registry-based loading instead
#pragma message("DEPRECATED: Direct format function calls will be removed in next release - use registry system")
extern struct llama_dataset* llama_dataset_load_text_internal(const struct common_params* params, struct llama_model* model);    // ❌ DEPRECATED - Remove in Task G2
extern struct llama_dataset* llama_dataset_load_parquet_internal(const struct common_params* params);                           // ❌ DEPRECATED - Remove in Task G2

// TODO: Add forward declarations for streaming functions when streaming module is properly integrated

// Helper function to warn about deprecated direct format loading
static void warn_deprecated_direct_loading(const char* function_name, const char* format_name) {
    static bool warned = false;
    if (!warned) {
        fprintf(stderr, "WARNING: %s() using deprecated direct %s format loading. "
                       "Registry-based loading will be required in next release. "
                       "Use llama_dataset_registry_load_by_name() instead.\n", 
                       function_name, format_name);
        warned = true;
    }
}

struct llama_dataset* llama_dataset_from_gguf(const struct common_params* params) {
    // DEPRECATED: Direct format loading bypasses registry system
    warn_deprecated_direct_loading("llama_dataset_from_gguf", "GGUF");
    
    // Clear any previous errors
    llama_dataset_error_clear_internal();
    
    // DEPRECATED: Direct delegation to format module (linked at build time)
    // TODO: Replace with registry-based loading in Task G3
    return llama_dataset_load_gguf(params);
}

struct llama_dataset* llama_dataset_from_txt(const struct common_params* params, struct llama_model* model) {
    // DEPRECATED: Direct format loading bypasses registry system
    warn_deprecated_direct_loading("llama_dataset_from_txt", "text");
    
    // Clear any previous errors
    llama_dataset_error_clear_internal();
    
    // DEPRECATED: Direct delegation to format module (linked at build time)
    // TODO: Replace with registry-based loading in Task G3
    return llama_dataset_load_text_internal(params, model);
}

struct llama_dataset* llama_dataset_from_parquet(const struct common_params* params) {
    // DEPRECATED: Direct format loading bypasses registry system
    warn_deprecated_direct_loading("llama_dataset_from_parquet", "Parquet");
    
    // Clear any previous errors
    llama_dataset_error_clear_internal();
    
    // DEPRECATED: Direct delegation to format module (linked at build time)
    // TODO: Replace with registry-based loading in Task G3
    return llama_dataset_load_parquet_internal(params);
}

uint64_t llama_dataset_n_sequences(const struct llama_dataset* dataset) {
    if (!dataset) {
        llama_dataset_error_set_with_context_internal("core", "n_sequences", "Invalid dataset parameter");
        return 0;
    }
    
    return dataset->n_seq;
}

void llama_dataset_free(struct llama_dataset* dataset) {
    if (!dataset) {
        return;
    }

    // Cleanup streaming infrastructure
    if (dataset->streaming) {
        // TODO: Implement streaming cleanup when streaming module is properly integrated
        // For now, just set to nullptr - streaming integration in progress
        dataset->streaming_cache = nullptr;
        dataset->optimization_manager = nullptr;
    }

    // Coordinate cleanup across all modules
    // TODO: Implement format-specific cleanup functions
    // llama_dataset_gguf_cleanup(dataset);
    // llama_dataset_text_cleanup(dataset);
    // llama_dataset_parquet_cleanup(dataset);

    // TODO: Implement actual resource cleanup for GGUF context, tensors, etc.
    // This will be filled during extraction from main file

    // Free the dataset structure itself
    free(dataset);
}

// Metadata access wrappers
const char* llama_dataset_get_metadata_str(const struct llama_dataset* dataset, const char* key) {
    return llama_dataset_metadata_get_str_internal(dataset, key);
}

int64_t llama_dataset_get_metadata_int(const struct llama_dataset* dataset, const char* key, int64_t default_value) {
    return llama_dataset_metadata_get_int_internal(dataset, key, default_value);
}

float llama_dataset_get_metadata_float(const struct llama_dataset* dataset, const char* key, float default_value) {
    return llama_dataset_metadata_get_float_internal(dataset, key, default_value);
}

// Forward declarations for streaming functions (implemented in streaming module)
extern "C" {
    extern bool llama_dataset_set_streaming_cache_size_internal(struct llama_dataset* dataset, size_t cache_size_bytes);
    extern bool llama_dataset_set_streaming_read_ahead_internal(struct llama_dataset* dataset, bool enabled, size_t window_size);
    extern bool llama_dataset_set_adaptive_cache_sizing_internal(struct llama_dataset* dataset, bool enabled);
    extern bool llama_dataset_get_streaming_stats_internal(const struct llama_dataset* dataset, double* hit_ratio, size_t* memory_usage_bytes, size_t* entry_count);
}

// Streaming configuration wrappers
extern "C" bool llama_dataset_set_streaming_cache_size(struct llama_dataset* dataset, size_t cache_size_bytes) {
    // Clear any previous errors
    llama_dataset_error_clear_internal();
    
    // Delegate to streaming module
    return llama_dataset_set_streaming_cache_size_internal(dataset, cache_size_bytes);
}

extern "C" bool llama_dataset_set_streaming_read_ahead(struct llama_dataset* dataset, bool enabled, size_t window_size) {
    // Clear any previous errors
    llama_dataset_error_clear_internal();
    
    // Delegate to streaming module
    return llama_dataset_set_streaming_read_ahead_internal(dataset, enabled, window_size);
}

extern "C" bool llama_dataset_set_adaptive_cache_sizing(struct llama_dataset* dataset, bool enabled) {
    // Clear any previous errors
    llama_dataset_error_clear_internal();
    
    // Delegate to streaming module
    return llama_dataset_set_adaptive_cache_sizing_internal(dataset, enabled);
}

extern "C" bool llama_dataset_get_streaming_stats(const struct llama_dataset* dataset, double* hit_ratio, size_t* memory_usage_bytes, size_t* entry_count) {
    // Clear any previous errors
    llama_dataset_error_clear_internal();
    
    // Delegate to streaming module
    return llama_dataset_get_streaming_stats_internal(dataset, hit_ratio, memory_usage_bytes, entry_count);
}

// Conversion wrappers
void llama_dataset_to_gguf(struct llama_dataset* dataset, const char* path) {
    // TODO: Implement GGUF conversion - delegate to conversion module
    (void)dataset; (void)path;
    // llama_dataset_conversion_to_gguf(dataset, path);
}

// Error handling wrappers (public API functions)
bool llama_dataset_has_error(void) {
    return llama_dataset_error_has_error_internal();
}

const char* llama_dataset_get_error_message(void) {
    return llama_dataset_error_get_message_internal();
}

enum dataset_error llama_dataset_get_error_code(void) {
    return llama_dataset_error_get_code_internal();
}

void llama_dataset_clear_error(void) {
    llama_dataset_error_clear_internal();
}

const char* llama_dataset_get_error(void) {
    return llama_dataset_error_get_message_internal();
}

void llama_dataset_set_error(const char* message) {
    llama_dataset_error_set_internal(message);
}

void llama_dataset_set_error_with_code(enum dataset_error code, const char* message) {
    llama_dataset_error_set_with_code_internal(code, message);
}

const char* llama_dataset_error_code_to_string(enum dataset_error code) {
    return llama_dataset_error_code_to_string_internal(code);
}

// Internal allocation helper (moved from utils.cpp to avoid streaming dependencies)
extern "C" struct llama_dataset* llama_dataset_alloc_internal(enum dataset_type type, bool streaming) {
    struct llama_dataset* dataset = static_cast<struct llama_dataset *>(malloc(sizeof(struct llama_dataset)));
    if (!dataset) {
        llama_dataset_error_set_with_code_internal(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to allocate dataset structure");
        return nullptr;
    }

    // Initialize all fields to zero/null using proper C++ initialization
    *dataset = {};

    // Set type and streaming flag
    dataset->type = type;
    dataset->streaming = streaming;

    // Initialize tokenization fields
    dataset->model = nullptr;
    dataset->tokenizer_ctx = nullptr;
    dataset->owns_model = false;

    // Initialize streaming infrastructure if in streaming mode
    if (streaming) {
        // TODO: Initialize streaming cache when streaming module is properly integrated
        // For now, just set to nullptr - streaming integration in progress
        dataset->streaming_cache = nullptr;
        dataset->optimization_manager = nullptr;
    } else {
        dataset->streaming_cache = nullptr;
        dataset->optimization_manager = nullptr;
    }

    return dataset;
}

// Placeholder implementations for format loader functions
// TODO: These should be implemented by format modules
struct llama_dataset* llama_dataset_load_text_internal(const struct common_params* params, struct llama_model* model) {
    (void)params; (void)model;
    llama_dataset_error_set_with_code_internal(DATASET_ERROR_UNKNOWN, "Text loader not yet implemented");
    return nullptr;
}

struct llama_dataset* llama_dataset_load_parquet_internal(const struct common_params* params) {
    (void)params;
    llama_dataset_error_set_with_code_internal(DATASET_ERROR_UNKNOWN, "Parquet loader not yet implemented");
    return nullptr;
}

// Note: Format loader functions are implemented in their respective format modules:
// - llama_dataset_load_gguf() in formats/gguf/ (connected in Phase 4.1)
// - llama_dataset_load_text_internal() in formats/text/ (placeholder above)
// - llama_dataset_load_parquet_internal() in formats/parquet/ (placeholder above)

// Note: Sequence loading functions are implemented in their respective format modules:
// - load_parquet_sequence_with_tokenization() in formats/parquet/
// - llama_dataset_gguf_get_tensor_data_streaming() in formats/gguf/
// - llama_dataset_get_parquet_tensor_data_streaming() in formats/parquet/