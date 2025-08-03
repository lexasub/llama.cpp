#pragma once

/**
 * @file llama-dataset-internal.h
 * @brief Internal dataset structures and private interfaces for the dataset converter framework.
 *
 * This header defines the internal implementation details of the dataset converter framework,
 * including private data structures, internal function declarations, and implementation-specific
 * interfaces that are not exposed through the public API. It serves as the bridge between the
 * public interface and the modular implementation architecture.
 *
 * ## Architecture Role
 *
 * This module plays a central role in the dataset converter architecture by:
 * - **Structure Definition**: Defining the internal llama_dataset structure with all private members
 * - **Format Abstraction**: Providing unified internal interfaces for different format implementations
 * - **Resource Management**: Managing internal resources, contexts, and format-specific data
 * - **Streaming Integration**: Integrating with streaming cache and optimization subsystems
 * - **Cross-Module Communication**: Facilitating communication between core, format, and streaming modules
 *
 * ## Internal Data Structures
 *
 * The core `llama_dataset` structure contains:
 * - **Universal GGUF Context**: All formats are internally represented as GGUF for consistency
 * - **GGML Integration**: Direct tensor access and memory management through GGML contexts
 * - **Format-Specific Data**: Opaque pointers to format-specific state and configuration
 * - **Streaming Infrastructure**: Cache management and optimization components
 * - **Tokenization Support**: Model and context management for text processing
 *
 * ## Memory Management Strategy
 *
 * The internal structure implements a sophisticated memory management strategy:
 * - **Lazy Loading**: Tensors are loaded on-demand in streaming mode
 * - **Reference Counting**: Shared resources use reference counting for safe cleanup
 * - **Cache Management**: LRU cache with adaptive sizing based on memory pressure
 * - **Format Isolation**: Format-specific resources are isolated and managed independently
 * - **RAII Principles**: C++ components follow RAII for automatic resource management
 *
 * ## Streaming Architecture Integration
 *
 * The internal structure integrates deeply with the streaming subsystem:
 * - **Cache Coordination**: Direct integration with streaming cache for optimal performance
 * - **Optimization Management**: Embedded optimization manager for adaptive performance tuning
 * - **Memory Monitoring**: Integration with memory pressure monitoring for adaptive behavior
 * - **Read-Ahead Buffering**: Support for predictive loading based on access patterns
 *
 * ## Format-Specific Implementation Details
 *
 * ### GGUF Format
 * - Direct GGUF context usage with minimal overhead
 * - Native tensor access through cached tensor pointers
 * - Metadata preserved from original GGUF structure
 * - Streaming support through selective tensor loading
 *
 * ### Text Format
 * - Tokenization through integrated llama model and context
 * - Dynamic sequence generation and caching
 * - Memory-efficient token storage and retrieval
 * - Streaming tokenization for large text files
 *
 * ### Parquet Format
 * - Apache Arrow integration through format_data pointer
 * - Schema analysis and column type detection
 * - Batch processing with configurable chunk sizes
 * - Streaming support through Arrow's streaming interfaces
 *
 * ## Thread Safety Considerations
 *
 * The internal structure is designed with thread safety in mind:
 * - **Read-Only Access**: Multiple threads can safely read from the same dataset
 * - **Cache Synchronization**: Streaming cache uses appropriate locking mechanisms
 * - **Format Isolation**: Format-specific data is isolated to prevent cross-contamination
 * - **Context Management**: GGML and tokenizer contexts are thread-local when needed
 *
 * ## Error Handling and Diagnostics
 *
 * Internal error handling provides detailed diagnostics:
 * - **Error State Tracking**: Each component maintains detailed error state
 * - **Resource Cleanup**: Automatic cleanup on error conditions
 * - **Diagnostic Information**: Rich error context for debugging and troubleshooting
 * - **Validation Integration**: Deep integration with validation subsystem
 *
 * ## Performance Optimization
 *
 * The internal structure enables various performance optimizations:
 * - **Cache Locality**: Data structures optimized for cache-friendly access patterns
 * - **Memory Pooling**: Efficient memory allocation through pooling strategies
 * - **Vectorization**: Support for SIMD operations where applicable
 * - **Adaptive Algorithms**: Dynamic optimization based on usage patterns
 *
 * ## Usage Guidelines
 *
 * This header should only be included by:
 * - Core implementation files (llama-dataset.cpp, llama-dataset-sequence.cpp)
 * - Format implementation files (gguf/*.cpp, text/*.cpp, parquet/*.cpp)
 * - Streaming implementation files (streaming/*.cpp)
 * - Validation implementation files (validation/*.cpp)
 * - Internal utility implementations (llama-dataset-utils.cpp)
 *
 * **Important**: This header must never be included by:
 * - Public header files
 * - External applications or libraries
 * - Test files that should use the public API
 * - Tools that should use the public interface
 *
 * ## Integration Points
 *
 * This module integrates with:
 * - **streaming/**: Provides cache and optimization manager instances
 * - **formats/**: Receives format-specific data and implementations
 * - **validation/**: Provides internal structure access for validation
 * - **platform/**: Uses platform-specific memory and threading primitives
 * - **core/**: Implements the public API defined in llama-dataset.h
 *
 * ## Future Extensibility
 *
 * The internal structure is designed for future extensibility:
 * - **Plugin Architecture**: Support for dynamically loaded format plugins
 * - **Custom Optimizations**: Extensible optimization manager interface
 * - **Advanced Caching**: Support for distributed and persistent caching
 * - **Hardware Acceleration**: Integration points for GPU and specialized hardware
 *
 * @warning This header contains implementation details that may change between versions.
 *          External code should never depend on these internal structures.
 *
 * @see llama-dataset.h for the public API interface
 * @see streaming/streaming-cache.h for streaming cache implementation
 * @see formats/ directory for format-specific implementations
 * @see validation/llama-dataset-validation.h for validation integration
 *
 * @version 1.0
 * @since 2024
 */

#include "llama-dataset.h"
#include "ggml/include/gguf.h"
#include "ggml/include/ggml.h"
#include "llama.h"

#ifdef __cplusplus
#include <string>
#endif

// Forward declarations for Windows compatibility
#ifdef __cplusplus
class llama_dataset_streaming_cache;
extern "C" {
#endif

/**
 * @brief Core internal data structure for unified dataset representation.
 *
 * This structure serves as the central data container for all dataset formats within
 * the converter framework. It provides a unified internal representation that abstracts
 * format-specific details while maintaining optimal performance and memory efficiency.
 *
 * ## Design Philosophy
 *
 * The structure follows a hybrid approach combining:
 * - **Universal Representation**: All formats are internally converted to GGUF for consistency
 * - **Format Preservation**: Original format-specific data is preserved for fidelity
 * - **Streaming Optimization**: Designed for efficient streaming and caching operations
 * - **Resource Management**: Comprehensive resource tracking and automatic cleanup
 *
 * ## Memory Layout and Access Patterns
 *
 * The structure is optimized for common access patterns:
 * - **Sequential Access**: Cached tensors array provides O(1) sequence access
 * - **Random Access**: Streaming cache optimizes for random access patterns
 * - **Memory Locality**: Related data is grouped for cache-friendly access
 * - **Lazy Loading**: Resources are loaded on-demand to minimize memory footprint
 *
 * ## Lifecycle Management
 *
 * The structure lifecycle follows these phases:
 * 1. **Initialization**: Basic structure setup with format detection
 * 2. **Loading**: Format-specific loading with metadata extraction
 * 3. **Optimization**: Cache setup and streaming configuration
 * 4. **Active Use**: Sequence access with dynamic optimization
 * 5. **Cleanup**: Automatic resource deallocation and context cleanup
 *
 * ## Thread Safety Model
 *
 * The structure supports concurrent access with these guarantees:
 * - **Read Operations**: Multiple threads can safely read simultaneously
 * - **Cache Operations**: Streaming cache handles concurrent access internally
 * - **Format Data**: Format-specific data access is protected by format modules
 * - **Context Safety**: GGML and tokenizer contexts are thread-safe for read operations
 *
 * ## Error Recovery and Resilience
 *
 * The structure includes comprehensive error recovery mechanisms:
 * - **Partial Failure Handling**: Individual sequence failures don't affect the entire dataset
 * - **Resource Cleanup**: Automatic cleanup on error conditions prevents resource leaks
 * - **State Validation**: Internal consistency checks detect and handle corruption
 * - **Graceful Degradation**: Fallback mechanisms maintain functionality under adverse conditions
 */
struct llama_dataset {
    // Core GGUF Infrastructure
    // All formats are internally represented as GGUF for consistency and interoperability
    struct gguf_context * ctx;              ///< Universal GGUF representation (never NULL after successful load)
    struct ggml_context * ggml_ctx;         ///< GGML tensor context (NULL in streaming mode for memory efficiency)
    struct ggml_tensor ** cached_tensors;   ///< Fast O(1) sequence access cache (array of n_seq pointers)
    uint64_t n_seq;                         ///< Total number of sequences (cached for performance)
    
    // Format and Configuration
    enum dataset_type type;                 ///< Original format type (GGUF, text, Parquet)
    bool streaming;                         ///< Streaming mode flag (affects memory management strategy)
    void * format_data;                     ///< Format-specific state and configuration (opaque pointer)
    
    // Tokenization Infrastructure
    // Used for text format and any format requiring tokenization services
    struct llama_model * model;             ///< Llama model for tokenization (shared or owned)
    struct llama_context * tokenizer_ctx;  ///< Dedicated tokenizer context (optimized for batch processing)
    bool owns_model;                        ///< Resource ownership flag (determines cleanup responsibility)
    
    // Streaming and Optimization Infrastructure
    // C++ components for advanced streaming and optimization capabilities
#ifdef __cplusplus
    llama_dataset_streaming_cache * streaming_cache;       ///< LRU cache with adaptive sizing (NULL if streaming disabled)
    void * optimization_manager;            ///< Streaming optimization manager (handles adaptive algorithms)
    std::string column;                     ///< Active column name for Parquet datasets (empty for other formats)
#else
    void * streaming_cache;                 ///< Opaque pointer for C compatibility (cast to streaming_cache in C++)
    void * optimization_manager;            ///< Opaque pointer for optimization manager (cast in C++)
#endif
};

#ifdef __cplusplus
}
#endif
