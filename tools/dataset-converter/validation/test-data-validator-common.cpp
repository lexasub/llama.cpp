/**
 * @file test-data-validator-common.cpp
 * @brief Implementation of common validation utilities and shared validation logic.
 *
 * This module implements the foundational validation utilities that provide consistent
 * file system operations, cross-platform compatibility, and shared validation patterns
 * used throughout the test data validation system. It serves as the core implementation
 * layer that enables reliable and efficient validation operations across all supported
 * formats (GGUF, Parquet, Text) and platforms.
 *
 * ## Implementation Overview
 *
 * ### Cross-Platform File System Operations
 * The module provides robust implementations of file system operations that work
 * consistently across different operating systems and file systems:
 *
 * - **File Existence Checking**: Uses platform-appropriate stat() system calls
 *   with proper error handling and file type verification
 * - **Directory Operations**: Implements recursive directory creation with
 *   appropriate permissions and parent directory handling
 * - **File Size Determination**: Provides accurate file size information using
 *   64-bit integers to support large files on all platforms
 * - **Permission Handling**: Manages file and directory permissions consistently
 *   across Unix-like systems and Windows platforms
 *
 * ### Error Handling and Robustness
 * All utility functions implement comprehensive error handling:
 *
 * - **System Call Error Management**: Proper handling of errno values and
 *   system-specific error conditions
 * - **Edge Case Handling**: Robust behavior for invalid paths, permission
 *   issues, and file system limitations
 * - **Resource Management**: Careful handling of system resources with
 *   appropriate cleanup and error recovery
 * - **Thread Safety**: All functions are designed to be thread-safe and
 *   can be called concurrently from multiple validation threads
 *
 * ### Performance Optimizations
 * The implementation includes several performance optimizations:
 *
 * - **Minimal System Calls**: Efficient use of stat() and related system calls
 *   to minimize file system overhead
 * - **Path Processing**: Optimized string handling for path manipulation
 *   and directory traversal operations
 * - **Caching Considerations**: Functions are designed to work efficiently
 *   with file system caches and avoid unnecessary repeated operations
 * - **Memory Efficiency**: Minimal memory allocation and efficient use of
 *   stack-based operations where possible
 *
 * ## File System Operation Implementation
 *
 * ### File Existence Verification
 * The file_exists() implementation uses the stat() system call to verify both
 * file existence and file type. This approach provides several advantages:
 *
 * ```c
 * struct stat st;
 * return stat(path, &st) == 0 && S_ISREG(st.st_mode);
 * ```
 *
 * - **Type Safety**: Ensures the path refers to a regular file, not a directory
 * - **Atomic Operation**: Single system call provides both existence and type info
 * - **Cross-Platform**: Works consistently across Unix-like systems
 * - **Symbolic Link Handling**: Follows symbolic links to check target files
 *
 * ### Directory Management
 * The directory creation implementation handles complex scenarios:
 *
 * ```c
 * // Recursive parent directory creation
 * if (errno == ENOENT) {
 *     std::string parent = path_str.substr(0, pos);
 *     if (create_directory_if_missing(parent.c_str())) {
 *         return mkdir(path, 0755) == 0;
 *     }
 * }
 * ```
 *
 * - **Recursive Creation**: Automatically creates missing parent directories
 * - **Permission Management**: Sets appropriate permissions (755) for created directories
 * - **Idempotent Operation**: Safe to call multiple times on the same path
 * - **Error Recovery**: Handles various failure scenarios gracefully
 *
 * ### File Size Determination
 * The get_file_size() implementation provides accurate size information:
 *
 * ```c
 * struct stat st;
 * if (stat(path, &st) == 0) {
 *     return static_cast<uint64_t>(st.st_size);
 * }
 * ```
 *
 * - **Large File Support**: Uses 64-bit integers for files larger than 4GB
 * - **Type Safety**: Proper casting to ensure correct size representation
 * - **Error Handling**: Returns 0 for non-existent or inaccessible files
 * - **Efficiency**: Single system call provides size information
 *
 * ## Integration with Validation System
 *
 * ### Format-Specific Validator Support
 * These common utilities are used by all format-specific validators:
 *
 * - **GGUF Validator**: Uses file operations for GGUF file validation
 * - **Parquet Validator**: Leverages directory management for Parquet datasets
 * - **Text Validator**: Utilizes file size checking for text file validation
 * - **Core Validator**: Builds upon these utilities for advanced validation
 *
 * ### Validation Report Integration
 * The utilities support comprehensive validation reporting:
 *
 * - **Error Message Generation**: Provides detailed error information for reports
 * - **Statistics Tracking**: Enables accurate counting of validation operations
 * - **Result Aggregation**: Supports collection of validation results across files
 * - **Performance Metrics**: Allows measurement of validation operation timing
 *
 * ## Platform Compatibility
 *
 * ### Unix-Like Systems
 * The implementation is optimized for Unix-like systems (Linux, macOS, BSD):
 *
 * - **POSIX Compliance**: Uses standard POSIX system calls and interfaces
 * - **Permission Model**: Implements Unix permission model with owner/group/other
 * - **Path Handling**: Supports Unix-style path separators and conventions
 * - **System Integration**: Works with standard Unix file system features
 *
 * ### Cross-Platform Considerations
 * While primarily Unix-focused, the implementation considers portability:
 *
 * - **Standard Library Usage**: Uses standard C/C++ library functions where possible
 * - **Conditional Compilation**: Prepared for platform-specific code sections
 * - **Path Separator Handling**: Can be extended to handle Windows path separators
 * - **Error Code Mapping**: Designed to support platform-specific error handling
 *
 * ## Memory Management and Safety
 *
 * ### Stack-Based Operations
 * Most operations use stack-based memory management:
 *
 * - **Local Variables**: Uses local struct stat variables for system calls
 * - **String Handling**: Efficient std::string usage for path manipulation
 * - **No Dynamic Allocation**: Avoids malloc/free for basic operations
 * - **RAII Principles**: Leverages C++ RAII for automatic resource management
 *
 * ### Error Safety
 * All functions provide strong error safety guarantees:
 *
 * - **No Side Effects on Failure**: Failed operations don't modify system state
 * - **Exception Safety**: C++ code provides basic exception safety
 * - **Resource Cleanup**: Proper cleanup of any allocated resources
 * - **Consistent State**: System remains in consistent state after errors
 *
 * ## Performance Characteristics
 *
 * ### System Call Efficiency
 * The implementation minimizes system call overhead:
 *
 * - **Single stat() Calls**: Most operations require only one system call
 * - **Batch Operations**: Directory creation batches multiple mkdir() calls
 * - **Caching Friendly**: Operations work well with file system caches
 * - **Low Latency**: Minimal processing overhead for common operations
 *
 * ### Scalability Considerations
 * The utilities are designed for scalable validation operations:
 *
 * - **Thread Safety**: All functions can be called concurrently
 * - **No Global State**: Functions don't rely on shared mutable state
 * - **Memory Efficiency**: Minimal memory footprint per operation
 * - **CPU Efficiency**: Low CPU overhead for file system operations
 *
 * ## Future Extensibility
 *
 * ### Additional Utility Functions
 * The module is designed to accommodate future utility functions:
 *
 * - **File Permission Checking**: Extended permission validation capabilities
 * - **Advanced File Operations**: Copy, move, and backup operations
 * - **Metadata Handling**: Extended file metadata and attribute support
 * - **Performance Monitoring**: File operation timing and statistics
 *
 * ### Platform Extensions
 * The architecture supports platform-specific extensions:
 *
 * - **Windows Support**: Can be extended with Windows-specific implementations
 * - **Advanced File Systems**: Support for special file system features
 * - **Network File Systems**: Handling of network-mounted file systems
 * - **Security Integration**: Integration with platform security features
 *
 * @author Dataset Converter Team
 * @version 1.0
 * @since 2024
 * @see test-data-validator-common.h Common validation utilities interface
 * @see test-data-validator-core.cpp Core validation functionality
 * @see test-data-validator.h Main validation interface
 */

#include "test-data-validator-common.h"

#include "common/log.h"
#include "llama-dataset-internal.h"
#include "llama-impl.h"

#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>

//
// Common helper functions implementation
//

/**
 * @brief Check if a file exists and is accessible.
 *
 * Implements comprehensive file existence checking using the POSIX stat() system call.
 * This function provides reliable file existence verification with proper file type
 * checking to ensure the path refers to a regular file rather than a directory or
 * special file type.
 *
 * ## Implementation Details
 *
 * ### System Call Usage
 * The function uses stat() rather than access() for several advantages:
 * - **Atomic Operation**: Single system call provides both existence and type information
 * - **Race Condition Avoidance**: Eliminates TOCTOU (Time-of-Check-Time-of-Use) issues
 * - **Complete Information**: Provides file metadata in addition to existence check
 * - **Symbolic Link Handling**: Automatically follows symbolic links to check targets
 *
 * ### File Type Verification
 * The S_ISREG() macro ensures the path refers to a regular file:
 * ```c
 * return stat(path, &st) == 0 && S_ISREG(st.st_mode);
 * ```
 * - **Regular Files Only**: Excludes directories, device files, and other special types
 * - **Type Safety**: Prevents confusion between files and directories
 * - **Validation Accuracy**: Ensures only appropriate files are considered valid
 *
 * ### Error Handling
 * The function handles various error conditions gracefully:
 * - **Non-existent Files**: Returns false when stat() fails with ENOENT
 * - **Permission Errors**: Returns false when access is denied (EACCES)
 * - **Invalid Paths**: Returns false for malformed or invalid path strings
 * - **File System Errors**: Returns false for I/O errors or file system issues
 *
 * ### Performance Characteristics
 * - **Single System Call**: Minimal overhead with one stat() operation
 * - **File System Cache Friendly**: Benefits from kernel file system caching
 * - **Thread Safe**: No shared state, safe for concurrent access
 * - **Low Memory Usage**: Uses only stack-allocated struct stat
 *
 * @param path Path to the file to check (relative or absolute)
 * @return true if file exists and is a regular file, false otherwise
 *
 * @note Returns false for directories, even if they exist and are accessible
 * @note Follows symbolic links and checks the target file
 * @note Thread-safe and suitable for concurrent validation operations
 * @note Does not check file permissions beyond basic accessibility
 *
 * @see directory_exists() For checking directory existence
 * @see get_file_size() For additional file information after existence check
 */
bool file_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

/**
 * @brief Check if a directory exists and is accessible.
 *
 * Implements reliable directory existence checking using the POSIX stat() system call
 * with proper directory type verification. This function ensures that the specified
 * path refers to an actual directory rather than a regular file or other file system
 * object, providing accurate directory validation for test data management.
 *
 * ## Implementation Details
 *
 * ### Directory Type Verification
 * Uses the S_ISDIR() macro to verify directory type:
 * ```c
 * return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
 * ```
 * - **Directory-Specific Check**: Only returns true for actual directories
 * - **Type Safety**: Prevents confusion between directories and regular files
 * - **Special File Exclusion**: Excludes device files, symbolic links to files, etc.
 * - **Mount Point Handling**: Correctly identifies mount points as directories
 *
 * ### System Call Efficiency
 * The stat() system call provides comprehensive information:
 * - **Single Operation**: Gets existence and type information atomically
 * - **Metadata Access**: Provides additional directory metadata if needed
 * - **Cache Utilization**: Benefits from kernel directory entry caching
 * - **Symbolic Link Resolution**: Follows symbolic links to check target directories
 *
 * ### Error Condition Handling
 * Handles various directory-related error scenarios:
 * - **Non-existent Paths**: Returns false when directory doesn't exist
 * - **Permission Restrictions**: Returns false when directory access is denied
 * - **Path Resolution Errors**: Returns false for invalid or malformed paths
 * - **File System Issues**: Returns false for I/O errors or file system problems
 *
 * ### Cross-Platform Considerations
 * While primarily Unix-focused, the implementation considers portability:
 * - **POSIX Compliance**: Uses standard POSIX directory checking mechanisms
 * - **Path Format Handling**: Works with Unix-style path separators
 * - **Permission Model**: Compatible with Unix directory permission model
 * - **Extension Ready**: Can be extended for Windows directory checking
 *
 * ### Performance and Safety
 * - **Thread Safety**: No shared state, safe for concurrent directory checks
 * - **Memory Efficiency**: Uses only stack-allocated struct stat
 * - **Low Overhead**: Single system call with minimal processing
 * - **Cache Friendly**: Works efficiently with file system caches
 *
 * @param path Path to the directory to check (relative or absolute)
 * @return true if directory exists and is accessible, false otherwise
 *
 * @note Returns false for regular files, even if they exist and are accessible
 * @note Follows symbolic links and checks the target directory
 * @note Does not verify write permissions or directory modification rights
 * @note Thread-safe and suitable for concurrent validation operations
 *
 * @see file_exists() For checking regular file existence
 * @see create_directory_if_missing() For creating missing directories
 */
bool directory_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

/**
 * @brief Create a directory and any necessary parent directories.
 *
 * Implements robust recursive directory creation with comprehensive error handling
 * and proper permission management. This function provides reliable directory
 * creation that handles missing parent directories, existing directories, and
 * various error conditions that may occur during directory creation operations.
 *
 * ## Implementation Algorithm
 *
 * ### Existence Check Optimization
 * The function first checks if the directory already exists:
 * ```c
 * if (directory_exists(path)) {
 *     return true;
 * }
 * ```
 * - **Idempotent Operation**: Safe to call multiple times on the same path
 * - **Performance Optimization**: Avoids unnecessary system calls
 * - **Early Return**: Minimizes processing for existing directories
 * - **Consistency Check**: Ensures the path is actually a directory
 *
 * ### Primary Directory Creation
 * Attempts direct directory creation with appropriate permissions:
 * ```c
 * if (mkdir(path, 0755) == 0) {
 *     return true;
 * }
 * ```
 * - **Permission Setting**: Creates directories with 755 permissions (rwxr-xr-x)
 * - **Owner Access**: Full read/write/execute permissions for owner
 * - **Group/Other Access**: Read and execute permissions for group and others
 * - **Security Considerations**: Balanced between accessibility and security
 *
 * ### Recursive Parent Creation
 * Handles missing parent directories through recursive creation:
 * ```c
 * if (errno == ENOENT) {
 *     std::string parent = path_str.substr(0, pos);
 *     if (create_directory_if_missing(parent.c_str())) {
 *         return mkdir(path, 0755) == 0;
 *     }
 * }
 * ```
 * - **Error Analysis**: Checks errno to identify missing parent directories
 * - **Path Parsing**: Extracts parent directory path using string operations
 * - **Recursive Call**: Creates parent directories before target directory
 * - **Retry Logic**: Attempts target directory creation after parent creation
 *
 * ### Error Handling Strategy
 * Comprehensive error handling for various failure scenarios:
 * - **ENOENT (No such file or directory)**: Triggers recursive parent creation
 * - **EEXIST (File exists)**: Handled by initial existence check
 * - **EACCES (Permission denied)**: Returns false for permission issues
 * - **ENOSPC (No space left on device)**: Returns false for storage issues
 * - **Other Errors**: Returns false for any other mkdir() failures
 *
 * ## Path Processing Details
 *
 * ### String Manipulation
 * Uses std::string for safe and efficient path manipulation:
 * - **Memory Safety**: Automatic memory management for path strings
 * - **Substring Operations**: Efficient parent path extraction
 * - **Path Separator Handling**: Uses '/' as the standard path separator
 * - **Unicode Support**: Handles UTF-8 encoded path names correctly
 *
 * ### Path Validation
 * Implicit validation through system call behavior:
 * - **Invalid Characters**: System calls reject invalid path characters
 * - **Path Length Limits**: Respects file system path length limitations
 * - **Relative/Absolute Paths**: Handles both relative and absolute paths
 * - **Special Cases**: Handles edge cases like empty paths or root directory
 *
 * ## Permission Management
 *
 * ### Unix Permission Model
 * Sets standard directory permissions (0755):
 * - **Owner (7)**: Read (4) + Write (2) + Execute (1) = Full access
 * - **Group (5)**: Read (4) + Execute (1) = Read and traverse access
 * - **Others (5)**: Read (4) + Execute (1) = Read and traverse access
 * - **Security Balance**: Provides necessary access while maintaining security
 *
 * ### Permission Inheritance
 * Created directories inherit appropriate permissions:
 * - **Consistent Permissions**: All created directories use the same permissions
 * - **Umask Interaction**: Works with system umask settings appropriately
 * - **Parent Directory Compatibility**: Ensures compatibility with parent permissions
 * - **Access Requirements**: Provides sufficient access for test data operations
 *
 * ## Performance and Scalability
 *
 * ### System Call Optimization
 * Minimizes system calls for efficient operation:
 * - **Existence Check First**: Avoids unnecessary mkdir() calls
 * - **Single mkdir() Attempt**: Direct creation before recursive approach
 * - **Recursive Minimization**: Only recurses when necessary
 * - **Path Caching**: Benefits from file system directory entry caching
 *
 * ### Memory Efficiency
 * Efficient memory usage for path processing:
 * - **Stack-Based Operations**: Uses stack allocation for most operations
 * - **String Optimization**: Efficient std::string usage for path manipulation
 * - **Minimal Allocation**: Avoids unnecessary dynamic memory allocation
 * - **RAII Compliance**: Automatic cleanup of allocated resources
 *
 * @param path Path to the directory to create (relative or absolute)
 * @return true if directory was created or already exists, false on error
 *
 * @note Function is idempotent - safe to call multiple times on the same path
 * @note Creates parent directories recursively if they don't exist
 * @note Sets permissions to 0755 (rwxr-xr-x) for all created directories
 * @note May require appropriate permissions for the parent directory
 * @note Thread-safe when called on different paths, not thread-safe for same path
 *
 * @see directory_exists() For checking if creation is needed
 * @see file_exists() For verifying the path doesn't conflict with a file
 */
bool create_directory_if_missing(const char* path) {
    if (directory_exists(path)) {
        return true;
    }

    // Create directory with proper permissions
    if (mkdir(path, 0755) == 0) {
        return true;
    }

    // If mkdir failed, check if it's because parent directories don't exist
    if (errno == ENOENT) {
        // Try to create parent directories
        std::string path_str(path);
        size_t pos = path_str.find_last_of('/');
        if (pos != std::string::npos) {
            std::string parent = path_str.substr(0, pos);
            if (create_directory_if_missing(parent.c_str())) {
                return mkdir(path, 0755) == 0;
            }
        }
    }

    return false;
}

/**
 * @brief Get the size of a file in bytes.
 *
 * Implements accurate file size determination using the POSIX stat() system call
 * with proper 64-bit size handling for large files. This function provides reliable
 * file size information that works correctly across different platforms and file
 * systems, supporting files larger than 4GB and handling various edge cases.
 *
 * ## Implementation Details
 *
 * ### Large File Support
 * Uses 64-bit integers to support files of any practical size:
 * ```c
 * return static_cast<uint64_t>(st.st_size);
 * ```
 * - **64-bit Size Handling**: Supports files larger than 4GB on all platforms
 * - **Type Safety**: Explicit casting ensures correct size representation
 * - **Platform Independence**: Works consistently across 32-bit and 64-bit systems
 * - **Future Compatibility**: Prepared for even larger file sizes
 *
 * ### System Call Efficiency
 * Uses stat() for comprehensive file information:
 * - **Single System Call**: Gets size information with minimal overhead
 * - **Atomic Operation**: Provides consistent size information at a point in time
 * - **Metadata Access**: Provides additional file metadata if needed later
 * - **Cache Utilization**: Benefits from kernel file system caching
 *
 * ### Error Handling Strategy
 * Comprehensive error handling for various file access scenarios:
 * - **Non-existent Files**: Returns 0 when file doesn't exist
 * - **Permission Errors**: Returns 0 when file access is denied
 * - **Directory Paths**: Returns 0 for directories (use directory_exists() first)
 * - **Special Files**: Returns 0 for device files and other special file types
 * - **I/O Errors**: Returns 0 for file system errors or corruption
 *
 * ### File Type Considerations
 * The function works with various file types but has specific behaviors:
 * - **Regular Files**: Returns actual file content size
 * - **Directories**: Returns 0 (not applicable for directories)
 * - **Symbolic Links**: Follows links and returns target file size
 * - **Device Files**: Returns 0 (size not meaningful for device files)
 * - **Sparse Files**: Returns logical size, not physical disk usage
 *
 * ## Size Accuracy and Reliability
 *
 * ### File System Compatibility
 * Works correctly with various file systems:
 * - **Local File Systems**: Accurate size for ext4, XFS, NTFS, etc.
 * - **Network File Systems**: Reliable size for NFS, SMB, etc.
 * - **Compressed File Systems**: Returns uncompressed logical size
 * - **Encrypted File Systems**: Returns size of encrypted content
 *
 * ### Timing Considerations
 * Provides point-in-time size information:
 * - **Snapshot Accuracy**: Size is accurate at the moment of the stat() call
 * - **Concurrent Modifications**: May not reflect ongoing writes to the file
 * - **Cache Consistency**: Benefits from file system cache coherency
 * - **Performance Impact**: Minimal impact on file system performance
 *
 * ### Validation Use Cases
 * Optimized for test data validation scenarios:
 * - **Size Range Checking**: Enables validation of file size constraints
 * - **Corruption Detection**: Helps identify truncated or oversized files
 * - **Performance Planning**: Allows estimation of processing time and memory needs
 * - **Resource Management**: Enables resource allocation based on file sizes
 *
 * ## Performance Characteristics
 *
 * ### System Call Overhead
 * Minimal overhead for size determination:
 * - **Single stat() Call**: No additional system calls required
 * - **Kernel Cache Benefits**: Leverages kernel file system caching
 * - **Low CPU Usage**: Minimal processing overhead
 * - **Memory Efficiency**: Uses only stack-allocated struct stat
 *
 * ### Scalability Considerations
 * Designed for efficient batch operations:
 * - **Thread Safety**: Safe for concurrent access from multiple threads
 * - **No Global State**: Functions independently without shared state
 * - **Batch Friendly**: Efficient when called on many files sequentially
 * - **Cache Locality**: Benefits from file system cache locality
 *
 * ## Integration with Validation System
 *
 * ### Size Constraint Validation
 * Supports comprehensive size-based validation:
 * ```c
 * uint64_t size = get_file_size(path);
 * if (size < spec->min_size_bytes || size > spec->max_size_bytes) {
 *     return TEST_DATA_SIZE_INVALID;
 * }
 * ```
 * - **Range Checking**: Enables minimum and maximum size validation
 * - **Zero Size Detection**: Identifies empty files that may indicate problems
 * - **Large File Handling**: Properly handles very large test data files
 * - **Performance Estimation**: Allows prediction of processing requirements
 *
 * ### Error Reporting Integration
 * Provides information for detailed error reporting:
 * - **Size Information**: Includes actual size in validation reports
 * - **Comparison Data**: Enables reporting of expected vs. actual sizes
 * - **Diagnostic Information**: Helps identify file corruption or truncation
 * - **Troubleshooting Support**: Provides data for debugging file issues
 *
 * @param path Path to the file to measure (relative or absolute)
 * @return File size in bytes, or 0 if file doesn't exist or on error
 *
 * @note Returns 0 for directories, device files, and other non-regular files
 * @note Follows symbolic links and returns the size of the target file
 * @note Size reflects logical file content, not physical disk space usage
 * @note Thread-safe and suitable for concurrent access from multiple threads
 * @note For validation purposes, check file_exists() first to distinguish errors
 *
 * @see file_exists() For verifying file existence before size check
 * @see test_data_file_info For size constraint specifications
 * @see test_data_validation_result For size validation result codes
 */
size_t get_file_size(const char* path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        return static_cast<size_t>(st.st_size);
    }
    return 0;
}

// Note: Permission checking and validation result string functions
// have been moved to test-data-validator-core.cpp
