/**
 * @file platform-compat.cpp
 * @brief Cross-platform compatibility layer implementation for Windows/Unix systems
 * 
 * This module provides the implementation of cross-platform functionality to ensure
 * the dataset-converter module builds and runs correctly on both Windows and Unix-like
 * systems. It abstracts away platform-specific differences in file system operations,
 * process management, memory monitoring, and system resource access.
 * 
 * The module implements a comprehensive platform abstraction layer that includes:
 * - File system operations (directory creation, file existence checks, path handling)
 * - Process management (process creation, termination, waiting)
 * - Memory monitoring and resource usage tracking
 * - High-resolution timing and sleep operations
 * - Directory traversal with unified interface
 * - Thread and synchronization primitives abstraction
 * 
 * Platform-specific implementations are conditionally compiled based on preprocessor
 * macros (_WIN32 for Windows, otherwise Unix/Linux). The module ensures consistent
 * behavior across platforms while leveraging native system APIs for optimal performance.
 * 
 * Key design principles:
 * - Unified API surface regardless of underlying platform
 * - Efficient use of native system calls and APIs
 * - Proper error handling and resource cleanup
 * - Thread-safe operations where applicable
 * - Minimal overhead abstraction layer
 * 
 * Memory management considerations:
 * - All allocated resources are properly cleaned up
 * - Platform-specific handles are managed correctly
 * - Error conditions are handled gracefully with appropriate cleanup
 * 
 * Performance characteristics:
 * - Direct mapping to native APIs where possible
 * - Minimal abstraction overhead
 * - Efficient resource usage tracking
 * - Optimized for both single-threaded and multi-threaded usage
 * 
 * @author Dataset Converter Team
 * @version 1.0
 * @since 2024
 */

#include "platform-compat.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>

#ifdef _WIN32
    #include <psapi.h>
    #include <tlhelp32.h>
    #pragma comment(lib, "psapi.lib")
#else
    #include <sys/time.h>
    #include <sys/sysinfo.h>
#endif

// Cross-platform implementations

/**
 * @brief Get current working directory in a cross-platform way
 * 
 * This function provides a unified interface for obtaining the current working
 * directory across different platforms. On Windows, it uses _getcwd(), while
 * on Unix systems it uses the standard getcwd() function.
 * 
 * @param buffer Buffer to store the current working directory path
 * @param size Size of the buffer in bytes
 * @return Pointer to buffer containing the current directory path on success,
 *         NULL on failure (buffer too small, permission denied, etc.)
 * 
 * @note The buffer must be large enough to hold the complete path including
 *       the null terminator. On Windows, the maximum path length is typically
 *       260 characters (MAX_PATH), while Unix systems may support longer paths.
 * 
 * @warning The returned pointer is the same as the input buffer parameter.
 *          Do not attempt to free() the returned pointer.
 */
char* platform_getcwd(char* buffer, size_t size) {
#ifdef _WIN32
    return _getcwd(buffer, static_cast<int>(size));
#else
    return getcwd(buffer, size);
#endif
}

/**
 * @brief Check if a file or directory exists in a cross-platform way
 * 
 * This function provides a unified interface for checking file/directory existence
 * across different platforms. It uses the appropriate access() function variant
 * for each platform to test for file existence without requiring read permissions.
 * 
 * @param path Path to the file or directory to check
 * @return 1 if the file/directory exists and is accessible, 0 otherwise
 * 
 * @note This function only checks for existence, not readability or other permissions.
 *       On Windows, it uses _access() with mode 0 (existence check only).
 *       On Unix systems, it uses access() with F_OK flag.
 * 
 * @warning Returns 0 for NULL path parameter. Does not distinguish between
 *          "file does not exist" and "permission denied" cases.
 */
int platform_file_exists(const char* path) {
    if (!path) return 0;
    
#ifdef _WIN32
    return _access(path, 0) == 0 ? 1 : 0;
#else
    return access(path, F_OK) == 0 ? 1 : 0;
#endif
}

/**
 * @brief Get file size in bytes using cross-platform stat() function
 * 
 * This function retrieves the size of a file in bytes using the standard
 * stat() function, which is available on both Windows and Unix platforms.
 * The function handles both regular files and provides appropriate error
 * handling for various failure conditions.
 * 
 * @param path Path to the file whose size should be determined
 * @return File size in bytes on success, -1 on error (file not found,
 *         permission denied, path is directory, etc.)
 * 
 * @note For directories, the behavior is platform-dependent. Some systems
 *       return the directory size, others may return an error.
 * 
 * @warning Large files (>2GB) are supported through long long return type.
 *          Returns -1 for NULL path parameter or any stat() failure.
 */
long long platform_file_size(const char* path) {
    if (!path) return -1;
    
    struct stat st;
    if (stat(path, &st) != 0) {
        return -1;
    }
    
    return static_cast<long long>(st.st_size);
}

/**
 * @brief Create directory hierarchy recursively in a cross-platform way
 * 
 * This function creates a complete directory path, including all intermediate
 * directories that don't exist. It handles platform-specific path formats
 * and directory creation APIs while providing a unified interface.
 * 
 * Algorithm:
 * 1. Create a working copy of the input path
 * 2. Skip platform-specific root elements (drive letters on Windows, root slash on Unix)
 * 3. Iterate through path components, creating each directory level
 * 4. Handle existing directories gracefully (EEXIST is not an error)
 * 5. Clean up allocated memory on both success and failure paths
 * 
 * @param path Complete directory path to create (can include multiple levels)
 * @return 0 on success (all directories created or already exist), -1 on error
 * 
 * @note On Windows, uses _mkdir() which doesn't take permission parameter.
 *       On Unix, creates directories with 0755 permissions (rwxr-xr-x).
 * 
 * @warning The function modifies a copy of the input path during processing
 *          but leaves the original path unchanged. Memory allocation failure
 *          or permission issues will cause the function to fail.
 */
int platform_mkdir_recursive(const char* path) {
    if (!path) return -1;
    
    char* path_copy = strdup(path);
    if (!path_copy) return -1;
    
    char* p = path_copy;
    
    // Skip root slash on Unix or drive letter on Windows
#ifdef _WIN32
    if (strlen(p) >= 2 && p[1] == ':') {
        p += 2; // Skip "C:"
    }
#endif
    if (*p == PATH_SEPARATOR) {
        p++; // Skip leading slash
    }
    
    while (*p) {
        char* next_sep = strchr(p, PATH_SEPARATOR);
        if (next_sep) {
            *next_sep = '\0';
        }
        
        // Create directory
#ifdef _WIN32
        if (_mkdir(path_copy) != 0 && errno != EEXIST) {
#else
        if (mkdir(path_copy, 0755) != 0 && errno != EEXIST) {
#endif
            free(path_copy);
            return -1;
        }
        
        if (next_sep) {
            *next_sep = PATH_SEPARATOR;
            p = next_sep + 1;
        } else {
            break;
        }
    }
    
    free(path_copy);
    return 0;
}

/**
 * @brief Get system memory information in a cross-platform way
 * 
 * This function retrieves current system memory statistics including total
 * physical memory and currently available memory. It uses platform-specific
 * APIs to provide accurate, real-time memory information.
 * 
 * Platform-specific implementation details:
 * - Windows: Uses GlobalMemoryStatusEx() to get MEMORYSTATUSEX structure
 *   containing detailed memory statistics including physical and virtual memory
 * - Linux/Unix: Uses sysinfo() system call to get system information
 *   including memory statistics with proper unit conversion
 * 
 * @param total_memory Pointer to store total physical memory in bytes
 * @param available_memory Pointer to store currently available memory in bytes
 * @return 0 on success, -1 on error (invalid parameters or system call failure)
 * 
 * @note Available memory represents memory that can be immediately allocated
 *       without swapping. This includes free memory plus memory that can be
 *       quickly reclaimed (buffers, caches on Linux).
 * 
 * @warning Both output parameters must be valid pointers. The function will
 *          fail if either parameter is NULL. Memory values are in bytes and
 *          may be very large on modern systems.
 */
int platform_get_memory_info(size_t* total_memory, size_t* available_memory) {
    if (!total_memory || !available_memory) return -1;
    
#ifdef _WIN32
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    
    if (!GlobalMemoryStatusEx(&mem_status)) {
        return -1;
    }
    
    *total_memory = static_cast<size_t>(mem_status.ullTotalPhys);
    *available_memory = static_cast<size_t>(mem_status.ullAvailPhys);
    
#else
    struct sysinfo info;
    if (sysinfo(&info) != 0) {
        return -1;
    }
    
    *total_memory = static_cast<size_t>(info.totalram * info.mem_unit);
    *available_memory = static_cast<size_t>(info.freeram * info.mem_unit);
#endif
    
    return 0;
}

/**
 * @brief Sleep for specified number of milliseconds in a cross-platform way
 * 
 * This function provides a unified interface for sleeping/pausing execution
 * for a specified duration. It uses the most appropriate sleep function
 * available on each platform.
 * 
 * Platform-specific implementation:
 * - Windows: Uses Sleep() function which takes milliseconds directly
 * - Unix/Linux: Uses usleep() function, converting milliseconds to microseconds
 * 
 * @param milliseconds Number of milliseconds to sleep (must be non-negative)
 * 
 * @note The actual sleep duration may be slightly longer than requested due
 *       to system scheduling and timer resolution. On some systems, very short
 *       sleep durations may be rounded up to the minimum timer resolution.
 * 
 * @warning Negative values are passed through to the underlying system call
 *          and may cause undefined behavior. The function does not validate
 *          the input parameter.
 */
void platform_sleep_ms(int milliseconds) {
#ifdef _WIN32
    Sleep(static_cast<DWORD>(milliseconds));
#else
    usleep(milliseconds * 1000);
#endif
}

/**
 * @brief Get high-resolution timestamp in microseconds
 * 
 * This function provides a high-resolution timestamp suitable for performance
 * measurement and timing operations. It uses the best available timing source
 * on each platform to provide microsecond precision.
 * 
 * Platform-specific implementation:
 * - Windows: Uses QueryPerformanceCounter() with QueryPerformanceFrequency()
 *   to get the highest resolution timer available. Converts performance counter
 *   ticks to microseconds using the system frequency.
 * - Unix/Linux: Uses gettimeofday() to get current time with microsecond precision
 *   based on system clock.
 * 
 * @return Current timestamp in microseconds since an arbitrary epoch
 *         (typically system boot or Unix epoch depending on platform)
 * 
 * @note The timestamp is monotonic on Windows (performance counter) but may
 *       be affected by system clock adjustments on Unix systems. For relative
 *       timing measurements, the absolute value is not important.
 * 
 * @warning On Windows, the function assumes QueryPerformanceFrequency() succeeds.
 *          Division by zero is theoretically possible but extremely unlikely on
 *          modern systems.
 */
long long platform_get_timestamp_us(void) {
#ifdef _WIN32
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    
    return (counter.QuadPart * 1000000LL) / frequency.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<long long>(tv.tv_sec) * 1000000LL + tv.tv_usec;
#endif
}

#ifdef _WIN32
// Windows-specific implementations

/**
 * @brief Terminate a process by PID on Windows
 * 
 * This function provides a Windows implementation of process termination
 * that mimics the Unix kill() system call behavior. It opens a handle to
 * the target process and terminates it using the Windows API.
 * 
 * Implementation details:
 * 1. Opens process handle with PROCESS_TERMINATE access rights
 * 2. Calls TerminateProcess() with the signal value as exit code
 * 3. Properly closes the process handle to avoid resource leaks
 * 
 * @param pid Process ID of the target process to terminate
 * @param signal Signal/exit code to use for termination (used as exit code)
 * @return 0 on success, -1 on failure (process not found, access denied, etc.)
 * 
 * @note Unlike Unix signals, Windows process termination is immediate and
 *       forceful. The signal parameter is used as the process exit code.
 * 
 * @warning This function performs forceful process termination without
 *          allowing the target process to clean up resources. Use with caution.
 */
int platform_kill_process(platform_pid_t pid, int signal) {
    HANDLE process = OpenProcess(PROCESS_TERMINATE, FALSE, pid);
    if (!process) {
        return -1;
    }
    
    BOOL result = TerminateProcess(process, static_cast<UINT>(signal));
    CloseHandle(process);
    
    return result ? 0 : -1;
}

/**
 * @brief Get current process ID on Windows
 * 
 * This function returns the process identifier (PID) of the current process
 * using the Windows GetCurrentProcessId() API function.
 * 
 * @return Current process ID as a DWORD value
 * 
 * @note On Windows, process IDs are DWORD values and are guaranteed to be
 *       unique system-wide at any given time, though they may be reused
 *       after a process terminates.
 */
platform_pid_t platform_get_pid(void) {
    return GetCurrentProcessId();
}

/**
 * @brief Wait for a process to terminate and get its exit status on Windows
 * 
 * This function provides a Windows implementation of process waiting that
 * mimics the Unix waitpid() system call behavior. It waits for the specified
 * process to terminate and optionally retrieves its exit code.
 * 
 * Implementation details:
 * 1. Opens process handle with SYNCHRONIZE and PROCESS_QUERY_INFORMATION rights
 * 2. Uses WaitForSingleObject() to wait indefinitely for process termination
 * 3. Retrieves exit code using GetExitCodeProcess() if status pointer provided
 * 4. Properly closes process handle to avoid resource leaks
 * 
 * @param pid Process ID of the process to wait for
 * @param status Pointer to store the process exit code (can be NULL)
 * @return Process ID on success, -1 on failure (process not found, access denied, etc.)
 * 
 * @note This function waits indefinitely for the process to terminate.
 *       If the process is already terminated, it returns immediately.
 * 
 * @warning The function will block indefinitely if the target process never
 *          terminates. Consider using a timeout mechanism for production code.
 */
int platform_wait_process(platform_pid_t pid, int* status) {
    HANDLE process = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_INFORMATION, FALSE, pid);
    if (!process) {
        return -1;
    }
    
    DWORD wait_result = WaitForSingleObject(process, INFINITE);
    if (wait_result != WAIT_OBJECT_0) {
        CloseHandle(process);
        return -1;
    }
    
    if (status) {
        DWORD exit_code;
        if (GetExitCodeProcess(process, &exit_code)) {
            *status = static_cast<int>(exit_code);
        } else {
            *status = -1;
        }
    }
    
    CloseHandle(process);
    return pid;
}

/**
 * @brief Get resource usage information on Windows (equivalent to getrusage)
 * 
 * This function provides a Windows implementation of resource usage tracking
 * that mimics the Unix getrusage() system call. It collects memory and CPU
 * usage statistics for the current process using Windows-specific APIs.
 * 
 * Implementation details:
 * 1. Uses GetProcessMemoryInfo() to get memory statistics including peak working set
 * 2. Uses GetProcessTimes() to get CPU time information in FILETIME format
 * 3. Converts FILETIME (100-nanosecond intervals) to seconds and microseconds
 * 4. Handles API failures gracefully by setting appropriate default values
 * 
 * Resource information collected:
 * - ru_maxrss: Peak working set size in kilobytes
 * - ru_utime_sec/ru_utime_usec: User CPU time in seconds and microseconds
 * - ru_stime_sec/ru_stime_usec: System CPU time in seconds and microseconds
 * 
 * @param who Resource usage target (ignored on Windows, always current process)
 * @param usage Pointer to platform_rusage structure to fill with resource data
 * @return 0 on success, -1 on failure (invalid usage pointer)
 * 
 * @note The 'who' parameter is ignored on Windows as the implementation only
 *       supports current process resource usage (equivalent to RUSAGE_SELF).
 * 
 * @warning Memory usage is converted from bytes to kilobytes to match Unix
 *          getrusage() behavior. CPU times are high-precision but may have
 *          platform-specific accuracy limitations.
 */
int platform_getrusage(int who, struct platform_rusage* usage) {
    if (!usage) return -1;
    
    HANDLE process = GetCurrentProcess();
    PROCESS_MEMORY_COUNTERS pmc;
    FILETIME creation_time, exit_time, kernel_time, user_time;
    
    // Get memory information
    if (GetProcessMemoryInfo(process, &pmc, sizeof(pmc))) {
        usage->ru_maxrss = static_cast<long>(pmc.PeakWorkingSetSize / 1024); // Convert to KB
    } else {
        usage->ru_maxrss = 0;
    }
    
    // Get CPU time information
    if (GetProcessTimes(process, &creation_time, &exit_time, &kernel_time, &user_time)) {
        // Convert FILETIME to seconds and microseconds
        ULARGE_INTEGER user_time_64, kernel_time_64;
        user_time_64.LowPart = user_time.dwLowDateTime;
        user_time_64.HighPart = user_time.dwHighDateTime;
        kernel_time_64.LowPart = kernel_time.dwLowDateTime;
        kernel_time_64.HighPart = kernel_time.dwHighDateTime;
        
        // FILETIME is in 100-nanosecond intervals
        long long user_us = user_time_64.QuadPart / 10;
        long long kernel_us = kernel_time_64.QuadPart / 10;
        
        usage->ru_utime_sec = static_cast<long>(user_us / 1000000);
        usage->ru_utime_usec = static_cast<long>(user_us % 1000000);
        usage->ru_stime_sec = static_cast<long>(kernel_us / 1000000);
        usage->ru_stime_usec = static_cast<long>(kernel_us % 1000000);
    } else {
        usage->ru_utime_sec = usage->ru_utime_usec = 0;
        usage->ru_stime_sec = usage->ru_stime_usec = 0;
    }
    
    return 0;
}

/**
 * @brief Open directory for reading on Windows (equivalent to opendir)
 * 
 * This function provides a Windows implementation of directory opening that
 * mimics the Unix opendir() function behavior. It uses the Windows FindFirstFile
 * API to begin directory enumeration and maintains state for subsequent reads.
 * 
 * Implementation details:
 * 1. Allocates platform_DIR structure to maintain enumeration state
 * 2. Constructs search pattern by appending "\\*" to the directory path
 * 3. Calls FindFirstFileA() to begin enumeration and get first entry
 * 4. Sets up state tracking for subsequent platform_readdir() calls
 * 5. Handles allocation and API failures with proper cleanup
 * 
 * @param path Directory path to open for reading
 * @return Pointer to platform_DIR structure on success, NULL on failure
 *         (path not found, access denied, memory allocation failure, etc.)
 * 
 * @note The returned platform_DIR pointer must be closed with platform_closedir()
 *       to avoid resource leaks. The function allocates memory that must be freed.
 * 
 * @warning The path parameter must not exceed MAX_PATH length on Windows.
 *          The function does not validate path length and may fail silently
 *          or cause buffer overflow for very long paths.
 */
platform_DIR* platform_opendir(const char* path) {
    if (!path) return nullptr;
    
    platform_DIR* dir = static_cast<platform_DIR*>(malloc(sizeof(platform_DIR)));
    if (!dir) return nullptr;
    
    char search_path[MAX_PATH];
    snprintf(search_path, sizeof(search_path), "%s\\*", path);
    
    dir->handle = FindFirstFileA(search_path, &dir->find_data);
    if (dir->handle == INVALID_HANDLE_VALUE) {
        free(dir);
        return nullptr;
    }
    
    dir->first_call = 1;
    return dir;
}

/**
 * @brief Read next directory entry on Windows (equivalent to readdir)
 * 
 * This function provides a Windows implementation of directory entry reading
 * that mimics the Unix readdir() function behavior. It uses the Windows
 * FindNextFile API to enumerate directory contents sequentially.
 * 
 * Implementation details:
 * 1. Handles first call specially (FindFirstFile already retrieved first entry)
 * 2. For subsequent calls, uses FindNextFileA() to get next directory entry
 * 3. Copies filename from WIN32_FIND_DATA to platform_dirent structure
 * 4. Ensures null termination of filename string for safety
 * 5. Returns pointer to internal dirent structure (valid until next call)
 * 
 * @param dir Pointer to platform_DIR structure from platform_opendir()
 * @return Pointer to platform_dirent structure containing entry information,
 *         NULL when no more entries or on error
 * 
 * @note The returned pointer is valid only until the next call to platform_readdir()
 *       or platform_closedir(). The caller should not free the returned pointer.
 * 
 * @warning The function returns NULL both for end-of-directory and error conditions.
 *          Use GetLastError() to distinguish between these cases if needed.
 */
struct platform_dirent* platform_readdir(platform_DIR* dir) {
    if (!dir || dir->handle == INVALID_HANDLE_VALUE) {
        return nullptr;
    }
    
    if (dir->first_call) {
        dir->first_call = 0;
    } else {
        if (!FindNextFileA(dir->handle, &dir->find_data)) {
            return nullptr;
        }
    }
    
    strncpy(dir->entry.d_name, dir->find_data.cFileName, sizeof(dir->entry.d_name) - 1);
    dir->entry.d_name[sizeof(dir->entry.d_name) - 1] = '\0';
    
    return &dir->entry;
}

/**
 * @brief Close directory handle on Windows (equivalent to closedir)
 * 
 * This function provides a Windows implementation of directory closing that
 * mimics the Unix closedir() function behavior. It properly releases the
 * Windows FindFirstFile handle and frees allocated memory.
 * 
 * Implementation details:
 * 1. Validates the directory pointer parameter
 * 2. Closes the Windows FindFirstFile handle using FindClose()
 * 3. Frees the allocated platform_DIR structure memory
 * 4. Handles invalid handles gracefully (INVALID_HANDLE_VALUE)
 * 
 * @param dir Pointer to platform_DIR structure to close
 * @return 0 on success, -1 if dir parameter is NULL
 * 
 * @note This function should be called for every successful platform_opendir()
 *       call to avoid resource leaks. It's safe to call with NULL pointer.
 * 
 * @warning After calling this function, the dir pointer becomes invalid and
 *          should not be used. The function frees the memory pointed to by dir.
 */
int platform_closedir(platform_DIR* dir) {
    if (!dir) return -1;
    
    if (dir->handle != INVALID_HANDLE_VALUE) {
        FindClose(dir->handle);
    }
    
    free(dir);
    return 0;
}

#endif // _WIN32
