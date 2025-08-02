#pragma once

/**
 * @file platform-compat.h
 * @brief Cross-platform compatibility layer for Windows/Unix systems
 * 
 * This header provides compatibility macros and function declarations
 * to ensure the dataset-converter module builds on both Windows and Unix systems.
 */

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <io.h>
    #include <direct.h>
    #include <process.h>
    #include <sys/stat.h>
    
    // Windows-specific definitions
    #define PLATFORM_WINDOWS 1
    #define PATH_SEPARATOR '\\'
    #define PATH_SEPARATOR_STR "\\"
    
    // POSIX compatibility macros for Windows
    #define access _access
    #define mkdir(path, mode) _mkdir(path)
    #define rmdir _rmdir
    #define unlink _unlink
    #define getcwd _getcwd
    #define chdir _chdir
    #define stat _stat
    #define fstat _fstat
    
    // Process/thread compatibility
    typedef HANDLE platform_thread_t;
    typedef CRITICAL_SECTION platform_mutex_t;
    typedef DWORD platform_pid_t;
    
    // Thread-local storage
    #define THREAD_LOCAL __declspec(thread)
    
    // Function declarations for Windows compatibility
    int platform_kill_process(platform_pid_t pid, int signal);
    platform_pid_t platform_get_pid(void);
    int platform_wait_process(platform_pid_t pid, int* status);
    
    // Memory resource monitoring (Windows equivalent of getrusage)
    struct platform_rusage {
        long ru_maxrss;     // Maximum resident set size
        long ru_utime_sec;  // User CPU time (seconds)
        long ru_utime_usec; // User CPU time (microseconds)
        long ru_stime_sec;  // System CPU time (seconds)
        long ru_stime_usec; // System CPU time (microseconds)
    };
    
    int platform_getrusage(int who, struct platform_rusage* usage);
    
    // Directory traversal
    struct platform_dirent {
        char d_name[256];
    };
    
    typedef struct platform_dir {
        HANDLE handle;
        WIN32_FIND_DATAA find_data;
        struct platform_dirent entry;
        int first_call;
    } platform_DIR;
    
    platform_DIR* platform_opendir(const char* path);
    struct platform_dirent* platform_readdir(platform_DIR* dir);
    int platform_closedir(platform_DIR* dir);
    
#else
    // Unix/Linux systems
    #include <unistd.h>
    #include <dirent.h>
    #include <sys/wait.h>
    #include <sys/resource.h>
    #include <sys/stat.h>
    #include <signal.h>
    #include <pthread.h>
    
    #define PLATFORM_UNIX 1
    #define PATH_SEPARATOR '/'
    #define PATH_SEPARATOR_STR "/"
    
    // Unix compatibility types
    typedef pthread_t platform_thread_t;
    typedef pthread_mutex_t platform_mutex_t;
    typedef pid_t platform_pid_t;
    
    // Thread-local storage
    #define THREAD_LOCAL __thread
    
    // Direct mappings for Unix
    #define platform_kill_process(pid, sig) kill(pid, sig)
    #define platform_get_pid() getpid()
    #define platform_wait_process(pid, status) waitpid(pid, status, 0)
    
    // Resource usage
    typedef struct rusage platform_rusage;
    #define platform_getrusage(who, usage) getrusage(who, usage)
    
    // Directory operations
    typedef struct dirent platform_dirent;
    typedef DIR platform_DIR;
    #define platform_opendir(path) opendir(path)
    #define platform_readdir(dir) readdir(dir)
    #define platform_closedir(dir) closedir(dir)
    
#endif

// Common cross-platform utilities
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get current working directory in a cross-platform way
 * @param buffer Buffer to store the path
 * @param size Size of the buffer
 * @return Pointer to buffer on success, NULL on failure
 */
char* platform_getcwd(char* buffer, size_t size);

/**
 * @brief Check if a file exists in a cross-platform way
 * @param path Path to check
 * @return 1 if file exists, 0 otherwise
 */
int platform_file_exists(const char* path);

/**
 * @brief Get file size in a cross-platform way
 * @param path Path to file
 * @return File size in bytes, -1 on error
 */
long long platform_file_size(const char* path);

/**
 * @brief Create directory recursively in a cross-platform way
 * @param path Directory path to create
 * @return 0 on success, -1 on error
 */
int platform_mkdir_recursive(const char* path);

/**
 * @brief Get system memory information
 * @param total_memory Pointer to store total system memory
 * @param available_memory Pointer to store available memory
 * @return 0 on success, -1 on error
 */
int platform_get_memory_info(size_t* total_memory, size_t* available_memory);

/**
 * @brief Sleep for specified milliseconds
 * @param milliseconds Number of milliseconds to sleep
 */
void platform_sleep_ms(int milliseconds);

/**
 * @brief Get high-resolution timestamp in microseconds
 * @return Timestamp in microseconds
 */
long long platform_get_timestamp_us(void);

#ifdef __cplusplus
}
#endif