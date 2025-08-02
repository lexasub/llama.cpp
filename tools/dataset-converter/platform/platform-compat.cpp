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

char* platform_getcwd(char* buffer, size_t size) {
#ifdef _WIN32
    return _getcwd(buffer, static_cast<int>(size));
#else
    return getcwd(buffer, size);
#endif
}

int platform_file_exists(const char* path) {
    if (!path) return 0;
    
#ifdef _WIN32
    return _access(path, 0) == 0 ? 1 : 0;
#else
    return access(path, F_OK) == 0 ? 1 : 0;
#endif
}

long long platform_file_size(const char* path) {
    if (!path) return -1;
    
    struct stat st;
    if (stat(path, &st) != 0) {
        return -1;
    }
    
    return static_cast<long long>(st.st_size);
}

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

void platform_sleep_ms(int milliseconds) {
#ifdef _WIN32
    Sleep(static_cast<DWORD>(milliseconds));
#else
    usleep(milliseconds * 1000);
#endif
}

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

int platform_kill_process(platform_pid_t pid, int signal) {
    HANDLE process = OpenProcess(PROCESS_TERMINATE, FALSE, pid);
    if (!process) {
        return -1;
    }
    
    BOOL result = TerminateProcess(process, static_cast<UINT>(signal));
    CloseHandle(process);
    
    return result ? 0 : -1;
}

platform_pid_t platform_get_pid(void) {
    return GetCurrentProcessId();
}

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

int platform_closedir(platform_DIR* dir) {
    if (!dir) return -1;
    
    if (dir->handle != INVALID_HANDLE_VALUE) {
        FindClose(dir->handle);
    }
    
    free(dir);
    return 0;
}

#endif // _WIN32