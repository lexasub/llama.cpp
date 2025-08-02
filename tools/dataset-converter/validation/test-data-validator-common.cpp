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

bool file_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

bool directory_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

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

uint64_t get_file_size(const char* path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        return static_cast<uint64_t>(st.st_size);
    }
    return 0;
}

// Note: Permission checking and validation result string functions
// have been moved to test-data-validator-core.cpp