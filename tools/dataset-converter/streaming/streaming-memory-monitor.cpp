#include "streaming-memory-monitor.h"

#include <sys/resource.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <string>
#include <thread>

#include "common/log.h"
#include "llama-impl.h"

llama_dataset_streaming_memory_monitor::llama_dataset_streaming_memory_monitor(
    size_t interval_ms,
    double high_threshold,
    double low_threshold)
    : running(false),
      check_interval_ms(interval_ms),
      high_pressure_threshold(high_threshold),
      low_pressure_threshold(low_threshold) {
}

llama_dataset_streaming_memory_monitor::~llama_dataset_streaming_memory_monitor() {
    stop();
}

void llama_dataset_streaming_memory_monitor::monitor_loop() {
    LLAMA_LOG_DEBUG("Memory monitor thread started\n");

    bool high_pressure_notified = false;
    bool low_pressure_notified = false;

    while (running) {
        // Sleep for the check interval
        {
            std::unique_lock lock(mutex);
            cv.wait_for(lock, std::chrono::milliseconds(check_interval_ms),
                [this] { return !running; });

            if (!running) {
                break;
            }
        }

        // Check memory pressure
        double pressure = get_memory_pressure();

        // Only notify if pressure crosses thresholds
        if (pressure >= high_pressure_threshold && !high_pressure_notified) {
            if (pressure_callback) {
                pressure_callback(pressure);
            }
            high_pressure_notified = true;
            low_pressure_notified = false;
            LLAMA_LOG_DEBUG("High memory pressure detected: %.2f\n", pressure);
        } else if (pressure <= low_pressure_threshold && !low_pressure_notified) {
            if (pressure_callback) {
                pressure_callback(pressure);
            }
            high_pressure_notified = false;
            low_pressure_notified = true;
            LLAMA_LOG_DEBUG("Low memory pressure detected: %.2f\n", pressure);
        }
    }

    LLAMA_LOG_DEBUG("Memory monitor thread stopped");
}

double llama_dataset_streaming_memory_monitor::get_memory_pressure() {
    // Get process memory usage
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);

    // Get system memory info from /proc/meminfo
    size_t total_memory = 0;
    size_t available_memory = 0;

    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.find("MemTotal:") == 0) {
                size_t pos = line.find_first_of("0123456789");
                if (pos != std::string::npos) {
                    total_memory = std::stoull(line.substr(pos)) * 1024; // Convert to bytes
                }
            } else if (line.find("MemAvailable:") == 0) {
                size_t pos = line.find_first_of("0123456789");
                if (pos != std::string::npos) {
                    available_memory = std::stoull(line.substr(pos)) * 1024; // Convert to bytes
                }
            }

            if (total_memory > 0 && available_memory > 0) {
                break;
            }
        }
        meminfo.close();
    }

    // Calculate memory pressure
    double pressure = 0.0;
    if (total_memory > 0) {
        pressure = 1.0 - static_cast<double>(available_memory) / total_memory;
    }

    return pressure;
}

void llama_dataset_streaming_memory_monitor::start() {
    if (running) {
        return;
    }

    running = true;
    monitor_thread = std::thread(&llama_dataset_streaming_memory_monitor::monitor_loop, this);

    LLAMA_LOG_DEBUG("Started memory monitor with interval %zu ms", check_interval_ms);
}

void llama_dataset_streaming_memory_monitor::stop() {
    if (!running) {
        return;
    }

    {
        std::lock_guard lock(mutex);
        running = false;
    }

    // Wake up the monitor thread
    cv.notify_all();

    // Wait for the monitor thread to finish
    if (monitor_thread.joinable()) {
        monitor_thread.join();
    }

    LLAMA_LOG_DEBUG("Stopped memory monitor\n");
}

void llama_dataset_streaming_memory_monitor::set_pressure_callback(MemoryPressureCallback callback) {
    std::lock_guard lock(mutex);
    pressure_callback = callback;
}

void llama_dataset_streaming_memory_monitor::set_check_interval(size_t interval_ms) {
    std::lock_guard lock(mutex);
    check_interval_ms = interval_ms;
    LLAMA_LOG_DEBUG("Set memory monitor check interval to %zu ms\n", check_interval_ms);
}

void llama_dataset_streaming_memory_monitor::set_pressure_thresholds(double high_threshold, double low_threshold) {
    std::lock_guard<std::mutex> lock(mutex);
    high_pressure_threshold = high_threshold;
    low_pressure_threshold = low_threshold;
    LLAMA_LOG_DEBUG("Set memory monitor pressure thresholds to high=%.2f, low=%.2f\n", high_threshold, low_threshold);
}

llama_dataset_streaming_memory_monitor::MemoryInfo llama_dataset_streaming_memory_monitor::get_memory_info() const {
    MemoryInfo info;
    info.total_physical_memory = 0;
    info.available_physical_memory = 0;
    info.process_physical_memory = 0;
    info.memory_pressure = 0.0;

    // Get process memory usage
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    info.process_physical_memory = usage.ru_maxrss * 1024; // Convert to bytes

    // Get system memory info from /proc/meminfo
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.find("MemTotal:") == 0) {
                size_t pos = line.find_first_of("0123456789");
                if (pos != std::string::npos) {
                    info.total_physical_memory = std::stoull(line.substr(pos)) * 1024; // Convert to bytes
                }
            } else if (line.find("MemAvailable:") == 0) {
                size_t pos = line.find_first_of("0123456789");
                if (pos != std::string::npos) {
                    info.available_physical_memory = std::stoull(line.substr(pos)) * 1024; // Convert to bytes
                }
            }

            if (info.total_physical_memory > 0 && info.available_physical_memory > 0) {
                break;
            }
        }
        meminfo.close();
    }

    // Calculate memory pressure
    if (info.total_physical_memory > 0) {
        info.memory_pressure = 1.0 - static_cast<double>(info.available_physical_memory) / info.total_physical_memory;
    }

    return info;
}
