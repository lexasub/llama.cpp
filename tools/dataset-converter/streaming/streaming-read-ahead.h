#pragma once

#include <cstdint>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <unordered_set>

/**
 * @brief Read-ahead buffer manager for streaming datasets
 * 
 * This class manages asynchronous prefetching of sequences to improve
 * sequential access performance in streaming mode.
 */
class llama_dataset_streaming_read_ahead {
public:
    using PrefetchCallback = std::function<void(uint64_t)>;
    
private:
    std::thread worker_thread;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::queue<uint64_t> prefetch_queue;
    std::unordered_set<uint64_t> in_progress;
    std::atomic<bool> running;
    std::atomic<bool> paused;
    
    size_t window_size;
    size_t max_queue_size;
    PrefetchCallback prefetch_callback;
    
    void worker_loop();
    
public:
    llama_dataset_streaming_read_ahead(size_t window = 5, size_t max_queue = 20);
    ~llama_dataset_streaming_read_ahead();
    
    // Start the prefetch worker thread
    void start();
    
    // Stop the prefetch worker thread
    void stop();
    
    // Pause prefetching temporarily
    void pause();
    
    // Resume prefetching
    void resume();
    
    // Set the prefetch callback function
    void set_prefetch_callback(PrefetchCallback callback);
    
    // Request prefetching for a sequence and its window
    void prefetch(uint64_t sequence_id);
    
    // Clear the prefetch queue
    void clear_queue();
    
    // Set the prefetch window size
    void set_window_size(size_t window);
    
    // Get current status
    struct Status {
        bool is_running;
        bool is_paused;
        size_t queue_size;
        size_t in_progress_count;
        size_t window_size;
    };
    
    Status get_status() const;
};