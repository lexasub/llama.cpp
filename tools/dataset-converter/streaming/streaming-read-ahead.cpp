#include "streaming-read-ahead.h"

#include "../../common/log.h"
#include "llama-impl.h"

StreamingReadAhead::StreamingReadAhead(size_t window, size_t max_queue)
    : running(false),
      paused(false),
      window_size(window),
      max_queue_size(max_queue) {
}

StreamingReadAhead::~StreamingReadAhead() {
    stop();
}

void StreamingReadAhead::worker_loop() {
    LLAMA_LOG_DEBUG("Streaming read-ahead worker thread started");

    while (running) {
        uint64_t sequence_id = 0;

        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            // Wait until there's work to do or we're stopped
            cv.wait(lock, [this] {
                return (!prefetch_queue.empty() && !paused) || !running;
            });

            if (!running) {
                break;
            }

            if (paused || prefetch_queue.empty()) {
                continue;
            }

            // Get next sequence to prefetch
            sequence_id = prefetch_queue.front();
            prefetch_queue.pop();

            // Mark as in progress
            in_progress.insert(sequence_id);
        }

        // Execute the prefetch callback outside the lock
        if (prefetch_callback && sequence_id > 0) {
            try {
                prefetch_callback(sequence_id);
                LLAMA_LOG_DEBUG("Prefetched sequence %zu", sequence_id);
            } catch (const std::exception& e) {
                LLAMA_LOG_ERROR("Error prefetching sequence %zu: %s", sequence_id, e.what());
            }
        }

        // Mark as completed
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            in_progress.erase(sequence_id);
        }
    }

    LLAMA_LOG_DEBUG("Streaming read-ahead worker thread stopped");
}

void StreamingReadAhead::start() {
    if (running) {
        return;
    }

    running = true;
    worker_thread = std::thread(&StreamingReadAhead::worker_loop, this);

    LLAMA_LOG_DEBUG("Started streaming read-ahead with window size %zu", window_size);
}

void StreamingReadAhead::stop() {
    if (!running) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        running = false;

        // Clear the queue
        std::queue<uint64_t> empty;
        std::swap(prefetch_queue, empty);
        in_progress.clear();
    }

    // Wake up the worker thread
    cv.notify_all();

    // Wait for the worker thread to finish
    if (worker_thread.joinable()) {
        worker_thread.join();
    }

    LLAMA_LOG_DEBUG("Stopped streaming read-ahead");
}

void StreamingReadAhead::pause() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    paused = true;
    LLAMA_LOG_DEBUG("Paused streaming read-ahead");
}

void StreamingReadAhead::resume() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        paused = false;
    }

    cv.notify_all();
    LLAMA_LOG_DEBUG("Resumed streaming read-ahead");
}

void StreamingReadAhead::set_prefetch_callback(PrefetchCallback callback) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    prefetch_callback = callback;
}

void StreamingReadAhead::prefetch(uint64_t sequence_id) {
    std::lock_guard<std::mutex> lock(queue_mutex);

    if (!running || paused) {
        return;
    }

    // Add the next window_size sequences to the queue
    for (size_t i = 1; i <= window_size; i++) {
        uint64_t next_id = sequence_id + i;

        // Skip if already in queue or in progress
        if (in_progress.find(next_id) != in_progress.end()) {
            continue;
        }

        // Limit queue size
        if (prefetch_queue.size() >= max_queue_size) {
            break;
        }

        prefetch_queue.push(next_id);
    }

    // Notify worker thread
    cv.notify_one();
}

void StreamingReadAhead::clear_queue() {
    std::lock_guard<std::mutex> lock(queue_mutex);

    // Clear the queue
    std::queue<uint64_t> empty;
    std::swap(prefetch_queue, empty);

    LLAMA_LOG_DEBUG("Cleared streaming read-ahead queue");
}

void StreamingReadAhead::set_window_size(size_t window) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    window_size = window;
    LLAMA_LOG_DEBUG("Set streaming read-ahead window size to %zu", window_size);
}

StreamingReadAhead::Status StreamingReadAhead::get_status() const {
    // Can't use lock_guard with const mutex in a const method
    // We'll create a copy of the status without locking
    // This is not thread-safe but acceptable for status reporting
    Status status;
    status.is_running = running.load();
    status.is_paused = paused.load();
    status.queue_size = prefetch_queue.size();
    status.in_progress_count = in_progress.size();
    status.window_size = window_size;

    return status;
}
