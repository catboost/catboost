#ifndef CONTOURPY_THREADED_H
#define CONTOURPY_THREADED_H

#include "base.h"
#include <condition_variable>
#include <mutex>

namespace contourpy {

class ThreadedContourGenerator : public BaseContourGenerator<ThreadedContourGenerator>
{
public:
    ThreadedContourGenerator(
        const CoordinateArray& x, const CoordinateArray& y, const CoordinateArray& z,
        const MaskArray& mask, bool corner_mask, LineType line_type, FillType fill_type,
        bool quad_as_tri, ZInterp z_interp, index_t x_chunk_size, index_t y_chunk_size,
        index_t n_threads);

    index_t get_thread_count() const;

private:
    friend class BaseContourGenerator<ThreadedContourGenerator>;

    // Lock class is used to lock access to a single thread when creating/modifying Python objects.
    // Also acquires the GIL for the duration of the lock.
    class Lock
    {
    public:
        explicit Lock(ThreadedContourGenerator& contour_generator)
            : _lock(contour_generator._python_mutex)
        {}

        // Non-copyable and non-moveable.
        Lock(const Lock& other) = delete;
        Lock(const Lock&& other) = delete;
        Lock& operator=(const Lock& other) = delete;
        Lock& operator=(const Lock&& other) = delete;

    private:
        std::unique_lock<std::mutex> _lock;
        py::gil_scoped_acquire _gil;
    };

    // Write points and offsets/codes to output numpy arrays.
    void export_filled(const ChunkLocal& local, std::vector<py::list>& return_lists);

    // Write points and offsets/codes to output numpy arrays.
    void export_lines(const ChunkLocal& local, std::vector<py::list>& return_lists);

    static index_t limit_n_threads(index_t n_threads, index_t n_chunks);

    void march(std::vector<py::list>& return_lists);

    void thread_function(std::vector<py::list>& return_lists);



    // Multithreading member variables.
    index_t _n_threads;        // Number of threads used.
    index_t _next_chunk;       // Next available chunk for thread to process.
    index_t _finished_count;   // Count of threads that have finished the cache init.
    std::mutex _chunk_mutex;   // Locks access to _next_chunk/_finished_count.
    std::mutex _python_mutex;  // Locks access to Python objects.
    std::condition_variable _condition_variable;  // Implements multithreaded barrier.
};

} // namespace contourpy

#endif // CONTOURPY_THREADED_H
