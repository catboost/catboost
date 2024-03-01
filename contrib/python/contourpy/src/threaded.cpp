#include "base_impl.h"
#include "converter.h"
#include "threaded.h"
#include "util.h"
#include <thread>

namespace contourpy {

ThreadedContourGenerator::ThreadedContourGenerator(
    const CoordinateArray& x, const CoordinateArray& y, const CoordinateArray& z,
    const MaskArray& mask, bool corner_mask, LineType line_type, FillType fill_type,
    bool quad_as_tri, ZInterp z_interp, index_t x_chunk_size, index_t y_chunk_size,
    index_t n_threads)
    : BaseContourGenerator(x, y, z, mask, corner_mask, line_type, fill_type, quad_as_tri, z_interp,
                           x_chunk_size, y_chunk_size),
      _n_threads(limit_n_threads(n_threads, get_n_chunks())),
      _next_chunk(0)
{}

void ThreadedContourGenerator::export_filled(
    const ChunkLocal& local, std::vector<py::list>& return_lists)
{
    // Reimplementation of SerialContourGenerator::export_filled() to separate out the creation of
    // numpy arrays (which requires a thread lock) from the population of those arrays (which does
    // not). This minimises the time that the lock is used for.

    assert(local.total_point_count > 0);

    switch (get_fill_type())
    {
        case FillType::OuterCode:
        case FillType::OuterOffset: {
            assert(!has_direct_points() && !has_direct_line_offsets());

            auto outer_count = local.line_count - local.hole_count;
            bool outer_code = (get_fill_type() == FillType::OuterCode);
            std::vector<PointArray::value_type*> points_ptrs(outer_count);
            std::vector<CodeArray::value_type*> codes_ptrs(outer_code ? outer_count: 0);
            std::vector<OffsetArray::value_type*> offsets_ptrs(outer_code ? 0 : outer_count);

            {
                Lock lock(*this);  // cppcheck-suppress unreadVariable
                for (decltype(outer_count) i = 0; i < outer_count; ++i) {
                    auto outer_start = local.outer_offsets.start[i];
                    auto outer_end = local.outer_offsets.start[i+1];
                    auto point_start = local.line_offsets.start[outer_start];
                    auto point_end = local.line_offsets.start[outer_end];
                    auto point_count = point_end - point_start;
                    assert(point_count > 2);

                    index_t points_shape[2] = {static_cast<index_t>(point_count), 2};
                    PointArray point_array(points_shape);
                    return_lists[0].append(point_array);
                    points_ptrs[i] = point_array.mutable_data();

                    if (outer_code) {
                        index_t codes_shape = static_cast<index_t>(point_count);
                        CodeArray code_array(codes_shape);
                        return_lists[1].append(code_array);
                        codes_ptrs[i] = code_array.mutable_data();
                    }
                    else {
                        index_t offsets_shape = static_cast<index_t>(outer_end - outer_start + 1);
                        OffsetArray offset_array(offsets_shape);
                        return_lists[1].append(offset_array);
                        offsets_ptrs[i] = offset_array.mutable_data();
                    }
                }
            }

            for (decltype(outer_count) i = 0; i < outer_count; ++i) {
                auto outer_start = local.outer_offsets.start[i];
                auto outer_end = local.outer_offsets.start[i+1];
                auto point_start = local.line_offsets.start[outer_start];
                auto point_end = local.line_offsets.start[outer_end];
                auto point_count = point_end - point_start;
                assert(point_count > 2);

                Converter::convert_points(
                    point_count, local.points.start + 2*point_start, points_ptrs[i]);

                if (outer_code)
                    Converter::convert_codes(
                        point_count, outer_end - outer_start + 1,
                        local.line_offsets.start + outer_start, point_start, codes_ptrs[i]);
                else
                    Converter::convert_offsets(
                        outer_end - outer_start + 1, local.line_offsets.start + outer_start,
                        point_start, offsets_ptrs[i]);
            }
            break;
        }
        case FillType::ChunkCombinedCode:
        case FillType::ChunkCombinedCodeOffset: {
            assert(has_direct_points() && !has_direct_line_offsets());
            // return_lists[0][local_chunk] already contains combined points.
            // If ChunkCombinedCodeOffset. return_lists[2][local.chunk] already contains outer
            //    offsets.

            index_t codes_shape = static_cast<index_t>(local.total_point_count);
            CodeArray::value_type* codes_ptr = nullptr;

            {
                Lock lock(*this);  // cppcheck-suppress unreadVariable
                CodeArray code_array(codes_shape);
                return_lists[1][local.chunk] = code_array;
                codes_ptr = code_array.mutable_data();
            }

            Converter::convert_codes(
                local.total_point_count, local.line_count + 1, local.line_offsets.start, 0,
                codes_ptr);
            break;
        }
        case FillType::ChunkCombinedOffset:
        case FillType::ChunkCombinedOffsetOffset:
            assert(has_direct_points() && has_direct_line_offsets());
            if (get_fill_type() == FillType::ChunkCombinedOffsetOffset) {
                assert(has_direct_outer_offsets());
            }
            // return_lists[0][local_chunk] already contains combined points.
            // return_lists[1][local.chunk] already contains line offsets.
            // If ChunkCombinedOffsetOffset, return_lists[2][local.chunk] already contains
            //      outer offsets.
            break;
    }
}

void ThreadedContourGenerator::export_lines(
    const ChunkLocal& local, std::vector<py::list>& return_lists)
{
    // Reimplementation of SerialContourGenerator::export_lines() to separate out the creation of
    // numpy arrays (which requires a thread lock) from the population of those arrays (which does
    // not). This minimises the time that the lock is used for.

    assert(local.total_point_count > 0);

    switch (get_line_type())
    {
        case LineType::Separate:
        case LineType::SeparateCode: {
            assert(!has_direct_points() && !has_direct_line_offsets());

            bool separate_code = (get_line_type() == LineType::SeparateCode);
            std::vector<PointArray::value_type*> points_ptrs(local.line_count);
            std::vector<CodeArray::value_type*> codes_ptrs(separate_code ? local.line_count: 0);

            {
                Lock lock(*this);  // cppcheck-suppress unreadVariable
                for (decltype(local.line_count) i = 0; i < local.line_count; ++i) {
                    auto point_start = local.line_offsets.start[i];
                    auto point_end = local.line_offsets.start[i+1];
                    auto point_count = point_end - point_start;
                    assert(point_count > 1);

                    index_t points_shape[2] = {static_cast<index_t>(point_count), 2};
                    PointArray point_array(points_shape);

                    return_lists[0].append(point_array);
                    points_ptrs[i] = point_array.mutable_data();

                    if (separate_code) {
                        index_t codes_shape = static_cast<index_t>(point_count);
                        CodeArray code_array(codes_shape);
                        return_lists[1].append(code_array);
                        codes_ptrs[i] = code_array.mutable_data();
                    }
                }
            }

            for (decltype(local.line_count) i = 0; i < local.line_count; ++i) {
                auto point_start = local.line_offsets.start[i];
                auto point_end = local.line_offsets.start[i+1];
                auto point_count = point_end - point_start;
                assert(point_count > 1);

                Converter::convert_points(
                    point_count, local.points.start + 2*point_start, points_ptrs[i]);

                if (separate_code) {
                    Converter::convert_codes_check_closed_single(
                        point_count, local.points.start + 2*point_start, codes_ptrs[i]);
                }
            }
            break;
        }
        case LineType::ChunkCombinedCode: {
            assert(has_direct_points() && !has_direct_line_offsets());
            // return_lists[0][local.chunk] already contains points.

            index_t codes_shape = static_cast<index_t>(local.total_point_count);
            CodeArray::value_type* codes_ptr = nullptr;

            {
                Lock lock(*this);  // cppcheck-suppress unreadVariable
                CodeArray code_array(codes_shape);
                return_lists[1][local.chunk] = code_array;
                codes_ptr = code_array.mutable_data();
            }

            Converter::convert_codes_check_closed(
                local.total_point_count, local.line_count + 1, local.line_offsets.start,
                local.points.start, codes_ptr);
            break;
        }
        case LineType::ChunkCombinedOffset:
            assert(has_direct_points() && has_direct_line_offsets());
            // return_lists[0][local.chunk] already contains points.
            // return_lists[1][local.chunk] already contains line offsets.
            break;
        case LineType::ChunkCombinedNan:
            assert(has_direct_points());
            // return_lists[0][local.chunk] already contains points.
            break;
    }
}

index_t ThreadedContourGenerator::get_thread_count() const
{
    return _n_threads;
}

index_t ThreadedContourGenerator::limit_n_threads(index_t n_threads, index_t n_chunks)
{
    index_t max_threads = std::max<index_t>(Util::get_max_threads(), 1);
    if (n_threads == 0)
        return std::min(max_threads, n_chunks);
    else
        return std::min({max_threads, n_chunks, n_threads});
}

void ThreadedContourGenerator::march(std::vector<py::list>& return_lists)
{
    // Each thread executes thread_function() which has two stages:
    //   1) Initialise cache z-levels and starting locations
    //   2) Trace contours
    // Each stage is performed on a chunk by chunk basis.  There is a barrier between the two stages
    // to synchronise the threads so the cache setup is complete before being used by the trace.
    _next_chunk = 0;      // Next available chunk index.
    _finished_count = 0;  // Count of threads that have finished the cache init.

    // Main thread releases GIL for remainder of this function.
    // It is temporarily reacquired as necessary within the scope of threaded Lock objects.
    py::gil_scoped_release release;

    // Create (_n_threads-1) new worker threads.
    std::vector<std::thread> threads;
    threads.reserve(_n_threads-1);
    for (index_t i = 0; i < _n_threads-1; ++i)
        threads.emplace_back(
            &ThreadedContourGenerator::thread_function, this, std::ref(return_lists));

    thread_function(std::ref(return_lists));  // Main thread work.

    for (auto& thread : threads)
        thread.join();
    assert(_next_chunk == 2*get_n_chunks());
    threads.clear();
}

void ThreadedContourGenerator::thread_function(std::vector<py::list>& return_lists)
{
    // Function that is executed by each of the threads.
    // _next_chunk starts at zero and increases up to 2*_n_chunks.  A thread in need of work reads
    // _next_chunk and incremements it, then processes that chunk.  For _next_chunk < _n_chunks this
    // is stage 1 (init cache levels and starting locations) and for _next_chunk >= _n_chunks this
    // is stage 2 (trace contours).  There is a synchronisation barrier between the two stages so
    // that the cache initialisation is complete before being used by the contour trace.

    auto n_chunks = get_n_chunks();
    index_t chunk;
    ChunkLocal local;

    // Stage 1: Initialise cache z-levels and starting locations.
    while (true) {
        {
            std::lock_guard<std::mutex> guard(_chunk_mutex);
            if (_next_chunk < n_chunks)
                chunk = _next_chunk++;
            else
                break;  // No more work to do.
        }

        get_chunk_limits(chunk, local);
        init_cache_levels_and_starts(&local);
        local.clear();
    }

    {
        // Implementation of multithreaded barrier.  Each thread increments the shared counter.
        // Last thread to finish notifies the other threads that they can all continue.
        std::unique_lock<std::mutex> lock(_chunk_mutex);
        _finished_count++;
        if (_finished_count == _n_threads)
            _condition_variable.notify_all();
        else
            _condition_variable.wait(lock);
    }

    // Stage 2: Trace contours.
    while (true) {
        {
            std::lock_guard<std::mutex> guard(_chunk_mutex);
            if (_next_chunk < 2*n_chunks)
                chunk = _next_chunk++ - n_chunks;
            else
                break;  // No more work to do.
        }

        get_chunk_limits(chunk, local);
        march_chunk(local, return_lists);
        local.clear();
    }
}

} // namespace contourpy
