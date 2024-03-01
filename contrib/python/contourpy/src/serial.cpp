#include "base_impl.h"
#include "converter.h"
#include "serial.h"

namespace contourpy {

SerialContourGenerator::SerialContourGenerator(
    const CoordinateArray& x, const CoordinateArray& y, const CoordinateArray& z,
    const MaskArray& mask, bool corner_mask, LineType line_type, FillType fill_type,
    bool quad_as_tri, ZInterp z_interp, index_t x_chunk_size, index_t y_chunk_size)
    : BaseContourGenerator(x, y, z, mask, corner_mask, line_type, fill_type, quad_as_tri, z_interp,
                           x_chunk_size, y_chunk_size)
{}

void SerialContourGenerator::export_filled(
    const ChunkLocal& local, std::vector<py::list>& return_lists)
{
    assert(local.total_point_count > 0);

    switch (get_fill_type())
    {
        case FillType::OuterCode:
        case FillType::OuterOffset: {
            assert(!has_direct_points() && !has_direct_line_offsets());
            auto outer_count = local.line_count - local.hole_count;

            for (decltype(outer_count) i = 0; i < outer_count; ++i) {
                auto outer_start = local.outer_offsets.start[i];
                auto outer_end = local.outer_offsets.start[i+1];
                auto point_start = local.line_offsets.start[outer_start];
                auto point_end = local.line_offsets.start[outer_end];
                auto point_count = point_end - point_start;
                assert(point_count > 2);

                return_lists[0].append(Converter::convert_points(
                    point_count, local.points.start + 2*point_start));

                if (get_fill_type() == FillType::OuterCode)
                    return_lists[1].append(Converter::convert_codes(
                        point_count, outer_end - outer_start + 1,
                        local.line_offsets.start + outer_start, point_start));
                else
                    return_lists[1].append(Converter::convert_offsets(
                        outer_end - outer_start + 1, local.line_offsets.start + outer_start,
                        point_start));
            }
            break;
        }
        case FillType::ChunkCombinedCode:
        case FillType::ChunkCombinedCodeOffset: {
            assert(has_direct_points() && !has_direct_line_offsets());

            // return_lists[0][local_chunk] already contains combined points.
            // If ChunkCombinedCodeOffset. return_lists[2][local.chunk] already contains outer
            //    offsets.
            return_lists[1][local.chunk] = Converter::convert_codes(
                local.total_point_count, local.line_count + 1, local.line_offsets.start, 0);
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

void SerialContourGenerator::export_lines(
    const ChunkLocal& local, std::vector<py::list>& return_lists)
{
    assert(local.total_point_count > 0);

    switch (get_line_type())
    {
        case LineType::Separate:
        case LineType::SeparateCode: {
            assert(!has_direct_points() && !has_direct_line_offsets());

            bool separate_code = (get_line_type() == LineType::SeparateCode);

            for (decltype(local.line_count) i = 0; i < local.line_count; ++i) {
                auto point_start = local.line_offsets.start[i];
                auto point_end = local.line_offsets.start[i+1];
                auto point_count = point_end - point_start;
                assert(point_count > 1);

                return_lists[0].append(Converter::convert_points(
                    point_count, local.points.start + 2*point_start));

                if (separate_code) {
                    return_lists[1].append(
                        Converter::convert_codes_check_closed_single(
                            point_count, local.points.start + 2*point_start));
                }
            }
            break;
        }
        case LineType::ChunkCombinedCode: {
            assert(has_direct_points() && !has_direct_line_offsets());

            // return_lists[0][local.chunk] already contains points.
            return_lists[1][local.chunk] = Converter::convert_codes_check_closed(
                local.total_point_count, local.line_count + 1, local.line_offsets.start,
                local.points.start);
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

void SerialContourGenerator::march(std::vector<py::list>& return_lists)
{
    auto n_chunks = get_n_chunks();
    bool single_chunk = (n_chunks == 1);

    if (single_chunk) {
        // Stage 1: If single chunk, initialise cache z-levels and starting locations for whole
        // domain.
        init_cache_levels_and_starts();
    }

    // Stage 2: Trace contours.
    ChunkLocal local;
    for (index_t chunk = 0; chunk < n_chunks; ++chunk) {
        get_chunk_limits(chunk, local);
        if (!single_chunk)
            init_cache_levels_and_starts(&local);
        march_chunk(local, return_lists);
        local.clear();
    }
}

} // namespace contourpy
