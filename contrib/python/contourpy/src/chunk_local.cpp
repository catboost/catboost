#include "chunk_local.h"
#include <iostream>
#include <limits>

namespace contourpy {

ChunkLocal::ChunkLocal()
{
    look_up_quads.reserve(100);
    clear();
}

void ChunkLocal::clear()
{
    chunk = -1;
    istart = iend = jstart = jend = -1;
    pass = -1;

    total_point_count = 0;
    line_count = 0;
    hole_count = 0;

    points.clear();
    line_offsets.clear();
    outer_offsets.clear();

    look_up_quads.clear();
}

std::ostream &operator<<(std::ostream &os, const ChunkLocal& local)
{
    os << "ChunkLocal:"
        << " chunk=" << local.chunk
        << " istart=" << local.istart
        << " iend=" << local.iend
        << " jstart=" << local.jstart
        << " jend=" << local.jend
        << " total_point_count=" << local.total_point_count
        << " line_count=" << local.line_count
        << " hole_count=" << local.hole_count;

    if (local.line_offsets.start != nullptr) {
        os << " line_offsets=";
        for (count_t i = 0; i < local.line_count + 1; ++i) {
            os << local.line_offsets.start[i] << " ";
        }
    }

    if (local.outer_offsets.start != nullptr) {
        os << " outer_offsets=";
        for (count_t i = 0; i < local.line_count - local.hole_count + 1; ++i) {
            os << local.outer_offsets.start[i] << " ";
        }
    }

    return os;
}

} // namespace contourpy
