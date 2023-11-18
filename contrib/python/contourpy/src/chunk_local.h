#ifndef CONTOURPY_CHUNK_LOCAL_H
#define CONTOURPY_CHUNK_LOCAL_H

#include "output_array.h"
#include <iosfwd>

namespace contourpy {

struct ChunkLocal
{
    ChunkLocal();

    void clear();

    friend std::ostream &operator<<(std::ostream &os, const ChunkLocal& local);



    index_t chunk;                       // Index in range 0 to _n_chunks-1.
    index_t istart, iend, jstart, jend;  // Chunk limits, inclusive.
    int pass;

    // Data for whole pass.
    count_t total_point_count;           // Includes nan separators if used.
    count_t line_count;                  // Count of all lines
    count_t hole_count;                  // Count of holes only.

    // Output arrays that are initialised at the end of pass 0 and written to during pass 1.
    OutputArray<double> points;
    OutputArray<offset_t> line_offsets;  // Into array of points.
    OutputArray<offset_t> outer_offsets; // Into array of points or line offsets depending on
                                         //   fill_type.

    // Data for current outer.
    std::vector<index_t> look_up_quads;  // To find holes of current outer.
};

} // namespace contourpy

#endif // CONTOURPY_CHUNK_LOCAL_H
