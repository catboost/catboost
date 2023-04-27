#ifndef CONTOURPY_SERIAL_H
#define CONTOURPY_SERIAL_H

#include "base.h"

namespace contourpy {

class SerialContourGenerator : public BaseContourGenerator<SerialContourGenerator>
{
public:
    SerialContourGenerator(
        const CoordinateArray& x, const CoordinateArray& y, const CoordinateArray& z,
        const MaskArray& mask, bool corner_mask, LineType line_type, FillType fill_type,
        bool quad_as_tri, ZInterp z_interp, index_t x_chunk_size, index_t y_chunk_size);

private:
    friend class BaseContourGenerator<SerialContourGenerator>;

    // Dummy Lock class which is the single-threaded version of ThreadedContourGenerator::Lock and
    // does not do anything, allowing base class code to use Lock objects for both serial and
    // multithreaded code.
    class Lock
    {
    public:
        explicit Lock(SerialContourGenerator& contour_generator)
        {}
    };

    // Write points and offsets/codes to output numpy arrays.
    void export_filled(const ChunkLocal& local, std::vector<py::list>& return_lists);

    // Write points and offsets/codes to output numpy arrays.
    void export_lines(const ChunkLocal& local, std::vector<py::list>& return_lists);

    void march(std::vector<py::list>& return_lists);
};

} // namespace contourpy

#endif // CONTOURPY_SERIAL_H
