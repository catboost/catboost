#ifndef CONTOURPY_MPL_2005_H
#define CONTOURPY_MPL_2005_H

#include "contour_generator.h"
#include "mpl2005_original.h"

namespace contourpy {

class Mpl2005ContourGenerator : public ContourGenerator
{
public:
    Mpl2005ContourGenerator(
        const CoordinateArray& x, const CoordinateArray& y, const CoordinateArray& z,
        const MaskArray& mask, index_t x_chunk_size, index_t y_chunk_size);

    virtual ~Mpl2005ContourGenerator();

    py::tuple filled(double lower_level, double upper_level) override;

    py::tuple get_chunk_count() const;  // Return (y_chunk_count, x_chunk_count)
    py::tuple get_chunk_size() const;   // Return (y_chunk_size, x_chunk_size)

    py::sequence lines(double level) override;

private:
    CoordinateArray _x, _y, _z;
    Csite *_site;
};

} // namespace contourpy

#endif // CONTOURPY_MPL_2005_H
