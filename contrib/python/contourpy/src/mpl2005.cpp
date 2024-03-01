#include "mpl2005.h"

namespace contourpy {

Mpl2005ContourGenerator::Mpl2005ContourGenerator(
    const CoordinateArray& x, const CoordinateArray& y, const CoordinateArray& z,
    const MaskArray& mask, index_t x_chunk_size, index_t y_chunk_size)
    : _x(x),
      _y(y),
      _z(z),
      _site(cntr_new())
{
    if (_x.ndim() != 2 || _y.ndim() != 2 || _z.ndim() != 2)
        throw std::invalid_argument("x, y and z must all be 2D arrays");

    auto nx = _z.shape(1);
    auto ny = _z.shape(0);

    if (_x.shape(1) != nx || _x.shape(0) != ny ||
        _y.shape(1) != nx || _y.shape(0) != ny)
        throw std::invalid_argument("x, y and z arrays must have the same shape");

    if (nx < 2 || ny < 2)
        throw std::invalid_argument("x, y and z must all be at least 2x2 arrays");

    if (mask.ndim() != 0) {  // ndim == 0 if mask is not set, which is valid.
        if (mask.ndim() != 2)
            throw std::invalid_argument("mask array must be a 2D array");

        if (mask.shape(1) != nx || mask.shape(0) != ny)
            throw std::invalid_argument(
                "If mask is set it must be a 2D array with the same shape as z");
    }

    if (x_chunk_size < 0 || y_chunk_size < 0)
        throw std::invalid_argument("x_chunk_size and y_chunk_size cannot be negative");

    const bool* mask_data = (mask.ndim() > 0 ? mask.data() : nullptr);

    cntr_init(
        _site, nx, ny, _x.data(), _y.data(), _z.data(), mask_data, x_chunk_size, y_chunk_size);
}

Mpl2005ContourGenerator::~Mpl2005ContourGenerator()
{
    cntr_del(_site);
}

py::tuple Mpl2005ContourGenerator::filled(const double& lower_level, const double& upper_level)
{
    if (lower_level >= upper_level)
        throw std::invalid_argument("upper_level must be larger than lower_level");

    double levels[2] = {lower_level, upper_level};
    return cntr_trace(_site, levels, 2);
}

py::tuple Mpl2005ContourGenerator::get_chunk_count() const
{
    long nx_chunks = (long)(ceil((_site->imax-1.0) / _site->i_chunk_size));
    long ny_chunks = (long)(ceil((_site->jmax-1.0) / _site->j_chunk_size));
    return py::make_tuple(ny_chunks, nx_chunks);
}

py::tuple Mpl2005ContourGenerator::get_chunk_size() const
{
    return py::make_tuple(_site->j_chunk_size, _site->i_chunk_size);
}

py::tuple Mpl2005ContourGenerator::lines(const double& level)
{
    double levels[2] = {level, 0.0};
    return cntr_trace(_site, levels, 1);
}

} // namespace contourpy
