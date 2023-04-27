#ifndef CONTOURPY_Z_INTERP_H
#define CONTOURPY_Z_INTERP_H

#include <iosfwd>
#include <string>

namespace contourpy {

// Enum for type of interpolation used to find intersection of contour lines
// with grid cell edges.

// C++11 scoped enum, must be fully qualified to use.
enum class ZInterp
{
    Linear = 1,
    Log = 2
};

std::ostream &operator<<(std::ostream &os, const ZInterp& z_interp);

} // namespace contourpy

#endif // CONTOURPY_ZINTERP_H
