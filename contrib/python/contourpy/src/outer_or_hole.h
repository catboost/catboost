#ifndef CONTOURPY_OUTER_OR_HOLE_H
#define CONTOURPY_OUTER_OR_HOLE_H

#include <iosfwd>

namespace contourpy {

typedef enum
{
    Outer,
    Hole
} OuterOrHole;

std::ostream &operator<<(std::ostream &os, const OuterOrHole& outer_or_hole);

} // namespace contourpy

#endif // CONTOURPY_OUTER_OR_HOLE_H
