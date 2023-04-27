#include "outer_or_hole.h"
#include <iostream>

namespace contourpy {

std::ostream &operator<<(std::ostream &os, const OuterOrHole& outer_or_hole)
{
    switch (outer_or_hole) {
        case Outer:
            os << "Outer";
            break;
        case Hole:
            os << "Hole";
            break;
    }
    return os;
}

} // namespace contourpy
