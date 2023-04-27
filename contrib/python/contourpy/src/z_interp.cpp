#include "z_interp.h"
#include <iostream>

namespace contourpy {

std::ostream &operator<<(std::ostream &os, const ZInterp& z_interp)
{
    switch (z_interp) {
        case ZInterp::Linear:
            os << "Linear";
            break;
        case ZInterp::Log:
            os << "Log";
            break;
    }
    return os;
}

} // namespace contourpy
