#include "line_type.h"
#include <iostream>

namespace contourpy {

std::ostream &operator<<(std::ostream &os, const LineType& line_type)
{
    switch (line_type) {
        case LineType::Separate:
            os << "Separate";
            break;
        case LineType::SeparateCode:
            os << "SeparateCode";
            break;
        case LineType::ChunkCombinedCode:
            os << "ChunkCombinedCode";
            break;
        case LineType::ChunkCombinedOffset:
            os << "ChunkCombinedOffset";
            break;
        case LineType::ChunkCombinedNan:
            os << "ChunkCombinedNan";
            break;
    }
    return os;
}

} // namespace contourpy
