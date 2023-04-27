#include "fill_type.h"
#include <iostream>

namespace contourpy {

std::ostream &operator<<(std::ostream &os, const FillType& fill_type)
{
    switch (fill_type) {
        case FillType::OuterCode:
            os << "OuterCode";
            break;
        case FillType::OuterOffset:
            os << "OuterOffset";
            break;
        case FillType::ChunkCombinedCode:
            os << "ChunkCombinedCode";
            break;
        case FillType::ChunkCombinedOffset:
            os << "ChunkCombinedOffset";
            break;
        case FillType::ChunkCombinedCodeOffset:
            os << "ChunkCombinedCodeOffset";
            break;
        case FillType::ChunkCombinedOffsetOffset:
            os << "ChunkCombinedOffsetOffset";
            break;
    }
    return os;
}

} // namespace contourpy
