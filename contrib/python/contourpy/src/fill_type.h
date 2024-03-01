#ifndef CONTOURPY_FILL_TYPE_H
#define CONTOURPY_FILL_TYPE_H

#include <iosfwd>
#include <string>

namespace contourpy {

// C++11 scoped enum, must be fully qualified to use.
enum class FillType
{
    OuterCode = 201,
    OuterOffset = 202,
    ChunkCombinedCode = 203,
    ChunkCombinedOffset = 204,
    ChunkCombinedCodeOffset = 205,
    ChunkCombinedOffsetOffset = 206,
};

std::ostream &operator<<(std::ostream &os, const FillType& fill_type);

} // namespace contourpy

#endif // CONTOURPY_FILL_TYPE_H
