#ifndef CONTOURPY_LINE_TYPE_H
#define CONTOURPY_LINE_TYPE_H

#include <iosfwd>
#include <string>

namespace contourpy {

// C++11 scoped enum, must be fully qualified to use.
enum class LineType
{
    Separate = 101,
    SeparateCode = 102,
    ChunkCombinedCode = 103,
    ChunkCombinedOffset = 104,
    ChunkCombinedNan = 105,
};

std::ostream &operator<<(std::ostream &os, const LineType& line_type);

} // namespace contourpy

#endif // CONTOURPY_LINE_TYPE_H
