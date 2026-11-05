#ifndef SOURCE_LOCATION_INL_H_
#error "Direct inclusion of this file is not allowed, include source_location.h"
// For the sake of sane code completion.
#include "source_location.h"
#endif

#include <string>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <size_t N>
consteval TSourceLocationLite<N> MakeSourceLocationLite(const char* fileName, i64 line)
{
    TSourceLocationLite<N> result{.FileName = {}, .Line = line};
    std::char_traits<char>::copy(result.FileName, fileName, N + 1);
    return result;
}

////////////////////////////////////////////////////////////////////////////////

inline TSourceLocation::TSourceLocation(const char* fileName, int line)
    : FileName_(fileName)
    , Line_(line)
{ }

#ifdef __cpp_lib_source_location
inline TSourceLocation::TSourceLocation(const std::source_location& location)
    : FileName_(location.file_name())
    , Line_(location.line())
{ }
#endif // __cpp_lib_source_location

template <auto LocationLite>
TSourceLocation TSourceLocation::FromLite()
{
    return TSourceLocation(LocationLite.FileName, LocationLite.Line);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
