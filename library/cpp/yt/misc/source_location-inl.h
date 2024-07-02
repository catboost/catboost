#ifndef SOURCE_LOCATION_INL_H_
#error "Direct inclusion of this file is not allowed, include source_location.h"
// For the sake of sane code completion.
#include "source_location.h"
#endif

namespace NYT {

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

////////////////////////////////////////////////////////////////////////////////

} // namespace std
