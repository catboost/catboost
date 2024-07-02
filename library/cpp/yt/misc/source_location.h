#pragma once

#include <util/generic/string.h>

#ifdef __cpp_lib_source_location
#include <source_location>
#endif // __cpp_lib_source_location

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

class TStringBuilderBase;

// TODO(dgolear): Drop when LLVM-14 is eradicated.
#ifdef __cpp_lib_source_location
void FormatValue(TStringBuilderBase* builder, const std::source_location& location, TStringBuf /*spec*/);
#endif // __cpp_lib_source_location

////////////////////////////////////////////////////////////////////////////////

class TSourceLocation
{
public:
    TSourceLocation() = default;
    TSourceLocation(const char* fileName, int line);
#ifdef __cpp_lib_source_location
    explicit TSourceLocation(const std::source_location& location);
#endif // __cpp_lib_source_location

    const char* GetFileName() const;
    int GetLine() const;
    bool IsValid() const;

    bool operator<(const TSourceLocation& other) const;
    bool operator==(const TSourceLocation& other) const;

private:
    const char* FileName_ = nullptr;
    int Line_ = -1;
};

//! Defines a macro to record the current source location.
#ifdef __cpp_lib_source_location
#define YT_CURRENT_SOURCE_LOCATION ::NYT::TSourceLocation(std::source_location::current())
#else
#define YT_CURRENT_SOURCE_LOCATION ::NYT::TSourceLocation(__FILE__, __LINE__)
#endif // __cpp_lib_source_location

void FormatValue(TStringBuilderBase* builder, const TSourceLocation& location, TStringBuf spec);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define SOURCE_LOCATION_INL_H_
#include "source_location-inl.h"
#undef SOURCE_LOCATION_INL_H_
