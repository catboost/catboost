#pragma once

#include <util/generic/string.h>

#ifdef __cpp_lib_source_location
#include <source_location>
#endif // __cpp_lib_source_location

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

// TODO(dgolear): Drop when LLVM-14 is eradicated.
#ifdef __cpp_lib_source_location

class TStringBuilderBase;

void FormatValue(TStringBuilderBase* builder, const std::source_location& location, TStringBuf /*format*/);
TString ToString(const std::source_location& location);

#endif // __cpp_lib_source_location

////////////////////////////////////////////////////////////////////////////////

class TSourceLocation
{
public:
    TSourceLocation()
        : FileName_(nullptr)
        , Line_(-1)
    { }

    TSourceLocation(const char* fileName, int line)
        : FileName_(fileName)
        , Line_(line)
    { }

    const char* GetFileName() const;
    int GetLine() const;
    bool IsValid() const;

    bool operator<(const TSourceLocation& other) const;
    bool operator==(const TSourceLocation& other) const;

private:
    const char* FileName_;
    int Line_;

};

//! Defines a macro to record the current source location.
#define FROM_HERE ::NYT::TSourceLocation(__FILE__, __LINE__)

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
