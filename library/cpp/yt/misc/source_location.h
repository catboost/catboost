#pragma once

namespace NYT {

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
