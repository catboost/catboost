#pragma once

#include <util/generic/string.h>

#include <array>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

struct TThreadName
{
    TThreadName() = default;
    TThreadName(const TString& name);

    TStringBuf ToStringBuf() const;

    static constexpr int BufferCapacity = 16; // including zero terminator
    std::array<char, BufferCapacity> Buffer{}; // zero-terminated
    int Length = 0; // not including zero terminator
};

TThreadName GetCurrentThreadName();

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
