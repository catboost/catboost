#pragma once

#include <util/generic/string.h>

#include <array>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

struct TThreadName
{
    static constexpr int BufferCapacity = 16; // including zero terminator
    std::array<char, BufferCapacity> Buffer{}; // zero-terminated
    int Length; // not including zero terminator

    TString ToString() const;
};

TThreadName GetCurrentThreadName();

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
