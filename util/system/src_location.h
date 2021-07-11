#pragma once

#include "src_root.h"

#include <util/generic/strbuf.h>

struct TSourceLocation {
    constexpr TSourceLocation(const TStringBuf f, int l) noexcept
        : File(f)
        , Line(l)
    {
    }

    TStringBuf File;
    int Line;
};

// __SOURCE_FILE__ should be used instead of __FILE__
#if !defined(__NVCC__)
    #define __SOURCE_FILE__ (__SOURCE_FILE_IMPL__.As<TStringBuf>())
#else
    #define __SOURCE_FILE__ (__SOURCE_FILE_IMPL__.template As<TStringBuf>())
#endif

#define __LOCATION__ ::TSourceLocation(__SOURCE_FILE__, __LINE__)
