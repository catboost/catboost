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

//should be used instead of __FILE__
#define __SOURCE_FILE__ (__SOURCE_FILE_IMPL__.As<TStringBuf>())
#define __LOCATION__ ::TSourceLocation(__SOURCE_FILE__, __LINE__)
