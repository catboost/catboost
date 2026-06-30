#pragma once

#include "preprocessor.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! Static initializer will be invoked prior to entering |main|.
//! The exact order of these invocations is, of course, undefined.
#define YT_STATIC_INITIALIZER(...) \
    [[maybe_unused]] static inline const void* PP_ANONYMOUS_VARIABLE(StaticInitializer) = [] { \
        __VA_ARGS__; \
        return nullptr; \
    } ()

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
