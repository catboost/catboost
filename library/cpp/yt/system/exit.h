#pragma once

#include <library/cpp/yt/misc/enum.h>

#include <type_traits>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

DEFINE_ENUM(EProcessExitCode,
    ((OK)                  (0))
    ((ArgumentsError)      (1))
    ((GenericError)        (2))
    ((IOError)             (3))
    ((InternalError)       (4))
    ((OutOfMemory)         (9))
);

//! Invokes _exit to abort the process immediately without calling any cleanup code
//! and without printing any details to stderr.
[[noreturn]] void AbortProcessSilently(int exitCode);

//! A typed version of #AbortProcessSilently.
template <class E>
    requires std::is_enum_v<E>
[[noreturn]] void AbortProcessSilently(E exitCode);

//! Invokes _exit to abort the process immediately
//! without calling any cleanup code but printing error message to stderr.
[[noreturn]] void AbortProcessDramatically(
    int exitCode,
    TStringBuf exitCodeStr,
    TStringBuf message);

//! A typed version of #AbortProcessDramatically.
template <class E>
    requires std::is_enum_v<E>
[[noreturn]] void AbortProcessDramatically(
    E exitCode,
    TStringBuf message);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define EXIT_INL_H_
#include "exit-inl.h"
#undef EXIT_INL_H_
