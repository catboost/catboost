#pragma once

#include <library/cpp/yt/misc/enum.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

DEFINE_ENUM(EProcessExitCode,
    ((OK)                  (0))
    ((ArgumentsError)      (1))
    ((GenericError)        (2))
    ((IOError)             (3))
    ((OutOfMemory)         (9))
);

//! Invokes _exit to abort the process immediately without calling any cleanup code.
[[noreturn]] void AbortProcess(int exitCode);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
