#include "new.h"

#include <library/cpp/yt/system/exit.h>

namespace NYT::NDetail {

////////////////////////////////////////////////////////////////////////////////

void AbortOnOom()
{
    AbortProcessDramatically(
        EProcessExitCode::OutOfMemory,
        "Out-of-memory during object allocation");
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NDetail
