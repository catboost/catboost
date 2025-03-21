#pragma once

#include "error_code.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
class TErrorOr;

using TError = TErrorOr<void>;

struct TErrorAttribute;
class TErrorAttributes;
struct TOriginAttributes;

////////////////////////////////////////////////////////////////////////////////

YT_DEFINE_ERROR_ENUM(
    ((OK)                    (0))
    ((Generic)               (1))
    ((Canceled)              (2))
    ((Timeout)               (3))
    ((FutureCombinerFailure) (4))
    ((FutureCombinerShortcut)(5))
);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
