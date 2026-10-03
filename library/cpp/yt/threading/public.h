#pragma once

#include <library/cpp/yt/system/public.h>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

// TODO(babenko): Drop these re-exports; use NYT::TThreadId and NYT::InvalidThreadId instead.
using ::NYT::InvalidThreadId;
using ::NYT::TThreadId;

class TExecutionStack;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
