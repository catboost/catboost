#pragma once

#include <string>
#include <utility>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

using TAllocationTagKey = std::string;
using TAllocationTagValue = std::string;
using TAllocationTag = std::pair<TAllocationTagKey, TAllocationTagValue>;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
