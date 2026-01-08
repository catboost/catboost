#pragma once

#include <util/generic/strbuf.h>

#include <optional>
#include <string>
#include <vector>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

// NB: Currently not supported on other platforms.
#if defined(_linux_) || defined(_darwin_)
std::vector<std::string> GetEnvironNameValuePairs();
#endif

std::pair<TStringBuf, std::optional<TStringBuf>> ParseEnvironNameValuePair(TStringBuf pair);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
