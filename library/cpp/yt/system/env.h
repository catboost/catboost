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

std::optional<std::string> TryGetEnvValue(TStringBuf name);
std::string GetEnvValueOrThrow(TStringBuf name);

//! Returns the value of the environment variable #name parsed as #T (via |FromString|).
//! Throws if the variable is not set or its value cannot be parsed.
template <class T>
T GetEnvValueOrThrow(TStringBuf name);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define ENV_INL_H_
#include "env-inl.h"
#undef ENV_INL_H_
