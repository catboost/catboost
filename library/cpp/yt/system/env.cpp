#include "env.h"

#include <library/cpp/yt/exception/exception.h>

#include <util/string/printf.h>

#include <util/system/platform.h>
#include <util/system/env.h>

#include <util/generic/maybe.h>

#ifdef _darwin_
    #include <crt_externs.h>
    #define environ (*_NSGetEnviron())
#endif

#ifdef _linux_
    #include <unistd.h>
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

#if defined(_linux_) || defined(_darwin_)

std::vector<std::string> GetEnvironNameValuePairs()
{
    std::vector<std::string> result;
    for (char** envIt = environ; *envIt; ++envIt) {
        result.emplace_back(*envIt);
    }
    return result;
}

#endif

std::pair<TStringBuf, std::optional<TStringBuf>> ParseEnvironNameValuePair(TStringBuf pair)
{
    if (auto pos = pair.find('='); pos != std::string::npos) {
        return {pair.substr(0, pos), pair.substr(pos + 1)};
    } else {
        return {pair, std::nullopt};
    }
}

std::optional<std::string> TryGetEnvValue(TStringBuf name)
{
    auto result = TryGetEnv(TString(name));
    return result ? std::optional<std::string>(*result) : std::nullopt;
}

std::string GetEnvValueOrThrow(TStringBuf name)
{
    auto value = TryGetEnvValue(name);
    if (!value) {
        throw TSimpleException(Sprintf("Environment variable \"%s\" is not set", name.data()));
    }
    return *value;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
