#include "env.h"

#include <util/system/platform.h>

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

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
