#include "httpreqdata.h"

#include <library/cpp/case_insensitive_string/case_insensitive_string.h>

#include <util/stream/mem.h>
#include <util/string/join.h>

#include <array>

#ifdef _sse4_2_
#include <smmintrin.h>
#endif

TBaseServerRequestData::TBaseServerRequestData(SOCKET s)
    : Socket_(s)
    , BeginTime_(MicroSeconds())
{
}

TBaseServerRequestData::TBaseServerRequestData(TStringBuf qs, SOCKET s)
    : Query_(qs)
    , OrigQuery_(Query_)
    , Socket_(s)
    , BeginTime_(MicroSeconds())
{
}

void TBaseServerRequestData::AppendQueryString(TStringBuf str) {
    if (Y_UNLIKELY(!Query_.empty())) {
        TStringBuf separator = !Query_.EndsWith('&') && !str.StartsWith('&') ? "&"sv : ""sv;
        ModifiedQueryString_ = TString::Join(Query_, separator, str);
     } else {
        ModifiedQueryString_ = str;
     }
    Query_ = ModifiedQueryString_;
}

void TBaseServerRequestData::SetRemoteAddr(TStringBuf addr) {
    Addr_.ConstructInPlace(addr.substr(0, INET6_ADDRSTRLEN - 1));
}

TStringBuf TBaseServerRequestData::RemoteAddr() const {
    if (!Addr_) {
        auto& addr = Addr_.ConstructInPlace();
        addr.ReserveAndResize(INET6_ADDRSTRLEN);
        if (GetRemoteAddr(Socket_, addr.begin(), addr.size())) {
            if (auto pos = addr.find('\0'); pos != TString::npos) {
                addr.resize(pos);
            }
        } else {
            addr.clear();
        }
     }

    return *Addr_;
 }

const TString* TBaseServerRequestData::HeaderIn(TStringBuf key) const {
    return HeadersIn_.FindPtr(key);
}

TStringBuf TBaseServerRequestData::HeaderInOrEmpty(TStringBuf key) const {
    const auto* ptr = HeaderIn(key);
    return ptr ? TStringBuf{*ptr} : TStringBuf{};
}

TString TBaseServerRequestData::HeaderByIndex(size_t n) const noexcept {
    if (n >= HeadersIn_.size()) {
        return {};
    }

    const auto& [key, value] = *std::next(HeadersIn_.begin(), n);

    return TString::Join(key, ": ", value);
}

TStringBuf TBaseServerRequestData::Environment(TStringBuf key) const {
    TCaseInsensitiveStringBuf ciKey(key.data(), key.size());
    if (ciKey == "REMOTE_ADDR") {
        const auto ip = HeaderIn("X-Real-IP");
        return ip ? *ip : RemoteAddr();
    } else if (ciKey == "QUERY_STRING") {
        return Query();
    } else if (ciKey == "SERVER_NAME") {
        return ServerName();
    } else if (ciKey == "SERVER_PORT") {
        return ServerPort();
    } else if (ciKey == "SCRIPT_NAME") {
        return ScriptName();
    }
    return {};
}

 void TBaseServerRequestData::Clear() {
    HeadersIn_.clear();
    Addr_ = Nothing();
    Path_.clear();
    Query_ = {};
    OrigQuery_ = {};
    Host_.clear();
    Port_.clear();
    CurPage_.remove();
    ParseBuf_.clear();
    BeginTime_ = MicroSeconds();
}

const TString& TBaseServerRequestData::GetCurPage() const {
    if (!CurPage_ && Host_) {
        std::array<TStringBuf, 7> fragments;
        auto fragmentIt = fragments.begin();
        *fragmentIt++ = "http://"sv;
        *fragmentIt++ = Host_;
        if (Port_) {
            *fragmentIt++ = ":"sv;
            *fragmentIt++ = Port_;
        }
        *fragmentIt++ = Path_;
        if (!Query_.empty()) {
            *fragmentIt++ = "?"sv;
            *fragmentIt++ = Query_;
        }

        CurPage_ = JoinRange(""sv, fragments.begin(), fragmentIt);
    }
    return CurPage_;
}

bool TBaseServerRequestData::Parse(TStringBuf origReq) {
    ParseBuf_.reserve(origReq.size() + 16);
    ParseBuf_.assign(origReq.begin(), origReq.end());
    ParseBuf_.insert(ParseBuf_.end(), 15, ' ');
    ParseBuf_.push_back('\0');
    char* req = ParseBuf_.data();

    while (*req == ' ' || *req == '\t')
        req++;
    if (*req != '/')
        return false;     // we are not a proxy
    while (req[1] == '/') // remove redundant slashes
        req++;

    char* pathBegin = req;
    char* queryBegin = nullptr;

#ifdef _sse4_2_
    const __m128i simdSpace = _mm_set1_epi8(' ');
    const __m128i simdTab = _mm_set1_epi8('\t');
    const __m128i simdHash = _mm_set1_epi8('#');
    const __m128i simdQuestion = _mm_set1_epi8('?');

    auto isEnd = [=](__m128i x) {
        const auto v = _mm_or_si128(
                _mm_or_si128(
                    _mm_cmpeq_epi8(x, simdSpace), _mm_cmpeq_epi8(x, simdTab)),
                _mm_cmpeq_epi8(x, simdHash));
        return !_mm_testz_si128(v, v);
    };

    // No need for the range check because we have padding of spaces at the end.
    for (;; req += 16) {
        const auto x = _mm_loadu_si128(reinterpret_cast<const __m128i *>(req));
        const auto isQuestionSimd = _mm_cmpeq_epi8(x, simdQuestion);
        const auto isQuestion = !_mm_testz_si128(isQuestionSimd, isQuestionSimd);
        if (isEnd(x)) {
            if (isQuestion) {
                // The prospective query end and a question sign are both in the
                // current block. Need to find out which comes first.
                for (;*req != ' ' && *req != '\t' && *req != '#'; ++req) {
                    if (*req == '?') {
                        queryBegin = req + 1;
                        break;
                    }
                }
            }
            break;
        }
        if (isQuestion) {
            // Find the exact query beginning
            for (queryBegin = req; *queryBegin != '?'; ++queryBegin) {
            }
            ++queryBegin;

            break;
        }
    }

    // If we bailed out because we found query string begin. Now look for the the end of the query
    if (queryBegin) {
        for (;; req += 16) {
            const auto x = _mm_loadu_si128(reinterpret_cast<const __m128i *>(req));
            if (isEnd(x)) {
                break;
            }
        }
    }
#else
    for (;*req != ' ' && *req != '\t' && *req != '#'; ++req) {
        if (*req == '?') {
            queryBegin = req + 1;
            break;
        }
    }
#endif

    while (*req != ' ' && *req != '\t' && *req != '#') {
        ++req;
    }

    char* pathEnd = queryBegin ? queryBegin - 1 : req;
    // Make sure Path_ and Query_ are actually zero-reminated.
    *pathEnd = '\0';
    *req = '\0';
    Path_ = TStringBuf{pathBegin, pathEnd};
    if (queryBegin) {
        Query_ = TStringBuf{queryBegin, req};
        OrigQuery_ = Query_;
    } else {
        Query_ = {};
        OrigQuery_ = {};
    }

    return true;
}

void TBaseServerRequestData::AddHeader(const TString& name, const TString& value) {
    HeadersIn_[name] = value;

    if (stricmp(name.data(), "Host") == 0) {
        size_t hostLen = strcspn(value.data(), ":");
        if (value[hostLen] == ':')
            Port_ = value.substr(hostLen + 1);
        Host_ = value.substr(0, hostLen);
    }
}

void TBaseServerRequestData::SetPath(TString path) {
    Path_ = std::move(path);
}
