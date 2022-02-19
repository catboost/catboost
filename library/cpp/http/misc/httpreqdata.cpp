#include "httpreqdata.h"

#include <library/cpp/case_insensitive_string/case_insensitive_string.h>
#include <util/stream/mem.h>

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
        CurPage_ = "http://";
        CurPage_ += Host_;
        if (Port_) {
            CurPage_ += ':';
            CurPage_ += Port_;
        }
        CurPage_ += Path_;
        if (Query_) {
            CurPage_ += '?';
            CurPage_ += Query_;
        }
    }
    return CurPage_;
}

bool TBaseServerRequestData::Parse(TStringBuf origReqBuf) {
    ParseBuf_.reserve(origReqBuf.size() + 1);
    ParseBuf_.assign(origReqBuf.begin(), origReqBuf.end());
    ParseBuf_.push_back('\0');
    char* req = ParseBuf_.data();

    while (*req == ' ' || *req == '\t')
        req++;
    if (*req != '/')
        return false;     // we are not a proxy
    while (req[1] == '/') // remove redundant slashes
        req++;

    // detect url end (can contain some garbage after whitespace, e.g. 'HTTP 1.1')
    char* urlEnd = req;
    while (*urlEnd && *urlEnd != ' ' && *urlEnd != '\t')
        urlEnd++;
    if (*urlEnd)
        *urlEnd = 0;

    // cut fragment if exists
    char* fragment = strchr(req, '#');
    if (fragment)
        *fragment = 0; // ignore fragment
    else
        fragment = urlEnd;
    char* path = req;

    // calculate Search length without additional strlen-ing
    char* query = strchr(path, '?');
    if (query) {
        *query++ = 0;
        ptrdiff_t delta = fragment - query;
        // indeed, second case is a parse error
        Query_ = {query, static_cast<size_t>(delta >= 0 ? delta : (urlEnd - query))};
    } else {
        Query_ = {};
    }
    Path_ = path;
    OrigQuery_ = Query_;

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
