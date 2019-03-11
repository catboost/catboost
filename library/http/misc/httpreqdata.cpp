#include "httpreqdata.h"

#include <util/stream/mem.h>

TBaseServerRequestData::TBaseServerRequestData(SOCKET s)
    : Addr(nullptr)
    , Host()
    , Port()
    , Path(nullptr)
    , Search(nullptr)
    , SearchLength(0)
    , Socket(s)
    , BeginTime(MicroSeconds())
{
}

TBaseServerRequestData::TBaseServerRequestData(const char* qs, SOCKET s)
    : Addr(nullptr)
    , Host()
    , Port()
    , Path(nullptr)
    , Search((char*)qs)
    , SearchLength(qs ? strlen(qs) : 0)
    , OrigSearch(Search, SearchLength)
    , Socket(s)
    , BeginTime(MicroSeconds())
{
}

void TBaseServerRequestData::AppendQueryString(const char* str, size_t length) {
    if (Y_UNLIKELY(Search)) {
        Y_ASSERT(strlen(Search) == SearchLength);
        ModifiedQueryString.Reserve(SearchLength + length + 2);
        ModifiedQueryString.Assign(Search, SearchLength);
        if (SearchLength > 0 && Search[SearchLength - 1] != '&' &&
            length > 0 && str[0] != '&') {
            ModifiedQueryString.Append('&');
        }
        ModifiedQueryString.Append(str, length);
    } else {
        ModifiedQueryString.Reserve(length + 1);
        ModifiedQueryString.Assign(str, length);
    }
    ModifiedQueryString.Append('\0');
    Search = ModifiedQueryString.data();
    SearchLength = ModifiedQueryString.size() - 1; // ignore terminator
}

void TBaseServerRequestData::SetRemoteAddr(TStringBuf addr) {
    TMemoryOutput out(AddrData, Y_ARRAY_SIZE(AddrData) - 1);
    out.Write(addr.substr(0, Y_ARRAY_SIZE(AddrData) - 1));
    *out.Buf() = '\0';

    Addr = AddrData;
}

const char* TBaseServerRequestData::RemoteAddr() const {
    if (!Addr) {
        *AddrData = 0;
        GetRemoteAddr(Socket, AddrData, sizeof(AddrData));
        Addr = AddrData;
    }

    return Addr;
}

const char* TBaseServerRequestData::HeaderIn(const char* key) const {
    auto it = HeadersIn_.find(key);

    if (it == HeadersIn_.end()) {
        return nullptr;
    }

    return it->second.data();
}

TString TBaseServerRequestData::HeaderByIndex(size_t n) const noexcept {
    if (n >= HeadersCount()) {
        return nullptr;
    }

    HeaderInHash::const_iterator i = HeadersIn_.begin();

    while (n) {
        ++i;
        --n;
    }

    return TString(i->first) + AsStringBuf(": ") + i->second;
}

const char* TBaseServerRequestData::Environment(const char* key) const {
    if (stricmp(key, "REMOTE_ADDR") == 0) {
        const char* ip = HeaderIn("X-Real-IP");
        if (ip)
            return ip;
        return RemoteAddr();
    } else if (stricmp(key, "QUERY_STRING") == 0) {
        return QueryString();
    } else if (stricmp(key, "SERVER_NAME") == 0) {
        return ServerName().data();
    } else if (stricmp(key, "SERVER_PORT") == 0) {
        return ServerPort().data();
    } else if (stricmp(key, "SCRIPT_NAME") == 0) {
        return ScriptName();
    }
    return nullptr;
}

void TBaseServerRequestData::Clear() {
    HeadersIn_.clear();
    Addr = Path = Search = nullptr;
    OrigSearch = {};
    SearchLength = 0;
    Host.clear();
    Port.clear();
    CurPage.remove();
    ParseBuf.Clear();
    BeginTime = MicroSeconds();
}

const char* TBaseServerRequestData::GetCurPage() const {
    if (!CurPage && Host) {
        CurPage = "http://";
        CurPage += Host;
        if (Port) {
            CurPage += ':';
            CurPage += Port;
        }
        CurPage += Path;
        if (Search) {
            CurPage += '?';
            CurPage += Search;
        }
    }
    return CurPage.data();
}

bool TBaseServerRequestData::Parse(const char* origReq) {
    size_t origReqLength = strlen(origReq);
    ParseBuf.Assign(origReq, origReqLength + 1);
    char* req = ParseBuf.Data();

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
    Path = req;

    // calculate Search length without additional strlen-ing
    Search = strchr(Path, '?');
    if (Search) {
        *Search++ = 0;
        ptrdiff_t delta = fragment - Search;
        // indeed, second case is a parse error
        SearchLength = (delta >= 0) ? delta : (urlEnd - Search);
        Y_ASSERT(strlen(Search) == SearchLength);
    } else {
        SearchLength = 0;
    }
    OrigSearch = {Search, SearchLength};

    return true;
}

void TBaseServerRequestData::AddHeader(const TString& name, const TString& value) {
    HeadersIn_[name] = value;

    if (stricmp(name.data(), "Host") == 0) {
        size_t hostLen = strcspn(value.data(), ":");
        if (value[hostLen] == ':')
            Port = value.substr(hostLen + 1);
        Host = value.substr(0, hostLen);
    }
}

void TBaseServerRequestData::SetPath(const TString& path) {
    PathStorage = TBuffer(path.data(), path.size() + 1);
    Path = PathStorage.Data();
}
