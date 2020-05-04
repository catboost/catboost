#pragma once

#include <library/cpp/digest/lower_case/hash_ops.h>

#include <util/str_stl.h>

#include <util/system/defaults.h>
#include <util/string/cast.h>
#include <library/cpp/cgiparam/cgiparam.h>
#include <util/network/address.h>
#include <util/network/socket.h>
#include <util/generic/hash.h>
#include <util/system/yassert.h>
#include <util/generic/string.h>
#include <util/datetime/base.h>
#include <util/generic/buffer.h>

class TBaseServerRequestData {
    typedef THashMap<TString, TString, TCIOps, TCIOps> HeaderInHash;

public:
    TBaseServerRequestData(SOCKET s = INVALID_SOCKET);
    TBaseServerRequestData(const char* qs, SOCKET s = INVALID_SOCKET);

    void SetHost(const TString& host, ui16 port) {
        Host = host;
        Port = ToString(port);
    }

    const TString& ServerName() const {
        return Host;
    }

    NAddr::IRemoteAddrPtr ServerAddress() const {
        return NAddr::GetSockAddr(Socket);
    }

    const TString& ServerPort() const {
        return Port;
    }

    const char* ScriptName() const {
        return Path;
    }

    const char* QueryString() const {
        return Search;
    }

    TStringBuf QueryStringBuf() const {
        return TStringBuf(Search, SearchLength);
    }

    TStringBuf OrigQueryStringBuf() const {
        return OrigSearch;
    }

    void AppendQueryString(const char* str, size_t length);
    const char* RemoteAddr() const;
    void SetRemoteAddr(TStringBuf addr);
    const char* HeaderIn(const char* key) const;

    const HeaderInHash& HeadersIn() const {
        return HeadersIn_;
    }

    inline size_t HeadersCount() const noexcept {
        return HeadersIn_.size();
    }

    TString HeaderByIndex(size_t n) const noexcept;
    const char* Environment(const char* key) const;

    void Clear();

    void SetSocket(SOCKET s) noexcept {
        Socket = s;
    }

    ui64 RequestBeginTime() const noexcept {
        return BeginTime;
    }

    void SetPath(const TString& path);
    const char* GetCurPage() const;
    bool Parse(const char* req);
    void AddHeader(const TString& name, const TString& value);

private:
    TBuffer PathStorage;
    mutable char* Addr;
    TString Host;
    TString Port;
    char* Path;
    char* Search;
    size_t SearchLength; // length of Search
    TStringBuf OrigSearch;
    HeaderInHash HeadersIn_;
    mutable char AddrData[INET6_ADDRSTRLEN];
    SOCKET Socket;
    ui64 BeginTime;
    mutable TString CurPage;
    TBuffer ParseBuf;
    TBuffer ModifiedQueryString;
};

class TServerRequestData: public TBaseServerRequestData {
public:
    TServerRequestData(SOCKET s = INVALID_SOCKET)
        : TBaseServerRequestData(s)
    {
    }
    TServerRequestData(const char* qs, SOCKET s = INVALID_SOCKET)
        : TBaseServerRequestData(qs, s)
    {
        Scan();
    }

    void Scan() {
        CgiParam.Scan(QueryStringBuf());
    }

public:
    TCgiParameters CgiParam;
};
