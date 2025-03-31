#pragma once

#include <library/cpp/digest/lower_case/hash_ops.h>

#include <util/str_stl.h>

#include <util/system/defaults.h>
#include <util/string/cast.h>
#include <library/cpp/cgiparam/cgiparam.h>
#include <util/memory/blob.h>
#include <util/network/address.h>
#include <util/network/socket.h>
#include <util/generic/hash.h>
#include <util/system/yassert.h>
#include <util/generic/string.h>
#include <util/datetime/base.h>
#include <util/generic/vector.h>
#include <util/generic/maybe.h>

using THttpHeadersContainer = THashMap<TString, TString, TCIOps, TCIOps>;

class TBaseServerRequestData {
public:
    TBaseServerRequestData(SOCKET s = INVALID_SOCKET);
    TBaseServerRequestData(TStringBuf qs, SOCKET s = INVALID_SOCKET);

    void SetHost(const TString& host, ui16 port) {
        Host_ = host;
        Port_ = ToString(port);
    }

    const TString& ServerName() const {
        return Host_;
    }

    NAddr::IRemoteAddrPtr ServerAddress() const {
        return NAddr::GetSockAddr(Socket_);
    }

    const TString& ServerPort() const {
        return Port_;
    }

    TStringBuf ScriptName() const {
        return Path_;
    }

    TStringBuf Query() const {
        return Query_;
    }

    TStringBuf OrigQuery() const {
        return OrigQuery_;
    }

    TStringBuf Body() const {
        return Body_.AsStringBuf();
    }

    void AppendQueryString(TStringBuf str);
    TStringBuf RemoteAddr() const;
    void SetRemoteAddr(TStringBuf addr);
    // Returns nullptr when the header does not exist
    const TString* HeaderIn(TStringBuf key) const;
    // Throws on missing header
    TStringBuf HeaderInOrEmpty(TStringBuf key) const;

    const THttpHeadersContainer& HeadersIn() const {
        return HeadersIn_;
    }

    inline size_t HeadersCount() const noexcept {
        return HeadersIn_.size();
    }

    TString HeaderByIndex(size_t n) const noexcept;
    TStringBuf Environment(TStringBuf key) const;

    void Clear();

    void SetSocket(SOCKET s) noexcept {
        Socket_ = s;
    }

    void SetBody(const TBlob& body) noexcept {
        Body_ = body;
    }

    ui64 RequestBeginTime() const noexcept {
        return BeginTime_;
    }

    void SetPath(TString path);
    const TString& GetCurPage() const;
    bool Parse(TStringBuf req);
    void AddHeader(const TString& name, const TString& value);

private:
    mutable TMaybe<TString> Addr_;
    TString Host_;
    TString Port_;
    TString Path_;
    TStringBuf Query_;
    TStringBuf OrigQuery_;
    TBlob Body_;
    THttpHeadersContainer HeadersIn_;
    SOCKET Socket_;
    ui64 BeginTime_;
    mutable TString CurPage_;
    TVector<char> ParseBuf_;
    TString ModifiedQueryString_;
};

class TServerRequestData: public TBaseServerRequestData {
public:
    TServerRequestData(SOCKET s = INVALID_SOCKET)
        : TBaseServerRequestData(s)
    {
    }
    TServerRequestData(TStringBuf qs, SOCKET s = INVALID_SOCKET)
        : TBaseServerRequestData(qs, s)
    {
        Scan();
    }

    void Scan() {
        CgiParam.Scan(Query());
    }

public:
    TCgiParameters CgiParam;
};
