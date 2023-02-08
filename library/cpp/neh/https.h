#pragma once

#include <openssl/ossl_typ.h>

#include <util/generic/string.h>
#include <util/generic/strbuf.h>

#include <functional>

namespace NNeh {
    class IProtocol;
    struct TParsedLocation;

    IProtocol* SSLGetProtocol();
    IProtocol* SSLPostProtocol();
    IProtocol* SSLFullProtocol();

    /// if exceed soft limit, reduce quantity unused connections in cache
    void SetHttpOutputConnectionsLimits(size_t softLimit, size_t hardLimit);

    /// if exceed soft limit, reduce keepalive time for unused connections
    void SetHttpInputConnectionsLimits(size_t softLimit, size_t hardLimit);

    /// unused input sockets keepalive timeouts
    /// real(used) timeout:
    ///   - max, if not reached soft limit
    ///   - min, if reached hard limit
    ///   - approx. linear changed[max..min], while conn. count in range [soft..hard]
    void SetHttpInputConnectionsTimeouts(unsigned minSeconds, unsigned maxSeconds);

    struct THttpsOptions {
        using TVerifyCallback = int (*)(int, X509_STORE_CTX*);
        using TPasswordCallback = std::function<TString (const TParsedLocation&, const TString&, const TString&)>;
        static TString CAFile;
        static TString CAPath;
        static TString ClientCertificate;
        static TString ClientPrivateKey;
        static TString ClientPrivateKeyPassword;
        static bool CheckCertificateHostname;
        static bool EnableSslServerDebug;
        static bool EnableSslClientDebug;
        static TVerifyCallback ClientVerifyCallback;
        static TPasswordCallback KeyPasswdCallback;
        static bool RedirectionNotError;
        static bool Set(TStringBuf name, TStringBuf value);
    };
}
