#pragma once

#include <contrib/libs/openssl/include/openssl/ossl_typ.h>

#include <util/generic/string.h>
#include <util/generic/strbuf.h>

#include <functional>

namespace NNeh {
    class IProtocol;

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
        static TString CAFile;
        static TString CAPath;
        static bool CheckCertificateHostname;
        static bool EnableSslServerDebug;
        static bool EnableSslClientDebug;
        static TVerifyCallback ClientVerifyCallback;
        static bool Set(TStringBuf name, TStringBuf value);
    };
}
