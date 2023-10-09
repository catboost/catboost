#include "https.h"

#include "details.h"
#include "factory.h"
#include "http_common.h"
#include "jobqueue.h"
#include "location.h"
#include "multi.h"
#include "pipequeue.h"
#include "utils.h"

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/bio.h>
#include <openssl/x509v3.h>

#include <library/cpp/openssl/init/init.h>
#include <library/cpp/openssl/method/io.h>
#include <library/cpp/coroutine/listener/listen.h>
#include <library/cpp/dns/cache.h>
#include <library/cpp/http/misc/parsed_request.h>
#include <library/cpp/http/misc/httpcodes.h>
#include <library/cpp/http/io/stream.h>

#include <util/generic/cast.h>
#include <util/generic/list.h>
#include <util/generic/utility.h>
#include <util/network/socket.h>
#include <util/stream/str.h>
#include <util/stream/zlib.h>
#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/system/condvar.h>
#include <util/system/error.h>
#include <util/system/types.h>
#include <util/thread/factory.h>

#include <atomic>

#if defined(_unix_)
#include <sys/ioctl.h>
#endif

#if defined(_linux_)
#undef SIOCGSTAMP
#undef SIOCGSTAMPNS
#include <linux/sockios.h>
#define FIONWRITE SIOCOUTQ
#endif

using namespace NDns;
using namespace NAddr;

namespace NNeh {
    TString THttpsOptions::CAFile;
    TString THttpsOptions::CAPath;
    TString THttpsOptions::ClientCertificate;
    TString THttpsOptions::ClientPrivateKey;
    TString THttpsOptions::ClientPrivateKeyPassword;
    bool THttpsOptions::EnableSslServerDebug = false;
    bool THttpsOptions::EnableSslClientDebug = false;
    bool THttpsOptions::CheckCertificateHostname = false;
    THttpsOptions::TVerifyCallback THttpsOptions::ClientVerifyCallback = nullptr;
    THttpsOptions::TPasswordCallback THttpsOptions::KeyPasswdCallback = nullptr;
    bool THttpsOptions::RedirectionNotError = false;

    bool THttpsOptions::Set(TStringBuf name, TStringBuf value) {
#define YNDX_NEH_HTTPS_TRY_SET(optName)                 \
    if (name == TStringBuf(#optName)) {                 \
        optName = FromString<decltype(optName)>(value); \
        return true;                                    \
    }

        YNDX_NEH_HTTPS_TRY_SET(CAFile);
        YNDX_NEH_HTTPS_TRY_SET(CAPath);
        YNDX_NEH_HTTPS_TRY_SET(ClientCertificate);
        YNDX_NEH_HTTPS_TRY_SET(ClientPrivateKey);
        YNDX_NEH_HTTPS_TRY_SET(ClientPrivateKeyPassword);
        YNDX_NEH_HTTPS_TRY_SET(EnableSslServerDebug);
        YNDX_NEH_HTTPS_TRY_SET(EnableSslClientDebug);
        YNDX_NEH_HTTPS_TRY_SET(CheckCertificateHostname);
        YNDX_NEH_HTTPS_TRY_SET(RedirectionNotError);

#undef YNDX_NEH_HTTPS_TRY_SET

        return false;
    }
}

namespace NNeh {
    namespace NHttps {
        namespace {
            // force ssl_write/ssl_read functions to return this value via BIO_method_read/write that means request is canceled
            constexpr int SSL_RVAL_TIMEOUT = -42;

            struct TInputConnections {
                TInputConnections()
                    : Counter(0)
                    , MaxUnusedConnKeepaliveTimeout(120)
                    , MinUnusedConnKeepaliveTimeout(10)
                {
                }

                inline size_t ExceedSoftLimit() const noexcept {
                    return NHttp::TFdLimits::ExceedLimit(Counter.Val(), Limits.Soft());
                }

                inline size_t ExceedHardLimit() const noexcept {
                    return NHttp::TFdLimits::ExceedLimit(Counter.Val(), Limits.Hard());
                }

                inline size_t DeltaLimit() const noexcept {
                    return Limits.Delta();
                }

                unsigned UnusedConnKeepaliveTimeout() const {
                    if (size_t e = ExceedSoftLimit()) {
                        size_t d = DeltaLimit();
                        size_t leftAvailableFd = NHttp::TFdLimits::ExceedLimit(d, e);
                        unsigned r = static_cast<unsigned>(MaxUnusedConnKeepaliveTimeout.load(std::memory_order_acquire) * leftAvailableFd / (d + 1));
                        return Max(r, (unsigned)MinUnusedConnKeepaliveTimeout.load(std::memory_order_acquire));
                    }
                    return MaxUnusedConnKeepaliveTimeout.load(std::memory_order_acquire);
                }

                void SetFdLimits(size_t soft, size_t hard) {
                    Limits.SetSoft(soft);
                    Limits.SetHard(hard);
                }

                NHttp::TFdLimits Limits;
                TAtomicCounter Counter;
                std::atomic<unsigned> MaxUnusedConnKeepaliveTimeout; //in seconds
                std::atomic<unsigned> MinUnusedConnKeepaliveTimeout; //in seconds
            };

            TInputConnections* InputConnections() {
                return Singleton<TInputConnections>();
            }

            struct TSharedSocket: public TSocketHolder, public TAtomicRefCount<TSharedSocket> {
                inline TSharedSocket(TSocketHolder& s)
                    : TSocketHolder(s.Release())
                {
                    InputConnections()->Counter.Inc();
                }

                ~TSharedSocket() {
                    InputConnections()->Counter.Dec();
                }
            };

            using TSocketRef = TIntrusivePtr<TSharedSocket>;

            struct TX509Deleter {
                static void Destroy(X509* cert) {
                    X509_free(cert);
                }
            };
            using TX509Holder = THolder<X509, TX509Deleter>;

            struct TSslSessionDeleter {
                static void Destroy(SSL_SESSION* sess) {
                    SSL_SESSION_free(sess);
                }
            };
            using TSslSessionHolder = THolder<SSL_SESSION, TSslSessionDeleter>;

            struct TSslDeleter {
                static void Destroy(SSL* ssl) {
                    SSL_free(ssl);
                }
            };
            using TSslHolder = THolder<SSL, TSslDeleter>;

            // read from bio and write via operator<<() to dst
            template <typename T>
            class TBIOInput : public NOpenSSL::TAbstractIO {
            public:
                TBIOInput(T& dst)
                    : Dst_(dst)
                {
                }

                int Write(const char* data, size_t dlen, size_t* written) override {
                    Dst_ << TStringBuf(data, dlen);
                    *written = dlen;
                    return 1;
                }

                int Read(char* data, size_t dlen, size_t* readbytes) override {
                    Y_UNUSED(data);
                    Y_UNUSED(dlen);
                    Y_UNUSED(readbytes);
                    return -1;
                }

                int Puts(const char* buf) override {
                    Y_UNUSED(buf);
                    return -1;
                }

                int Gets(char* buf, int len) override {
                    Y_UNUSED(buf);
                    Y_UNUSED(len);
                    return -1;
                }

                void Flush() override {
                }

            private:
                T& Dst_;
            };
        }

        class TSslException: public yexception {
        public:
            TSslException() = default;

            TSslException(TStringBuf f) {
                *this << f << Endl;
                InitErr();
            }

            TSslException(TStringBuf f, const SSL* ssl, int ret) {
                *this << f << TStringBuf(" error type: ");
                const int etype = SSL_get_error(ssl, ret);
                switch (etype) {
                    case SSL_ERROR_ZERO_RETURN:
                        *this << TStringBuf("SSL_ERROR_ZERO_RETURN");
                        break;
                    case SSL_ERROR_WANT_READ:
                        *this << TStringBuf("SSL_ERROR_WANT_READ");
                        break;
                    case SSL_ERROR_WANT_WRITE:
                        *this << TStringBuf("SSL_ERROR_WANT_WRITE");
                        break;
                    case SSL_ERROR_WANT_CONNECT:
                        *this << TStringBuf("SSL_ERROR_WANT_CONNECT");
                        break;
                    case SSL_ERROR_WANT_ACCEPT:
                        *this << TStringBuf("SSL_ERROR_WANT_ACCEPT");
                        break;
                    case SSL_ERROR_WANT_X509_LOOKUP:
                        *this << TStringBuf("SSL_ERROR_WANT_X509_LOOKUP");
                        break;
                    case SSL_ERROR_SYSCALL:
                        *this << TStringBuf("SSL_ERROR_SYSCALL ret: ") << ret << TStringBuf(", errno: ") << errno;
                        break;
                    case SSL_ERROR_SSL:
                        *this << TStringBuf("SSL_ERROR_SSL");
                        break;
                }
                *this << ' ';
                InitErr();
            }

        private:
            void InitErr() {
                TBIOInput<TSslException> bio(*this);
                ERR_print_errors(bio);
            }
        };

        namespace {
            enum EMatchResult {
                MATCH_FOUND,
                NO_MATCH,
                NO_EXTENSION,
                ERROR
            };
            bool EqualNoCase(TStringBuf a, TStringBuf b) {
                return (a.size() == b.size()) && ToString(a).to_lower() == ToString(b).to_lower();
            }
            bool MatchDomainName(TStringBuf tmpl, TStringBuf name) {
                // match wildcards only in the left-most part
                // do not support (optional according to RFC) partial wildcards (ww*.yandex.ru)
                // see RFC-6125
                TStringBuf tmplRest = tmpl;
                TStringBuf tmplFirst = tmplRest.NextTok('.');
                if (tmplFirst == "*") {
                    tmpl = tmplRest;
                    name.NextTok('.');
                }
                return EqualNoCase(tmpl, name);
            }

            EMatchResult MatchCertAltNames(X509* cert, TStringBuf hostname) {
                EMatchResult result = NO_MATCH;
                STACK_OF(GENERAL_NAME)* names = (STACK_OF(GENERAL_NAME)*)X509_get_ext_d2i(cert, NID_subject_alt_name, nullptr, NULL);
                if (!names) {
                    return NO_EXTENSION;
                }

                int namesCt = sk_GENERAL_NAME_num(names);
                for (int i = 0; i < namesCt; ++i) {
                    const GENERAL_NAME* name = sk_GENERAL_NAME_value(names, i);

                    if (name->type == GEN_DNS) {
                        TStringBuf dnsName((const char*)ASN1_STRING_get0_data(name->d.dNSName), ASN1_STRING_length(name->d.dNSName));
                        if (MatchDomainName(dnsName, hostname)) {
                            result = MATCH_FOUND;
                            break;
                        }
                    }
                }
                sk_GENERAL_NAME_pop_free(names, GENERAL_NAME_free);
                return result;
            }

            EMatchResult MatchCertCommonName(X509* cert, TStringBuf hostname) {
                int commonNameLoc = X509_NAME_get_index_by_NID(X509_get_subject_name(cert), NID_commonName, -1);
                if (commonNameLoc < 0) {
                    return ERROR;
                }

                X509_NAME_ENTRY* commonNameEntry = X509_NAME_get_entry(X509_get_subject_name(cert), commonNameLoc);
                if (!commonNameEntry) {
                    return ERROR;
                }

                ASN1_STRING* commonNameAsn1 = X509_NAME_ENTRY_get_data(commonNameEntry);
                if (!commonNameAsn1) {
                    return ERROR;
                }

                TStringBuf commonName((const char*)ASN1_STRING_get0_data(commonNameAsn1), ASN1_STRING_length(commonNameAsn1));

                return MatchDomainName(commonName, hostname)
                           ? MATCH_FOUND
                           : NO_MATCH;
            }

            bool CheckCertHostname(X509* cert, TStringBuf hostname) {
                switch (MatchCertAltNames(cert, hostname)) {
                    case MATCH_FOUND:
                        return true;
                        break;
                    case NO_EXTENSION:
                        return MatchCertCommonName(cert, hostname) == MATCH_FOUND;
                        break;
                    default:
                        return false;
                }
            }

            void ParseUserInfo(const TParsedLocation& loc, TString& cert, TString& pvtKey) {
                if (!loc.UserInfo) {
                    return;
                }

                TStringBuf kws = loc.UserInfo;
                while (kws) {
                    TStringBuf name = kws.NextTok('=');
                    TStringBuf value = kws.NextTok(';');
                    if (TStringBuf("cert") == name) {
                        cert = value;
                    } else if (TStringBuf("key") == name) {
                        pvtKey = value;
                    }
                }
            }

            struct TSSLInit {
                inline TSSLInit() {
                    InitOpenSSL();
                }
            } SSL_INIT;
        }

        static inline void PrepareSocket(SOCKET s) {
            SetNoDelay(s, true);
        }

        class TConnCache;
        static TConnCache* ConnectionCache();

        class TConnCache: public IThreadFactory::IThreadAble {
        public:
            struct TConnection;
            typedef TAutoPtr<TSocketHolder> TSocketRef;
            typedef THolder<TConnection> TConnectionHolder;
            typedef TAutoPtr<TConnectionHolder> TConnectionRef;
            typedef TAutoLockFreeQueue<TConnectionHolder> TConnList;

            struct TConnection {
                inline TConnection(TSocketRef& s, const TResolvedHost host) noexcept
                    : Socket(s)
                    , Host(host)
                {
                    ConnectionCache()->ActiveConnections.Inc();
                }

                inline ~TConnection() {
                    if (!!Socket && IsNotSocketClosedByOtherSide(*Socket)) {
                        if (!!Ssl) {
                            ResetBIO();
                            // do not wait for shutdown confirmation
                            Y_UNUSED(SSL_shutdown(Ssl.Get()));
                        }
                    }
                    ConnectionCache()->ActiveConnections.Dec();
                }

                void ResetBIO() {
                    if (!!Socket) {
                        BIO* bio = BIO_new_socket(*Socket, 0);
                        SSL_set_bio(Ssl.Get(), bio, bio);
                    }
                }

                bool HasSsl() const {
                    return Ssl.Get();
                }

                TSslHolder&& MoveSsl() {
                    return std::move(Ssl);
                }

                void SetSsl(TSslHolder&& ssl) {
                    Ssl = std::move(ssl);
                }

                bool ShutdownReceived() {
                    if (!Ssl) {
                        return false;
                    }
                    char buffer;
                    int rval = SSL_peek(Ssl.Get(), &buffer, sizeof(buffer));
                    if (rval) {
                        return false;
                    }
                    return (SSL_get_shutdown(Ssl.Get()) & SSL_RECEIVED_SHUTDOWN);
                }

                SOCKET Fd() {
                    return *Socket;
                }

            protected:
                friend class TConnCache;
                TSslHolder Ssl;
                TSocketRef Socket;

            public:
                const TResolvedHost Host;
            };

            TConnCache()
                : InPurging_(0)
                , MaxConnId_(0)
                , Shutdown_(false)
            {
                T_ = SystemThreadFactory()->Run(this);
            }

            ~TConnCache() override {
                {
                    TGuard<TMutex> g(PurgeMutex_);

                    Shutdown_ = true;
                    CondPurge_.Signal();
                }

                T_->Join();
            }

            TConnectionRef Connect(TCont* c, const TString& msgAddr, const TResolvedHost& addr, TErrorRef* error) {
                if (ExceedHardLimit()) {
                    if (error) {
                        *error = new TError("neh::https output connections limit reached", TError::TType::UnknownType);
                    }
                    return nullptr;
                }

                TConnectionRef res;
                TConnList& connList = ConnList(addr);

                while (connList.Dequeue(&res)) {
                    ActiveConnections.Inc();
                    CachedConnections.Dec();
                    if (IsNotSocketClosedByOtherSide((*res)->Fd()) && !(*res)->ShutdownReceived()) {
                        return res;
                    }
                }

                if (!c) {
                    if (error) {
                        *error = new TError("directo connection failed");
                    }
                    return nullptr;
                }

                const TInstant now(TInstant::Now());
                const TInstant deadline(now + TDuration::Seconds(10));
                TDuration delay = TDuration::MilliSeconds(8);
                TInstant checkpoint = Min(deadline, delay.ToDeadLine());

                TNetworkAddress::TIterator ait = addr.Addr.Begin();
                TSocketRef socket(new TSocketHolder(NCoro::Socket(*ait)));
                int ret = NCoro::ConnectD(c, *socket, *ait, deadline);
                res.Reset(new TConnectionHolder);
                res->Reset(new TConnection(socket, addr));

                if (ret) {
                    do {
                        if ((ret == ETIMEDOUT || ret == EINTR) && checkpoint < deadline) {
                            delay += delay;
                            checkpoint = Min(deadline, now + delay);
                            TConnectionRef res2;
                            if (connList.Dequeue(&res2)) {
                                ActiveConnections.Inc();
                                CachedConnections.Dec();
                                if (IsNotSocketClosedByOtherSide((*res2)->Fd()) && !(*res)->ShutdownReceived()) {
                                    return res2;
                                }
                            }
                        } else {
                            if (error) {
                                *error = new TError(TStringBuilder() << TStringBuf("can not connect to ") << msgAddr);
                            }
                            return nullptr;
                        }
                    } while (ret = NCoro::PollD(c, (*res)->Fd(), CONT_POLL_WRITE, checkpoint));
                }
                PrepareSocket((*res)->Fd());
                return res;
            }

            inline void Release(TConnectionRef conn) {
                if (!ExceedHardLimit()) {
                    size_t maxConnId = MaxConnId_.load(std::memory_order_acquire);

                    while (maxConnId < (*conn)->Host.Id) {
                        MaxConnId_.compare_exchange_strong(
                            maxConnId,
                            (*conn)->Host.Id,
                            std::memory_order_seq_cst,
                            std::memory_order_seq_cst);
                        maxConnId = MaxConnId_.load(std::memory_order_acquire);
                    }
                    ConnList((*conn)->Host).Enqueue(conn);
                    CachedConnections.Inc();
                    ActiveConnections.Dec();
                }

                if (CachedConnections.Val() && ExceedSoftLimit()) {
                    SuggestPurgeCache();
                }
            }

            void SetFdLimits(size_t soft, size_t hard) {
                Limits.SetSoft(soft);
                Limits.SetHard(hard);
            }

        private:
            void SuggestPurgeCache() {
                if (AtomicTryLock(&InPurging_)) {
                    //evaluate the usefulness of purging the cache
                    //если в кеше мало соединений (< MaxConnId_/16 или 64), не чистим кеш
                    if ((size_t)CachedConnections.Val() > (Min((size_t)MaxConnId_.load(std::memory_order_acquire), (size_t)1024U) >> 4)) {
                        //по мере приближения к hardlimit нужда в чистке cache приближается к 100%
                        size_t closenessToHardLimit256 = ((ActiveConnections.Val() + 1) << 8) / (Limits.Delta() + 1);
                        //чем больше соединений в кеше, а не в работе, тем менее нужен кеш (можно его почистить)
                        size_t cacheUselessness256 = ((CachedConnections.Val() + 1) << 8) / (ActiveConnections.Val() + 1);

                        //итого, - пороги срабатывания:
                        //при достижении soft-limit, если соединения в кеше, а не в работе
                        //на полпути от soft-limit к hard-limit, если в кеше больше половины соединений
                        //при приближении к hardlimit пытаться почистить кеш почти постоянно
                        if ((closenessToHardLimit256 + cacheUselessness256) >= 256U) {
                            TGuard<TMutex> g(PurgeMutex_);

                            CondPurge_.Signal();
                            return; //memo: thread MUST unlock InPurging_ (see DoExecute())
                        }
                    }
                    AtomicUnlock(&InPurging_);
                }
            }

            void DoExecute() override {
                while (true) {
                    {
                        TGuard<TMutex> g(PurgeMutex_);

                        if (Shutdown_)
                            return;

                        CondPurge_.WaitI(PurgeMutex_);
                    }

                    PurgeCache();

                    AtomicUnlock(&InPurging_);
                }
            }

            void PurgeCache() noexcept {
                //try remove at least ExceedSoftLimit() oldest connections from cache
                //вычисляем долю кеша, которую нужно почистить (в 256 долях) (но не менее 1/32 кеша)
                const size_t frac256 = Min<size_t>(256, Max<size_t>(8, (ExceedSoftLimit() << 8) / (CachedConnections.Val() + 1)));
                TConnectionRef tmp;

                for (size_t i = 0; i < MaxConnId_.load(std::memory_order_acquire) && !Shutdown_; i++) {
                    TConnList& tc = Lst_.Get(i);
                    if (size_t qsize = tc.Size()) {
                        //в каждой очереди чистим вычисленную долю
                        size_t purgeCounter = ((qsize * frac256) >> 8);

                        if (!purgeCounter && qsize) {
                            if (qsize == 1) {
                                // check lifeness
                                TConnectionRef res;
                                if (tc.Dequeue(&res)) {
                                    // if connection valid put it back
                                    if (IsNotSocketClosedByOtherSide((*res)->Fd()) && !(*res)->ShutdownReceived()) {
                                        tc.Enqueue(res);
                                    } else {
                                        ActiveConnections.Inc();
                                        CachedConnections.Dec();
                                    }
                                }
                            } else {
                                // drop at least one connection from queue with at least 2 connections
                                purgeCounter = 1;
                            }
                        }

                        while (purgeCounter-- && tc.Dequeue(&tmp)) {
                            ActiveConnections.Inc();
                            CachedConnections.Dec();
                            tmp->Reset(nullptr);
                        }
                    }
                }
            }

            inline TConnList& ConnList(const TResolvedHost& addr) {
                return Lst_.Get(addr.Id);
            }

            inline size_t TotalConnections() const noexcept {
                return ActiveConnections.Val() + CachedConnections.Val();
            }

            inline size_t ExceedSoftLimit() const noexcept {
                return NHttp::TFdLimits::ExceedLimit(TotalConnections(), Limits.Soft());
            }

            inline size_t ExceedHardLimit() const noexcept {
                return NHttp::TFdLimits::ExceedLimit(TotalConnections(), Limits.Hard());
            }

            NHttp::TFdLimits Limits;
            TAtomicCounter ActiveConnections;
            TAtomicCounter CachedConnections;

            NHttp::TLockFreeSequence<TConnList> Lst_;

            TAtomic InPurging_;
            std::atomic<size_t> MaxConnId_;

            TAutoPtr<IThreadFactory::IThread> T_;
            TCondVar CondPurge_;
            TMutex PurgeMutex_;
            TAtomicBool Shutdown_;
        };

        class TSslCtx: public TThrRefBase {
        protected:
            TSslCtx()
                : SslCtx_(nullptr)
            {
            }

        public:
            ~TSslCtx() override {
                SSL_CTX_free(SslCtx_);
            }

            operator SSL_CTX*() {
                return SslCtx_;
            }

        protected:
            SSL_CTX* SslCtx_;
        };
        using TSslCtxPtr = TIntrusivePtr<TSslCtx>;

        class TSslCtxServer: public TSslCtx {
            struct TPasswordCallbackUserData {
                TParsedLocation Location;
                TString         CertFileName;
                TString         KeyFileName;
            };
            class TUserDataHolder {
            public:
                TUserDataHolder(SSL_CTX* ctx, const TParsedLocation& location, const TString& certFileName, const TString& keyFileName)
                    : SslCtx_(ctx)
                    , Data_{location, certFileName, keyFileName}
                {
                    SSL_CTX_set_default_passwd_cb_userdata(SslCtx_, &Data_);
                }
                ~TUserDataHolder() {
                    SSL_CTX_set_default_passwd_cb_userdata(SslCtx_, nullptr);
                }
            private:
                SSL_CTX* SslCtx_;
                TPasswordCallbackUserData Data_;
            };
        public:
            TSslCtxServer(const TParsedLocation& loc) {
                const SSL_METHOD* method = SSLv23_server_method();
                if (Y_UNLIKELY(!method)) {
                    ythrow TSslException(TStringBuf("SSLv23_server_method"));
                }

                SslCtx_ = SSL_CTX_new(method);
                if (Y_UNLIKELY(!SslCtx_)) {
                    ythrow TSslException(TStringBuf("SSL_CTX_new(server)"));
                }

                TString cert, key;
                ParseUserInfo(loc, cert, key);

                TUserDataHolder holder(SslCtx_, loc, cert, key);

                SSL_CTX_set_default_passwd_cb(SslCtx_, [](char* buf, int size, int rwflag, void* userData) -> int {
                    Y_UNUSED(rwflag);
                    Y_UNUSED(userData);

                    if (THttpsOptions::KeyPasswdCallback == nullptr || userData == nullptr) {
                        return 0;
                    }

                    auto data = static_cast<TPasswordCallbackUserData*>(userData);
                    const auto& passwd = THttpsOptions::KeyPasswdCallback(data->Location, data->CertFileName, data->KeyFileName);

                    if (size < static_cast<int>(passwd.size())) {
                        return -1;
                    }

                    return passwd.copy(buf, size, 0);
                });

                if (!cert || !key) {
                    ythrow TSslException() << TStringBuf("no certificate or private key is specified for server");
                }

                if (1 != SSL_CTX_use_certificate_chain_file(SslCtx_, cert.data())) {
                    ythrow TSslException(TStringBuf("SSL_CTX_use_certificate_chain_file (server)"));
                }

                if (1 != SSL_CTX_use_PrivateKey_file(SslCtx_, key.data(), SSL_FILETYPE_PEM)) {
                    ythrow TSslException(TStringBuf("SSL_CTX_use_PrivateKey_file (server)"));
                }

                if (1 != SSL_CTX_check_private_key(SslCtx_)) {
                    ythrow TSslException(TStringBuf("SSL_CTX_check_private_key (server)"));
                }
            }
        };

        class TSslCtxClient: public TSslCtx {
        public:
            TSslCtxClient() {
                const SSL_METHOD* method = SSLv23_client_method();
                if (Y_UNLIKELY(!method)) {
                    ythrow TSslException(TStringBuf("SSLv23_client_method"));
                }

                SslCtx_ = SSL_CTX_new(method);
                if (Y_UNLIKELY(!SslCtx_)) {
                    ythrow TSslException(TStringBuf("SSL_CTX_new(client)"));
                }

                const TString& caFile = THttpsOptions::CAFile;
                const TString& caPath = THttpsOptions::CAPath;
                if (caFile || caPath) {
                    if (!SSL_CTX_load_verify_locations(SslCtx_, caFile ? caFile.data() : nullptr, caPath ? caPath.data() : nullptr)) {
                        ythrow TSslException(TStringBuf("SSL_CTX_load_verify_locations(client)"));
                    }
                }

                SSL_CTX_set_options(SslCtx_, SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3 | SSL_OP_NO_COMPRESSION);
                if (THttpsOptions::ClientVerifyCallback) {
                    SSL_CTX_set_verify(SslCtx_, SSL_VERIFY_PEER, THttpsOptions::ClientVerifyCallback);
                } else {
                    SSL_CTX_set_verify(SslCtx_, SSL_VERIFY_NONE, nullptr);
                }

                const TString& clientCertificate = THttpsOptions::ClientCertificate;
                const TString& clientPrivateKey = THttpsOptions::ClientPrivateKey;
                if (clientCertificate && clientPrivateKey) {
                    SSL_CTX_set_default_passwd_cb(SslCtx_, [](char* buf, int size, int rwflag, void* userData) -> int {
                        Y_UNUSED(rwflag);
                        Y_UNUSED(userData);

                        const TString& clientPrivateKeyPwd = THttpsOptions::ClientPrivateKeyPassword;
                        if (!clientPrivateKeyPwd) {
                            return 0;
                        }
                        if (size < static_cast<int>(clientPrivateKeyPwd.size())) {
                            return -1;
                        }

                        return clientPrivateKeyPwd.copy(buf, size, 0);
                    });
                    if (1 != SSL_CTX_use_certificate_chain_file(SslCtx_, clientCertificate.c_str())) {
                        ythrow TSslException(TStringBuf("SSL_CTX_use_certificate_chain_file (client)"));
                    }
                    if (1 != SSL_CTX_use_PrivateKey_file(SslCtx_, clientPrivateKey.c_str(), SSL_FILETYPE_PEM)) {
                        ythrow TSslException(TStringBuf("SSL_CTX_use_PrivateKey_file (client)"));
                    }
                    if (1 != SSL_CTX_check_private_key(SslCtx_)) {
                        ythrow TSslException(TStringBuf("SSL_CTX_check_private_key (client)"));
                    }
                } else if (clientCertificate || clientPrivateKey) {
                    ythrow TSslException() << TStringBuf("both certificate and private key must be specified for client");
                }
            }

            static TSslCtxClient& Instance() {
                return *Singleton<TSslCtxClient>();
            }
        };

        class TContBIO : public NOpenSSL::TAbstractIO {
        public:
            TContBIO(SOCKET s, const TAtomicBool* canceled = nullptr)
                : Timeout_(TDuration::MicroSeconds(10000))
                , S_(s)
                , Canceled_(canceled)
                , Cont_(nullptr)
            {
            }

            SOCKET Socket() {
                return S_;
            }

            int PollT(int what, const TDuration& timeout) {
                return NCoro::PollT(Cont_, Socket(), what, timeout);
            }

            void WaitUntilWritten() {
#if defined(FIONWRITE)
                if (Y_LIKELY(Cont_)) {
                    int err;
                    int nbytes = Max<int>();
                    TDuration tout = TDuration::MilliSeconds(10);

                    while (((err = ioctl(S_, FIONWRITE, &nbytes)) == 0) && nbytes) {
                        err = NCoro::PollT(Cont_, S_, CONT_POLL_READ, tout);

                        if (!err) {
                            //wait complete, cause have some data
                            break;
                        }

                        if (err != ETIMEDOUT) {
                            ythrow TSystemError(err) << TStringBuf("request failed");
                        }

                        tout = tout * 2;
                    }

                    if (err) {
                        ythrow TSystemError() << TStringBuf("ioctl() failed");
                    }
                } else {
                    ythrow TSslException() << TStringBuf("No cont available");
                }
#endif
            }

            void AcquireCont(TCont* c) {
                Cont_ = c;
            }
            void ReleaseCont() {
                Cont_ = nullptr;
            }

            int Write(const char* data, size_t dlen, size_t* written) override {
                if (Y_UNLIKELY(!Cont_)) {
                    return -1;
                }

                while (true) {
                    auto done = NCoro::WriteI(Cont_, S_, data, dlen);
                    if (done.Status() != EAGAIN) {
                        *written = done.Checked();
                        return 1;
                    }
                }
            }

            int Read(char* data, size_t dlen, size_t* readbytes) override {
                if (Y_UNLIKELY(!Cont_)) {
                    return -1;
                }

                if (!Canceled_) {
                    while (true) {
                        auto done = NCoro::ReadI(Cont_, S_, data, dlen);
                        if (EAGAIN != done.Status()) {
                            *readbytes = done.Processed();
                            return 1;
                        }
                    }
                }

                while (true) {
                    if (*Canceled_) {
                        return SSL_RVAL_TIMEOUT;
                    }

                    TContIOStatus ioStat(NCoro::ReadT(Cont_, S_, data, dlen, Timeout_));
                    if (ioStat.Status() == ETIMEDOUT) {
                        //increase to 1.5 times every iteration (to 1sec floor)
                        Timeout_ = TDuration::MicroSeconds(Min<ui64>(1000000, Timeout_.MicroSeconds() + (Timeout_.MicroSeconds() >> 1)));
                        continue;
                    }

                    *readbytes = ioStat.Processed();
                    return 1;
                }
            }

            int Puts(const char* buf) override {
                Y_UNUSED(buf);
                return -1;
            }

            int Gets(char* buf, int size) override {
                Y_UNUSED(buf);
                Y_UNUSED(size);
                return -1;
            }

            void Flush() override {
            }

        private:
            TDuration Timeout_;
            SOCKET S_;
            const TAtomicBool* Canceled_;
            TCont* Cont_;
        };

        class TSslIOStream: public IInputStream, public IOutputStream {
        protected:
            TSslIOStream(TSslCtx& sslCtx, TAutoPtr<TContBIO> connection)
                : Connection_(connection)
                , SslCtx_(sslCtx)
                , Ssl_(nullptr)
            {
            }

            virtual void Handshake() = 0;

        public:
            void WaitUntilWritten() {
                if (Connection_) {
                    Connection_->WaitUntilWritten();
                }
            }

            int PollReadT(const TDuration& timeout) {
                if (!Connection_) {
                    return -1;
                }

                while (true) {
                    const int rpoll = Connection_->PollT(CONT_POLL_READ, timeout);
                    if (!Ssl_ || rpoll) {
                        return rpoll;
                    }

                    char c = 0;
                    const int rpeek = SSL_peek(Ssl_.Get(), &c, sizeof(c));
                    if (rpeek < 0) {
                        return -1;
                    } else if (rpeek > 0) {
                        return 0;
                    } else {
                        if ((SSL_get_shutdown(Ssl_.Get()) & SSL_RECEIVED_SHUTDOWN) != 0) {
                            Shutdown(); // wait until shutdown is finished
                            return EIO;
                        }
                    }
                }
            }

            void Shutdown() {
                if (Ssl_ && Connection_) {
                    for (size_t i = 0; i < 2; ++i) {
                        bool rval = SSL_shutdown(Ssl_.Get());
                        if (0 == rval) {
                            continue;
                        } else if (1 == rval) {
                            break;
                        }
                    }
                }
            }

            inline void AcquireCont(TCont* c) {
                if (Y_UNLIKELY(!Connection_)) {
                    ythrow TSslException() << TStringBuf("no connection provided");
                }

                Connection_->AcquireCont(c);
            }

            inline void ReleaseCont() {
                if (Connection_) {
                    Connection_->ReleaseCont();
                }
            }

            TContIOStatus WriteVectorI(const TList<IOutputStream::TPart>& vec) {
                for (const auto& p : vec) {
                    Write(p.buf, p.len);
                }
                return TContIOStatus::Success(vec.size());
            }

            SOCKET Socket() {
                if (Y_UNLIKELY(!Connection_)) {
                    ythrow TSslException() << TStringBuf("no connection provided");
                }

                return Connection_->Socket();
            }

        private:
            void DoWrite(const void* buf, size_t len) override {
                if (Y_UNLIKELY(!Connection_)) {
                    ythrow TSslException() << TStringBuf("DoWrite() no connection provided");
                }

                const int rval = SSL_write(Ssl_.Get(), buf, len);
                if (rval <= 0) {
                    ythrow TSslException(TStringBuf("SSL_write"), Ssl_.Get(), rval);
                }
            }

            size_t DoRead(void* buf, size_t len) override {
                if (Y_UNLIKELY(!Connection_)) {
                    ythrow TSslException() << TStringBuf("DoRead() no connection provided");
                }

                const int rval = SSL_read(Ssl_.Get(), buf, len);
                if (rval < 0) {
                    if (SSL_RVAL_TIMEOUT == rval) {
                        ythrow TSystemError(ECANCELED) << TStringBuf(" http request canceled");
                    }
                    ythrow TSslException(TStringBuf("SSL_read"), Ssl_.Get(), rval);
                } else if (0 == rval) {
                    if ((SSL_get_shutdown(Ssl_.Get()) & SSL_RECEIVED_SHUTDOWN) != 0) {
                        return rval;
                    } else {
                        const int err = SSL_get_error(Ssl_.Get(), rval);
                        if (SSL_ERROR_ZERO_RETURN != err) {
                            ythrow TSslException(TStringBuf("SSL_read"), Ssl_.Get(), rval);
                        }
                    }
                }

                return static_cast<size_t>(rval);
            }

        protected:
            // just for ssl debug
            static void InfoCB(const SSL* s, int where, int ret) {
                TStringBuf str;
                const int w = where & ~SSL_ST_MASK;
                if (w & SSL_ST_CONNECT) {
                    str = TStringBuf("SSL_connect");
                } else if (w & SSL_ST_ACCEPT) {
                    str = TStringBuf("SSL_accept");
                } else {
                    str = TStringBuf("undefined");
                }

                if (where & SSL_CB_LOOP) {
                    Cerr << str << ':' << SSL_state_string_long(s) << Endl;
                } else if (where & SSL_CB_ALERT) {
                    Cerr << TStringBuf("SSL3 alert ") << ((where & SSL_CB_READ) ? TStringBuf("read") : TStringBuf("write")) << ' ' << SSL_alert_type_string_long(ret) << ':' << SSL_alert_desc_string_long(ret) << Endl;
                } else if (where & SSL_CB_EXIT) {
                    if (ret == 0) {
                        Cerr << str << TStringBuf(":failed in ") << SSL_state_string_long(s) << Endl;
                    } else if (ret < 0) {
                        Cerr << str << TStringBuf(":error in ") << SSL_state_string_long(s) << Endl;
                    }
                }
            }

        protected:
            THolder<TContBIO> Connection_;
            TSslCtx& SslCtx_;
            TSslHolder Ssl_;
        };

        class TContBIOWatcher {
        public:
            TContBIOWatcher(TSslIOStream& io, TCont* c) noexcept
                : IO_(io)
            {
                IO_.AcquireCont(c);
            }

            ~TContBIOWatcher() noexcept {
                IO_.ReleaseCont();
            }

        private:
            TSslIOStream& IO_;
        };

        class TSslClientIOStream: public TSslIOStream {
        public:
            TSslClientIOStream(TSslCtxClient& sslCtx, const TParsedLocation& loc, SOCKET s, const TAtomicBool* canceled)
                : TSslIOStream(sslCtx, new TContBIO(s, canceled))
                , Location_(loc)
            {
            }

            void SetSsl(TSslHolder&& ssl) {
                Ssl_ = std::move(ssl);
                BIO_up_ref(*Connection_);
                SSL_set_bio(Ssl_.Get(), *Connection_, *Connection_);
            }

            TSslHolder&& MoveSsl() {
                return std::move(Ssl_);
            }

            void Handshake() override {
                Ssl_.Reset(SSL_new(SslCtx_));
                if (THttpsOptions::EnableSslClientDebug) {
                    SSL_set_info_callback(Ssl_.Get(), InfoCB);
                }

                BIO_up_ref(*Connection_); // SSL_set_bio consumes only one reference if rbio and wbio are the same
                SSL_set_bio(Ssl_.Get(), *Connection_, *Connection_);

                const TString hostname(Location_.Host);
                const int rev = SSL_set_tlsext_host_name(Ssl_.Get(), hostname.data());
                if (Y_UNLIKELY(1 != rev)) {
                    ythrow TSslException(TStringBuf("SSL_set_tlsext_host_name(client)"), Ssl_.Get(), rev);
                }

                TString cert, pvtKey;
                ParseUserInfo(Location_, cert, pvtKey);

                if (cert && (1 != SSL_use_certificate_file(Ssl_.Get(), cert.data(), SSL_FILETYPE_PEM))) {
                    ythrow TSslException(TStringBuf("SSL_use_certificate_file(client)"));
                }

                if (pvtKey) {
                    if (1 != SSL_use_PrivateKey_file(Ssl_.Get(), pvtKey.data(), SSL_FILETYPE_PEM)) {
                        ythrow TSslException(TStringBuf("SSL_use_PrivateKey_file(client)"));
                    }

                    if (1 != SSL_check_private_key(Ssl_.Get())) {
                        ythrow TSslException(TStringBuf("SSL_check_private_key(client)"));
                    }
                }

                SSL_set_connect_state(Ssl_.Get());

                // TODO restore session if reconnect
                const int rval = SSL_do_handshake(Ssl_.Get());
                if (1 != rval) {
                    if (rval == SSL_RVAL_TIMEOUT) {
                        ythrow TSystemError(ECANCELED) << TStringBuf("canceled");
                    } else {
                        ythrow TSslException(TStringBuf("BIO_do_handshake(client)"), Ssl_.Get(), rval);
                    }
                }

                if (THttpsOptions::CheckCertificateHostname) {
                    TX509Holder peerCert(SSL_get_peer_certificate(Ssl_.Get()));
                    if (!peerCert) {
                        ythrow TSslException(TStringBuf("SSL_get_peer_certificate(client)"));
                    }

                    if (!CheckCertHostname(peerCert.Get(), Location_.Host)) {
                        ythrow TSslException(TStringBuf("CheckCertHostname(client)"));
                    }
                }
            }

        private:
            const TParsedLocation Location_;
            //TSslSessionHolder Session_;
        };

        static TConnCache* ConnectionCache() {
            return Singleton<TConnCache>();
        }

        //some templates magic
        template <class T>
        static inline TAutoPtr<T> AutoPtr(T* t) noexcept {
            return t;
        }

        static inline TString ReadAll(THttpInput& in) {
            TString ret;
            ui64 clin;

            if (in.GetContentLength(clin)) {
                const size_t cl = SafeIntegerCast<size_t>(clin);

                ret.ReserveAndResize(cl);
                size_t sz = in.Load(ret.begin(), cl);
                if (sz != cl) {
                    throw yexception() << TStringBuf("not full content: ") << sz << TStringBuf(" bytes from ") << cl;
                }
            } else if (in.HasContent()) {
                TVector<char> buff(9500); //common jumbo frame size

                while (size_t len = in.Read(buff.data(), buff.size())) {
                    ret.AppendNoAlias(buff.data(), len);
                }
            }

            return ret;
        }

        template <class TRequestType>
        class THttpsRequest: public IJob {
        public:
            inline THttpsRequest(TSimpleHandleRef hndl, TMessage msg)
                : Hndl_(hndl)
                , Msg_(std::move(msg))
                , Loc_(Msg_.Addr)
                , Addr_(CachedThrResolve(TResolveInfo(Loc_.Host, Loc_.GetPort())))
            {
            }

            void DoRun(TCont* c) override {
                THolder<THttpsRequest> This(this);

                if (c->Cancelled()) {
                    Hndl_->NotifyError(new TError("canceled", TError::TType::Cancelled));
                    return;
                }

                TErrorRef error;
                TConnCache::TConnectionRef connection(ConnectionCache()->Connect(c, Msg_.Addr, *Addr_, &error));
                if (!connection) {
                    Hndl_->NotifyError(error);
                    return;
                }

                TSslClientIOStream io(TSslCtxClient::Instance(), Loc_, (*connection)->Fd(), Hndl_->CanceledPtr());
                TContBIOWatcher w(io, c);
                TString received;
                THttpHeaders headers;
                TString firstLine;

                try {
                    if ((*connection)->HasSsl()) {
                        io.SetSsl((*connection)->MoveSsl());
                    } else {
                        io.Handshake();
                    }
                    RequestData().SendTo(io);
                    Req_.Destroy();
                    error = ProcessRecv(io, &received, &headers, &firstLine);
                    (*connection)->SetSsl(io.MoveSsl());
                    (*connection)->ResetBIO();
                } catch (const TSystemError& e) {
                    if (c->Cancelled() || e.Status() == ECANCELED) {
                        error = new TError("canceled", TError::TType::Cancelled);
                    } else {
                        error = new TError(CurrentExceptionMessage());
                    }
                } catch (...) {
                    if (c->Cancelled()) {
                        error = new TError("canceled", TError::TType::Cancelled);
                    } else {
                        error = new TError(CurrentExceptionMessage());
                    }
                }

                if (error) {
                    Hndl_->NotifyError(error, received, firstLine, headers);
                } else {
                    ConnectionCache()->Release(connection);
                    Hndl_->NotifyResponse(received, firstLine, headers);
                }
            }

            TErrorRef ProcessRecv(TSslClientIOStream& io, TString* data, THttpHeaders* headers, TString* firstLine) {
                io.WaitUntilWritten();

                Hndl_->SetSendComplete();

                THttpInput in(&io);
                *data = ReadAll(in);
                *firstLine = in.FirstLine();
                *headers = in.Headers();

                i32 code = ParseHttpRetCode(in.FirstLine());
                if (code < 200 || code > (!THttpsOptions::RedirectionNotError ? 299 : 399)) {
                    return new TError(TStringBuilder() << TStringBuf("request failed(") << in.FirstLine() << ')', TError::TType::ProtocolSpecific, code);
                }

                return nullptr;
            }

            const NHttp::TRequestData& RequestData() {
                if (!Req_) {
                    Req_ = TRequestType::Build(Msg_, Loc_);
                }
                return *Req_;
            }

        private:
            TSimpleHandleRef Hndl_;
            const TMessage Msg_;
            const TParsedLocation Loc_;
            const TResolvedHost* Addr_;
            NHttp::TRequestData::TPtr Req_;
        };

        class TServer: public IRequester, public TContListener::ICallBack {
            class TSslServerIOStream: public TSslIOStream, public TThrRefBase {
            public:
                TSslServerIOStream(TSslCtxServer& sslCtx, TSocketRef s)
                    : TSslIOStream(sslCtx, new TContBIO(*s))
                    , S_(s)
                {
                }

                void Close(bool shutdown) {
                    if (shutdown) {
                        Shutdown();
                    }
                    S_->Close();
                }

                void Handshake() override {
                    if (!Ssl_) {
                        Ssl_.Reset(SSL_new(SslCtx_));
                        if (THttpsOptions::EnableSslServerDebug) {
                            SSL_set_info_callback(Ssl_.Get(), InfoCB);
                        }

                        BIO_up_ref(*Connection_); // SSL_set_bio consumes only one reference if rbio and wbio are the same
                        SSL_set_bio(Ssl_.Get(), *Connection_, *Connection_);

                        const int rc = SSL_accept(Ssl_.Get());
                        if (1 != rc) {
                            ythrow TSslException(TStringBuf("SSL_accept"), Ssl_.Get(), rc);
                        }
                    }

                    if (!SSL_is_init_finished(Ssl_.Get())) {
                        const int rc = SSL_do_handshake(Ssl_.Get());
                        if (rc != 1) {
                            ythrow TSslException(TStringBuf("SSL_do_handshake"), Ssl_.Get(), rc);
                        }
                    }
                }

            private:
                TSocketRef S_;
            };

            class TJobsQueue: public TAutoOneConsumerPipeQueue<IJob>, public TThrRefBase {
            };

            typedef TIntrusivePtr<TJobsQueue> TJobsQueueRef;

            class TWrite: public IJob, public TData {
            private:
                template <class T>
                static void WriteHeader(IOutputStream& os, TStringBuf name, T value) {
                    os << name << TStringBuf(": ") << value << TStringBuf("\r\n");
                }

                static void WriteHttpCode(IOutputStream& os, TMaybe<IRequest::TResponseError> error) {
                    if (!error.Defined()) {
                        os << HttpCodeStrEx(HttpCodes::HTTP_OK);
                        return;
                    }

                    switch (*error) {
                        case IRequest::TResponseError::BadRequest:
                            os << HttpCodeStrEx(HttpCodes::HTTP_BAD_REQUEST);
                            break;
                        case IRequest::TResponseError::Forbidden:
                            os << HttpCodeStrEx(HttpCodes::HTTP_FORBIDDEN);
                            break;
                        case IRequest::TResponseError::NotExistService:
                            os << HttpCodeStrEx(HttpCodes::HTTP_NOT_FOUND);
                            break;
                        case IRequest::TResponseError::TooManyRequests:
                            os << HttpCodeStrEx(HttpCodes::HTTP_TOO_MANY_REQUESTS);
                            break;
                        case IRequest::TResponseError::InternalError:
                            os << HttpCodeStrEx(HttpCodes::HTTP_INTERNAL_SERVER_ERROR);
                            break;
                        case IRequest::TResponseError::NotImplemented:
                            os << HttpCodeStrEx(HttpCodes::HTTP_NOT_IMPLEMENTED);
                            break;
                        case IRequest::TResponseError::BadGateway:
                            os << HttpCodeStrEx(HttpCodes::HTTP_BAD_GATEWAY);
                            break;
                        case IRequest::TResponseError::ServiceUnavailable:
                            os << HttpCodeStrEx(HttpCodes::HTTP_SERVICE_UNAVAILABLE);
                            break;
                        case IRequest::TResponseError::BandwidthLimitExceeded:
                            os << HttpCodeStrEx(HttpCodes::HTTP_BANDWIDTH_LIMIT_EXCEEDED);
                            break;
                        case IRequest::TResponseError::MaxResponseError:
                            ythrow yexception() << TStringBuf("unknow type of error");
                    }
                }

            public:
                inline TWrite(TData& data, const TString& compressionScheme, TIntrusivePtr<TSslServerIOStream> io, TServer* server, const TString& headers, int httpCode)
                    : CompressionScheme_(compressionScheme)
                    , IO_(io)
                    , Server_(server)
                    , Error_(TMaybe<IRequest::TResponseError>())
                    , Headers_(headers)
                    , HttpCode_(httpCode)
                {
                    swap(data);
                }

                inline TWrite(TData& data, const TString& compressionScheme, TIntrusivePtr<TSslServerIOStream> io, TServer* server, IRequest::TResponseError error, const TString& headers)
                    : CompressionScheme_(compressionScheme)
                    , IO_(io)
                    , Server_(server)
                    , Error_(error)
                    , Headers_(headers)
                    , HttpCode_(0)
                {
                    swap(data);
                }

                void DoRun(TCont* c) override {
                    THolder<TWrite> This(this);

                    try {
                        TContBIOWatcher w(*IO_, c);

                        PrepareSocket(IO_->Socket());

                        char buf[128];
                        TMemoryOutput mo(buf, sizeof(buf));

                        mo << TStringBuf("HTTP/1.1 ");
                        if (HttpCode_) {
                            mo << HttpCodeStrEx(HttpCode_);
                        } else {
                            WriteHttpCode(mo, Error_);
                        }
                        mo << TStringBuf("\r\n");

                        if (!CompressionScheme_.empty()) {
                            WriteHeader(mo, TStringBuf("Content-Encoding"), TStringBuf(CompressionScheme_));
                        }
                        WriteHeader(mo, TStringBuf("Connection"), TStringBuf("Keep-Alive"));
                        WriteHeader(mo, TStringBuf("Content-Length"), size());

                        mo << Headers_;

                        mo << TStringBuf("\r\n");

                        IO_->Write(buf, mo.Buf() - buf);
                        if (size()) {
                            IO_->Write(data(), size());
                        }

                        Server_->Enqueue(new TRead(IO_, Server_));
                    } catch (...) {
                    }
                }

            private:
                const TString CompressionScheme_;
                TIntrusivePtr<TSslServerIOStream> IO_;
                TServer* Server_;
                TMaybe<IRequest::TResponseError> Error_;
                TString Headers_;
                int HttpCode_;
            };

            class TRequest: public IHttpRequest {
            public:
                inline TRequest(THttpInput& in, TIntrusivePtr<TSslServerIOStream> io, TServer* server)
                    : IO_(io)
                    , Tmp_(in.FirstLine())
                    , CompressionScheme_(in.BestCompressionScheme())
                    , RemoteHost_(PrintHostByRfc(*GetPeerAddr(IO_->Socket())))
                    , Headers_(in.Headers())
                    , H_(Tmp_)
                    , Server_(server)
                {
                }

                ~TRequest() override {
                    if (!!IO_) {
                        try {
                            Server_->Enqueue(new TFail(IO_, Server_));
                        } catch (...) {
                        }
                    }
                }

                TStringBuf Scheme() const override {
                    return TStringBuf("https");
                }

                TString RemoteHost() const override {
                    return RemoteHost_;
                }

                const THttpHeaders& Headers() const override {
                    return Headers_;
                }

                TStringBuf Method() const override {
                    return H_.Method;
                }

                TStringBuf Cgi() const override {
                    return H_.Cgi;
                }

                TStringBuf Service() const override {
                    return TStringBuf(H_.Path).Skip(1);
                }

                TStringBuf RequestId() const override {
                    return TStringBuf();
                }

                bool Canceled() const override {
                    if (!IO_) {
                        return false;
                    }
                    return !IsNotSocketClosedByOtherSide(IO_->Socket());
                }

                void SendReply(TData& data) override {
                    SendReply(data, TString(), HttpCodes::HTTP_OK);
                }

                void SendReply(TData& data, const TString& headers, int httpCode) override {
                    const bool compressed = Compress(data);
                    Server_->Enqueue(new TWrite(data, compressed ? CompressionScheme_ : TString(), IO_, Server_, headers, httpCode));
                    Y_UNUSED(IO_.Release());
                }

                void SendError(TResponseError error, const THttpErrorDetails& details) override {
                    TData data;
                    Server_->Enqueue(new TWrite(data, TString(), IO_, Server_, error, details.Headers));
                    Y_UNUSED(IO_.Release());
                }

            private:
                bool Compress(TData& data) const {
                    if (CompressionScheme_ == TStringBuf("gzip")) {
                        try {
                            TData gzipped(data.size());
                            TMemoryOutput out(gzipped.data(), gzipped.size());
                            TZLibCompress c(&out, ZLib::GZip);
                            c.Write(data.data(), data.size());
                            c.Finish();
                            gzipped.resize(out.Buf() - gzipped.data());
                            data.swap(gzipped);
                            return true;
                        } catch (yexception&) {
                            // gzipped data occupies more space than original data
                        }
                    }
                    return false;
                }

            private:
                TIntrusivePtr<TSslServerIOStream> IO_;
                const TString      Tmp_;
                const TString      CompressionScheme_;
                const TString      RemoteHost_;
                const THttpHeaders Headers_;

            protected:
                TParsedHttpFull H_;
                TServer* Server_;
            };

            class TGetRequest: public TRequest {
            public:
                inline TGetRequest(THttpInput& in, TIntrusivePtr<TSslServerIOStream> io, TServer* server)
                    : TRequest(in, io, server)
                {
                }

                TStringBuf Data() const override {
                    return H_.Cgi;
                }

                TStringBuf Body() const override {
                    return TStringBuf();
                }
            };

            class TPostRequest: public TRequest {
            public:
                inline TPostRequest(THttpInput& in, TIntrusivePtr<TSslServerIOStream> io, TServer* server)
                    : TRequest(in, io, server)
                    , Data_(ReadAll(in))
                {
                }

                TStringBuf Data() const override {
                    return Data_;
                }

                TStringBuf Body() const override {
                    return Data_;
                }

            private:
                TString Data_;
            };

            class TFail: public IJob {
            public:
                inline TFail(TIntrusivePtr<TSslServerIOStream> io, TServer* server)
                    : IO_(io)
                    , Server_(server)
                {
                }

                void DoRun(TCont* c) override {
                    THolder<TFail> This(this);
                    constexpr TStringBuf answer = "HTTP/1.1 503 Service unavailable\r\n"
                                                          "Content-Length: 0\r\n\r\n"sv;

                    try {
                        TContBIOWatcher w(*IO_, c);
                        IO_->Write(answer);
                        Server_->Enqueue(new TRead(IO_, Server_));
                    } catch (...) {
                    }
                }

            private:
                TIntrusivePtr<TSslServerIOStream> IO_;
                TServer* Server_;
            };

            class TRead: public IJob {
            public:
                TRead(TIntrusivePtr<TSslServerIOStream> io, TServer* server, bool selfRemove = false)
                    : IO_(io)
                    , Server_(server)
                    , SelfRemove(selfRemove)
                {
                }

                inline void operator()(TCont* c) {
                    try {
                        TContBIOWatcher w(*IO_, c);

                        if (IO_->PollReadT(TDuration::Seconds(InputConnections()->UnusedConnKeepaliveTimeout()))) {
                            IO_->Close(true);
                            return;
                        }

                        IO_->Handshake();
                        THttpInput in(IO_.Get());

                        const char sym = *in.FirstLine().data();

                        if (sym == 'p' || sym == 'P') {
                            Server_->OnRequest(new TPostRequest(in, IO_, Server_));
                        } else {
                            Server_->OnRequest(new TGetRequest(in, IO_, Server_));
                        }
                    } catch (...) {
                        IO_->Close(false);
                    }

                    if (SelfRemove) {
                        delete this;
                    }
                }

            private:
                void DoRun(TCont* c) override {
                    THolder<TRead> This(this);
                    (*this)(c);
                }

            private:
                TIntrusivePtr<TSslServerIOStream> IO_;
                TServer* Server_;
                bool SelfRemove = false;
            };

        public:
            inline TServer(IOnRequest* cb, const TParsedLocation& loc)
                : CB_(cb)
                , E_(RealStackSize(16000))
                , L_(new TContListener(this, &E_, TContListener::TOptions().SetDeferAccept(true)))
                , JQ_(new TJobsQueue())
                , SslCtx_(loc)
            {
                L_->Bind(TNetworkAddress(loc.GetPort()));
                E_.Create<TServer, &TServer::RunDispatcher>(this, "dispatcher");
                Thrs_.push_back(Spawn<TServer, &TServer::Run>(this));
            }

            ~TServer() override {
                JQ_->Enqueue(nullptr);

                for (size_t i = 0; i < Thrs_.size(); ++i) {
                    Thrs_[i]->Join();
                }
            }

            void Run() {
                //SetHighestThreadPriority();
                L_->Listen();
                E_.Execute();
            }

            inline void OnRequest(const IRequestRef& req) {
                CB_->OnRequest(req);
            }

            TJobsQueueRef& JobQueue() noexcept {
                return JQ_;
            }

            void Enqueue(IJob* j) {
                JQ_->EnqueueSafe(TAutoPtr<IJob>(j));
            }

            void RunDispatcher(TCont* c) {
                for (;;) {
                    TAutoPtr<IJob> job(JQ_->Dequeue(c));

                    if (!job) {
                        break;
                    }

                    try {
                        c->Executor()->Create(*job, "https-job");
                        Y_UNUSED(job.Release());
                    } catch (...) {
                    }
                }

                JQ_->Enqueue(nullptr);
                c->Executor()->Abort();
            }

            void OnAcceptFull(const TAcceptFull& a) override {
                try {
                    TSocketRef s(new TSharedSocket(*a.S));

                    if (InputConnections()->ExceedHardLimit()) {
                        s->Close();
                        return;
                    }

                    THolder<TRead> read(new TRead(new TSslServerIOStream(SslCtx_, s), this, /* selfRemove */ true));
                    E_.Create(*read, "https-response");
                    Y_UNUSED(read.Release());
                    E_.Running()->Yield();
                } catch (...) {
                }
            }

            void OnError() override {
                try {
                    throw;
                } catch (const TSystemError& e) {
                    //crutch for prevent 100% busyloop (simple suspend listener/accepter)
                    if (e.Status() == EMFILE) {
                        E_.Running()->SleepT(TDuration::MilliSeconds(500));
                    }
                }
            }

        private:
            IOnRequest* CB_;
            TContExecutor E_;
            THolder<TContListener> L_;
            TVector<TThreadRef> Thrs_;
            TJobsQueueRef JQ_;
            TSslCtxServer SslCtx_;
        };

        template <class T>
        class THttpsProtocol: public IProtocol {
        public:
            IRequesterRef CreateRequester(IOnRequest* cb, const TParsedLocation& loc) override {
                return new TServer(cb, loc);
            }

            THandleRef ScheduleRequest(const TMessage& msg, IOnRecv* fallback, TServiceStatRef& ss) override {
                TSimpleHandleRef ret(new TSimpleHandle(fallback, msg, !ss ? nullptr : new TStatCollector(ss)));
                try {
                    TAutoPtr<THttpsRequest<T>> req(new THttpsRequest<T>(ret, msg));
                    JobQueue()->Schedule(req);
                    return ret.Get();
                } catch (...) {
                    ret->ResetOnRecv();
                    throw;
                }
            }

            TStringBuf Scheme() const noexcept override {
                return T::Name();
            }

            bool SetOption(TStringBuf name, TStringBuf value) override {
                return THttpsOptions::Set(name, value);
            }
        };

        struct TRequestGet: public NHttp::TRequestGet {
            static inline TStringBuf Name() noexcept {
                return TStringBuf("https");
            }
        };

        struct TRequestFull: public NHttp::TRequestFull {
            static inline TStringBuf Name() noexcept {
                return TStringBuf("fulls");
            }
        };

        struct TRequestPost: public NHttp::TRequestPost {
            static inline TStringBuf Name() noexcept {
                return TStringBuf("posts");
            }
        };

    }
}

namespace NNeh {
    IProtocol* SSLGetProtocol() {
        return Singleton<NHttps::THttpsProtocol<NNeh::NHttps::TRequestGet>>();
    }

    IProtocol* SSLPostProtocol() {
        return Singleton<NHttps::THttpsProtocol<NNeh::NHttps::TRequestPost>>();
    }

    IProtocol* SSLFullProtocol() {
        return Singleton<NHttps::THttpsProtocol<NNeh::NHttps::TRequestFull>>();
    }

    void SetHttpOutputConnectionsLimits(size_t softLimit, size_t hardLimit) {
        Y_ABORT_UNLESS(
            hardLimit > softLimit,
            "invalid output fd limits; hardLimit=%" PRISZT ", softLimit=%" PRISZT,
            hardLimit, softLimit);

        NHttps::ConnectionCache()->SetFdLimits(softLimit, hardLimit);
    }

    void SetHttpInputConnectionsLimits(size_t softLimit, size_t hardLimit) {
        Y_ABORT_UNLESS(
            hardLimit > softLimit,
            "invalid output fd limits; hardLimit=%" PRISZT ", softLimit=%" PRISZT,
            hardLimit, softLimit);

        NHttps::InputConnections()->SetFdLimits(softLimit, hardLimit);
    }

    void SetHttpInputConnectionsTimeouts(unsigned minSec, unsigned maxSec) {
        Y_ABORT_UNLESS(
            maxSec > minSec,
            "invalid input fd limits timeouts; maxSec=%u, minSec=%u",
            maxSec, minSec);

        NHttps::InputConnections()->MinUnusedConnKeepaliveTimeout.store(minSec, std::memory_order_release);
        NHttps::InputConnections()->MaxUnusedConnKeepaliveTimeout.store(maxSec, std::memory_order_release);
    }
}
