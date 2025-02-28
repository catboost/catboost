#pragma once

#include <util/generic/array_ref.h>
#include <util/generic/flags.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/stream/mem.h>
#include <util/stream/output.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <library/cpp/http/misc/httpcodes.h>

#include "location.h"
#include "neh.h"
#include "rpc.h"

#include <atomic>

//common primitives for http/http2

namespace NNeh {
    struct THttpErrorDetails {
        TString Details = {};
        TString Headers = {};
    };

    class IHttpRequest: public IRequest {
    public:
        using IRequest::SendReply;
        virtual void SendReply(TData& data, const TString& headers, int httpCode = 200) = 0;
        virtual const THttpHeaders& Headers() const = 0;
        virtual TStringBuf Method() const = 0;
        virtual TStringBuf Body() const = 0;
        virtual TStringBuf Cgi() const = 0;
        void SendError(TResponseError err, const TString& details = TString()) override final {
            SendError(err, THttpErrorDetails{.Details = details});
        }

        virtual void SendError(TResponseError err, const THttpErrorDetails& details) = 0;
    };

    namespace NHttp {
        enum class EResolverType {
            ETCP = 0,
            EUNIXSOCKET = 1
        };

        struct TFdLimits {
        public:
            TFdLimits()
                : Soft_(10000)
                , Hard_(15000)
            {
            }

            TFdLimits(const TFdLimits& other)  {
                Soft_.store(other.Soft(), std::memory_order_release);
                Hard_.store(other.Hard(), std::memory_order_release);
            }

            inline size_t Delta() const noexcept {
                return ExceedLimit(Hard_.load(std::memory_order_acquire), Soft_.load(std::memory_order_acquire));
            }

            inline static size_t ExceedLimit(size_t val, size_t limit) noexcept {
                return val > limit ? val - limit : 0;
            }

            void SetSoft(size_t value) noexcept {
                Soft_.store(value, std::memory_order_release);
            }

            void SetHard(size_t value) noexcept {
                Hard_.store(value, std::memory_order_release);
            }

            size_t Soft() const noexcept {
                return Soft_.load(std::memory_order_acquire);
            }

            size_t Hard() const noexcept {
                return Hard_.load(std::memory_order_acquire);
            }

        private:
            std::atomic<size_t> Soft_;
            std::atomic<size_t> Hard_;
        };

        template <class T>
        class TLockFreeSequence {
        public:
            inline TLockFreeSequence() {
                memset((void*)T_, 0, sizeof(T_));
            }

            inline ~TLockFreeSequence() {
                for (size_t i = 0; i < Y_ARRAY_SIZE(T_); ++i) {
                    delete[] T_[i];
                }
            }

            inline T& Get(size_t n) {
                const size_t i = GetValueBitCount(n + 1) - 1;

                return GetList(i)[n + 1 - (((size_t)1) << i)];
            }

        private:
            inline T* GetList(size_t n) {
                T* volatile* t = T_ + n;

                T* result;
                while (!(result = AtomicGet(*t))) {
                    TArrayHolder<T> nt(new T[((size_t)1) << n]);

                    if (AtomicCas(t, nt.Get(), nullptr)) {
                        return nt.Release();
                    }
                }

                return result;
            }

        private:
            T* volatile T_[sizeof(size_t) * 8];
        };

        class TRequestData: public TNonCopyable {
        public:
            using TPtr = TAutoPtr<TRequestData>;
            using TParts = TVector<IOutputStream::TPart>;

            inline TRequestData(size_t memSize)
                : Mem(memSize)
            {
            }

            inline void SendTo(IOutputStream& io) const {
                io.Write(Parts_.data(), Parts_.size());
            }

            inline void AddPart(const void* buf, size_t len) noexcept {
                Parts_.push_back(IOutputStream::TPart(buf, len));
            }

            const TParts& Parts() const noexcept {
                return Parts_;
            }

            TVector<char> Mem;
            TString Data;
        private:
            TParts Parts_;
        };

        struct TRequestSettings {
            bool NoDelay = true;
            EResolverType ResolverType = EResolverType::ETCP;

            TRequestSettings& SetNoDelay(bool noDelay) {
                NoDelay = noDelay;
                return *this;
            }

            TRequestSettings& SetResolverType(EResolverType resolverType) {
                ResolverType = resolverType;
                return *this;
            }
        };

        struct TRequestGet {
            static TRequestData::TPtr Build(const TMessage& msg, const TParsedLocation& loc) {
                TRequestData::TPtr req(new TRequestData(50 + loc.Service.size() + msg.Data.size() + loc.Host.size()));
                TMemoryOutput out(req->Mem.data(), req->Mem.size());

                out << TStringBuf("GET /") << loc.Service;

                if (!!msg.Data) {
                    out << '?' << msg.Data;
                }

                out << TStringBuf(" HTTP/1.1\r\nHost: ") << loc.Host;

                if (!!loc.Port) {
                    out << TStringBuf(":") << loc.Port;
                }

                out << TStringBuf("\r\n\r\n");

                req->AddPart(req->Mem.data(), out.Buf() - req->Mem.data());
                return req;
            }

            static inline TStringBuf Name() noexcept {
                return TStringBuf("http");
            }

            static TRequestSettings RequestSettings() {
                return TRequestSettings();
            }
        };

        struct TRequestPost {
            static TRequestData::TPtr Build(const TMessage& msg, const TParsedLocation& loc) {
                TRequestData::TPtr req(new TRequestData(100 + loc.Service.size() + loc.Host.size()));
                TMemoryOutput out(req->Mem.data(), req->Mem.size());

                out << TStringBuf("POST /") << loc.Service
                    << TStringBuf(" HTTP/1.1\r\nHost: ") << loc.Host;

                if (!!loc.Port) {
                    out << TStringBuf(":") << loc.Port;
                }

                out << TStringBuf("\r\nContent-Length: ") << msg.Data.size() << TStringBuf("\r\n\r\n");

                req->AddPart(req->Mem.data(), out.Buf() - req->Mem.data());
                req->AddPart(msg.Data.data(), msg.Data.size());
                req->Data = msg.Data;
                return req;
            }

            static inline TStringBuf Name() noexcept {
                return TStringBuf("post");
            }

            static TRequestSettings RequestSettings() {
                return TRequestSettings();
            }
        };

        struct TRequestFull {
            static TRequestData::TPtr Build(const TMessage& msg, const TParsedLocation&) {
                TRequestData::TPtr req(new TRequestData(0));
                req->AddPart(msg.Data.data(), msg.Data.size());
                req->Data = msg.Data;
                return req;
            }

            static inline TStringBuf Name() noexcept {
                return TStringBuf("full");
            }

            static TRequestSettings RequestSettings() {
                return TRequestSettings();
            }
        };

        enum class ERequestType {
            Any = 0 /* "ANY" */,
            Post    /* "POST" */,
            Get     /* "GET" */,
            Put     /* "PUT" */,
            Delete  /* "DELETE" */,
            Patch   /* "PATCH" */,
        };

        enum class ERequestFlag {
            None = 0,
            /** use absoulte uri for proxy requests in the first request line
         * POST http://ya.ru HTTP/1.1
         * @see https://www.w3.org/Protocols/rfc2616/rfc2616-sec5.html#sec5.1.2
         */
            AbsoluteUri = 1,
        };

        Y_DECLARE_FLAGS(ERequestFlags, ERequestFlag);
        Y_DECLARE_OPERATORS_FOR_FLAGS(ERequestFlags);

        static constexpr ERequestType DefaultRequestType = ERequestType::Any;

        extern const TStringBuf DefaultContentType;

        /// @brief `MakeFullRequest` transmutes http/post/http2/post2 message to full/full2 with
        /// additional HTTP headers and/or content data.
        ///
        /// If reqType is `Any`, then request type is POST, unless content is empty and schema
        /// prefix is http/https/http2, in that case request type is GET.
        ///
        /// @msg[in]        Will get URL from `msg.Data`.
        bool MakeFullRequest(TMessage& msg, TStringBuf headers, TStringBuf content, TStringBuf contentType = DefaultContentType, ERequestType reqType = DefaultRequestType, ERequestFlags flags = ERequestFlag::None);

        /// @see `MakeFullrequest`.
        ///
        /// @urlParts[in]       Will construct url from `urlParts`, `msg.Data` is not used.
        bool MakeFullRequest(TMessage& msg, TConstArrayRef<TString> urlParts, TStringBuf headers, TStringBuf content, TStringBuf contentType = DefaultContentType, ERequestType reqType = DefaultRequestType, ERequestFlags flags = ERequestFlag::None);

        /// Same as `MakeFullRequest` but it will add ERequestFlag::AbsoluteUri to the @a flags
        /// and replace msg.Addr with @a proxyAddr
        ///
        /// @see `MakeFullrequest`.
        bool MakeFullProxyRequest(TMessage& msg, TStringBuf proxyAddr, TStringBuf headers, TStringBuf content, TStringBuf contentType = DefaultContentType, ERequestType reqType = DefaultRequestType, ERequestFlags flags = ERequestFlag::None);

        size_t GetUrlPartsLength(TConstArrayRef<TString> urlParts);
        //part1&part2&...
        void JoinUrlParts(TConstArrayRef<TString> urlParts, IOutputStream& out);
        //'?' + JoinUrlParts
        void WriteUrlParts(TConstArrayRef<TString> urlParts, IOutputStream& out);

        bool IsHttpScheme(TStringBuf scheme);
    }

    HttpCodes GetHttpCode(const IRequest::TResponseError&);
}
