#include <library/testing/benchmark/bench.h>

#include <util/random/fast.h>
#include <util/random/random.h>
#include <util/generic/singleton.h>
#include <util/generic/vector.h>
#include <util/charset/wide.h>

#include <cmath>

namespace {
    template <size_t N>
    struct TRandomAsciiString: public TVector<char> {
        inline TRandomAsciiString() {
            reserve(N);
            for (size_t i = 0; i < N; ++i) {
                push_back(RandomNumber<char>(127));
            }
        }
    };

    template <size_t N>
    struct TRandomRuString: public TVector<char> {
        inline TRandomRuString() {
            TVector<unsigned char> data(N * 2);
            unsigned char* textEnd = data.begin();
            for (size_t i = 0; i < N; ++i) {
                size_t runeLen;
                WriteUTF8Char(RandomNumber<ui32>(0x7FF) + 1, runeLen, textEnd);
                textEnd += runeLen;
            }
            assign(reinterpret_cast<const char*>(data.begin()), reinterpret_cast<const char*>(textEnd));
        }
    };

    using RAS1 = TRandomAsciiString<1>;
    using RAS10 = TRandomAsciiString<10>;
    using RAS50 = TRandomAsciiString<50>;
    using RAS1000 = TRandomAsciiString<1000>;
    using RAS1000000 = TRandomAsciiString<1000000>;

    using RRS1 = TRandomRuString<1>;
    using RRS10 = TRandomRuString<10>;
    using RRS1000 = TRandomRuString<1000>;
    using RRS1000000 = TRandomRuString<1000000>;
}

#ifdef _sse2_
#define IS_ASCII_BENCHMARK(length)                                                                                                                                           \
    Y_CPU_BENCHMARK(IsStringASCII##length, iface) {                                                                                                                          \
        const auto& data = *Singleton<RAS##length>();                                                                                                                        \
        for (size_t x = 0; x < iface.Iterations(); ++x) {                                                                                                                    \
            Y_DO_NOT_OPTIMIZE_AWAY(::NDetail::DoIsStringASCII(data.begin(), data.end()));                                                                                    \
        }                                                                                                                                                                    \
    }                                                                                                                                                                        \
    Y_CPU_BENCHMARK(IsStringASCIISlow##length, iface) {                                                                                                                      \
        const auto& data = *Singleton<RAS##length>();                                                                                                                        \
        for (size_t x = 0; x < iface.Iterations(); ++x) {                                                                                                                    \
            Y_DO_NOT_OPTIMIZE_AWAY(::NDetail::DoIsStringASCIISlow(data.begin(), data.end()));                                                                                \
        }                                                                                                                                                                    \
    }                                                                                                                                                                        \
    Y_CPU_BENCHMARK(IsStringASCIISSE##length, iface) {                                                                                                                       \
        const auto& data = *Singleton<RAS##length>();                                                                                                                        \
        for (size_t x = 0; x < iface.Iterations(); ++x) {                                                                                                                    \
            Y_DO_NOT_OPTIMIZE_AWAY(::NDetail::DoIsStringASCIISSE(reinterpret_cast<const unsigned char*>(data.begin()), reinterpret_cast<const unsigned char*>(data.end()))); \
        }                                                                                                                                                                    \
    }
#else //no sse
#define IS_ASCII_BENCHMARK(length)                                                            \
    Y_CPU_BENCHMARK(IsStringASCIIScalar##length, iface) {                                     \
        const auto& data = *Singleton<RAS##length>();                                         \
        for (size_t x = 0; x < iface.Iterations(); ++x) {                                     \
            Y_DO_NOT_OPTIMIZE_AWAY(::NDetail::DoIsStringASCII(data.begin(), data.end()));     \
        }                                                                                     \
    }                                                                                         \
    Y_CPU_BENCHMARK(IsStringASCIISlow##length, iface) {                                       \
        const auto& data = *Singleton<RAS##length>();                                         \
        for (size_t x = 0; x < iface.Iterations(); ++x) {                                     \
            Y_DO_NOT_OPTIMIZE_AWAY(::NDetail::DoIsStringASCIISlow(data.begin(), data.end())); \
        }                                                                                     \
    }
#endif

IS_ASCII_BENCHMARK(1);
IS_ASCII_BENCHMARK(10);
IS_ASCII_BENCHMARK(50);
IS_ASCII_BENCHMARK(1000);
IS_ASCII_BENCHMARK(1000000);

template <bool robust, typename TCharType>
inline size_t UTF8ToWideImplScalar(const char* text, size_t len, TCharType* dest, size_t& written) {
    const unsigned char* cur = reinterpret_cast<const unsigned char*>(text);
    const unsigned char* last = cur + len;
    TCharType* p = dest;

    ::NDetail::UTF8ToWideImplScalar<robust>(cur, last, p);
    written = p - dest;
    return cur - reinterpret_cast<const unsigned char*>(text);
}

template <bool robust, typename TCharType>
inline size_t UTF8ToWideImplSSE(const char* text, size_t len, TCharType* dest, size_t& written) {
    return UTF8ToWideImpl(text, len, dest, written);
}

static wchar16 WBUF_UTF16[10000000];
static wchar32 WBUF_UTF32[10000000];

#define UTF8_TO_WIDE_SCALAR_BENCHMARK_ASCII(impl, length, to)                                                   \
    Y_CPU_BENCHMARK(UTF8ToWideASCII##impl##length##to, iface) {                                                 \
        const auto& data = *Singleton<RAS##length>();                                                           \
        for (size_t x = 0; x < iface.Iterations(); ++x) {                                                       \
            size_t written = 0;                                                                                 \
            Y_DO_NOT_OPTIMIZE_AWAY(UTF8ToWideImpl##impl<false>(data.begin(), data.size(), WBUF_##to, written)); \
        }                                                                                                       \
    }

#define UTF8_TO_WIDE_SCALAR_BENCHMARK_RU(impl, length, to)                                                      \
    Y_CPU_BENCHMARK(UTF8ToWideRU##impl##length##to, iface) {                                                    \
        const auto& data = *Singleton<RRS##length>();                                                           \
        for (size_t x = 0; x < iface.Iterations(); ++x) {                                                       \
            size_t written = 0;                                                                                 \
            Y_DO_NOT_OPTIMIZE_AWAY(UTF8ToWideImpl##impl<false>(data.begin(), data.size(), WBUF_##to, written)); \
        }                                                                                                       \
    }

UTF8_TO_WIDE_SCALAR_BENCHMARK_ASCII(Scalar, 1, UTF16);
UTF8_TO_WIDE_SCALAR_BENCHMARK_ASCII(SSE, 1, UTF16);
UTF8_TO_WIDE_SCALAR_BENCHMARK_ASCII(Scalar, 10, UTF16);
UTF8_TO_WIDE_SCALAR_BENCHMARK_ASCII(SSE, 10, UTF16);
UTF8_TO_WIDE_SCALAR_BENCHMARK_ASCII(Scalar, 1000, UTF16);
UTF8_TO_WIDE_SCALAR_BENCHMARK_ASCII(SSE, 1000, UTF16);
UTF8_TO_WIDE_SCALAR_BENCHMARK_ASCII(Scalar, 1000000, UTF16);
UTF8_TO_WIDE_SCALAR_BENCHMARK_ASCII(SSE, 1000000, UTF16);

UTF8_TO_WIDE_SCALAR_BENCHMARK_RU(Scalar, 1, UTF16);
UTF8_TO_WIDE_SCALAR_BENCHMARK_RU(SSE, 1, UTF16);
UTF8_TO_WIDE_SCALAR_BENCHMARK_RU(Scalar, 10, UTF16);
UTF8_TO_WIDE_SCALAR_BENCHMARK_RU(SSE, 10, UTF16);
UTF8_TO_WIDE_SCALAR_BENCHMARK_RU(Scalar, 1000, UTF16);
UTF8_TO_WIDE_SCALAR_BENCHMARK_RU(SSE, 1000, UTF16);
UTF8_TO_WIDE_SCALAR_BENCHMARK_RU(Scalar, 1000000, UTF16);
UTF8_TO_WIDE_SCALAR_BENCHMARK_RU(SSE, 1000000, UTF16);

UTF8_TO_WIDE_SCALAR_BENCHMARK_ASCII(Scalar, 1, UTF32);
UTF8_TO_WIDE_SCALAR_BENCHMARK_ASCII(SSE, 1, UTF32);
UTF8_TO_WIDE_SCALAR_BENCHMARK_ASCII(Scalar, 10, UTF32);
UTF8_TO_WIDE_SCALAR_BENCHMARK_ASCII(SSE, 10, UTF32);
UTF8_TO_WIDE_SCALAR_BENCHMARK_ASCII(Scalar, 1000, UTF32);
UTF8_TO_WIDE_SCALAR_BENCHMARK_ASCII(SSE, 1000, UTF32);
UTF8_TO_WIDE_SCALAR_BENCHMARK_ASCII(Scalar, 1000000, UTF32);
UTF8_TO_WIDE_SCALAR_BENCHMARK_ASCII(SSE, 1000000, UTF32);

UTF8_TO_WIDE_SCALAR_BENCHMARK_RU(Scalar, 1, UTF32);
UTF8_TO_WIDE_SCALAR_BENCHMARK_RU(SSE, 1, UTF32);
UTF8_TO_WIDE_SCALAR_BENCHMARK_RU(Scalar, 10, UTF32);
UTF8_TO_WIDE_SCALAR_BENCHMARK_RU(SSE, 10, UTF32);
UTF8_TO_WIDE_SCALAR_BENCHMARK_RU(Scalar, 1000, UTF32);
UTF8_TO_WIDE_SCALAR_BENCHMARK_RU(SSE, 1000, UTF32);
UTF8_TO_WIDE_SCALAR_BENCHMARK_RU(Scalar, 1000000, UTF32);
UTF8_TO_WIDE_SCALAR_BENCHMARK_RU(SSE, 1000000, UTF32);
