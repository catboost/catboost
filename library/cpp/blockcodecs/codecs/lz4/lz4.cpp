#include <library/cpp/blockcodecs/core/codecs.h>
#include <library/cpp/blockcodecs/core/common.h>
#include <library/cpp/blockcodecs/core/register.h>

#include <contrib/libs/lz4/lz4.h>
#include <contrib/libs/lz4/lz4hc.h>

using namespace NBlockCodecs;

namespace {
    struct TLz4Base {
        static inline size_t DoMaxCompressedLength(size_t in) {
            return LZ4_compressBound(SafeIntegerCast<int>(in));
        }
    };

    struct TLz4FastCompress {
        inline TLz4FastCompress(int memory)
            : Memory(memory)
        {
        }

        inline size_t DoCompress(const TData& in, void* buf) const {
            return LZ4_compress_default(in.data(), (char*)buf, in.size(), LZ4_compressBound(in.size()));
        }

        inline TString CPrefix() {
            return "fast" + ToString(Memory);
        }

        const int Memory;
    };

    struct TLz4BestCompress {
        inline size_t DoCompress(const TData& in, void* buf) const {
            return LZ4_compress_HC(in.data(), (char*)buf, in.size(), LZ4_compressBound(in.size()), 0);
        }

        static inline TString CPrefix() {
            return "hc";
        }
    };

    struct TLz4FastDecompress {
        inline void DoDecompress(const TData& in, void* out, size_t len) const {
            ssize_t res = LZ4_decompress_fast(in.data(), (char*)out, len);
            if (res < 0) {
                ythrow TDecompressError(res);
            }
        }

        static inline TStringBuf DPrefix() {
            return TStringBuf("fast");
        }
    };

    struct TLz4SafeDecompress {
        inline void DoDecompress(const TData& in, void* out, size_t len) const {
            ssize_t res = LZ4_decompress_safe(in.data(), (char*)out, in.size(), len);
            if (res < 0) {
                ythrow TDecompressError(res);
            }
        }

        static inline TStringBuf DPrefix() {
            return TStringBuf("safe");
        }
    };

    template <class TC, class TD>
    struct TLz4Codec: public TAddLengthCodec<TLz4Codec<TC, TD>>, public TLz4Base, public TC, public TD {
        inline TLz4Codec()
            : MyName("lz4-" + TC::CPrefix() + "-" + TD::DPrefix())
        {
        }

        template <class T>
        inline TLz4Codec(const T& t)
            : TC(t)
            , MyName("lz4-" + TC::CPrefix() + "-" + TD::DPrefix())
        {
        }

        TStringBuf Name() const noexcept override {
            return MyName;
        }

        const TString MyName;
    };

    struct TLz4Registrar {
        TLz4Registrar() {
            for (int i = 10; i <= 20; ++i) {
                typedef TLz4Codec<TLz4FastCompress, TLz4FastDecompress> T1;
                typedef TLz4Codec<TLz4FastCompress, TLz4SafeDecompress> T2;

                THolder<T1> t1(new T1(i));
                THolder<T2> t2(new T2(i));

                RegisterCodec(std::move(t1));
                RegisterCodec(std::move(t2));
            }

            RegisterCodec(MakeHolder<TLz4Codec<TLz4BestCompress, TLz4FastDecompress>>());
            RegisterCodec(MakeHolder<TLz4Codec<TLz4BestCompress, TLz4SafeDecompress>>());

            RegisterAlias("lz4-fast-safe", "lz4-fast14-safe");
            RegisterAlias("lz4-fast-fast", "lz4-fast14-fast");
            RegisterAlias("lz4", "lz4-fast-safe");
            RegisterAlias("lz4fast", "lz4-fast-fast");
            RegisterAlias("lz4hc", "lz4-hc-safe");
        }
    };
    const TLz4Registrar Registrar{};
}
