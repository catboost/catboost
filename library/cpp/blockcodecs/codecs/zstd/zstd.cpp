#include <library/cpp/blockcodecs/core/codecs.h>
#include <library/cpp/blockcodecs/core/common.h>
#include <library/cpp/blockcodecs/core/register.h>

#define ZSTD_STATIC_LINKING_ONLY
#include <contrib/libs/zstd/include/zstd.h>

using namespace NBlockCodecs;

namespace {
    struct TZStd08Codec: public TAddLengthCodec<TZStd08Codec> {
        inline TZStd08Codec(unsigned level)
            : Level(level)
            , MyName(TStringBuf("zstd08_") + ToString(Level))
        {
        }

        static inline size_t CheckError(size_t ret, const char* what) {
            if (ZSTD_isError(ret)) {
                ythrow yexception() << what << TStringBuf(" zstd error: ") << ZSTD_getErrorName(ret);
            }

            return ret;
        }

        static inline size_t DoMaxCompressedLength(size_t l) noexcept {
            return ZSTD_compressBound(l);
        }

        inline size_t DoCompress(const TData& in, void* out) const {
            return CheckError(ZSTD_compress(out, DoMaxCompressedLength(in.size()), in.data(), in.size(), Level), "compress");
        }

        inline void DoDecompress(const TData& in, void* out, size_t dsize) const {
            const size_t res = CheckError(ZSTD_decompress(out, dsize, in.data(), in.size()), "decompress");

            if (res != dsize) {
                ythrow TDecompressError(dsize, res);
            }
        }

        TStringBuf Name() const noexcept override {
            return MyName;
        }

        const unsigned Level;
        const TString MyName;
    };

    struct TZStd08Registrar {
        TZStd08Registrar() {
            for (int i = 1; i <= ZSTD_maxCLevel(); ++i) {
                RegisterCodec(MakeHolder<TZStd08Codec>(i));
                RegisterAlias("zstd_" + ToString(i), "zstd08_" + ToString(i));
            }
        }
    };
    const TZStd08Registrar Registrar{};
}
