#include <library/cpp/blockcodecs/core/codecs.h>
#include <library/cpp/blockcodecs/core/common.h>
#include <library/cpp/blockcodecs/core/register.h>

#include <contrib/libs/zstd06/common/zstd.h>
#include <contrib/libs/zstd06/common/zstd_static.h>

using namespace NBlockCodecs;

namespace {
    struct TZStd06Codec: public TAddLengthCodec<TZStd06Codec> {
        inline TZStd06Codec(unsigned level)
            : Level(level)
            , MyName(TStringBuf("zstd06_") + ToString(Level))
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

    struct TZStd06Registrar {
        TZStd06Registrar() {
            for (unsigned i = 1; i <= ZSTD_maxCLevel(); ++i) {
                RegisterCodec(MakeHolder<TZStd06Codec>(i));
            }
        }
    };
    const TZStd06Registrar Registrar{};
}
