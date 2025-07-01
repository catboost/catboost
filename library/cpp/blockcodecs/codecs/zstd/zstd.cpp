#include <library/cpp/blockcodecs/core/codecs.h>
#include <library/cpp/blockcodecs/core/common.h>
#include <library/cpp/blockcodecs/core/register.h>

#define ZSTD_STATIC_LINKING_ONLY
#include <contrib/libs/zstd/include/zstd.h>

#include <utility>

using namespace NBlockCodecs;

namespace {
    struct TZStd08Codec: public TAddLengthCodec<TZStd08Codec> {
        TZStd08Codec(int level, TString name)
            : Level(level)
            , MyName(std::move(name))
        {
        }

        static size_t CheckError(size_t ret, const char* what) {
            if (Y_UNLIKELY(ZSTD_isError(ret))) {
                ythrow yexception() << what << TStringBuf(" zstd error: ") << ZSTD_getErrorName(ret);
            }

            return ret;
        }

        static size_t DoMaxCompressedLength(size_t l) noexcept {
            return ZSTD_compressBound(l);
        }

        size_t DoCompress(const TData& in, void* out) const {
            return CheckError(ZSTD_compress(out, DoMaxCompressedLength(in.size()), in.data(), in.size(), Level), "compress");
        }

        static void DoDecompress(const TData& in, void* out, size_t dsize) {
            const size_t res = CheckError(ZSTD_decompress(out, dsize, in.data(), in.size()), "decompress");

            if (Y_UNLIKELY(res != dsize)) {
                ythrow TDecompressError(dsize, res);
            }
        }

        TStringBuf Name() const noexcept override {
            return MyName;
        }

        const int Level;
        const TString MyName;
    };

    struct TZStd08Registrar {
        TZStd08Registrar() {
            for (int i = 1; i <= ZSTD_maxCLevel(); ++i) {
                const TString name = "zstd08_"sv + ToString(i);
                RegisterCodec(MakeHolder<TZStd08Codec>(i, name));
                RegisterAlias("zstd_"sv + ToString(i), name);
            }

            for (int i = 1; i <= 7; ++i) {
                RegisterCodec(MakeHolder<TZStd08Codec>(-i, "zstd_fast_"sv + ToString(i)));
            }
        }
    };
    const TZStd08Registrar Registrar{};
}
