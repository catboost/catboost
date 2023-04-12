#include <library/cpp/blockcodecs/core/codecs.h>
#include <library/cpp/blockcodecs/core/common.h>
#include <library/cpp/blockcodecs/core/register.h>

#include <zlib.h>

using namespace NBlockCodecs;

namespace {
    struct TZLibCodec: public TAddLengthCodec<TZLibCodec> {
        inline TZLibCodec(int level)
            : MyName("zlib-" + ToString(level))
            , Level(level)
        {
        }

        static inline size_t DoMaxCompressedLength(size_t in) noexcept {
            return compressBound(in);
        }

        TStringBuf Name() const noexcept override {
            return MyName;
        }

        inline size_t DoCompress(const TData& in, void* buf) const {
            //TRASH detected
            uLong ret = Max<unsigned int>();

            int cres = compress2((Bytef*)buf, &ret, (const Bytef*)in.data(), in.size(), Level);

            if (cres != Z_OK) {
                ythrow TCompressError(cres);
            }

            return ret;
        }

        inline void DoDecompress(const TData& in, void* out, size_t len) const {
            uLong ret = len;

            int uncres = uncompress((Bytef*)out, &ret, (const Bytef*)in.data(), in.size());
            if (uncres != Z_OK) {
                ythrow TDecompressError(uncres);
            }

            if (ret != len) {
                ythrow TDecompressError(len, ret);
            }
        }

        const TString MyName;
        const int Level;
    };

    struct TZLibRegistrar {
        TZLibRegistrar() {
            for (int i = 0; i < 10; ++i) {
                RegisterCodec(MakeHolder<TZLibCodec>(i));
            }
            RegisterAlias("zlib", "zlib-6");
        }
    };
    const TZLibRegistrar Registrar{};
}
