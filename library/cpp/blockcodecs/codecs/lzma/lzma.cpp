#include <library/cpp/blockcodecs/core/codecs.h>
#include <library/cpp/blockcodecs/core/common.h>
#include <library/cpp/blockcodecs/core/register.h>

#include <contrib/libs/lzmasdk/LzmaLib.h>

using namespace NBlockCodecs;

namespace {
    struct TLzmaCodec: public TAddLengthCodec<TLzmaCodec> {
        inline TLzmaCodec(int level)
            : Level(level)
            , MyName("lzma-" + ToString(Level))
        {
        }

        static inline size_t DoMaxCompressedLength(size_t in) noexcept {
            return Max<size_t>(in + in / 20, 128) + LZMA_PROPS_SIZE;
        }

        TStringBuf Name() const noexcept override {
            return MyName;
        }

        inline size_t DoCompress(const TData& in, void* buf) const {
            unsigned char* props = (unsigned char*)buf;
            unsigned char* data = props + LZMA_PROPS_SIZE;
            size_t destLen = Max<size_t>();
            size_t outPropsSize = LZMA_PROPS_SIZE;

            const int ret = LzmaCompress(data, &destLen, (const unsigned char*)in.data(), in.size(), props, &outPropsSize, Level, 0, -1, -1, -1, -1, -1);

            if (ret != SZ_OK) {
                ythrow TCompressError(ret);
            }

            return destLen + LZMA_PROPS_SIZE;
        }

        inline void DoDecompress(const TData& in, void* out, size_t len) const {
            if (in.size() <= LZMA_PROPS_SIZE) {
                ythrow TDataError() << TStringBuf("broken lzma stream");
            }

            const unsigned char* props = (const unsigned char*)in.data();
            const unsigned char* data = props + LZMA_PROPS_SIZE;
            size_t destLen = len;
            SizeT srcLen = in.size() - LZMA_PROPS_SIZE;

            const int res = LzmaUncompress((unsigned char*)out, &destLen, data, &srcLen, props, LZMA_PROPS_SIZE);

            if (res != SZ_OK) {
                ythrow TDecompressError(res);
            }

            if (destLen != len) {
                ythrow TDecompressError(len, destLen);
            }
        }

        const int Level;
        const TString MyName;
    };

    struct TLzmaRegistrar {
        TLzmaRegistrar() {
            for (int i = 0; i < 10; ++i) {
                RegisterCodec(MakeHolder<TLzmaCodec>(i));
            }
            RegisterAlias("lzma", "lzma-5");
        }
    };
    const TLzmaRegistrar Registrar{};
}
