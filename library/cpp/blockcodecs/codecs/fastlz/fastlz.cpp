#include <library/cpp/blockcodecs/core/codecs.h>
#include <library/cpp/blockcodecs/core/common.h>
#include <library/cpp/blockcodecs/core/register.h>

#include <contrib/libs/fastlz/fastlz.h>

using namespace NBlockCodecs;

namespace {
    struct TFastLZCodec: public TAddLengthCodec<TFastLZCodec> {
        inline TFastLZCodec(int level)
            : MyName("fastlz-" + ToString(level))
            , Level(level)
        {
        }

        static inline size_t DoMaxCompressedLength(size_t in) noexcept {
            return Max<size_t>(in + in / 20, 128);
        }

        TStringBuf Name() const noexcept override {
            return MyName;
        }

        inline size_t DoCompress(const TData& in, void* buf) const {
            if (Level) {
                return fastlz_compress_level(Level, in.data(), in.size(), buf);
            }

            return fastlz_compress(in.data(), in.size(), buf);
        }

        inline void DoDecompress(const TData& in, void* out, size_t len) const {
            const int ret = fastlz_decompress(in.data(), in.size(), out, len);

            if (ret < 0 || (size_t)ret != len) {
                ythrow TDataError() << TStringBuf("can not decompress");
            }
        }

        const TString MyName;
        const int Level;
    };

    struct TFastLZRegistrar {
        TFastLZRegistrar() {
            for (int i = 0; i < 3; ++i) {
                RegisterCodec(MakeHolder<TFastLZCodec>(i));
            }
            RegisterAlias("fastlz", "fastlz-0");
        }
    };
    const TFastLZRegistrar Registrar{};
}
