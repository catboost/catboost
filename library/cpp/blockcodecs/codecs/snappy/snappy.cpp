#include <library/cpp/blockcodecs/core/codecs.h>
#include <library/cpp/blockcodecs/core/common.h>
#include <library/cpp/blockcodecs/core/register.h>

#include <contrib/libs/snappy/snappy.h>

using namespace NBlockCodecs;

namespace {
    struct TSnappyCodec: public ICodec {
        size_t DecompressedLength(const TData& in) const override {
            size_t ret;

            if (snappy::GetUncompressedLength(in.data(), in.size(), &ret)) {
                return ret;
            }

            ythrow TDecompressError(0);
        }

        size_t MaxCompressedLength(const TData& in) const override {
            return snappy::MaxCompressedLength(in.size());
        }

        size_t Compress(const TData& in, void* out) const override {
            size_t ret;

            snappy::RawCompress(in.data(), in.size(), (char*)out, &ret);

            return ret;
        }

        size_t Decompress(const TData& in, void* out) const override {
            if (snappy::RawUncompress(in.data(), in.size(), (char*)out)) {
                return DecompressedLength(in);
            }

            ythrow TDecompressError(0);
        }

        TStringBuf Name() const noexcept override {
            return "snappy";
        }
    };

    struct TSnappyRegistrar {
        TSnappyRegistrar() {
            RegisterCodec(MakeHolder<TSnappyCodec>());
        }
    };
    const TSnappyRegistrar Registrar{};
}
