#include <library/cpp/blockcodecs/core/codecs.h>
#include <library/cpp/blockcodecs/core/common.h>
#include <library/cpp/blockcodecs/core/register.h>

#include <contrib/libs/brotli/include/brotli/encode.h>
#include <contrib/libs/brotli/include/brotli/decode.h>

using namespace NBlockCodecs;

namespace {
    struct TBrotliCodec : public TAddLengthCodec<TBrotliCodec> {
        static constexpr int BEST_QUALITY = 11;

        inline TBrotliCodec(ui32 level)
            : Quality(level)
            , MyName(TStringBuf("brotli_") + ToString(level))
        {
        }

        static inline size_t DoMaxCompressedLength(size_t l) noexcept {
            return BrotliEncoderMaxCompressedSize(l);
        }

        inline size_t DoCompress(const TData& in, void* out) const {
            size_t resultSize = MaxCompressedLength(in);
            auto result = BrotliEncoderCompress(
                                /*quality*/ Quality,
                                /*window*/ BROTLI_DEFAULT_WINDOW,
                                /*mode*/ BrotliEncoderMode::BROTLI_MODE_GENERIC,
                                /*input_size*/ in.size(),
                                /*input_buffer*/ (const unsigned char*)(in.data()),
                                /*encoded_size*/ &resultSize,
                                /*encoded_buffer*/ static_cast<unsigned char*>(out));
            if (result != BROTLI_TRUE) {
                ythrow yexception() << "internal brotli error during compression";
            }

            return resultSize;
        }

        inline void DoDecompress(const TData& in, void* out, size_t dsize) const {
            size_t decoded = dsize;
            auto result = BrotliDecoderDecompress(in.size(), (const unsigned char*)in.data(), &decoded, static_cast<unsigned char*>(out));
            if (result != BROTLI_DECODER_RESULT_SUCCESS) {
                ythrow yexception() << "internal brotli error during decompression";
            } else if (decoded != dsize) {
                ythrow TDecompressError(dsize, decoded);
            }
        }

        TStringBuf Name() const noexcept override {
            return MyName;
        }

        const int Quality = BEST_QUALITY;
        const TString MyName;
    };

    struct TBrotliRegistrar {
        TBrotliRegistrar() {
            for (int i = 1; i <= TBrotliCodec::BEST_QUALITY; ++i) {
                RegisterCodec(MakeHolder<TBrotliCodec>(i));
            }
        }
    };
    const TBrotliRegistrar Registrar{};
}
