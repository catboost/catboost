#include <library/cpp/blockcodecs/core/codecs.h>
#include <library/cpp/blockcodecs/core/common.h>
#include <library/cpp/blockcodecs/core/register.h>
#include <library/cpp/streams/zstd/zstd.h>

#include <util/generic/maybe.h>
#include <util/stream/str.h>

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

    // A streamed Zstd frame is readable by TZStd08Codec if it has the usual TAddLengthCodec prefix.
    class TZstdBlockEncoder final : public IOutputStream {
    public:
        TZstdBlockEncoder(TString& output, int level)
            : StringOutput_(TStringOutput(output))
            , String_(&output)
            , PrefixOffset_(output.size())
            , Compressor_(&*StringOutput_, level)
        {
            output.append(sizeof(ui64), '\0');
        }

        TZstdBlockEncoder(IOutputStream& output, ui64 uncompressedSize, int level)
            : ExpectedSize_(uncompressedSize)
            , Compressor_(&output, level)
        {
            output.Write(&uncompressedSize, sizeof(uncompressedSize));
        }

    private:
        void DoWrite(const void* buffer, size_t size) override {
            Compressor_.Write(buffer, size);
            Size_ += size;
        }

        void DoFlush() override {
            Compressor_.Flush();
        }

        void DoFinish() override {
            if (Finished_) {
                return;
            }
            Y_ENSURE(!ExpectedSize_ || *ExpectedSize_ == Size_, "Unexpected uncompressed size");
            Compressor_.Finish();
            if (String_) {
                WriteUnaligned<ui64>(String_->begin() + PrefixOffset_, Size_);
            }
            Finished_ = true;
        }

        TMaybe<TStringOutput> StringOutput_;
        TString* const String_ = nullptr;
        const size_t PrefixOffset_ = 0;
        const TMaybe<ui64> ExpectedSize_;
        TZstdCompress Compressor_;
        ui64 Size_ = 0;
        bool Finished_ = false;
    };

    const TStreamingBlockEncoderFactory StreamingBlockEncoderFactory{
        .ForString = [](const ICodec& codec, TString& output) -> THolder<IOutputStream> {
            return MakeHolder<TZstdBlockEncoder>(output, static_cast<const TZStd08Codec&>(codec).Level);
        },
        .ForKnownSize = [](const ICodec& codec, IOutputStream& output, ui64 uncompressedSize) -> THolder<IOutputStream> {
            return MakeHolder<TZstdBlockEncoder>(output, uncompressedSize, static_cast<const TZStd08Codec&>(codec).Level);
        },
    };

    struct TZStd08Registrar {
        TZStd08Registrar() {
            for (int i = 1; i <= ZSTD_maxCLevel(); ++i) {
                const TString name = "zstd08_"sv + ToString(i);
                RegisterCodec(MakeHolder<TZStd08Codec>(i, name), StreamingBlockEncoderFactory);
                RegisterAlias("zstd_"sv + ToString(i), name);
            }

            for (int i = 1; i <= 7; ++i) {
                RegisterCodec(MakeHolder<TZStd08Codec>(-i, "zstd_fast_"sv + ToString(i)), StreamingBlockEncoderFactory);
            }
        }
    };
    const TZStd08Registrar Registrar{};
}
