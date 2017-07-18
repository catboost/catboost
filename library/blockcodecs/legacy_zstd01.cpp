#include "legacy.h"
#include "codecs.h"

#include <contrib/libs/zstd01/zstd.h>

#include <util/ysaveload.h>
#include <util/stream/mem.h>
#include <util/generic/yexception.h>

using namespace NBlockCodecs;

namespace {
    // legacy zstd codec
    struct TLegacyZStdCodec: public ICodec {
        size_t DecompressedLength(const TData& in) const override {
            TMemoryInput mi(in);
            ui32 ret;

            ::Load(&mi, ret);

            return ret;
        }

        static inline size_t CheckError(size_t ret) {
            if (ZSTD_isError(ret)) {
                ythrow yexception() << STRINGBUF("zstd error: ") << ZSTD_getErrorName(ret);
            }

            return ret;
        }

        static inline size_t MaxLen(size_t l) noexcept {
            return ZSTD_compressBound(l) + sizeof(ui32);
        }

        size_t MaxCompressedLength(const TData& in) const override {
            return MaxLen(+in);
        }

        size_t Compress(const TData& in, void* out) const override {
            TMemoryOutput mo(out, MaxLen(+in));

            ::Save(&mo, (ui32)(+in));

            return CheckError(ZSTD_compress(mo.Buf(), mo.Avail(), ~in, +in)) + sizeof(ui32);
        }

        size_t Decompress(const TData& in, void* out) const override {
            TMemoryInput mi(in);
            ui32 dsize;

            ::Load(&mi, dsize);

            return CheckError(ZSTD_decompress(out, dsize, mi.Buf(), mi.Avail()));
        }

        TStringBuf Name() const noexcept override {
            return STRINGBUF("zstd_legacy");
        }
    };
}

THolder<ICodec> NBlockCodecs::LegacyZStdCodec() {
    return new TLegacyZStdCodec();
}
