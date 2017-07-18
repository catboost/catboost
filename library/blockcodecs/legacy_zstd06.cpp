#include "codecs.h"
#include "legacy.h"

#include <contrib/libs/zstd06/common/zstd.h>
#include <contrib/libs/zstd06/common/zstd_static.h>

#include <util/ysaveload.h>
#include <util/stream/null.h>
#include <util/stream/mem.h>
#include <util/string/cast.h>
#include <util/string/join.h>
#include <util/system/align.h>
#include <util/system/unaligned_mem.h>
#include <util/generic/hash.h>
#include <util/generic/cast.h>
#include <util/generic/buffer.h>
#include <util/generic/region.h>
#include <util/generic/singleton.h>
#include <util/generic/algorithm.h>
#include <util/generic/mem_copy.h>

using namespace NBlockCodecs;

namespace {

    struct TZStd06Codec: public TAddLengthCodec<TZStd06Codec> {
        inline TZStd06Codec(unsigned level)
            : Level(level)
            , MyName(STRINGBUF("zstd_") + ToString(Level))
        {
        }

        static inline size_t CheckError(size_t ret, const char* what) {
            if (ZSTD_isError(ret)) {
                ythrow yexception() << what << STRINGBUF(" zstd error: ") << ZSTD_getErrorName(ret);
            }

            return ret;
        }

        static inline size_t DoMaxCompressedLength(size_t l) noexcept {
            return ZSTD_compressBound(l);
        }

        inline size_t DoCompress(const TData& in, void* out) const {
            return CheckError(ZSTD_compress(out, DoMaxCompressedLength(+in), ~in, +in, Level), "compress");
        }

        inline void DoDecompress(const TData& in, void* out, size_t dsize) const {
            const size_t res = CheckError(ZSTD_decompress(out, dsize, ~in, +in), "decompress");

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

}

namespace NBlockCodecs {
    yvector<TCodecPtr> LegacyZStd06Codec() {
        yvector<TCodecPtr> codecs;

        for (unsigned i = 1; i <= ZSTD_maxCLevel(); ++i) {
            codecs.emplace_back(new TZStd06Codec(i));
        }

        return codecs;
    }
}
