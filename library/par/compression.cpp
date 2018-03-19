#include "par_log.h"

#include <library/blockcodecs/codecs.h>
#include <library/logger/global/global.h>
#include <util/system/env.h>
#include <util/generic/utility.h>

namespace NPar {
    const int MIN_SIZE_TO_PACK = 4000;
    const int N_SIGNATURE = 0x21a9e395;
    const int SIZEOF_SIGNATURE = sizeof(int);

    const unsigned int BLOCK_SIZE = 2000000000;
    using TBlockLen = unsigned int;

    using namespace NBlockCodecs;

    static const ICodec* GetCodec() {
        static const ICodec* codecPtr = nullptr;
        if (!codecPtr) {
            TString codecName = GetEnv("PAR_CODEC");
            if (!codecName) {
                codecName = "lz4fast";
            }
            codecPtr = Codec(codecName);
        }
        return codecPtr;
    }

    void QuickLZCompress(TVector<char>* dst) {
        if (!dst || dst->empty())
            return;
        TStringBuf srcdata{~*dst, +*dst};
        if (+srcdata > MIN_SIZE_TO_PACK || (+srcdata >= SIZEOF_SIGNATURE && *(int*)~srcdata == N_SIGNATURE)) {
            const ICodec* usedCodec = GetCodec();
            TVector<char> packed;
            packed.yresize(SIZEOF_SIGNATURE);
            *(int*)(~packed) = N_SIGNATURE;
            /* Blocking for data >2Gb as most codecs don't support such big data chunks.
               Stream format as follows:
               | signature 4 bytes | LengthOfBlock1 4 bytes | Block1 compressed data | LengthOfBlock2 4 bytes| Block2 compressed data | ... |
            */
            for (size_t i = 0; i < +srcdata; i += BLOCK_SIZE) {
                TStringBuf srcDataBlock{~*dst + i, Min((size_t)(+*dst - i), (size_t)BLOCK_SIZE)};
                size_t packedPreviousSize = +packed;
                packed.yresize(packedPreviousSize + sizeof(TBlockLen) + usedCodec->MaxCompressedLength(srcDataBlock));
                TBlockLen packedBlockSize = usedCodec->Compress(srcDataBlock, (~packed + packedPreviousSize + sizeof(TBlockLen)));
                *(TBlockLen*)(~packed + packedPreviousSize) = packedBlockSize;
                packed.yresize(packedPreviousSize + sizeof(TBlockLen) + packedBlockSize);
            }
            dst->swap(packed);
        }
    }

    void QuickLZDecompress(TVector<char>* dst) {
        if (!dst) {
            return;
        }
        int srcSize = dst->ysize();
        if (srcSize < SIZEOF_SIGNATURE) {
            return;
        }
        if (*(int*)~*dst != N_SIGNATURE) {
            return;
        }
        TVector<char> unpacked;
        const ICodec* usedCodec = GetCodec();
        for (size_t i = SIZEOF_SIGNATURE; i < +(*dst);) {
            TBlockLen packedBlockSize = *(TBlockLen*)(~*dst + i);
            TStringBuf srcDataBlock{~*dst + i + sizeof(TBlockLen), packedBlockSize};
            size_t unpackedPreviousSize = +unpacked;
            unpacked.yresize(unpackedPreviousSize + usedCodec->DecompressedLength(srcDataBlock));
            unpacked.yresize(unpackedPreviousSize + usedCodec->Decompress(srcDataBlock, ~unpacked + unpackedPreviousSize));
            i += packedBlockSize + sizeof(TBlockLen);
        }
        dst->swap(unpacked);
    }
}
