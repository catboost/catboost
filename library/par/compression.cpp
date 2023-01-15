#include "par_log.h"

#include <library/cpp/blockcodecs/codecs.h>
#include <library/cpp/logger/global/global.h>
#include <util/system/env.h>
#include <util/generic/singleton.h>
#include <util/generic/utility.h>

using namespace NBlockCodecs;

namespace {
    struct TCompressionHolder {
        TCompressionHolder() {
            CodecPtr = Codec(GetEnv("PAR_CODEC", "lz4fast"));
        }
        const ICodec* CodecPtr = nullptr;
    };
}

namespace NPar {
    const int MIN_SIZE_TO_PACK = 4000;
    const int N_SIGNATURE = 0x21a9e395;
    const int SIZEOF_SIGNATURE = sizeof(int);

    const unsigned int BLOCK_SIZE = 2000000000;
    using TBlockLen = unsigned int;

    static const ICodec* GetCodec() {
        return Singleton<TCompressionHolder>()->CodecPtr;
    }

    void QuickLZCompress(TVector<char>* dst) {
        if (!dst || dst->empty())
            return;
        TStringBuf srcdata{dst->data(), dst->size()};
        if (srcdata.size() > MIN_SIZE_TO_PACK || (srcdata.size() >= SIZEOF_SIGNATURE && *(int*)srcdata.data() == N_SIGNATURE)) {
            const ICodec* usedCodec = GetCodec();
            TVector<char> packed;
            packed.yresize(SIZEOF_SIGNATURE);
            *(int*)(packed.data()) = N_SIGNATURE;
            /* Blocking for data >2Gb as most codecs don't support such big data chunks.
               Stream format as follows:
               | signature 4 bytes | LengthOfBlock1 4 bytes | Block1 compressed data | LengthOfBlock2 4 bytes| Block2 compressed data | ... |
            */
            for (size_t i = 0; i < srcdata.size(); i += BLOCK_SIZE) {
                TStringBuf srcDataBlock{dst->data() + i, Min((size_t)(dst->size() - i), (size_t)BLOCK_SIZE)};
                size_t packedPreviousSize = packed.size();
                packed.yresize(packedPreviousSize + sizeof(TBlockLen) + usedCodec->MaxCompressedLength(srcDataBlock));
                TBlockLen packedBlockSize = usedCodec->Compress(srcDataBlock, (packed.data() + packedPreviousSize + sizeof(TBlockLen)));
                *(TBlockLen*)(packed.data() + packedPreviousSize) = packedBlockSize;
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
        if (*(int*)dst->data() != N_SIGNATURE) {
            return;
        }
        TVector<char> unpacked;
        const ICodec* usedCodec = GetCodec();
        for (size_t i = SIZEOF_SIGNATURE; i < (*dst).size();) {
            TBlockLen packedBlockSize = *(TBlockLen*)(dst->data() + i);
            TStringBuf srcDataBlock{dst->data() + i + sizeof(TBlockLen), packedBlockSize};
            size_t unpackedPreviousSize = unpacked.size();
            unpacked.yresize(unpackedPreviousSize + usedCodec->DecompressedLength(srcDataBlock));
            unpacked.yresize(unpackedPreviousSize + usedCodec->Decompress(srcDataBlock, unpacked.data() + unpackedPreviousSize));
            i += packedBlockSize + sizeof(TBlockLen);
        }
        dst->swap(unpacked);
    }
}
