#pragma once

#include <catboost/libs/helpers/exception.h>

#include <library/cpp/binsaver/bin_saver.h>
#include <library/cpp/dbg_output/dump.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/array_ref.h>
#include <util/generic/cast.h>
#include <util/generic/utility.h>
#include <util/system/types.h>
#include <util/system/yassert.h>

#include <climits>
#include <tuple>
#include <type_traits>


namespace NCB {

    static_assert(CHAR_BIT == 8, "CatBoost requires CHAR_BIT == 8");

    using TBinaryFeaturesPack = ui8;

    // 2d index: [PackIdx][BitIdx]
    struct TPackedBinaryIndex {
        ui32 PackIdx;
        ui8 BitIdx;

        ui8 BitsPerPack;

    public:
        // needed for BinSaver
        explicit TPackedBinaryIndex(
            ui32 packIdx = 0,
            ui32 bitIdx = 0,
            ui8 bitsPerPack = sizeof(TBinaryFeaturesPack) * CHAR_BIT)
            : PackIdx(packIdx)
            , BitIdx(bitIdx)
            , BitsPerPack(bitsPerPack)
        {}

        static TPackedBinaryIndex FromLinearIdx(
            ui32 linearIdx,
            ui8 bitsPerPack = sizeof(TBinaryFeaturesPack) * CHAR_BIT)
        {
            return TPackedBinaryIndex(linearIdx / bitsPerPack, linearIdx % bitsPerPack, bitsPerPack);
        }

        bool operator==(const TPackedBinaryIndex rhs) const {
            return std::tie(PackIdx, BitIdx, BitsPerPack) ==
                std::tie(rhs.PackIdx, rhs.BitIdx, rhs.BitsPerPack);
        }

        SAVELOAD(PackIdx, BitIdx, BitsPerPack);

        ui32 GetLinearIdx() const {
            return BitsPerPack*PackIdx + BitIdx;
        }
    };

    inline void CheckBitIdxForPackedBinaryIndex(ui8 bitIdx) {
        CB_ENSURE_INTERNAL(
            bitIdx < sizeof(TBinaryFeaturesPack)*CHAR_BIT,
            "bitIdx=" << bitIdx << " is out of range (bitIdx exclusive upper bound for TBinaryFeaturesPack ="
            << sizeof(TBinaryFeaturesPack)*CHAR_BIT << ')'
        );
    }

    // Do not call for different bits in parallel!
    template <class TSrcElement>
    void SetBinaryFeatureInPackArray(
        TConstArrayRef<TSrcElement> srcFeature,
        ui8 bitIdx,
        bool needToClearDstBits,
        bool skipCheck, // skip checks if called for multiple parts in parallel
        TArrayRef<TBinaryFeaturesPack>* dstFeaturePacks) {

        static_assert(
            std::is_unsigned<TSrcElement>::value,
            "SetBinaryFeatureInPackArray requires unsigned source data"
        );

        if (skipCheck) {
            Y_ASSERT(bitIdx < sizeof(TBinaryFeaturesPack)*CHAR_BIT);
        } else {
            CheckBitIdxForPackedBinaryIndex(bitIdx);
        }

        TBinaryFeaturesPack* dstIt = dstFeaturePacks->data();

        if (needToClearDstBits) {
            const TBinaryFeaturesPack clearBitMask = ~(TBinaryFeaturesPack(1) << bitIdx);
            for (auto srcIt = srcFeature.begin(); srcIt != srcFeature.end(); ++srcIt, ++dstIt) {
                CB_ENSURE_INTERNAL(*srcIt < 2, "attempt to interpret non-binary feature as binary");
                *dstIt = (*dstIt & clearBitMask) | (TBinaryFeaturesPack(*srcIt) << bitIdx);
            }
        } else {
            for (auto srcIt = srcFeature.begin(); srcIt != srcFeature.end(); ++srcIt, ++dstIt) {
                CB_ENSURE_INTERNAL(*srcIt < 2, "attempt to interpret non-binary feature as binary");
                *dstIt |= TBinaryFeaturesPack(*srcIt) << bitIdx;
            }
        }
    }


    // Do not call for different bits in parallel!
    template <class TSrcElement>
    void ParallelSetBinaryFeatureInPackArray(
        TConstArrayRef<TSrcElement> srcFeature,
        ui8 bitIdx,
        bool needToClearDstBits,
        NPar::ILocalExecutor* localExecutor,
        TArrayRef<TBinaryFeaturesPack>* dstFeaturePacks) {

        CheckBitIdxForPackedBinaryIndex(bitIdx);

        int objectCount = SafeIntegerCast<int>(srcFeature.size());
        NPar::ILocalExecutor::TExecRangeParams rangeParams(0, objectCount);
        rangeParams.SetBlockCount(localExecutor->GetThreadCount() + 1);

        localExecutor->ExecRangeWithThrow(
            [&](int i) {
                int startIdx = i*rangeParams.GetBlockSize();
                int endIdx = Min(startIdx + rangeParams.GetBlockSize(), objectCount);

                TArrayRef<TBinaryFeaturesPack> dstPart = dstFeaturePacks->Slice(startIdx, endIdx - startIdx);
                SetBinaryFeatureInPackArray(
                    srcFeature.Slice(startIdx, endIdx - startIdx),
                    bitIdx,
                    needToClearDstBits,
                    /*skipCheck*/ true,
                    &dstPart);
            },
            0,
            rangeParams.GetBlockCount(),
            NPar::TLocalExecutor::WAIT_COMPLETE
        );
    }

}


template <>
struct TDumper<NCB::TPackedBinaryIndex> {
    template <class S>
    static inline void Dump(S& s, const NCB::TPackedBinaryIndex& packedBinaryIndex) {
        s << "PackIdx=" << ui32(packedBinaryIndex.PackIdx)
          << ",BitIdx=" << ui32(packedBinaryIndex.BitIdx)
          << ",BitsPerPack=" << ui32(packedBinaryIndex.BitsPerPack);
    }
};

