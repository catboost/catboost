#pragma once

#include "power_hash.h"

#include <library/cpp/binsaver/bin_saver.h>

#include <util/ysaveload.h>
#include <util/generic/typetraits.h>

namespace NTensor2 {
    const ui64 INVALID_DENSE_HASH_INDEX = 0xffffffffffffffffull;
    const ui64 TRASH_BUCKET_IDX = ui64(-1) - 1;

    namespace NDenseHashResizePolicy {
        struct TNoResize {
            static bool NeedResize(ui64 binCount, ui64 hashSize) {
                Y_UNUSED(binCount)
                , Y_UNUSED(hashSize);
                return false;
            }
            static ui64 GetNewHashSize(ui64 binCount, ui64 hashSize) {
                Y_UNUSED(binCount);
                return hashSize;
            }
        };

        struct TDoubleIfHalfFull {
            static bool NeedResize(ui64 binCount, ui64 hashSize) {
                return binCount * 2 > hashSize;
            }
            static ui64 GetNewHashSize(ui64 binCount, ui64 hashSize) {
                Y_UNUSED(binCount);
                return hashSize * 2;
            }
        };
    }
    struct TElem {
        ui64 Index;
        ui64 Bin;
    };
} // namespace NTensor2

Y_DECLARE_PODTYPE(NTensor2::TElem);

namespace NTensor2 {
    template <class TResizePolicy = NDenseHashResizePolicy::TNoResize>
    struct TDenseHash {
        TVector<TElem> Elems;
        ui64 HashMask;
        ui64 BinCount;

        static ui64 CallsCount;
        static ui64 ItersCount;

    public:
        TDenseHash()
            : HashMask(0)
            , BinCount(0)
        {
        }

        SAVELOAD(Elems, HashMask, BinCount);
        Y_SAVELOAD_DEFINE(Elems, HashMask, BinCount);

        void Clear() {
            Elems.resize(0);
            HashMask = 0;
            BinCount = 0;
        }
        void Init(ui64 hashSize) {
            Clear();

            TElem zz;
            zz.Index = INVALID_DENSE_HASH_INDEX;
            zz.Bin = 0;

            HashMask = 1;
            while (HashMask < hashSize) {
                HashMask <<= 1;
            }
            Elems.resize(HashMask, zz);
            HashMask--; // pow 2 - 1
        }

        ui64 GetIndex(ui64 idx) const {
            CallsCount++;
            ui64 zz = (idx * NPowerHash2::MAGIC_MULT) & HashMask;
            for (;;) {
                ItersCount++;
                const TElem& elem = Elems[zz];
                ui64 elemIdx = elem.Index;
                if (elemIdx == idx) {
                    return elem.Bin;
                }
                if (elemIdx == INVALID_DENSE_HASH_INDEX) {
                    return idx != TRASH_BUCKET_IDX ? GetIndex(TRASH_BUCKET_IDX) : INVALID_DENSE_HASH_INDEX;
                }
                zz = (zz + 1) & HashMask;
            }
        }
        ui64 AddIndex(ui64 idx) {
            ui64 zz = (idx * NPowerHash2::MAGIC_MULT) & HashMask;
            for (;;) {
                TElem& elem = Elems[zz];
                ui64 elemIdx = elem.Index;
                if (elemIdx == idx) {
                    return elem.Bin;
                }
                if (elemIdx == INVALID_DENSE_HASH_INDEX) {
                    elem.Index = idx;
                    elem.Bin = BinCount++;
                    ConsiderResize();
                    return BinCount - 1;
                }
                zz = (zz + 1) & HashMask;
            }
        }
        ui64 GetBinCount() const {
            return BinCount;
        }
        void Swap(TDenseHash& other) {
            Elems.swap(other.Elems);
            std::swap(HashMask, other.HashMask);
            std::swap(BinCount, other.BinCount);
        }
        void ConsiderResize() {
            if (TResizePolicy::NeedResize(BinCount, Elems.size())) {
                ui64 hashSize = TResizePolicy::GetNewHashSize(BinCount, Elems.size());
                Resize(hashSize);
            }
        }
        void Resize(ui64 hashSize) {
            TDenseHash resized;
            resized.Init(hashSize);
            for (const auto& elem : Elems) {
                if (elem.Index != INVALID_DENSE_HASH_INDEX) {
                    resized.AddIndex(elem.Index);
                }
            }
            this->Swap(resized);
        }
    };

    template <class TResizePolicy>
    ui64 TDenseHash<TResizePolicy>::CallsCount = 0;
    template <class TResizePolicy>
    ui64 TDenseHash<TResizePolicy>::ItersCount = 0;
}
