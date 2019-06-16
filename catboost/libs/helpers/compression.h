#pragma once

#include "array_subset.h"
#include "exception.h"
#include "maybe_owning_array_holder.h"

#include <library/threading/local_executor/local_executor.h>

#include <util/system/defaults.h>
#include <util/system/types.h>
#include <util/generic/yexception.h>
#include <util/string/builder.h>
#include <util/generic/array_ref.h>
#include <util/generic/ymath.h>
#include <util/generic/vector.h>

#include <climits>
#include <cmath>


template <class TStorageType>
class TIndexHelper {
public:
    TIndexHelper() = default;

    explicit TIndexHelper(ui32 bitsPerKey)
        : BitsPerKey(bitsPerKey)
    {
        CB_ENSURE(bitsPerKey <= 32, "Too many bits in key");
        EntriesPerType = sizeof(TStorageType) * CHAR_BIT / BitsPerKey;
    }

    inline ui64 Mask() const {
        return ((static_cast<TStorageType>(1) << BitsPerKey) - 1);
    }

    inline ui32 Offset(ui32 index) const {
        return index / EntriesPerType;
    }

    inline ui32 Shift(ui32 index) const {
        return (index % EntriesPerType) * BitsPerKey;
    }

    inline ui32 GetBitsPerKey() const {
        return BitsPerKey;
    }

    inline ui32 GetEntriesPerType() const {
        return EntriesPerType;
    }

    inline ui32 CompressedSize(ui32 size) const {
        return CeilDiv(size, EntriesPerType);
    }

    template <class T>
    inline T Extract(TConstArrayRef<TStorageType> compressedData, ui32 index) const {
        Y_ASSERT(sizeof(T) * CHAR_BIT >= BitsPerKey);
        const ui32 offset = Offset(index);
        const ui32 shift = Shift(index);
        return static_cast<T>((compressedData[offset] >> shift) & Mask());
    }

private:
    ui32 BitsPerKey = Max<ui32>();
    ui32 EntriesPerType = 0;
};


class TCompressedArray {
public:
    TCompressedArray() = default;

    TCompressedArray(ui64 size, ui32 bitsPerKey, NCB::TMaybeOwningArrayHolder<ui64> storage)
        : Size(size)
        , IndexHelper(bitsPerKey)
        , Storage(std::move(storage))
    {}

    ui64 GetSize() const {
        return Size;
    }

    ui32 GetBitsPerKey() const {
        return IndexHelper.GetBitsPerKey();
    }

    template <class T = ui32>
    T operator[](ui32 index) const {
        Y_ASSERT(index < Size);
        return IndexHelper.Extract<T>(*Storage, index);
    }

    // will throw exception if data cannot be interpreted as usual array as-is
    template <class T>
    void CheckIfCanBeInterpretedAsRawArray() const {
#if defined(_big_endian_)
        static_assert(
            (sizeof(T) == sizeof(ui64)),
            "Can't interpret TCompressedArray's data as raw array because of big-endian architecture"
        );
#endif
        static_assert(
            (sizeof(T) == sizeof(ui64)) || (alignof(ui64) == sizeof(ui64)),
            "Can't interpret TCompressedArray's data as raw array because of alignment"
        );
        CB_ENSURE(
            GetBitsPerKey() == sizeof(T)*CHAR_BIT,
            "Can't interpret TCompressedArray's data as raw array: elements are of size " << GetBitsPerKey()
            << " bits, but " << (sizeof(T)*CHAR_BIT) << " bits requested"
        );
    }

    // works only if BitsPerKey == sizeof(T)*CHAR_BIT
    template <class T>
    TConstArrayRef<T> GetRawArray() const {
        CheckIfCanBeInterpretedAsRawArray<T>();
        return TConstArrayRef<T>(reinterpret_cast<T*>((*Storage).data()), Size);
    }

    char* GetRawPtr() {
        return reinterpret_cast<char*>((*Storage).data());
    }

    const char* GetRawPtr() const {
        return reinterpret_cast<const char*>((*Storage).data());
    }

private:
    ui64 Size = 0;
    TIndexHelper<ui64> IndexHelper;
    NCB::TMaybeOwningArrayHolder<ui64> Storage;
};


template <class TStorageType, class T>
inline TVector<TStorageType> CompressVector(const T* data, ui32 size, ui32 bitsPerKey) {
    CB_ENSURE(bitsPerKey <= 32);
    CB_ENSURE(bitsPerKey, "Error: data with zero bits per key. Something went wrong");

    TVector<TStorageType> dst;
    TIndexHelper<TStorageType> indexHelper(bitsPerKey);
    dst.resize(indexHelper.CompressedSize(size));
    const auto mask = indexHelper.Mask();

    NPar::TLocalExecutor::TExecRangeParams params(0, size);
    //alignment by entries per int allows parallel compression
    params.SetBlockSize(indexHelper.GetEntriesPerType() * 8192);

    NPar::LocalExecutor().ExecRange(
        [&](int blockIdx) {
            NPar::TLocalExecutor::BlockedLoopBody(
                params,
                [&](int i) {
                    const ui32 offset = indexHelper.Offset((ui32)i);
                    const ui32 shift = indexHelper.Shift((ui32)i);
                    CB_ENSURE(
                        (data[i] & mask) == data[i],
                        TStringBuilder() << "Error: key contains too many bits: max bits per key: allowed "
                            << bitsPerKey << ", observe key " << static_cast<ui64>(data[i])
                    );
                    dst[offset] |= static_cast<ui64>(data[i]) << shift;
                }
            )(blockIdx);
        },
        0,
        params.GetBlockCount(),
        NPar::TLocalExecutor::WAIT_COMPLETE
    );

    return dst;
}

template <class TStorageType, class T>
inline TVector<TStorageType> CompressVector(const TVector<T>& data, ui32 bitsPerKey) {
    return CompressVector<TStorageType, T>(data.data(), data.size(), bitsPerKey);
}

template <class TStorageType, class T>
inline TVector<T> DecompressVector(const TVector<TStorageType>& compressedData, ui32 keys, ui32 bitsPerKey) {
    TVector<T> dst;
    CB_ENSURE(bitsPerKey < 32);
    CB_ENSURE(sizeof(T) <= sizeof(TStorageType));
    dst.clear();
    dst.resize(keys);
    const TIndexHelper<TStorageType> indexHelper(bitsPerKey);
    const auto mask = indexHelper.Mask();

    NPar::ParallelFor(
        0,
        keys,
        [&](int i) {
            const ui32 offset = indexHelper.Offset(i);
            const ui32 shift = indexHelper.Shift(i);
            dst[i] = (compressedData[offset] >> shift) & mask;
        }
    );

    return dst;
}

