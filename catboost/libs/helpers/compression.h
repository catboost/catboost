#pragma once

#include "array_subset.h"
#include "dynamic_iterator.h"
#include "exception.h"
#include "maybe_owning_array_holder.h"

#include <library/cpp/binsaver/bin_saver.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/system/defaults.h>
#include <util/system/types.h>
#include <util/generic/yexception.h>
#include <util/string/builder.h>
#include <util/generic/array_ref.h>
#include <util/generic/ymath.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>

#include <climits>
#include <cmath>
#include <type_traits>


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

    SAVELOAD(BitsPerKey, EntriesPerType);

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

    TCompressedArray(ui64 size, ui32 bitsPerKey, TVector<ui64>&& storage)
        : Size(size)
        , IndexHelper(bitsPerKey)
        , Storage(NCB::TMaybeOwningArrayHolder<ui64>::CreateOwning(std::move(storage)))
    {}

    // init later using GetRawArray or GetRawPtr
    static TCompressedArray CreateWithUninitializedData(ui64 size, ui32 bitsPerKey) {
        TIndexHelper<ui64> indexHelper(bitsPerKey);
        TVector<ui64> storage;
        storage.yresize(indexHelper.CompressedSize(size));
        return TCompressedArray(size, bitsPerKey, std::move(storage));
    }

    SAVELOAD(Size, IndexHelper, Storage);

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

    // comparison is strict by default, useful for unit tests
    bool operator==(const TCompressedArray& rhs) const {
        return EqualTo(rhs, /*strict*/ true);
    }

    template <class T>
    bool operator==(TConstArrayRef<T> rhs) const {
        static_assert(std::is_integral<T>::value);

        if (Size != rhs.size()) {
            return false;
        }

        for (auto i : xrange(Size)) {
            if ((*this)[i] != rhs[i]) {
                return false;
            }
        }
        return true;
    }

    // if strict is true compare bit-by-bit, else compare values even with different BitsPerKey
    bool EqualTo(const TCompressedArray& rhs, bool strict = true) const {
        if (Size != rhs.Size) {
            return false;
        }

        if (strict) {
            if ((GetBitsPerKey() != rhs.GetBitsPerKey()) || !(*Storage == *rhs.Storage)) {
                return false;
            }
        } else {
            for (auto i : xrange(Size)) {
                if ((*this)[i] != rhs[i]) {
                    return false;
                }
            }
        }
        return true;
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
            !(alignof(ui64) % alignof(T)),
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
    TArrayRef<T> GetRawArray() {
        CheckIfCanBeInterpretedAsRawArray<T>();
        return TArrayRef<T>(reinterpret_cast<T*>((*Storage).data()), Size);
    }

    template <class T>
    TConstArrayRef<T> GetRawArray() const {
        CheckIfCanBeInterpretedAsRawArray<T>();
        return TConstArrayRef<T>(reinterpret_cast<const T*>((*Storage).data()), Size);
    }

    char* GetRawPtr() {
        return reinterpret_cast<char*>((*Storage).data());
    }

    const char* GetRawPtr() const {
        return reinterpret_cast<const char*>((*Storage).data());
    }

    template<class T>
    NCB::IDynamicBlockWithExactIteratorPtr<T> GetTypedBlockIterator(ui64 offset) const;

    inline NCB::IDynamicBlockIteratorBasePtr GetBlockIterator(ui64 offset, ui64 count) const;

    inline NCB::IDynamicBlockIteratorBasePtr GetBlockIterator(ui64 offset, const NCB::TArraySubsetIndexing<ui32>* subsetIndexing) const;

    // calls generic f with 'const T' pointer to raw data of compressedArray with the appropriate T
    template <class F>
    inline void DispatchBitsPerKeyToDataType(
        const TStringBuf errorMessagePrefix,
        F&& f
    ) const {
        const auto bitsPerKey = GetBitsPerKey();
        const char* rawDataPtr = GetRawPtr();
        switch (bitsPerKey) {
            case 8:
                f((const ui8*)rawDataPtr);
                break;
            case 16:
                f((const ui16*)rawDataPtr);
                break;
            case 32:
                f((const ui32*)rawDataPtr);
                break;
            default:
                CB_ENSURE_INTERNAL(
                    false,
                    errorMessagePrefix << "unsupported bitsPerKey: " << bitsPerKey);
        }
    }

private:
    ui64 Size = 0;
    TIndexHelper<ui64> IndexHelper;
    NCB::TMaybeOwningArrayHolder<ui64> Storage;
};


template <class T>
class TGenericCompressedArrayBlockIterator final : public NCB::IDynamicBlockWithExactIterator<T> {
public:
    TGenericCompressedArrayBlockIterator(TCompressedArray compressedArray, ui64 offset = 0, ui64 count = 0)
        : CompressedArray(std::move(compressedArray))
        , Index(offset)
        , Count(count == 0 ? CompressedArray.GetSize() - Index : count)
    {
    }

    TConstArrayRef<T> Next(size_t size = Max<size_t>()) override {
        return NextExact(Min((ui64)size, Count));
    }

    TConstArrayRef<T> NextExact(size_t exactBlockSize) override {
        UncompressedBuffer.yresize(exactBlockSize);
        const ui64 blockEnd = Index + exactBlockSize;
        for (auto i : xrange(Index, blockEnd)) {
            UncompressedBuffer[i - Index] = CompressedArray.operator[]<T>(i);
        }
        Index = blockEnd;
        return UncompressedBuffer;
    }

private:
    TCompressedArray CompressedArray;
    ui64 Index;
    ui64 Count;
    TVector<T> UncompressedBuffer;
};

// TODO(kirillovs): This looks messy, maybe we don't want such ugly optimisations?
template <class T>
NCB::IDynamicBlockWithExactIteratorPtr<T> TCompressedArray::GetTypedBlockIterator(ui64 offset) const {
    static_assert(std::is_same_v<T, ui8> || std::is_same_v<T, ui16> || std::is_same_v<T, ui32>);

    CB_ENSURE(
        GetBitsPerKey() <= sizeof(T) * CHAR_BIT,
        "Compressed array can contain values outside of specified type range"
    );

    if (GetBitsPerKey() == sizeof(T) * CHAR_BIT) {
        return MakeHolder<NCB::TArrayBlockIterator<T>>(GetRawArray<T>().subspan(offset));
    }
    if (GetBitsPerKey() == 8) {
        return MakeHolder<NCB::TTypeCastingArrayBlockIterator<T, ui8>>(GetRawArray<ui8>().subspan(offset));
    }
    if (GetBitsPerKey() == 16) {
        return MakeHolder<NCB::TTypeCastingArrayBlockIterator<T, ui16>>(GetRawArray<ui16>().subspan(offset));
    }
    return MakeHolder<TGenericCompressedArrayBlockIterator<T>>(*this, offset);
}

inline NCB::IDynamicBlockIteratorBasePtr TCompressedArray::GetBlockIterator(ui64 offset, ui64 count) const {
    if (GetBitsPerKey() == 8) {
        return MakeHolder<NCB::TArrayBlockIterator<ui8>>(GetRawArray<ui8>().subspan(offset, count));
    }
    if (GetBitsPerKey() == 16) {
        return MakeHolder<NCB::TArrayBlockIterator<ui16>>(GetRawArray<ui16>().subspan(offset, count));
    }
    if (GetBitsPerKey() == 32) {
        return MakeHolder<NCB::TArrayBlockIterator<ui32>>(GetRawArray<ui32>().subspan(offset, count));
    }
    if (GetBitsPerKey() < 8) {
        return MakeHolder<TGenericCompressedArrayBlockIterator<ui8>>(*this, offset, count);
    }
    if (GetBitsPerKey() < 16) {
        return MakeHolder<TGenericCompressedArrayBlockIterator<ui16>>(*this, offset, count);
    }
    return MakeHolder<TGenericCompressedArrayBlockIterator<ui32>>(*this, offset, count);
}

inline NCB::IDynamicBlockIteratorBasePtr TCompressedArray::GetBlockIterator(ui64 offset, const NCB::TArraySubsetIndexing<ui32>* subsetIndexing) const {
    using namespace NCB;
    const ui32 size = subsetIndexing->Size();
    const ui32 remainingSize = size - offset;

    auto consecutiveSubsetBegin = subsetIndexing->GetConsecutiveSubsetBegin();
    if (consecutiveSubsetBegin.Defined()) {
        return GetBlockIterator(*consecutiveSubsetBegin + offset, remainingSize);
    }
    auto getIter = [&] (auto dataRef) -> NCB::IDynamicBlockIteratorBasePtr {
        using TInterfaceValue = std::remove_cvref_t<decltype(dataRef[0])>;
        return MakeArraySubsetBlockIterator<TInterfaceValue>(
            subsetIndexing,
            dataRef,
            offset
        );
    };
    if (GetBitsPerKey() == 8) {
        return getIter(GetRawArray<ui8>());
    }
    if (GetBitsPerKey() == 16) {
        return getIter(GetRawArray<ui16>());
    }
    if (GetBitsPerKey() == 32) {
        return getIter(GetRawArray<ui32>());
    }
    if (GetBitsPerKey() < 8) {
        return MakeArraySubsetBlockIterator<ui8>(
            subsetIndexing,
            *this,
            offset
        );
    }
    if (GetBitsPerKey() < 16) {
        return MakeArraySubsetBlockIterator<ui16>(
            subsetIndexing,
            *this,
            offset
        );
    }
    return MakeArraySubsetBlockIterator<ui32>(
        subsetIndexing,
        *this,
        offset
    );
}

template <class TStorageType, class T>
inline TVector<TStorageType> CompressVector(const T* data, ui32 size, ui32 bitsPerKey) {
    CB_ENSURE(bitsPerKey <= 32);
    CB_ENSURE(bitsPerKey, "Error: data with zero bits per key. Something went wrong");

    TVector<TStorageType> dst;
    TIndexHelper<TStorageType> indexHelper(bitsPerKey);
    dst.resize(indexHelper.CompressedSize(size));
    const auto mask = indexHelper.Mask();

    NPar::ILocalExecutor::TExecRangeParams params(0, size);
    //alignment by entries per int allows parallel compression
    params.SetBlockSize(indexHelper.GetEntriesPerType() * 8192);

    NPar::LocalExecutor().ExecRange(
        [&](int blockIdx) {
            NPar::ILocalExecutor::BlockedLoopBody(
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
        NPar::ILocalExecutor::WAIT_COMPLETE
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

