#pragma once

#include "fwd.h"
#include "ptr.h"
#include "bitops.h"
#include "typetraits.h"
#include "algorithm.h"
#include "utility.h"

#include <util/system/yassert.h>
#include <util/system/defaults.h>
#include <util/str_stl.h>
#include <util/ysaveload.h>

namespace NBitMapPrivate {
    // Returns number of bits set; result is in most significatnt byte
    inline ui64 ByteSums(ui64 x) {
        ui64 byteSums = x - ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1);

        byteSums = (byteSums & 0x3333333333333333ULL) + ((byteSums >> 2) & 0x3333333333333333ULL);
        byteSums = (byteSums + (byteSums >> 4)) & 0x0F0F0F0F0F0F0F0FULL;

        return byteSums * 0x0101010101010101ULL;
    }

    // better than intrinsics without -mpopcnt
    template <typename T>
    static unsigned CountBitsPrivate(T v) noexcept {
        return static_cast<unsigned>(ByteSums(v) >> 56);
    }

    template <typename TChunkType, size_t ExtraBits>
    struct TSanitizeMask {
        static constexpr TChunkType Value = ~((~TChunkType(0)) << ExtraBits);
    };

    template <typename TChunkType>
    struct TSanitizeMask<TChunkType, 0> {
        static constexpr TChunkType Value = (TChunkType)~TChunkType(0u);
    };

    template <typename TTargetChunk, typename TSourceChunk>
    struct TBigToSmallDataCopier {
        static_assert(sizeof(TTargetChunk) < sizeof(TSourceChunk), "expect sizeof(TTargetChunk) < sizeof(TSourceChunk)");
        static_assert(0 == sizeof(TSourceChunk) % sizeof(TTargetChunk), "expect 0 == sizeof(TSourceChunk) % sizeof(TTargetChunk)");

        static constexpr size_t BLOCK_SIZE = sizeof(TSourceChunk) / sizeof(TTargetChunk);

        union TCnv {
            TSourceChunk BigData;
            TTargetChunk SmallData[BLOCK_SIZE];
        };

        static inline void CopyChunk(TTargetChunk* target, TSourceChunk source) {
            TCnv c;
            c.BigData = source;
#if defined(_big_endian_)
            ::ReverseCopy(c.SmallData, c.SmallData + Y_ARRAY_SIZE(c.SmallData), target);
#else
            ::Copy(c.SmallData, c.SmallData + Y_ARRAY_SIZE(c.SmallData), target);
#endif
        }

        static inline void Copy(TTargetChunk* target, size_t targetSize, const TSourceChunk* source, size_t sourceSize) {
            Y_ASSERT(targetSize >= sourceSize * BLOCK_SIZE);
            if (targetSize > sourceSize * BLOCK_SIZE) {
                ::Fill(target + sourceSize * BLOCK_SIZE, target + targetSize, 0);
            }
            for (size_t i = 0; i < sourceSize; ++i) {
                CopyChunk(target + i * BLOCK_SIZE, source[i]);
            }
        }
    };

    template <typename TTargetChunk, typename TSourceChunk>
    struct TSmallToBigDataCopier {
        static_assert(sizeof(TTargetChunk) > sizeof(TSourceChunk), "expect sizeof(TTargetChunk) > sizeof(TSourceChunk)");
        static_assert(0 == sizeof(TTargetChunk) % sizeof(TSourceChunk), "expect 0 == sizeof(TTargetChunk) % sizeof(TSourceChunk)");

        static constexpr size_t BLOCK_SIZE = sizeof(TTargetChunk) / sizeof(TSourceChunk);

        union TCnv {
            TSourceChunk SmallData[BLOCK_SIZE];
            TTargetChunk BigData;
        };

        static inline TTargetChunk CopyFullChunk(const TSourceChunk* source) {
            TCnv c;
#if defined(_big_endian_)
            ::ReverseCopy(source, source + BLOCK_SIZE, c.SmallData);
#else
            ::Copy(source, source + BLOCK_SIZE, c.SmallData);
#endif
            return c.BigData;
        }

        static inline TTargetChunk CopyPartChunk(const TSourceChunk* source, size_t count) {
            Y_ASSERT(count <= BLOCK_SIZE);
            TCnv c;
            c.BigData = 0;
#if defined(_big_endian_)
            ::ReverseCopy(source, source + count, c.SmallData);
#else
            ::Copy(source, source + count, c.SmallData);
#endif
            return c.BigData;
        }

        static inline void Copy(TTargetChunk* target, size_t targetSize, const TSourceChunk* source, size_t sourceSize) {
            Y_ASSERT(targetSize * BLOCK_SIZE >= sourceSize);
            if (targetSize * BLOCK_SIZE > sourceSize) {
                ::Fill(target + sourceSize / BLOCK_SIZE, target + targetSize, 0);
            }
            size_t i = 0;
            for (; i < sourceSize / BLOCK_SIZE; ++i) {
                target[i] = CopyFullChunk(source + i * BLOCK_SIZE);
            }
            if (0 != sourceSize % BLOCK_SIZE) {
                target[i] = CopyPartChunk(source + i * BLOCK_SIZE, sourceSize % BLOCK_SIZE);
            }
        }
    };

    template <typename TChunk>
    struct TUniformDataCopier {
        static inline void Copy(TChunk* target, size_t targetSize, const TChunk* source, size_t sourceSize) {
            Y_ASSERT(targetSize >= sourceSize);
            for (size_t i = 0; i < sourceSize; ++i) {
                target[i] = source[i];
            }
            for (size_t i = sourceSize; i < targetSize; ++i) {
                target[i] = 0;
            }
        }
    };

    template <typename TFirst, typename TSecond>
    struct TIsSmaller {
        enum {
            Result = sizeof(TFirst) < sizeof(TSecond)
        };
    };

    template <typename TTargetChunk, typename TSourceChunk>
    struct TDataCopier: public std::conditional_t<std::is_same<TTargetChunk, TSourceChunk>::value, TUniformDataCopier<TTargetChunk>, std::conditional_t<TIsSmaller<TTargetChunk, TSourceChunk>::Result, TBigToSmallDataCopier<TTargetChunk, TSourceChunk>, TSmallToBigDataCopier<TTargetChunk, TSourceChunk>>> {
    };

    template <typename TTargetChunk, typename TSourceChunk>
    inline void CopyData(TTargetChunk* target, size_t targetSize, const TSourceChunk* source, size_t sourceSize) {
        TDataCopier<TTargetChunk, TSourceChunk>::Copy(target, targetSize, source, sourceSize);
    }

    template <size_t BitCount, typename TChunkType>
    struct TFixedStorage {
        using TChunk = TChunkType;

        static constexpr size_t Size = (BitCount + 8 * sizeof(TChunk) - 1) / (8 * sizeof(TChunk));

        TChunk Data[Size];

        TFixedStorage() {
            Zero(Data);
        }

        TFixedStorage(const TFixedStorage<BitCount, TChunkType>& st) {
            for (size_t i = 0; i < Size; ++i) {
                Data[i] = st.Data[i];
            }
        }

        template <typename TOtherChunk>
        TFixedStorage(const TOtherChunk* data, size_t size) {
            Y_ABORT_UNLESS(Size * sizeof(TChunk) >= size * sizeof(TOtherChunk), "Exceeding bitmap storage capacity");
            CopyData(Data, Size, data, size);
        }

        Y_FORCE_INLINE void Swap(TFixedStorage<BitCount, TChunkType>& st) {
            for (size_t i = 0; i < Size; ++i) {
                DoSwap(Data[i], st.Data[i]);
            }
        }

        Y_FORCE_INLINE static constexpr size_t GetBitCapacity() noexcept {
            return BitCount;
        }

        Y_FORCE_INLINE static constexpr size_t GetChunkCapacity() noexcept {
            return Size;
        }

        // Returns true if the resulting storage capacity is enough to fit the requested size
        Y_FORCE_INLINE static constexpr bool ExpandBitSize(const size_t bitSize) noexcept {
            return bitSize <= BitCount;
        }

        Y_FORCE_INLINE void Sanitize() {
            Data[Size - 1] &= TSanitizeMask<TChunk, BitCount % (8 * sizeof(TChunk))>::Value;
        }
    };

    // Dynamically expanded storage.
    // It uses "on stack" realization with no allocation for one chunk spaces
    template <typename TChunkType>
    struct TDynamicStorage {
        using TChunk = TChunkType;

        size_t Size;
        TChunk StackData;
        TArrayHolder<TChunk> ArrayData;
        TChunk* Data;

        TDynamicStorage()
            : Size(1)
            , StackData(0)
            , Data(&StackData)
        {
        }

        TDynamicStorage(const TDynamicStorage<TChunk>& st)
            : Size(1)
            , StackData(0)
            , Data(&StackData)
        {
            ExpandSize(st.Size, false);
            for (size_t i = 0; i < st.Size; ++i) {
                Data[i] = st.Data[i];
            }
            for (size_t i = st.Size; i < Size; ++i) {
                Data[i] = 0;
            }
        }

        template <typename TOtherChunk>
        TDynamicStorage(const TOtherChunk* data, size_t size)
            : Size(1)
            , StackData(0)
            , Data(&StackData)
        {
            ExpandBitSize(size * sizeof(TOtherChunk) * 8, false);
            CopyData(Data, Size, data, size);
        }

        Y_FORCE_INLINE void Swap(TDynamicStorage<TChunkType>& st) {
            DoSwap(Size, st.Size);
            DoSwap(StackData, st.StackData);
            DoSwap(ArrayData, st.ArrayData);
            Data = 1 == Size ? &StackData : ArrayData.Get();
            st.Data = 1 == st.Size ? &st.StackData : st.ArrayData.Get();
        }

        Y_FORCE_INLINE size_t GetBitCapacity() const {
            return Size * 8 * sizeof(TChunk);
        }

        Y_FORCE_INLINE size_t GetChunkCapacity() const {
            return Size;
        }

        // Returns true if the resulting storage capacity is enough to fit the requested size
        Y_FORCE_INLINE bool ExpandSize(size_t size, bool keepData = true) {
            if (size > Size) {
                size = Max(size, Size * 2);
                TArrayHolder<TChunk> newData(new TChunk[size]);
                if (keepData) {
                    for (size_t i = 0; i < Size; ++i) {
                        newData[i] = Data[i];
                    }
                    for (size_t i = Size; i < size; ++i) {
                        newData[i] = 0;
                    }
                }
                DoSwap(ArrayData, newData);
                Data = ArrayData.Get();
                Size = size;
            }
            return true;
        }

        Y_FORCE_INLINE bool ExpandBitSize(size_t bitSize, bool keepData = true) {
            return ExpandSize((bitSize + 8 * sizeof(TChunk) - 1) / (8 * sizeof(TChunk)), keepData);
        }

        Y_FORCE_INLINE void Sanitize() {
        }
    };

    template <size_t num>
    struct TDivCount {
        static constexpr size_t Value = 1 + TDivCount<(num >> 1)>::Value;
    };

    template <>
    struct TDivCount<0> {
        static constexpr size_t Value = 0;
    };

} // namespace NBitMapPrivate

template <size_t BitCount, typename TChunkType>
struct TFixedBitMapTraits {
    using TChunk = TChunkType;
    using TStorage = NBitMapPrivate::TFixedStorage<BitCount, TChunkType>;
};

template <typename TChunkType>
struct TDynamicBitMapTraits {
    using TChunk = TChunkType;
    using TStorage = NBitMapPrivate::TDynamicStorage<TChunkType>;
};

template <class TTraits>
class TBitMapOps {
public:
    using TChunk = typename TTraits::TChunk;
    using TThis = TBitMapOps<TTraits>;

private:
    static_assert(std::is_unsigned<TChunk>::value, "expect std::is_unsigned<TChunk>::value");

    static constexpr size_t BitsPerChunk = 8 * sizeof(TChunk);
    static constexpr TChunk ModMask = static_cast<TChunk>(BitsPerChunk - 1);
    static constexpr size_t DivCount = NBitMapPrivate::TDivCount<BitsPerChunk>::Value - 1;
    static constexpr TChunk FullChunk = (TChunk)~TChunk(0);

    template <class>
    friend class TBitMapOps;

    using TStorage = typename TTraits::TStorage;

    // The smallest unsigned type, which can be used in bit ops
    using TIntType = std::conditional_t<sizeof(TChunk) < sizeof(unsigned int), unsigned int, TChunk>;

    TStorage Mask;

public:
    class TReference {
    private:
        friend class TBitMapOps<TTraits>;

        TChunk* Chunk;
        size_t Offset;

        TReference(TChunk* c, size_t offset)
            : Chunk(c)
            , Offset(offset)
        {
        }

    public:
        ~TReference() = default;

        Y_FORCE_INLINE TReference& operator=(bool val) {
            if (val) {
                *Chunk |= static_cast<TChunk>(1) << Offset;
            } else {
                *Chunk &= ~(static_cast<TChunk>(1) << Offset);
            }

            return *this;
        }

        Y_FORCE_INLINE TReference& operator=(const TReference& ref) {
            if (ref) {
                *Chunk |= static_cast<TChunk>(1) << Offset;
            } else {
                *Chunk &= ~(static_cast<TChunk>(1) << Offset);
            }

            return *this;
        }

        Y_FORCE_INLINE bool operator~() const {
            return 0 == (*Chunk & (static_cast<TChunk>(1) << Offset));
        }

        Y_FORCE_INLINE operator bool() const {
            return 0 != (*Chunk & (static_cast<TChunk>(1) << Offset));
        }

        Y_FORCE_INLINE TReference& Flip() {
            *Chunk ^= static_cast<TChunk>(1) << Offset;
            return *this;
        }
    };

private:
    struct TSetOp {
        static constexpr TChunk Op(const TChunk src, const TChunk mask) noexcept {
            return src | mask;
        }
    };

    struct TResetOp {
        static constexpr TChunk Op(const TChunk src, const TChunk mask) noexcept {
            return src & ~mask;
        }
    };

    template <class TUpdateOp>
    void UpdateRange(size_t start, size_t end) {
        const size_t startChunk = start >> DivCount;
        const size_t startBitOffset = start & ModMask;

        const size_t endChunk = end >> DivCount;
        const size_t endBitOffset = end & ModMask;

        size_t bitOffset = startBitOffset;
        for (size_t chunk = startChunk; chunk <= endChunk; ++chunk) {
            TChunk updateMask = FullChunk << bitOffset;
            if (chunk == endChunk) {
                updateMask ^= FullChunk << endBitOffset;
                if (!updateMask) {
                    break;
                }
            }
            Mask.Data[chunk] = TUpdateOp::Op(Mask.Data[chunk], updateMask);
            bitOffset = 0;
        }
    }

public:
    TBitMapOps() = default;

    TBitMapOps(TChunk val) {
        Mask.Data[0] = val;
        Mask.Sanitize();
    }

    TBitMapOps(const TThis&) = default;

    template <class T>
    TBitMapOps(const TBitMapOps<T>& bitmap)
        : Mask(bitmap.Mask.Data, bitmap.Mask.GetChunkCapacity())
    {
        Mask.Sanitize();
    }

    template <class T>
    Y_FORCE_INLINE bool operator==(const TBitMapOps<T>& bitmap) const {
        return Equal(bitmap);
    }

    Y_FORCE_INLINE TThis& operator=(const TThis& bitmap) {
        if (this != &bitmap) {
            TThis bm(bitmap);
            Swap(bm);
        }
        return *this;
    }

    template <class T>
    Y_FORCE_INLINE TThis& operator=(const TBitMapOps<T>& bitmap) {
        TThis bm(bitmap);
        Swap(bm);
        return *this;
    }

    template <class T>
    Y_FORCE_INLINE TThis& operator&=(const TBitMapOps<T>& bitmap) {
        return And(bitmap);
    }

    Y_FORCE_INLINE TThis& operator&=(const TChunk& val) {
        return And(val);
    }

    template <class T>
    Y_FORCE_INLINE TThis& operator|=(const TBitMapOps<T>& bitmap) {
        return Or(bitmap);
    }

    Y_FORCE_INLINE TThis& operator|=(const TChunk& val) {
        return Or(val);
    }

    template <class T>
    Y_FORCE_INLINE TThis& operator^=(const TBitMapOps<T>& bitmap) {
        return Xor(bitmap);
    }

    Y_FORCE_INLINE TThis& operator^=(const TChunk& val) {
        return Xor(val);
    }

    template <class T>
    Y_FORCE_INLINE TThis& operator-=(const TBitMapOps<T>& bitmap) {
        return SetDifference(bitmap);
    }

    Y_FORCE_INLINE TThis& operator-=(const TChunk& val) {
        return SetDifference(val);
    }

    Y_FORCE_INLINE TThis& operator<<=(size_t pos) {
        return LShift(pos);
    }

    Y_FORCE_INLINE TThis& operator>>=(size_t pos) {
        return RShift(pos);
    }

    Y_FORCE_INLINE TThis operator<<(size_t pos) const {
        return TThis(*this).LShift(pos);
    }

    Y_FORCE_INLINE TThis operator>>(size_t pos) const {
        return TThis(*this).RShift(pos);
    }

    Y_FORCE_INLINE bool operator[](size_t pos) const {
        return Get(pos);
    }

    Y_FORCE_INLINE TReference operator[](size_t pos) {
        const bool fitStorage = Mask.ExpandBitSize(pos + 1);
        Y_ASSERT(fitStorage);
        return TReference(&Mask.Data[pos >> DivCount], ModMask & pos);
    }

    Y_FORCE_INLINE void Swap(TThis& bitmap) {
        DoSwap(Mask, bitmap.Mask);
    }

    Y_FORCE_INLINE TThis& Set(size_t pos) {
        const bool fitStorage = Mask.ExpandBitSize(pos + 1);
        Y_ASSERT(fitStorage);
        Mask.Data[pos >> DivCount] |= static_cast<TChunk>(1) << (pos & ModMask);
        return *this;
    }

    // Fills the specified [start, end) bit range by the 1. Other bits are kept unchanged
    TThis& Set(size_t start, size_t end) {
        Y_ASSERT(start <= end);
        if (start < end) {
            Reserve(end);
            UpdateRange<TSetOp>(start, end);
        }
        return *this;
    }

    Y_FORCE_INLINE TThis& Reset(size_t pos) {
        if ((pos >> DivCount) < Mask.GetChunkCapacity()) {
            Mask.Data[pos >> DivCount] &= ~(static_cast<TChunk>(1) << (pos & ModMask));
        }
        return *this;
    }

    // Clears the specified [start, end) bit range. Other bits are kept unchanged
    TThis& Reset(size_t start, size_t end) {
        Y_ASSERT(start <= end);
        if (start < end && (start >> DivCount) < Mask.GetChunkCapacity()) {
            UpdateRange<TResetOp>(start, Min(end, Mask.GetBitCapacity()));
        }
        return *this;
    }

    Y_FORCE_INLINE TThis& Flip(size_t pos) {
        const bool fitStorage = Mask.ExpandBitSize(pos + 1);
        Y_ASSERT(fitStorage);
        Mask.Data[pos >> DivCount] ^= static_cast<TChunk>(1) << (pos & ModMask);
        return *this;
    }

    Y_FORCE_INLINE bool Get(size_t pos) const {
        if ((pos >> DivCount) < Mask.GetChunkCapacity()) {
            return Mask.Data[pos >> DivCount] & (static_cast<TChunk>(1) << (pos & ModMask));
        }
        return false;
    }

    template <class TTo>
    void Export(size_t pos, TTo& to) const {
        static_assert(std::is_unsigned<TTo>::value, "expect std::is_unsigned<TTo>::value");
        to = 0;
        size_t chunkpos = pos >> DivCount;
        if (chunkpos >= Mask.GetChunkCapacity()) {
            return;
        }
        if ((pos & ModMask) == 0) {
            if (sizeof(TChunk) >= sizeof(TTo)) {
                to = (TTo)Mask.Data[chunkpos];
            } else { // if (sizeof(TChunk) < sizeof(TTo))
                NBitMapPrivate::CopyData(&to, 1, Mask.Data + chunkpos, Min(((sizeof(TTo) * 8) >> DivCount), Mask.GetChunkCapacity() - chunkpos));
            }
        } else if ((pos & (sizeof(TTo) * 8 - 1)) == 0 && sizeof(TChunk) >= 2 * sizeof(TTo)) {
            to = (TTo)(Mask.Data[chunkpos] >> (pos & ModMask));
        } else {
            static constexpr size_t copyToSize = (sizeof(TChunk) >= sizeof(TTo)) ? (sizeof(TChunk) / sizeof(TTo)) + 2 : 3;
            TTo temp[copyToSize] = {0, 0};
            // or use non defined by now TBitmap<copyToSize, TTo>::CopyData,RShift(pos & ModMask),Export(0,to)
            NBitMapPrivate::CopyData(temp, copyToSize, Mask.Data + chunkpos, Min((sizeof(TTo) / sizeof(TChunk)) + 1, Mask.GetChunkCapacity() - chunkpos));
            to = (temp[0] >> (pos & ModMask)) | (temp[1] << (8 * sizeof(TTo) - (pos & ModMask)));
        }
    }

    Y_FORCE_INLINE bool Test(size_t n) const {
        return Get(n);
    }

    Y_FORCE_INLINE TThis& Push(bool val) {
        LShift(1);
        return val ? Set(0) : *this;
    }

    Y_FORCE_INLINE bool Pop() {
        bool val = Get(0);
        return RShift(1), val;
    }

    // Clear entire bitmap. Current capacity is kept unchanged
    Y_FORCE_INLINE TThis& Clear() {
        for (size_t i = 0; i < Mask.GetChunkCapacity(); ++i) {
            Mask.Data[i] = 0;
        }
        return *this;
    }

    // Returns bits capacity
    Y_FORCE_INLINE constexpr size_t Size() const noexcept {
        return Mask.GetBitCapacity();
    }

    Y_FORCE_INLINE void Reserve(size_t bitCount) {
        Y_ABORT_UNLESS(Mask.ExpandBitSize(bitCount), "Exceeding bitmap storage capacity");
    }

    Y_FORCE_INLINE size_t ValueBitCount() const {
        size_t nonZeroChunk = Mask.GetChunkCapacity() - 1;
        while (nonZeroChunk != 0 && !Mask.Data[nonZeroChunk]) {
            --nonZeroChunk;
        }
        return nonZeroChunk || Mask.Data[nonZeroChunk]
                   ? nonZeroChunk * BitsPerChunk + GetValueBitCount(TIntType(Mask.Data[nonZeroChunk]))
                   : 0;
    }

    Y_PURE_FUNCTION Y_FORCE_INLINE bool Empty() const {
        for (size_t i = 0; i < Mask.GetChunkCapacity(); ++i) {
            if (Mask.Data[i]) {
                return false;
            }
        }
        return true;
    }

    bool HasAny(const TThis& bitmap) const {
        for (size_t i = 0; i < Min(Mask.GetChunkCapacity(), bitmap.Mask.GetChunkCapacity()); ++i) {
            if (0 != (Mask.Data[i] & bitmap.Mask.Data[i])) {
                return true;
            }
        }
        return false;
    }

    template <class T>
    Y_FORCE_INLINE bool HasAny(const TBitMapOps<T>& bitmap) const {
        return HasAny(TThis(bitmap));
    }

    Y_FORCE_INLINE bool HasAny(const TChunk& val) const {
        return 0 != (Mask.Data[0] & val);
    }

    bool HasAll(const TThis& bitmap) const {
        for (size_t i = 0; i < Min(Mask.GetChunkCapacity(), bitmap.Mask.GetChunkCapacity()); ++i) {
            if (bitmap.Mask.Data[i] != (Mask.Data[i] & bitmap.Mask.Data[i])) {
                return false;
            }
        }
        for (size_t i = Mask.GetChunkCapacity(); i < bitmap.Mask.GetChunkCapacity(); ++i) {
            if (bitmap.Mask.Data[i] != 0) {
                return false;
            }
        }
        return true;
    }

    template <class T>
    Y_FORCE_INLINE bool HasAll(const TBitMapOps<T>& bitmap) const {
        return HasAll(TThis(bitmap));
    }

    Y_FORCE_INLINE bool HasAll(const TChunk& val) const {
        return (Mask.Data[0] & val) == val;
    }

    TThis& And(const TThis& bitmap) {
        // Don't expand capacity here, because resulting bits in positions,
        // which are greater then size of one of these bitmaps, will be zero
        for (size_t i = 0; i < Min(bitmap.Mask.GetChunkCapacity(), Mask.GetChunkCapacity()); ++i) {
            Mask.Data[i] &= bitmap.Mask.Data[i];
        }
        // Clear bits if current bitmap size is greater than AND-ed one
        for (size_t i = bitmap.Mask.GetChunkCapacity(); i < Mask.GetChunkCapacity(); ++i) {
            Mask.Data[i] = 0;
        }
        return *this;
    }

    template <class T>
    Y_FORCE_INLINE TThis& And(const TBitMapOps<T>& bitmap) {
        return And(TThis(bitmap));
    }

    Y_FORCE_INLINE TThis& And(const TChunk& val) {
        Mask.Data[0] &= val;
        for (size_t i = 1; i < Mask.GetChunkCapacity(); ++i) {
            Mask.Data[i] = 0;
        }
        return *this;
    }

    TThis& Or(const TThis& bitmap) {
        const size_t valueBitCount = bitmap.ValueBitCount();
        if (valueBitCount) {
            // Memory optimization: expand size only for non-zero bits
            Reserve(valueBitCount);
            for (size_t i = 0; i < Min(bitmap.Mask.GetChunkCapacity(), Mask.GetChunkCapacity()); ++i) {
                Mask.Data[i] |= bitmap.Mask.Data[i];
            }
        }
        return *this;
    }

    template <class T>
    Y_FORCE_INLINE TThis& Or(const TBitMapOps<T>& bitmap) {
        return Or(TThis(bitmap));
    }

    Y_FORCE_INLINE TThis& Or(const TChunk& val) {
        Mask.Data[0] |= val;
        Mask.Sanitize();
        return *this;
    }

    TThis& Xor(const TThis& bitmap) {
        Reserve(bitmap.Size());
        for (size_t i = 0; i < bitmap.Mask.GetChunkCapacity(); ++i) {
            Mask.Data[i] ^= bitmap.Mask.Data[i];
        }
        return *this;
    }

    template <class T>
    Y_FORCE_INLINE TThis& Xor(const TBitMapOps<T>& bitmap) {
        return Xor(TThis(bitmap));
    }

    Y_FORCE_INLINE TThis& Xor(const TChunk& val) {
        Mask.Data[0] ^= val;
        Mask.Sanitize();
        return *this;
    }

    TThis& SetDifference(const TThis& bitmap) {
        for (size_t i = 0; i < Min(bitmap.Mask.GetChunkCapacity(), Mask.GetChunkCapacity()); ++i) {
            Mask.Data[i] &= ~bitmap.Mask.Data[i];
        }
        return *this;
    }

    template <class T>
    Y_FORCE_INLINE TThis& SetDifference(const TBitMapOps<T>& bitmap) {
        return SetDifference(TThis(bitmap));
    }

    Y_FORCE_INLINE TThis& SetDifference(const TChunk& val) {
        Mask.Data[0] &= ~val;
        return *this;
    }

    Y_FORCE_INLINE TThis& Flip() {
        for (size_t i = 0; i < Mask.GetChunkCapacity(); ++i) {
            Mask.Data[i] = ~Mask.Data[i];
        }
        Mask.Sanitize();
        return *this;
    }

    TThis& LShift(size_t shift) {
        if (shift != 0) {
            const size_t valueBitCount = ValueBitCount();
            // Do nothing for empty bitmap
            if (valueBitCount != 0) {
                const size_t eshift = shift / BitsPerChunk;
                const size_t offset = shift % BitsPerChunk;
                const size_t subOffset = BitsPerChunk - offset;

                // Don't verify expand result, so l-shift of fixed bitmap will work in the same way as for unsigned integer.
                Mask.ExpandBitSize(valueBitCount + shift);

                if (offset == 0) {
                    for (size_t i = Mask.GetChunkCapacity() - 1; i >= eshift; --i) {
                        Mask.Data[i] = Mask.Data[i - eshift];
                    }
                } else {
                    for (size_t i = Mask.GetChunkCapacity() - 1; i > eshift; --i) {
                        Mask.Data[i] = (Mask.Data[i - eshift] << offset) | (Mask.Data[i - eshift - 1] >> subOffset);
                    }
                    if (eshift < Mask.GetChunkCapacity()) {
                        Mask.Data[eshift] = Mask.Data[0] << offset;
                    }
                }
                for (size_t i = 0; i < Min(eshift, Mask.GetChunkCapacity()); ++i) {
                    Mask.Data[i] = 0;
                }

                // Cleanup extra high bits in the storage
                Mask.Sanitize();
            }
        }
        return *this;
    }

    TThis& RShift(size_t shift) {
        if (shift != 0) {
            const size_t eshift = shift / BitsPerChunk;
            const size_t offset = shift % BitsPerChunk;
            if (eshift >= Mask.GetChunkCapacity()) {
                Clear();

            } else {
                const size_t limit = Mask.GetChunkCapacity() - eshift - 1;

                if (offset == 0) {
                    for (size_t i = 0; i <= limit; ++i) {
                        Mask.Data[i] = Mask.Data[i + eshift];
                    }
                } else {
                    const size_t subOffset = BitsPerChunk - offset;
                    for (size_t i = 0; i < limit; ++i) {
                        Mask.Data[i] = (Mask.Data[i + eshift] >> offset) | (Mask.Data[i + eshift + 1] << subOffset);
                    }
                    Mask.Data[limit] = Mask.Data[Mask.GetChunkCapacity() - 1] >> offset;
                }

                for (size_t i = limit + 1; i < Mask.GetChunkCapacity(); ++i) {
                    Mask.Data[i] = 0;
                }
            }
        }
        return *this;
    }

    // Applies bitmap at the specified offset using OR operator.
    // This method is optimized combination of Or() and LShift(), which allows reducing memory allocation
    // when combining long dynamic bitmaps.
    TThis& Or(const TThis& bitmap, size_t offset) {
        if (0 == offset) {
            return Or(bitmap);
        }

        const size_t otherValueBitCount = bitmap.ValueBitCount();
        // Continue only if OR-ed bitmap have non-zero bits
        if (otherValueBitCount) {
            const size_t chunkShift = offset / BitsPerChunk;
            const size_t subShift = offset % BitsPerChunk;
            const size_t subOffset = BitsPerChunk - subShift;

            Reserve(otherValueBitCount + offset);

            if (subShift == 0) {
                for (size_t i = chunkShift; i < Min(bitmap.Mask.GetChunkCapacity() + chunkShift, Mask.GetChunkCapacity()); ++i) {
                    Mask.Data[i] |= bitmap.Mask.Data[i - chunkShift];
                }
            } else {
                Mask.Data[chunkShift] |= bitmap.Mask.Data[0] << subShift;
                size_t i = chunkShift + 1;
                for (; i < Min(bitmap.Mask.GetChunkCapacity() + chunkShift, Mask.GetChunkCapacity()); ++i) {
                    Mask.Data[i] |= (bitmap.Mask.Data[i - chunkShift] << subShift) | (bitmap.Mask.Data[i - chunkShift - 1] >> subOffset);
                }
                if (i < Mask.GetChunkCapacity()) {
                    Mask.Data[i] |= bitmap.Mask.Data[i - chunkShift - 1] >> subOffset;
                }
            }
        }

        return *this;
    }

    bool Equal(const TThis& bitmap) const {
        if (Mask.GetChunkCapacity() > bitmap.Mask.GetChunkCapacity()) {
            for (size_t i = bitmap.Mask.GetChunkCapacity(); i < Mask.GetChunkCapacity(); ++i) {
                if (0 != Mask.Data[i]) {
                    return false;
                }
            }
        } else if (Mask.GetChunkCapacity() < bitmap.Mask.GetChunkCapacity()) {
            for (size_t i = Mask.GetChunkCapacity(); i < bitmap.Mask.GetChunkCapacity(); ++i) {
                if (0 != bitmap.Mask.Data[i]) {
                    return false;
                }
            }
        }
        size_t size = Min(Mask.GetChunkCapacity(), bitmap.Mask.GetChunkCapacity());
        for (size_t i = 0; i < size; ++i) {
            if (Mask.Data[i] != bitmap.Mask.Data[i]) {
                return false;
            }
        }
        return true;
    }

    template <class T>
    Y_FORCE_INLINE bool Equal(const TBitMapOps<T>& bitmap) const {
        return Equal(TThis(bitmap));
    }

    int Compare(const TThis& bitmap) const {
        size_t size = Min(Mask.GetChunkCapacity(), bitmap.Mask.GetChunkCapacity());
        int res = ::memcmp(Mask.Data, bitmap.Mask.Data, size * sizeof(TChunk));
        if (0 != res || Mask.GetChunkCapacity() == bitmap.Mask.GetChunkCapacity()) {
            return res;
        }

        if (Mask.GetChunkCapacity() > bitmap.Mask.GetChunkCapacity()) {
            for (size_t i = bitmap.Mask.GetChunkCapacity(); i < Mask.GetChunkCapacity(); ++i) {
                if (0 != Mask.Data[i]) {
                    return 1;
                }
            }
        } else {
            for (size_t i = Mask.GetChunkCapacity(); i < bitmap.Mask.GetChunkCapacity(); ++i) {
                if (0 != bitmap.Mask.Data[i]) {
                    return -1;
                }
            }
        }
        return 0;
    }

    template <class T>
    Y_FORCE_INLINE int Compare(const TBitMapOps<T>& bitmap) const {
        return Compare(TThis(bitmap));
    }

    // For backward compatibility
    Y_FORCE_INLINE static int Compare(const TThis& l, const TThis& r) {
        return l.Compare(r);
    }

    size_t FirstNonZeroBit() const {
        for (size_t i = 0; i < Mask.GetChunkCapacity(); ++i) {
            if (Mask.Data[i]) {
                // CountTrailingZeroBits() expects unsigned types not smaller than unsigned int. So, convert before calling
                return BitsPerChunk * i + CountTrailingZeroBits(TIntType(Mask.Data[i]));
            }
        }
        return Size();
    }

    // Returns position of the next non-zero bit, which offset is greater than specified pos
    // Typical loop for iterating bits:
    // for (size_t pos = bits.FirstNonZeroBit(); pos != bits.Size(); pos = bits.NextNonZeroBit(pos)) {
    //     ...
    // }
    // See Y_FOR_EACH_BIT macro definition at the bottom
    size_t NextNonZeroBit(size_t pos) const {
        size_t i = (pos + 1) >> DivCount;
        if (i < Mask.GetChunkCapacity()) {
            const size_t offset = (pos + 1) & ModMask;
            // Process the current chunk
            if (offset) {
                // Zero already iterated trailing bits using mask
                const TChunk val = Mask.Data[i] & ((~TChunk(0)) << offset);
                if (val) {
                    return BitsPerChunk * i + CountTrailingZeroBits(TIntType(val));
                }
                // Continue with other chunks
                ++i;
            }

            for (; i < Mask.GetChunkCapacity(); ++i) {
                if (Mask.Data[i]) {
                    return BitsPerChunk * i + CountTrailingZeroBits(TIntType(Mask.Data[i]));
                }
            }
        }
        return Size();
    }

    Y_FORCE_INLINE size_t Count() const {
        size_t count = 0;
        for (size_t i = 0; i < Mask.GetChunkCapacity(); ++i) {
            count += ::NBitMapPrivate::CountBitsPrivate(Mask.Data[i]);
        }
        return count;
    }

    void Save(IOutputStream* out) const {
        ::Save(out, ui8(sizeof(TChunk)));
        ::Save(out, ui64(Size()));
        ::SavePodArray(out, Mask.Data, Mask.GetChunkCapacity());
    }

    void Load(IInputStream* inp) {
        ui8 chunkSize = 0;
        ::Load(inp, chunkSize);
        Y_ABORT_UNLESS(size_t(chunkSize) == sizeof(TChunk), "Chunk size is not the same");

        ui64 bitCount64 = 0;
        ::Load(inp, bitCount64);
        size_t bitCount = size_t(bitCount64);
        Reserve(bitCount);

        size_t chunkCount = 0;
        if (bitCount > 0) {
            chunkCount = ((bitCount - 1) >> DivCount) + 1;
            ::LoadPodArray(inp, Mask.Data, chunkCount);
        }

        if (chunkCount < Mask.GetChunkCapacity()) {
            ::memset(Mask.Data + chunkCount, 0, (Mask.GetChunkCapacity() - chunkCount) * sizeof(TChunk));
        }
        Mask.Sanitize();
    }

    inline size_t Hash() const {
        THash<TChunk> chunkHasher;

        size_t hash = chunkHasher(0);
        bool tailSkipped = false;
        for (size_t i = Mask.GetChunkCapacity(); i > 0; --i) {
            if (tailSkipped || Mask.Data[i - 1]) {
                hash = ::CombineHashes(hash, chunkHasher(Mask.Data[i - 1]));
                tailSkipped = true;
            }
        }

        return hash;
    }

    inline const TChunk* GetChunks() const {
        return Mask.Data;
    }

    constexpr size_t GetChunkCount() const noexcept {
        return Mask.GetChunkCapacity();
    }
};

template <class X, class Y>
inline TBitMapOps<X> operator&(const TBitMapOps<X>& x, const TBitMapOps<Y>& y) {
    return TBitMapOps<X>(x).And(y);
}

template <class X>
inline TBitMapOps<X> operator&(const TBitMapOps<X>& x, const typename TBitMapOps<X>::TChunk& y) {
    return TBitMapOps<X>(x).And(y);
}

template <class X>
inline TBitMapOps<X> operator&(const typename TBitMapOps<X>::TChunk& x, const TBitMapOps<X>& y) {
    return TBitMapOps<X>(x).And(y);
}

template <class X, class Y>
inline TBitMapOps<X> operator|(const TBitMapOps<X>& x, const TBitMapOps<Y>& y) {
    return TBitMapOps<X>(x).Or(y);
}

template <class X>
inline TBitMapOps<X> operator|(const TBitMapOps<X>& x, const typename TBitMapOps<X>::TChunk& y) {
    return TBitMapOps<X>(x).Or(y);
}

template <class X>
inline TBitMapOps<X> operator|(const typename TBitMapOps<X>::TChunk& x, const TBitMapOps<X>& y) {
    return TBitMapOps<X>(x).Or(y);
}

template <class X, class Y>
inline TBitMapOps<X> operator^(const TBitMapOps<X>& x, const TBitMapOps<Y>& y) {
    return TBitMapOps<X>(x).Xor(y);
}

template <class X>
inline TBitMapOps<X> operator^(const TBitMapOps<X>& x, const typename TBitMapOps<X>::TChunk& y) {
    return TBitMapOps<X>(x).Xor(y);
}

template <class X>
inline TBitMapOps<X> operator^(const typename TBitMapOps<X>::TChunk& x, const TBitMapOps<X>& y) {
    return TBitMapOps<X>(x).Xor(y);
}

template <class X, class Y>
inline TBitMapOps<X> operator-(const TBitMapOps<X>& x, const TBitMapOps<Y>& y) {
    return TBitMapOps<X>(x).SetDifference(y);
}

template <class X>
inline TBitMapOps<X> operator-(const TBitMapOps<X>& x, const typename TBitMapOps<X>::TChunk& y) {
    return TBitMapOps<X>(x).SetDifference(y);
}

template <class X>
inline TBitMapOps<X> operator-(const typename TBitMapOps<X>::TChunk& x, const TBitMapOps<X>& y) {
    return TBitMapOps<X>(x).SetDifference(y);
}

template <class X>
inline TBitMapOps<X> operator~(const TBitMapOps<X>& x) {
    return TBitMapOps<X>(x).Flip();
}

/////////////////// Specialization ///////////////////////////

template <size_t BitCount, typename TChunkType /*= ui64*/>
class TBitMap: public TBitMapOps<TFixedBitMapTraits<BitCount, TChunkType>> {
private:
    using TBase = TBitMapOps<TFixedBitMapTraits<BitCount, TChunkType>>;

public:
    TBitMap()
        : TBase()
    {
    }

    TBitMap(typename TBase::TChunk val)
        : TBase(val)
    {
    }

    TBitMap(const TBitMap&) = default;
    TBitMap& operator=(const TBitMap&) = default;

    template <class T>
    TBitMap(const TBitMapOps<T>& bitmap)
        : TBase(bitmap)
    {
    }
};

using TDynBitMap = TBitMapOps<TDynamicBitMapTraits<ui64>>;

#define Y_FOR_EACH_BIT(var, bitmap) for (size_t var = (bitmap).FirstNonZeroBit(); var != (bitmap).Size(); var = (bitmap).NextNonZeroBit(var))

template <typename TTraits>
struct THash<TBitMapOps<TTraits>> {
    size_t operator()(const TBitMapOps<TTraits>& elem) const {
        return elem.Hash();
    }
};
