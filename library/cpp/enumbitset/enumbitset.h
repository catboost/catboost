#pragma once

#include <util/ysaveload.h>
#include <util/generic/bitmap.h>
#include <util/generic/serialized_enum.h>
#include <util/generic/yexception.h>
#include <util/string/cast.h>
#include <util/string/printf.h>
#include <util/system/yassert.h>

#include <type_traits>

// Stack memory bitmask for TEnum values [begin, end).
// @end value is not included in the mask and is not necessarily defined as enum value.
// For example: enum EType { A, B, C } ==> TEnumBitSet<EType, A, C + 1>
// Uses the smallest ui8/ui16/ui32/ui64 chunk that fits end - begin; ranges wider
// than 64 bits use multiple ui64 chunks. Object size and alignment depend on this
// choice; Save/Load and hexadecimal formats do not. Raw object memory is not a
// serialization format.
template <typename TEnum, int mbegin, int mend>
class TEnumBitSet: private TBitMap<mend - mbegin,
    std::conditional_t<(mend - mbegin <= 8), ui8,
        std::conditional_t<(mend - mbegin <= 16), ui16,
            std::conditional_t<(mend - mbegin <= 32), ui32, ui64>>>> {
public:
    static constexpr int BeginIndex = mbegin;
    static constexpr int EndIndex = mend;
    static constexpr size_t BitsetSize = EndIndex - BeginIndex;

    using TParent = typename TEnumBitSet::TBitMap;
    using TChunk = typename TParent::TChunk;
    // Backward compatibility: hexadecimal input is parsed in 64-bit chunks.
    using TLegacyChunk = ui64;
    using TThis = TEnumBitSet<TEnum, mbegin, mend>;

    constexpr TEnumBitSet() noexcept
        : TParent(0)
    {
    }

    constexpr TEnumBitSet(const TThis&) noexcept = default;
    constexpr TEnumBitSet(TThis&&) noexcept = default;
    constexpr TThis& operator=(const TThis&) noexcept = default;
    constexpr TThis& operator=(TThis&&) noexcept = default;

    explicit constexpr TEnumBitSet(const TParent& p) noexcept
        : TParent(p)
    {
    }

    template <typename TSourceChunk>
    explicit constexpr TEnumBitSet(const TBitMap<BitsetSize, TSourceChunk>& p) noexcept {
        if constexpr (sizeof(TSourceChunk) > sizeof(TChunk)) {
            // Backward compatibility with construction from a ui64-based TBitMap.
            TChunk low = 0;
            p.Export(0, low);
            TParent::Or(low);
        } else {
            TParent::operator=(TParent(p));
        }
    }

    // Earlier updates remain applied if converting a later argument throws.
    template <class... R>
    constexpr void Init(R... r) noexcept((noexcept(this->Set(r)) && ...)) {
        (Set(r), ...);
    }

    explicit constexpr TEnumBitSet(TEnum c) noexcept {
        Init(c);
    }

    template <class... R>
    constexpr TEnumBitSet(TEnum c1, TEnum c2, R... r) noexcept(noexcept(Init(c1, c2, r...))) {
        Init(c1, c2, r...);
    }

    template <class TIt>
    constexpr TEnumBitSet(const TIt& begin_, const TIt& end_) {
        for (TIt p = begin_; p != end_; ++p)
            Set(*p);
    }

    static constexpr bool IsValid(TEnum c) noexcept {
        return int(c) >= BeginIndex && int(c) < EndIndex;
    }

    constexpr bool Test(TEnum c) const noexcept {
        return TParent::Test(Pos(c));
    }

    constexpr TThis& Flip(TEnum c) noexcept {
        TParent::Flip(Pos(c));
        return *this;
    }

    constexpr TThis& Flip() noexcept {
        TParent::Flip();
        return *this;
    }

    constexpr TThis& Reset(TEnum c) noexcept {
        TParent::Reset(Pos(c));
        return *this;
    }

    constexpr TThis& Reset() noexcept {
        TParent::Clear();
        return *this;
    }

    constexpr TThis& Set(TEnum c) noexcept {
        TParent::Set(Pos(c));
        return *this;
    }

    constexpr TThis& Set(TEnum c, bool val) noexcept {
        if (val)
            Set(c);
        else
            Reset(c);
        return *this;
    }

    constexpr bool SafeTest(TEnum c) const noexcept {
        if (IsValid(c))
            return Test(c);
        return false;
    }

    constexpr TThis& SafeFlip(TEnum c) noexcept {
        if (IsValid(c))
            return Flip(c);
        return *this;
    }

    constexpr TThis& SafeReset(TEnum c) noexcept {
        if (IsValid(c))
            return Reset(c);
        return *this;
    }

    constexpr TThis& SafeSet(TEnum c) noexcept {
        if (IsValid(c))
            return Set(c);
        return *this;
    }

    constexpr TThis& SafeSet(TEnum c, bool val) noexcept {
        if (IsValid(c))
            return Set(c, val);
        return *this;
    }

    static constexpr TThis SafeConstruct(TEnum c) noexcept {
        TThis ret;
        ret.SafeSet(c);
        return ret;
    }

    constexpr bool operator<(const TThis& right) const noexcept {
        Y_ASSERT(this->GetChunkCount() == right.GetChunkCount());
        for (size_t i = 0; i < this->GetChunkCount(); ++i) {
            if (this->GetChunks()[i] < right.GetChunks()[i])
                return true;
            else if (this->GetChunks()[i] > right.GetChunks()[i])
                return false;
        }
        return false;
    }

    constexpr bool operator!=(const TThis& right) const noexcept {
        return !(TParent::operator==(right));
    }

    constexpr bool operator==(const TThis& right) const noexcept {
        return TParent::operator==(right);
    }

    constexpr TThis& operator&=(const TThis& right) noexcept {
        TParent::operator&=(right);
        return *this;
    }

    constexpr TThis& operator|=(const TThis& right) noexcept {
        TParent::operator|=(right);
        return *this;
    }

    constexpr TThis& operator^=(const TThis& right) noexcept {
        TParent::operator^=(right);
        return *this;
    }

    constexpr TThis operator~() const noexcept {
        TThis r = *this;
        r.Flip();
        return r;
    }

    constexpr TThis operator|(const TThis& right) const noexcept {
        TThis ret = *this;
        ret |= right;
        return ret;
    }

    constexpr TThis operator&(const TThis& right) const noexcept {
        TThis ret = *this;
        ret &= right;
        return ret;
    }

    constexpr TThis operator^(const TThis& right) const noexcept {
        TThis ret = *this;
        ret ^= right;
        return ret;
    }


    constexpr TThis& operator&=(const TEnum c) noexcept {
        return TThis::operator&=(TThis(c));
    }

    constexpr TThis& operator|=(const TEnum c) noexcept {
        return Set(c);
    }

    constexpr TThis& operator^=(const TEnum c) noexcept {
        return Flip(c);
    }

    constexpr TThis operator&(const TEnum c) const noexcept {
        return TThis::operator&(TThis(c));
    }

    constexpr TThis operator|(const TEnum c) const noexcept {
        TThis ret = *this;
        ret.Set(c);
        return ret;
    }

    constexpr TThis operator^(const TEnum c) const noexcept {
        TThis ret = *this;
        ret.Flip(c);
        return ret;
    }

    constexpr auto operator[] (TEnum e) noexcept {
        return TParent::operator[](this->Pos(e));
    }

    constexpr auto operator[] (TEnum e) const noexcept {
        return TParent::operator[](this->Pos(e));
    }

    using TParent::Count;
    using TParent::Empty;

    explicit constexpr operator bool() const noexcept {
        return !Empty();
    }

    constexpr void Swap(TThis& bitmap) noexcept {
        TParent::Swap(bitmap);
    }

    constexpr size_t GetHash() const noexcept {
        return this->Hash();
    }

    constexpr bool HasAny(TEnum c) const noexcept {
        return Test(c);
    }

    constexpr bool HasAny(const TThis& mask) const noexcept {
        return TParent::HasAny(mask);
    }

    // Single-argument calls select an enum/mask overload, also for derived sets.
    template <class... R, std::enable_if_t<sizeof...(R) != 1, int> = 0>
    constexpr bool HasAny(R... r) const noexcept((noexcept(HasAny(r)) && ...)) {
        return (HasAny(r) || ...);
    }

    constexpr bool HasAll(TEnum c) const noexcept {
        return Test(c);
    }

    constexpr bool HasAll(const TThis& mask) const noexcept {
        return TParent::HasAll(mask);
    }

    template <class... R, std::enable_if_t<sizeof...(R) != 1, int> = 0>
    constexpr bool HasAll(R... r) const noexcept((noexcept(HasAll(r)) && ...)) {
        return (HasAll(r) && ...);
    }

    //serialization to/from stream
    void Save(IOutputStream* buffer) const {
        ::Save(buffer, (ui32)Count());
        for (TEnum bit : *this) {
            ::Save(buffer, (ui32)bit);
        }
    }

    void Load(IInputStream* buffer) {
        Reset();

        ui32 sz = 0;
        ::Load(buffer, sz);

        for (ui32 t = 0; t < sz; t++) {
            ui32 bit = 0;
            ::Load(buffer, bit);

            SafeSet((TEnum)bit);
        }
    }

    // Returns the complete integer bit mask only when the set fits in one chunk.
    template <size_t Size = BitsetSize, std::enable_if_t<Size == BitsetSize && Size <= sizeof(TChunk) * 8, int> = 0>
    constexpr TChunk ToMask() const noexcept {
        return this->GetChunks()[0];
    }

    constexpr ui64 Low() const noexcept {
        ui64 t = 0;
        this->Export(0, t);
        return t;
    }

    TString ToString() const {
        static_assert(sizeof(typename TParent::TChunk) <= sizeof(ui64), "expect sizeof(typename TParent::TChunk) <= sizeof(ui64)");
        static constexpr size_t chunkSize = sizeof(TChunk) * 8;
        static constexpr size_t numDig = chunkSize / 4;
        static constexpr size_t numOfChunks = (BitsetSize + chunkSize - 1) / chunkSize;
        TString ret;
        ret.reserve(numOfChunks * numDig);
        for (int pos = (numOfChunks - 1) * chunkSize; pos >= 0; pos -= chunkSize) {
            ui64 t = 0;
            this->Export(pos, t);
            fcat(ret, "%0*lX", static_cast<int>(numDig), t);
        }

        size_t n = 0;
        while (n + 1 < ret.length() && ret[n] == '0')
            ++n;
        ret.remove(0, n);
        return ret;
    }

    void FromString(TStringBuf s) {
        static constexpr size_t chunkSize = sizeof(TLegacyChunk) * 8;
        static constexpr size_t numDig = chunkSize / 4;
        static constexpr size_t highChunkBits = (BitsetSize + chunkSize - 1) % chunkSize + 1;
        static constexpr TLegacyChunk highChunkBitsMask = (highChunkBits == chunkSize ? 0 : (TLegacyChunk(1) << highChunkBits)) - 1;

        Reset();
        for (size_t prev = s.length(), n = s.length() - numDig, pos = 0; prev; n -= numDig, pos += chunkSize) {
            if (pos >= BitsetSize)
                ythrow yexception() << "too many digits";
            if (n > prev)
                n = 0;
            TLegacyChunk t = IntFromString<TLegacyChunk, 16, TStringBuf>(s.substr(n, prev - n));
            if (BitsetSize < pos + chunkSize && t > highChunkBitsMask)
                ythrow yexception() << "digit is too big";
            this->Or(TParent(static_cast<TChunk>(t)), pos);
            prev = n;
        }
    }

    // TODO: Get rid of exceptions at all
    bool TryFromString(TStringBuf s) noexcept {
        try {
            FromString(s);
        } catch (...) {
            Reset();
            return false;
        }
        return true;
    }

    constexpr bool any() const noexcept { // obsolete
        return !Empty();
    }

    constexpr bool none() const noexcept { // obsolete
        return Empty();
    }

    constexpr size_t count() const noexcept { // obsolete
        return Count();
    }

    class TIterator {
    public:
        constexpr TIterator(TEnum value, const TThis* bitmap) noexcept
            : Value(static_cast<int>(value))
            , BitMap(bitmap)
        {
        }

        constexpr TIterator(const TThis* bitmap) noexcept
            : Value(EndIndex)
            , BitMap(bitmap)
        {
        }

        constexpr TEnum operator*() const noexcept {
            Y_ASSERT(Value < EndIndex);
            return static_cast<TEnum>(Value);
        }

        constexpr bool operator!=(const TIterator& other) const noexcept {
            return Value != other.Value;
        }

        constexpr TIterator& operator++() noexcept {
            Y_ASSERT(Value < EndIndex);
            TEnum res{};
            if (BitMap->FindNext(static_cast<TEnum>(Value), res)) {
                Value = static_cast<int>(res);
            } else {
                Value = EndIndex;
            }

            return *this;
        }

    private:
        int Value;
        const TThis* BitMap;
    };

    constexpr TIterator begin() const noexcept {
        TEnum res{};
        return FindFirst(res) ? TIterator(res, this) : TIterator(this);
    }

    constexpr TIterator end() const noexcept {
        return TIterator(this);
    }

private:
    static constexpr size_t Pos(TEnum c) noexcept {
        Y_ASSERT(IsValid(c));
        return static_cast<size_t>(int(c) - BeginIndex);
    }

    constexpr bool FindFirst(TEnum& result) const noexcept {
        // finds first set item in bitset (or End if bitset is empty)
        const int index = int(this->FirstNonZeroBit()) + BeginIndex;
        if (index < EndIndex) {
            result = static_cast<TEnum>(index);
            return true;
        }
        return false;
    }

    constexpr bool FindNext(TEnum current, TEnum& result) const noexcept {
        // finds first set item in bitset (or End if bitset is empty)
        const int index = int(this->NextNonZeroBit(int(current) - BeginIndex)) + BeginIndex;
        if (index < EndIndex) {
            result = static_cast<TEnum>(index);
            return true;
        }
        return false;
    }
};

template <typename TEnum, TEnum mbegin, int mend>
class TSfEnumBitSet: public TEnumBitSet<TEnum, static_cast<int>(mbegin), mend> {
public:
    using TParent = typename TSfEnumBitSet::TEnumBitSet;

    constexpr TSfEnumBitSet() noexcept
        : TParent()
    {
    }

    constexpr TSfEnumBitSet(const TParent& p) noexcept
        : TParent(p)
    {
    }

    //! unsafe initialization from ui64, value must be shifted according to TParent::Begin
    explicit constexpr TSfEnumBitSet(ui64 val) noexcept
        : TParent(typename TParent::TParent(val))
    {
        //static_assert(TParent::BitsetSize <= 64, "expect TParent::BitsetSize <= 64");
    }

    // Earlier updates remain applied if converting a later argument throws.
    template <class... R>
    constexpr void Init(R... r) noexcept((noexcept(this->SafeSet(r)) && ...)) {
        (this->SafeSet(r), ...);
    }

    constexpr TSfEnumBitSet(TEnum c) noexcept {
        Init(c);
    }

    template <class... R>
    constexpr TSfEnumBitSet(TEnum c1, TEnum c2, R... r) noexcept(noexcept(Init(c1, c2, r...))) {
        Init(c1, c2, r...);
    }

    static TSfEnumBitSet GetFromString(TStringBuf s) {
        TSfEnumBitSet ebs;
        ebs.FromString(s);
        return ebs;
    }

    static TSfEnumBitSet TryGetFromString(TStringBuf s) noexcept {
        TSfEnumBitSet ebs;
        ebs.TryFromString(s);
        return ebs;
    }
};

/* For Enums with GENERATE_ENUM_SERIALIZATION_WITH_HEADER */
template <typename TEnum>
class TGeneratedEnumBitSet : public TEnumBitSet<TEnum, 0, GetEnumItemsCount<TEnum>()> {
public:
    using TParent = TEnumBitSet<TEnum, 0, GetEnumItemsCount<TEnum>()>;

    constexpr TGeneratedEnumBitSet() noexcept
        : TParent()
    {
    }

    explicit constexpr TGeneratedEnumBitSet(const TParent& p) noexcept
        : TParent(p)
    {
    }

    explicit constexpr TGeneratedEnumBitSet(TEnum c1) noexcept
        : TParent(c1)
    {
    }

    template <class... R>
    constexpr TGeneratedEnumBitSet(TEnum c1, TEnum c2, R... r) noexcept(noexcept(TParent(c1, c2, r...)))
        : TParent(c1, c2, r...)
    {
    }
};
