#pragma once

#include <util/ysaveload.h>
#include <util/generic/bitmap.h>
#include <util/generic/serialized_enum.h>
#include <util/generic/yexception.h>
#include <util/string/cast.h>
#include <util/string/printf.h>
#include <util/system/yassert.h>

// Stack memory bitmask for TEnum values [begin, end).
// @end value is not included in the mask and is not necessarily defined as enum value.
// For example: enum EType { A, B, C } ==> TEnumBitSet<EType, A, C + 1>
template <typename TEnum, int mbegin, int mend>
class TEnumBitSet: private TBitMap<mend - mbegin> {
public:
    static const int BeginIndex = mbegin;
    static const int EndIndex = mend;
    static const size_t BitsetSize = EndIndex - BeginIndex;

    typedef TBitMap<BitsetSize> TParent;
    typedef TEnumBitSet<TEnum, mbegin, mend> TThis;

    TEnumBitSet()
        : TParent(0)
    {
    }

    explicit TEnumBitSet(const TParent& p)
        : TParent(p)
    {
    }

    void Init(TEnum c) {
        Set(c);
    }

    template <class... R>
    void Init(TEnum c1, TEnum c2, R... r) {
        Set(c1);
        Init(c2, r...);
    }

    explicit TEnumBitSet(TEnum c) {
        Init(c);
    }

    template <class... R>
    TEnumBitSet(TEnum c1, TEnum c2, R... r) {
        Init(c1, c2, r...);
    }

    template <class TIt>
    TEnumBitSet(const TIt& begin_, const TIt& end_) {
        for (TIt p = begin_; p != end_; ++p)
            Set(*p);
    }

    static bool IsValid(TEnum c) {
        return int(c) >= BeginIndex && int(c) < EndIndex;
    }

    bool Test(TEnum c) const {
        return TParent::Test(Pos(c));
    }

    TThis& Flip(TEnum c) {
        TParent::Flip(Pos(c));
        return *this;
    }

    TThis& Flip() {
        TParent::Flip();
        return *this;
    }

    TThis& Reset(TEnum c) {
        TParent::Reset(Pos(c));
        return *this;
    }

    TThis& Reset() {
        TParent::Clear();
        return *this;
    }

    TThis& Set(TEnum c) {
        TParent::Set(Pos(c));
        return *this;
    }

    TThis& Set(TEnum c, bool val) {
        if (val)
            Set(c);
        else
            Reset(c);
        return *this;
    }

    bool SafeTest(TEnum c) const {
        if (IsValid(c))
            return Test(c);
        return false;
    }

    TThis& SafeFlip(TEnum c) {
        if (IsValid(c))
            return Flip(c);
        return *this;
    }

    TThis& SafeReset(TEnum c) {
        if (IsValid(c))
            return Reset(c);
        return *this;
    }

    TThis& SafeSet(TEnum c) {
        if (IsValid(c))
            return Set(c);
        return *this;
    }

    TThis& SafeSet(TEnum c, bool val) {
        if (IsValid(c))
            return Set(c, val);
        return *this;
    }

    static TThis SafeConstruct(TEnum c) {
        TThis ret;
        ret.SafeSet(c);
        return ret;
    }

    bool operator<(const TThis& right) const {
        Y_ASSERT(this->GetChunkCount() == right.GetChunkCount());
        for (size_t i = 0; i < this->GetChunkCount(); ++i) {
            if (this->GetChunks()[i] < right.GetChunks()[i])
                return true;
            else if (this->GetChunks()[i] > right.GetChunks()[i])
                return false;
        }
        return false;
    }

    bool operator!=(const TThis& right) const {
        return !(TParent::operator==(right));
    }

    bool operator==(const TThis& right) const {
        return TParent::operator==(right);
    }

    TThis& operator&=(const TThis& right) {
        TParent::operator&=(right);
        return *this;
    }

    TThis& operator|=(const TThis& right) {
        TParent::operator|=(right);
        return *this;
    }

    TThis& operator^=(const TThis& right) {
        TParent::operator^=(right);
        return *this;
    }

    TThis operator~() const {
        TThis r = *this;
        r.Flip();
        return r;
    }

    TThis operator|(const TThis& right) const {
        TThis ret = *this;
        ret |= right;
        return ret;
    }

    TThis operator&(const TThis& right) const {
        TThis ret = *this;
        ret &= right;
        return ret;
    }

    TThis operator^(const TThis& right) const {
        TThis ret = *this;
        ret ^= right;
        return ret;
    }


    TThis& operator&=(const TEnum c) {
        return TThis::operator&=(TThis(c));
    }

    TThis& operator|=(const TEnum c) {
        return TThis::operator|=(TThis(c));
    }

    TThis& operator^=(const TEnum c) {
        return TThis::operator^=(TThis(c));
    }

    TThis operator&(const TEnum c) const {
        return TThis::operator&(TThis(c));
    }

    TThis operator|(const TEnum c) const {
        return TThis::operator|(TThis(c));
    }

    TThis operator^(const TEnum c) const {
        return TThis::operator^(TThis(c));
    }

    auto operator[] (TEnum e) {
        return TParent::operator[](this->Pos(e));
    }

    auto operator[] (TEnum e) const {
        return TParent::operator[](this->Pos(e));
    }

    using TParent::Count;
    using TParent::Empty;

    explicit operator bool() const {
        return !Empty();
    }

    void Swap(TThis& bitmap) {
        TParent::Swap(bitmap);
    }

    size_t GetHash() const {
        return this->Hash();
    }

    bool HasAny(const TThis& mask) const {
        return TParent::HasAny(mask);
    }

    template <class... R>
    bool HasAny(TEnum c1, R... r) const {
        return Test(c1) || HasAny(r...);
    }

    bool HasAll(const TThis& mask) const {
        return TParent::HasAll(mask);
    }

    template <class... R>
    bool HasAll(TEnum c1, R... r) const {
        return Test(c1) && HasAll(r...);
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

            Set((TEnum)bit);
        }
    }

    ui64 Low() const {
        ui64 t = 0;
        this->Export(0, t);
        return t;
    }

    TString ToString() const {
        static_assert(sizeof(typename TParent::TChunk) <= sizeof(ui64), "expect sizeof(typename TParent::TChunk) <= sizeof(ui64)");
        static const size_t chunkSize = sizeof(typename TParent::TChunk) * 8;
        static const size_t numDig = chunkSize / 4;
        static const TString templ = Sprintf("%%0%lulX", numDig);
        static const size_t numOfChunks = (BitsetSize + chunkSize - 1) / chunkSize;
        TString ret;
        for (int pos = numOfChunks * chunkSize; pos >= 0; pos -= chunkSize) {
            ui64 t = 0;
            this->Export(pos, t);
            ret += Sprintf(templ.data(), t);
        }

        size_t n = 0;
        while (n + 1 < ret.length() && ret[n] == '0')
            ++n;
        ret.remove(0, n);
        return ret;
    }

    void FromString(TStringBuf s) {
        static const size_t chunkSize = sizeof(typename TParent::TChunk) * 8;
        static const size_t numDig = chunkSize / 4;
        static const size_t highChunkBits = (BitsetSize + chunkSize - 1) % chunkSize + 1;
        static const typename TParent::TChunk highChunkBitsMask = (typename TParent::TChunk(1) << highChunkBits) - 1;

        Reset();
        for (size_t prev = s.length(), n = s.length() - numDig, pos = 0; prev; n -= numDig, pos += chunkSize) {
            if (pos >= BitsetSize)
                ythrow yexception() << "too many digits";
            if (n > prev)
                n = 0;
            typename TParent::TChunk t = IntFromString<typename TParent::TChunk, 16, TStringBuf>(s.substr(n, prev - n));
            if (BitsetSize < pos + chunkSize && t > highChunkBitsMask)
                ythrow yexception() << "digit is too big";
            this->Or(TParent(t), pos);
            prev = n;
        }
    }

    // TODO: Get rid of exceptions at all
    bool TryFromString(TStringBuf s) {
        try {
            FromString(s);
        } catch (...) {
            Reset();
            return false;
        }
        return true;
    }

    bool any() const { // obsolete
        return !Empty();
    }

    bool none() const { // obsolete
        return Empty();
    }

    size_t count() const { // obsolete
        return Count();
    }

    class TIterator {
    public:
        TIterator(TEnum value, const TThis* bitmap) noexcept
            : Value(static_cast<int>(value))
            , BitMap(bitmap)
        {
        }

        TIterator(const TThis* bitmap) noexcept
            : Value(EndIndex)
            , BitMap(bitmap)
        {
        }

        TEnum operator*() const noexcept {
            Y_ASSERT(Value < EndIndex);
            return static_cast<TEnum>(Value);
        }

        bool operator!=(const TIterator& other) const noexcept {
            return Value != other.Value;
        }

        TIterator& operator++() noexcept {
            Y_ASSERT(Value < EndIndex);
            TEnum res;
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

    TIterator begin() const {
        TEnum res;
        return FindFirst(res) ? TIterator(res, this) : TIterator(this);
    }

    TIterator end() const {
        return TIterator(this);
    }

private:
    static size_t Pos(TEnum c) {
        Y_ASSERT(IsValid(c));
        return static_cast<size_t>(int(c) - BeginIndex);
    }

    bool HasAny(TEnum c) const {
        return Test(c);
    }

    bool HasAll(TEnum c) const {
        return Test(c);
    }

    bool FindFirst(TEnum& result) const {
        // finds first set item in bitset (or End if bitset is empty)
        const int index = int(this->FirstNonZeroBit()) + BeginIndex;
        if (index < EndIndex) {
            result = static_cast<TEnum>(index);
            return true;
        }
        return false;
    }

    bool FindNext(TEnum current, TEnum& result) const {
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
class TSfEnumBitSet: public TEnumBitSet<TEnum, mbegin, mend> {
public:
    typedef TEnumBitSet<TEnum, mbegin, mend> TParent;

    TSfEnumBitSet()
        : TParent()
    {
    }

    TSfEnumBitSet(const TParent& p)
        : TParent(p)
    {
    }

    //! unsafe initialization from ui64, value must be shifted according to TParent::Begin
    explicit TSfEnumBitSet(ui64 val)
        : TParent(typename TParent::TParent(val))
    {
        //static_assert(TParent::BitsetSize <= 64, "expect TParent::BitsetSize <= 64");
    }

    void Init(TEnum c) {
        this->SafeSet(c);
    }

    template <class... R>
    void Init(TEnum c1, TEnum c2, R... r) {
        this->SafeSet(c1);
        Init(c2, r...);
    }

    TSfEnumBitSet(TEnum c) {
        Init(c);
    }

    template <class... R>
    TSfEnumBitSet(TEnum c1, TEnum c2, R... r) {
        Init(c1, c2, r...);
    }

    static TSfEnumBitSet GetFromString(const TString& s) {
        TSfEnumBitSet ebs;
        ebs.FromString(s);
        return ebs;
    }

    static TSfEnumBitSet TryGetFromString(const TString& s) {
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

    TGeneratedEnumBitSet()
        : TParent()
    {
    }

    explicit TGeneratedEnumBitSet(const TParent& p)
        : TParent(p)
    {
    }

    explicit TGeneratedEnumBitSet(TEnum c1)
        : TParent(c1)
    {
    }

    template <class... R>
    TGeneratedEnumBitSet(TEnum c1, TEnum c2, R... r)
        : TParent(c1, c2, r...)
    {
    }
};
