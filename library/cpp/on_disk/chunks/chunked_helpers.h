#pragma once

#include <util/generic/vector.h>
#include <util/generic/buffer.h>
#include <util/generic/hash_set.h>
#include <util/generic/cast.h>
#include <util/generic/ymath.h>
#include <util/memory/blob.h>
#include <util/stream/buffer.h>
#include <util/stream/mem.h>
#include <util/system/unaligned_mem.h>
#include <util/ysaveload.h>

#include "reader.h"
#include "writer.h"

#include <cmath>
#include <cstddef>

template <typename T>
class TYVector {
private:
    ui32 Size;
    const T* Data;

public:
    TYVector(const TBlob& blob)
        : Size(IntegerCast<ui32>(ReadUnaligned<ui64>(blob.Data())))
        , Data((const T*)((const char*)blob.Data() + sizeof(ui64)))
    {
    }

    void Get(size_t idx, T& t) const {
        assert(idx < (size_t)Size);
        t = ReadUnaligned<T>(Data + idx);
    }

    const T& At(size_t idx) const {
        assert(idx < (size_t)Size);
        return Data[idx];
    }

    size_t GetSize() const {
        return Size;
    }

    size_t RealSize() const {
        return sizeof(ui64) + Size * sizeof(T);
    }

    ~TYVector() = default;
};

template <typename T>
class TYVectorWriter {
private:
    TVector<T> Vector;

public:
    TYVectorWriter() = default;

    void PushBack(const T& value) {
        Vector.push_back(value);
    }

    void Save(IOutputStream& out) const {
        ui64 uSize = (ui64)Vector.size();
        out.Write(&uSize, sizeof(uSize));
        out.Write(Vector.data(), Vector.size() * sizeof(T));
    }

    const T& At(size_t idx) const {
        assert(idx < Size());
        return Vector[idx];
    }

    T& At(size_t idx) {
        assert(idx < Size());
        return Vector[idx];
    }

    void Clear() {
        Vector.clear();
    }

    size_t Size() const {
        return Vector.size();
    }

    void Resize(size_t size) {
        Vector.resize(size);
    }

    void Resize(size_t size, const T& value) {
        Vector.resize(size, value);
    }
};

template <typename T, bool>
struct TYVectorG;

template <typename X>
struct TYVectorG<X, false> {
    typedef TYVector<X> T;
};

template <typename X>
struct TYVectorG<X, true> {
    typedef TYVectorWriter<X> T;
};

template <typename T>
struct TIsMemsetThisWithZeroesSupported {
    enum {
        Result = TTypeTraits<T>::IsPod
    };
};

#define MEMSET_THIS_WITH_ZEROES_SUPPORTED(type)     \
    template <>                                     \
    struct TIsMemsetThisWithZeroesSupported<type> { \
        enum {                                      \
            Result = true                           \
        };                                          \
    };

class TPlainHashCommon {
protected:
#pragma pack(push, 8)
    template <typename TKey, typename TValue>
    class TPackedPair {
    private:
        typedef TPackedPair<TKey, TValue> TThis;
        TKey Key;
        TValue Value;

    private:
        static_assert(TIsMemsetThisWithZeroesSupported<TKey>::Result, "expect TIsMemsetThisWithZeroesSupported<TKey>::Result");
        static_assert(TIsMemsetThisWithZeroesSupported<TValue>::Result, "expect TIsMemsetThisWithZeroesSupported<TValue>::Result");

        /// to aviod uninitialized bytes
        void Init(const TKey& key, const TValue& value) {
            memset(static_cast<TThis*>(this), 0, sizeof(TThis));
            Key = key;
            Value = value;
        }

    public:
        TPackedPair(typename TTypeTraits<TKey>::TFuncParam key, typename TTypeTraits<TValue>::TFuncParam value) {
            Init(key, value);
        }

        TPackedPair(const TThis& rhs) {
            Init(rhs.Key, rhs.Value);
        }

        TPackedPair& operator=(const TThis& rhs) {
            if (this != &rhs) {
                Init(rhs.Key, rhs.Value);
            }
            return *this;
        }

        TPackedPair() {
            Init(TKey(), TValue());
        }

        typename TTypeTraits<TKey>::TFuncParam First() const {
            return Key;
        }

        typename TTypeTraits<TValue>::TFuncParam Second() const {
            return Value;
        }

        static TKey GetFirst(const void* self) {
            static constexpr size_t offset = offsetof(TThis, Key);
            return ReadUnaligned<TKey>(reinterpret_cast<const char*>(self) + offset);
        }

        static TValue GetSecond(const void* self) {
            static constexpr size_t offset = offsetof(TThis, Value);
            return ReadUnaligned<TValue>(reinterpret_cast<const char*>(self) + offset);
        }
    };
#pragma pack(pop)

protected:
    static const ui16 VERSION_ID = 2;

#pragma pack(push, 8)
    struct TInterval {
        static const ui32 INVALID = (ui32)-1;
        ui32 Offset;
        ui32 Length;

        TInterval()
            : Offset(INVALID)
            , Length(INVALID)
        {
        }

        TInterval(ui32 offset, ui32 length)
            : Offset(offset)
            , Length(length)
        {
        }

        static inline ui32 GetOffset(const TInterval* self) {
            static constexpr size_t offset = offsetof(TInterval, Offset);
            return ReadUnaligned<ui32>(reinterpret_cast<const char*>(self) + offset);
        }

        static inline ui32 GetLength(const TInterval* self) {
            static constexpr size_t offset = offsetof(TInterval, Length);
            return ReadUnaligned<ui32>(reinterpret_cast<const char*>(self) + offset);
        }
    };
#pragma pack(pop)
    static_assert(8 == sizeof(TInterval), "expect 8 == sizeof(TInterval)");

    template <typename TKey>
    static ui32 KeyHash(typename TTypeTraits<TKey>::TFuncParam key, ui16 bits) {
        Y_ASSERT(bits < 32);
        const ui32 res = ui32(key) & ((ui32(1) << bits) - 1);

        Y_ASSERT(res < (ui32(1) << bits));
        return res;
    }
};

template <typename TKey, typename TValue>
class TPlainHashWriter : TPlainHashCommon {
private:
    typedef TPackedPair<TKey, TValue> TKeyValuePair;
    typedef TVector<TKeyValuePair> TData;
    TData Data;
    typedef TVector<TData> TData2;

    bool IsPlainEnought(ui16 bits) const {
        TVector<size_t> counts(1LL << bits, 0);
        for (size_t i = 0; i < Data.size(); ++i) {
            size_t& count = counts[KeyHash<TKey>(TKeyValuePair::GetFirst(&Data[i]), bits)];
            ++count;
            if (count > 2)
                return false;
        }
        return true;
    }

public:
    void Add(const TKey& key, const TValue& value) {
        Data.push_back(TKeyValuePair(key, value));
    }

    void Save(IOutputStream& out) const {
        Y_ASSERT(Data.size() < Max<ui32>());

        WriteBin<ui16>(&out, VERSION_ID);
        static const ui32 PAIR_SIZE = sizeof(TKeyValuePair);
        WriteBin<ui32>(&out, PAIR_SIZE);

        ui16 bits;
        if (!Data.empty()) {
            bits = (ui16)(log((float)Data.size()) / log(2.f));
            while ((bits < 22) && !IsPlainEnought(bits))
                ++bits;
        } else {
            bits = 0;
        }
        WriteBin<ui16>(&out, bits);
        WriteBin<ui32>(&out, (ui32)Data.size());

        const ui32 nBuckets = ui32(1) << bits;
        TData2 data2(nBuckets);
        for (size_t i = 0; i < Data.size(); ++i)
            data2[KeyHash<TKey>(TKeyValuePair::GetFirst(&Data[i]), bits)].push_back(Data[i]);

        typedef TVector<TInterval> TIntervals;
        TIntervals intervals(nBuckets);
        ui32 offset = 0;
        for (ui32 i = 0; i < nBuckets; ++i) {
            intervals[i].Offset = offset;
            intervals[i].Length = (ui32)data2[i].size();
            offset += (ui32)data2[i].size();
        }
#ifndef NDEBUG
        for (ui32 i = 0; i < nBuckets; ++i) {
            for (size_t j = 0; j < data2[i].size(); ++j)
                for (size_t k = j + 1; k < data2[i].size(); ++k)
                    if (TKeyValuePair::GetFirst(&data2[i][j]) == TKeyValuePair::GetFirst(&data2[i][k]))
                        ythrow yexception() << "key clash";
        }
#endif
        out.Write(intervals.data(), intervals.size() * sizeof(intervals[0]));
        for (ui32 i = 0; i < nBuckets; ++i)
            out.Write(data2[i].data(), data2[i].size() * sizeof(data2[i][0]));
    }
};

template <typename TKey, typename TValue>
class TPlainHash : TPlainHashCommon {
private:
    typedef TPackedPair<TKey, TValue> TKeyValuePair;

    const char* P;

    ui16 GetBits() const {
        return ReadUnaligned<ui16>(P + 6);
    }

    ui32 GetSize() const {
        return ReadUnaligned<ui32>(P + 8);
    }

    const TInterval* GetIntervals() const {
        return (const TInterval*)(P + 12);
    }

    const TKeyValuePair* GetData() const {
        return (const TKeyValuePair*)(GetIntervals() + (1ULL << GetBits()));
    }

    template <typename T>
    void Init(const T* p) {
        static_assert(sizeof(T) == 1, "expect sizeof(T) == 1");
        P = reinterpret_cast<const char*>(p);
#ifndef NDEBUG
        ui16 version = ReadUnaligned<ui16>(p);
        if (version != VERSION_ID)
            ythrow yexception() << "bad version: " << version;
        static const ui32 PAIR_SIZE = sizeof(TKeyValuePair);
        const ui32 size = ReadUnaligned<ui32>(p + 2);
        if (size != PAIR_SIZE)
            ythrow yexception() << "bad size " << size << " instead of " << PAIR_SIZE;
#endif
    }

public:
    typedef const TKeyValuePair* TConstIterator;

    TPlainHash(const char* p) {
        Init(p);
    }

    TPlainHash(const TBlob& blob) {
        Init(blob.Begin());
    }

    bool Find(typename TTypeTraits<TKey>::TFuncParam key, TValue* res) const {
        // Cerr << GetBits() << "\t" << (1 << GetBits()) << "\t" << GetSize() << Endl;
        const ui32 hash = KeyHash<TKey>(key, GetBits());
        const TInterval* intervalPtr = GetIntervals();
        const TKeyValuePair* pair = GetData() + TInterval::GetOffset(intervalPtr + hash);
        const ui32 length = TInterval::GetLength(intervalPtr + hash);
        for (ui32 i = 0; i < length; ++i, ++pair) {
            if (TKeyValuePair::GetFirst(pair) == key) {
                *res = TKeyValuePair::GetSecond(pair);
                return true;
            }
        }
        return false;
    }

    TValue Get(typename TTypeTraits<TKey>::TFuncParam key) const {
        TValue res;
        if (Find(key, &res))
            return res;
        else
            ythrow yexception() << "key not found";
    }

    TConstIterator Begin() const {
        return GetData();
    }

    TConstIterator End() const {
        return GetData() + GetSize();
    }

    const char* ByteEnd() const {
        return (const char*)(GetData() + GetSize());
    }

    size_t ByteSize() const {
        return 12 + sizeof(TInterval) * (size_t(1) << GetBits()) + sizeof(TKeyValuePair) * GetSize();
    }
};

template <typename Key, typename Value, bool>
struct TPlainHashG;

template <typename Key, typename Value>
struct TPlainHashG<Key, Value, false> {
    typedef TPlainHash<Key, Value> T;
};

template <typename Key, typename Value>
struct TPlainHashG<Key, Value, true> {
    typedef TPlainHashWriter<Key, Value> T;
};

template <typename T>
class TSingleValue {
private:
    const T* Value;

public:
    TSingleValue(const TBlob& blob) {
        Y_ASSERT(blob.Length() >= sizeof(T));
        Y_ASSERT(blob.Length() <= sizeof(T) + 16);
        Value = reinterpret_cast<const T*>(blob.Begin());
    }

    const T& Get() const {
        return *Value;
    }
};

template <typename T>
class TSingleValueWriter {
private:
    T Value;

public:
    TSingleValueWriter() = default;

    TSingleValueWriter(const T& value)
        : Value(value)
    {
    }

    void Set(const T& value) {
        Value = value;
    }

    void Save(IOutputStream& out) const {
        out.Write(&Value, sizeof(Value));
    }
};

TBlob GetBlock(const TBlob& data, size_t index);

template <class T>
void WriteBlock(TChunkedDataWriter& writer, const T& t) {
    writer.NewBlock();
    t.Save(writer);
}

template <class T>
void WriteBlock(TChunkedDataWriter& writer, T& t) {
    writer.NewBlock();
    t.Save(writer);
}

// Extends TChunkedDataWriter, allowing user to name blocks with arbitrary strings.
class TNamedChunkedDataWriter: public TChunkedDataWriter {
public:
    TNamedChunkedDataWriter(IOutputStream& slave);
    ~TNamedChunkedDataWriter() override;

    // Start a new unnamed block, overrides TChunkedDataReader::NewBlock().
    void NewBlock();

    // Start a new block with given name (possibly empty, in which case block is unnamed).
    // Throws an exception if name is a duplicate.
    void NewBlock(const TString& name);

    void WriteFooter();

private:
    TVector<TString> Names;
    THashMap<TString, size_t> NameToIndex;
};

class TNamedChunkedDataReader: public TChunkedDataReader {
public:
    TNamedChunkedDataReader(const TBlob& blob);

    inline bool HasBlock(const char* name) const {
        return NameToIndex.find(name) != NameToIndex.end();
    }

    inline size_t GetIndexByName(const char* name) const {
        THashMap<TString, size_t>::const_iterator it = NameToIndex.find(name);
        if (it == NameToIndex.end())
            throw yexception() << "Block \"" << name << "\" is not found";
        else
            return it->second;
    }

    // Returns number of blocks written to the file by user of TNamedChunkedDataReader.
    inline size_t GetBlocksCount() const {
        // Last block is for internal usage
        return TChunkedDataReader::GetBlocksCount() - 1;
    }

    inline const char* GetBlockName(size_t index) const {
        Y_ASSERT(index < GetBlocksCount());
        return Names[index].data();
    }

    inline const void* GetBlockByName(const char* name) const {
        return GetBlock(GetIndexByName(name));
    }

    inline size_t GetBlockLenByName(const char* name) const {
        return GetBlockLen(GetIndexByName(name));
    }

    inline TBlob GetBlobByName(const char* name) const {
        size_t id = GetIndexByName(name);
        return TBlob::NoCopy(GetBlock(id), GetBlockLen(id));
    }

    inline bool GetBlobByName(const char* name, TBlob& blob) const {
        THashMap<TString, size_t>::const_iterator it = NameToIndex.find(name);
        if (it == NameToIndex.end())
            return false;
        blob = TBlob::NoCopy(GetBlock(it->second), GetBlockLen(it->second));
        return true;
    }

private:
    TVector<TString> Names;
    THashMap<TString, size_t> NameToIndex;
};

template <class T>
struct TSaveLoadVectorNonPodElement {
    static inline void Save(IOutputStream* out, const T& t) {
        TSerializer<T>::Save(out, t);
    }

    static inline void Load(IInputStream* in, T& t, size_t elementSize) {
        Y_ASSERT(elementSize > 0);
        TSerializer<T>::Load(in, t);
    }
};

template <class T, bool isPod>
class TVectorTakingIntoAccountThePodType {
private:
    ui64 SizeofOffsets;
    const ui64* Offsets;
    const char* Data;

public:
    TVectorTakingIntoAccountThePodType(const TBlob& blob) {
        SizeofOffsets = ReadUnaligned<ui64>(blob.Begin());
        Y_ASSERT(SizeofOffsets > 0);
        Offsets = reinterpret_cast<const ui64*>(blob.Begin() + sizeof(ui64));
        Data = reinterpret_cast<const char*>(blob.Begin() + sizeof(ui64) + SizeofOffsets * sizeof(ui64));
    }

    size_t GetSize() const {
        return (size_t)(SizeofOffsets - 1);
    }

    size_t GetLength(ui64 index) const {
        if (index + 1 >= SizeofOffsets)
            ythrow yexception() << "bad offset";
        return IntegerCast<size_t>(ReadUnaligned<ui64>(Offsets + index + 1) - ReadUnaligned<ui64>(Offsets + index));
    }

    void Get(ui64 index, T& t) const {
        const size_t len = GetLength(index);
        TMemoryInput input(Data + ReadUnaligned<ui64>(Offsets + index), len);
        TSaveLoadVectorNonPodElement<T>::Load(&input, t, len);
    }

    T Get(ui64 index) const {
        T ret;
        Get(index, ret);
        return ret;
    }

    size_t RealSize() const {
        return sizeof(ui64) * (SizeofOffsets + 1) + ReadUnaligned<ui64>(Offsets + SizeofOffsets - 1);
    }
};

template <class T, bool isPod>
class TVectorTakingIntoAccountThePodTypeWriter : TNonCopyable {
private:
    typedef TVector<ui64> TOffsets;
    TOffsets Offsets;
    TBuffer Data;
    TBufferOutput DataStream;

public:
    TVectorTakingIntoAccountThePodTypeWriter()
        : DataStream(Data)
    {
    }

    void PushBack(const T& t) {
        Offsets.push_back((ui64) Data.size());
        TSaveLoadVectorNonPodElement<T>::Save(&DataStream, t);
    }

    size_t Size() const {
        return Offsets.size();
    }

    void Save(IOutputStream& out) const {
        ui64 sizeofOffsets = Offsets.size() + 1;
        out.Write(&sizeofOffsets, sizeof(sizeofOffsets));
        out.Write(Offsets.data(), Offsets.size() * sizeof(Offsets[0]));
        ui64 lastOffset = (ui64) Data.size();
        out.Write(&lastOffset, sizeof(lastOffset));
        out.Write(Data.data(), Data.size());
    }
};

template <class T>
class TVectorTakingIntoAccountThePodType<T, true>: public TYVector<T> {
public:
    TVectorTakingIntoAccountThePodType(const TBlob& blob)
        : TYVector<T>(blob)
    {
    }
};

template <class T>
class TVectorTakingIntoAccountThePodTypeWriter<T, true>: public TYVectorWriter<T> {
};

template <typename T>
class TGeneralVector: public TVectorTakingIntoAccountThePodType<T, TTypeTraits<T>::IsPod> {
    typedef TVectorTakingIntoAccountThePodType<T, TTypeTraits<T>::IsPod> TBase;

public:
    TGeneralVector(const TBlob& blob)
        : TBase(blob)
    {
    }
};

template <typename T>
class TGeneralVectorWriter: public TVectorTakingIntoAccountThePodTypeWriter<T, TTypeTraits<T>::IsPod> {
};

template <typename TItem, bool>
struct TGeneralVectorG;

template <typename TItem>
struct TGeneralVectorG<TItem, false> {
    typedef TGeneralVector<TItem> T;
};

template <typename TItem>
struct TGeneralVectorG<TItem, true> {
    typedef TGeneralVectorWriter<TItem> T;
};

template <>
struct TSaveLoadVectorNonPodElement<TString> {
    static inline void Save(IOutputStream* out, const TString& s) {
        out->Write(s.data(), s.size() + 1);
    }

    static inline void Load(TMemoryInput* in, TString& s, size_t elementSize) {
        Y_ASSERT(elementSize > 0 && in->Avail() >= elementSize);
        s.assign(in->Buf(), elementSize - 1); /// excluding 0 at the end
    }
};

template <bool G>
struct TStringsVectorG: public TGeneralVectorG<TString, G> {
};

using TStringsVector = TGeneralVector<TString>;
using TStringsVectorWriter = TGeneralVectorWriter<TString>;
