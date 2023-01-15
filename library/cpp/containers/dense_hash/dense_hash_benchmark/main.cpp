#include <library/cpp/containers/dense_hash/dense_hash.h>

#include <contrib/libs/sparsehash/src/sparsehash/dense_hash_map>

#include <util/datetime/cputimer.h>
#include <util/generic/hash.h>
#include <util/generic/hash_set.h>
#include <util/random/mersenne.h>
#include <util/random/shuffle.h>
#include <util/string/cast.h>

#include <unordered_map>

enum {
    ElementsCount = 2 * 1000 * 1000 * 10,
    MaxElementValue = ElementsCount * 10,
};

template <typename TKey, typename TValue>
class TDenseHashAdapter: public TDenseHash<TKey, TValue> {
private:
    using TBase = TDenseHash<TKey, TValue>;

public:
    TDenseHashAdapter(const TKey& emptyKey = TKey())
        : TBase(emptyKey)
    {
    }

    Y_FORCE_INLINE TValue& operator[](const TKey& key) {
        return TBase::operator[](key);
    }
};


template <typename TKey>
class TDenseHashSetAdapter: public TDenseHashSet<TKey> {
private:
    using TBase = TDenseHashSet<TKey>;

public:
    TDenseHashSetAdapter(const TKey& emptyKey = TKey())
        : TBase(emptyKey)
    {
    }

    Y_FORCE_INLINE void insert(const TKey& key) {
        this->Insert(key);
    }

    Y_FORCE_INLINE bool contains(const TKey& key) const {
        return this->Has(key);
    }
};

template <class TKey, class TValue>
struct TStdHash: public std::unordered_map<TKey, TValue>, public TMapOps<TStdHash<TKey, TValue>> {
};

template <class T1, class T2>
class TSerializer<TStdHash<T1, T2>>: public TMapSerializer<TStdHash<T1, T2>, false> {
};

template <class TKey, class TValue>
struct TGoogleDenseHash: public google::sparsehash::dense_hash_map<TKey, TValue>, public TMapOps<TGoogleDenseHash<TKey, TValue>> {
    TGoogleDenseHash() {
        this->set_empty_key((ui32)-1);
    }
};

template <class T1, class T2>
class TSerializer<TGoogleDenseHash<T1, T2>>: public TMapSerializer<TGoogleDenseHash<T1, T2>, false> {
};

template <typename THashMapType>
void BenchAddingToHashMap(THashMapType& hashMap, const TVector<ui32>& keys, const TString& title) {
    TSimpleTimer timer;
    for (const auto& key : keys) {
        hashMap[key] = key;
    }
    Cout << title << ": " << timer.Get() << "\n";
}

template <typename THashMapType>
size_t BenchGettingFromHashMap(const THashMapType& hashMap, const TVector<ui32>& keys, const TString& title) {
    TSimpleTimer timer;
    size_t foundCount = 0;
    for (const auto& key : keys) {
        foundCount += hashMap.FindPtr(key) != nullptr;
        foundCount += hashMap.FindPtr(key + MaxElementValue) != nullptr; // will not be found 100%
    }
    Cout << title << ": " << timer.Get() << "\n";
    return foundCount;
}

template <typename THashMapType>
void BenchAddingToHashSet(THashMapType& hashMap, const TString& title) {
    TSimpleTimer timer;
    for (ui32 value = 0; value < ElementsCount; ++value) {
        hashMap.insert(value);
    }
    Cout << title << ": " << timer.Get() << "\n";
}

template <typename THashMapType>
size_t BenchGettingFromHashSet(const THashMapType& hashMap, const TString& title) {
    TSimpleTimer timer;
    size_t foundCount = 0;
    for (ui32 value = 0; value < ElementsCount * 2; ++value) {
        foundCount += hashMap.contains(value);
    }
    Cout << title << ": " << timer.Get() << "\n";
    return foundCount;
}

template <typename THash>
void BenchSerialization(const THash& hash, const TString& title) {
    TSimpleTimer timer;

    TString indexStr;
    {
        TStringOutput index(indexStr);
        Save(&index, hash);
    }

    Cout << title << " serialization: " << timer.Get() << ", index string length: " << indexStr.size() << "\n";

    timer.Reset();

    THash newHash;
    {
        TStringInput index(indexStr);
        Load(&index, newHash);
    }
    Cout << title << " deserialization: " << timer.Get() << "\n";
}

void BenchMaps() {
    const ui32 seed = 19650218UL; // TODO: take from command line
    TMersenne<ui32> rng(seed);
    TVector<ui32> keys;
    for (size_t i = 0; i < ElementsCount; ++i) {
        keys.push_back(rng.GenRand() % MaxElementValue);
    }

    TVector<ui32> shuffledKeys(keys);
    Shuffle(shuffledKeys.begin(), shuffledKeys.begin(), rng);

    size_t yhashMapFound, denseHashMapFound, stdHashMapFound, googleDenseHashMapFound;

    {
        THashMap<ui32, ui32> yhashMap;
        BenchAddingToHashMap(yhashMap, keys, "adding to THashMap");
        yhashMapFound = BenchGettingFromHashMap(yhashMap, shuffledKeys, "getting from THashMap");
        BenchSerialization(yhashMap, "THashMap");
    }
    Cout << "---------------" << Endl;
    {
        TDenseHashAdapter<ui32, ui32> denseHash((ui32)-1);
        BenchAddingToHashMap(denseHash, keys, "adding to dense hash");
        denseHashMapFound = BenchGettingFromHashMap(denseHash, shuffledKeys, "getting from dense hash");
        BenchSerialization(denseHash, "dense hash");
    }
    Cout << "---------------" << Endl;
    {
        TStdHash<ui32, ui32> stdHash;
        BenchAddingToHashMap(stdHash, keys, "adding to std hash");
        stdHashMapFound = BenchGettingFromHashMap(stdHash, shuffledKeys, "getting from std hash");
        BenchSerialization(stdHash, "std hash");
    }
    Cout << "---------------" << Endl;
    {
        TGoogleDenseHash<ui32, ui32> googleDenseHash;
        BenchAddingToHashMap(googleDenseHash, keys, "adding to google dense hash");
        googleDenseHashMapFound = BenchGettingFromHashMap(googleDenseHash, shuffledKeys, "getting from google dense hash");
        BenchSerialization(googleDenseHash, "google dense hash");
    }
    Cout << "---------------" << Endl;

    Cout << "found in THashMap: " << yhashMapFound << "\n";
    Cout << "found in dense hash: " << denseHashMapFound << "\n";
    Cout << "found in std hash: " << stdHashMapFound << "\n";
    Cout << "found in google dense hash: " << googleDenseHashMapFound << "\n";

    Cout << "\n";
}

void BenchSets() {
    size_t yhashSetFound, denseHashSetFound;

    {
        THashSet<ui32> yhashSet;
        BenchAddingToHashSet(yhashSet, "adding to THashSet");
        yhashSetFound = BenchGettingFromHashSet(yhashSet, "getting from THashSet");
        BenchSerialization(yhashSet, "THashMap");
    }
    {
        TDenseHashSetAdapter<ui32> denseHashSet((ui32)-1);
        BenchAddingToHashSet(denseHashSet, "adding to dense hash set");
        denseHashSetFound = BenchGettingFromHashSet(denseHashSet, "getting from dense hash set");
        BenchSerialization(denseHashSet, "dense hash set");
    }
    Cout << "found in THashSet: " << yhashSetFound << "\n";
    Cout << "found in dense hash set: " << denseHashSetFound << "\n";

    Cout << "\n";
}

int main() {
    BenchMaps();
    BenchSets();
}
