#pragma once

#include <util/generic/hash.h>
#include <util/system/guard.h>
#include <util/system/spinlock.h>

// TValueType should be movable
template <typename TKeyType, typename TValueType, typename TKeyHashFcn, size_t BinCount = 32>
class TSpinLockedKeyValueStorage {
private:
    static inline size_t GetBinIndexForKey(const TKeyType& key) {
        return static_cast<size_t>(TKeyHashFcn()(key)) % BinCount;
    }

public:
    TSpinLockedKeyValueStorage() {
        Locks.resize(BinCount);
        Storages.resize(BinCount);
    }
    bool Empty() const {
        bool empty = true;
        for (size_t i = 0; i < BinCount; ++i) {
            with_lock (Locks[i]) {
                empty &= Storages[i].empty();
            }
            if (!empty) {
                return empty;
            }
        }
        return empty;
    }
    bool ApproximateSize() const {
        size_t accum = 0;
        for (size_t i = 0; i < BinCount; ++i) {
            with_lock (Locks[i]) {
                accum += Storages[i].size();
            }
        }
        return accum;
    }
    bool Has(const TKeyType& key) const {
        const auto binIndex = GetBinIndexForKey(key);
        with_lock (Locks[binIndex]) {
            return Storages[binIndex].has(key);
        }
    }
    bool EraseValueIfPresent(const TKeyType& key) {
        const auto binIndex = GetBinIndexForKey(key);
        with_lock (Locks[binIndex]) {
            auto& map = Storages[binIndex];
            if (!map.contains(key)) {
                return false;
            }
            map.erase(key);
            return true;
        }
    }
    bool ExtractValueIfPresent(const TKeyType& key, TValueType& value) {
        const auto binIndex = GetBinIndexForKey(key);
        with_lock (Locks[binIndex]) {
            auto& map = Storages[binIndex];
            if (!map.contains(key)) {
                return false;
            }
            value = std::move(map.at(key));
            map.erase(key);
            return true;
        }
    }

    TValueType ExtractValue(const TKeyType& key) {
        const auto binIndex = GetBinIndexForKey(key);
        with_lock (Locks[binIndex]) {
            auto& map = Storages[binIndex];
            auto val = std::move(map.at(key));
            map.erase(key);
            return val;
        }
    }

    template <typename... Args>
    void EmplaceValue(const TKeyType& key, Args&&... args) {
        const auto binIndex = GetBinIndexForKey(key);
        with_lock (Locks[binIndex]) {
            Y_ABORT_UNLESS(Storages[binIndex].emplace(key, std::forward<Args>(args)...).second, "emplacing non uniq value");
        }
    }
    bool LockedValueModify(const TKeyType& key, std::function<void(TValueType& value)> functor) {
        const auto binIndex = GetBinIndexForKey(key);
        with_lock (Locks[binIndex]) {
            auto& map = Storages[binIndex];
            if (!map.contains(key)) {
                return false;
            }
            functor(map.at(key));
            return true;
        }
    }
    void LockedIterateValues(std::function<void(const TKeyType& key, TValueType& value)> functor) {
        for (size_t i = 0; i < BinCount; ++i) {
            with_lock (Locks[i]) {
                for (auto& keyval : Storages[i]) {
                    functor(keyval.first, keyval.second);
                }
            }
        }
    }

private:
    using TInternalHash = THashMap<TKeyType, TValueType, TKeyHashFcn>;
    TVector<TInternalHash> Storages;
    mutable TVector<TSpinLock> Locks;
};
