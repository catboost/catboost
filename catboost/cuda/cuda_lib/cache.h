#pragma once

#include <util/system/types.h>
#include <util/generic/map.h>
#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/generic/guid.h>
#include <util/system/spinlock.h>
#include <util/system/mutex.h>

//cacheable type should has GetGuid() method. this stub is default implementation for this method
template <class TLock>
class TGuidHolderBase {
public:
    TGuidHolderBase()
        : Lock(new TLock)
    {
    }
    const TGUID& GetGuid() const {
        if (!HasGuid) {
            TGuard<TLock> guard(*Lock);
            if (!HasGuid) {
                CreateGuid(&Guid);
                HasGuid = true;
            }
        }
        return Guid;
    }

private:
    THolder<TLock> Lock;
    mutable TGUID Guid;
    mutable bool HasGuid = false;
};

using TGuidHolder = TGuidHolderBase<TFakeMutex>;
using TThreadSafeGuidHolder = TGuidHolderBase<TAdaptiveLock>;

class TScopedCacheHolder {
private:
    class IScopedCache {
    public:
        virtual ~IScopedCache() {
        }
    };

    template <class TKey, class TValue>
    class TScopedCache: public IScopedCache {
    public:
        TValue& Value(const TKey& key) {
            return Data.at(key);
        }

        const TValue& Value(const TKey& key) const {
            return Data.at(key);
        }

        void Set(const TKey& key,
                 TValue&& value) {
            Data[key] = std::move(value);
        }

        bool Has(const TKey& key) const {
            return Data.contains(key);
        }

    private:
        THashMap<TKey, TValue> Data;
    };

    template <class TScope, class TKey, class TValue>
    inline TScopedCache<TKey, TValue>* GetCachePtr(const TScope& scope) {
        using TCacheType = TScopedCache<TKey, TValue>;
        const ui64 typeHash = typeid(TCacheType).hash_code();
        auto& ptr = ScopeCaches[scope.GetGuid()][typeHash];
        if (ptr == nullptr) {
            ptr.Reset(new TCacheType());
        }
        return dynamic_cast<TCacheType*>(ptr.Get());
    }

private:
    THashMap<TGUID, TMap<ui64, THolder<IScopedCache>>> ScopeCaches;

public:
    template <class TScope, class TKey, class TBuilder>
    TScopedCacheHolder& CacheOnly(const TScope& scope,
                                  const TKey& key,
                                  TBuilder&& builder) {
        using TValue = decltype(builder());
        auto cachePtr = GetCachePtr<TScope, TKey, TValue>(scope);
        if (!cachePtr->Has(key)) {
            cachePtr->Set(key, builder());
        }
        return *this;
    }

    template <class TScope, class TKey, class TBuilder>
    auto Cache(const TScope& scope,
               const TKey& key,
               TBuilder&& builder) -> decltype(GetCachePtr<TScope, TKey, decltype(builder())>(scope)->Value(key)) {
        using TValue = decltype(builder());
        CacheOnly(scope, key, std::forward<TBuilder>(builder));
        return GetCachePtr<TScope, TKey, TValue>(scope)->Value(key);
    }
};
