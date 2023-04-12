#pragma once

#include "cache.h"

#include <util/generic/singleton.h>
#include <util/system/rwlock.h>

namespace NPrivate {
    // We are interested in getters promotion policy _here_ because of Read-Write-Lock optimizations.
    enum class EGettersPromotionPolicy {
        Promoted,   // LRU, TLRU, MRU, etc.
        Unpromoted  // FIFO, LIFO, LW, etc.
    };

    template <class Key, class Value, template <class, class> class List, EGettersPromotionPolicy GettersPromotionPolicy, class... TArgs>
    class TThreadSafeCache {
    public:
        using TPtr = TAtomicSharedPtr<Value>;

        class ICallbacks {
        public:
            using TKey = Key;
            using TValue = Value;
            using TOwner = TThreadSafeCache<Key, Value, List, GettersPromotionPolicy, TArgs...>;

        public:
            virtual ~ICallbacks() = default;
            virtual TKey GetKey(TArgs... args) const = 0;
            virtual TValue* CreateObject(TArgs... args) const = 0;
        };

    public:
        TThreadSafeCache(const ICallbacks& callbacks, size_t maxSize = Max<size_t>())
            : Callbacks(callbacks)
            , Cache(maxSize)
        {
        }

        bool Insert(const Key& key, const TPtr& value) {
            if (!Contains(key)) {
                TWriteGuard w(Mutex);
                return Cache.Insert(key, value);
            }
            return false;
        }

        void Update(const Key& key, const TPtr& value) {
            TWriteGuard w(Mutex);
            Cache.Update(key, value);
        }

        const TPtr Get(TArgs... args) const {
            return GetValue<true>(args...);
        }

        const TPtr GetUnsafe(TArgs... args) const {
            return GetValue<false>(args...);
        }

        void Clear() {
            TWriteGuard w(Mutex);
            Cache.Clear();
        }

        void Erase(TArgs... args) {
            Key key = Callbacks.GetKey(args...);
            if (!Contains(key)) {
                return;
            }
            TWriteGuard w(Mutex);
            typename TInternalCache::TIterator i = Cache.Find(key);
            if (i == Cache.End()) {
                return;
            }
            Cache.Erase(i);
        }

        bool Contains(const Key& key) const {
            TReadGuard r(Mutex);
            auto iter = Cache.FindWithoutPromote(key);
            return iter != Cache.End();
        }

        template <class TCallbacks>
        static const TPtr Get(TArgs... args) {
            return TThreadSafeCacheSingleton<TCallbacks>::Get(args...);
        }

        template <class TCallbacks>
        static const TPtr Erase(TArgs... args) {
            return TThreadSafeCacheSingleton<TCallbacks>::Erase(args...);
        }

        template <class TCallbacks>
        static void Clear() {
            return TThreadSafeCacheSingleton<TCallbacks>::Clear();
        }

        size_t GetMaxSize() const {
            TReadGuard w(Mutex);
            return Cache.GetMaxSize();
        }

        void SetMaxSize(size_t newSize) {
            TWriteGuard w(Mutex);
            Cache.SetMaxSize(newSize);
        }

    private:
        template <bool AllowNullValues>
        const TPtr GetValue(TArgs... args) const {
            Key key = Callbacks.GetKey(args...);
            switch (GettersPromotionPolicy) {
                case EGettersPromotionPolicy::Promoted:
                    break;
                case EGettersPromotionPolicy::Unpromoted: {
                    TReadGuard r(Mutex);
                    typename TInternalCache::TIterator i = Cache.FindWithoutPromote(key);
                    if (i != Cache.End()) {
                        return i.Value();
                    }
                    break;
                }
            }
            TWriteGuard w(Mutex);
            typename TInternalCache::TIterator i = Cache.Find(key);
            if (i != Cache.End()) {
                return i.Value();
            }
            TPtr value = Callbacks.CreateObject(args...);
            if (value || AllowNullValues) {
                Cache.Insert(key, value);
            }
            return value;
        }

    private:
        using TInternalCache = TCache<Key, TPtr, List<Key, TPtr>, TNoopDelete>;

        template <class TCallbacks>
        class TThreadSafeCacheSingleton {
        public:
            static const TPtr Get(TArgs... args) {
                return Singleton<TThreadSafeCacheSingleton>()->Cache.Get(args...);
            }

            static const TPtr Erase(TArgs... args) {
                return Singleton<TThreadSafeCacheSingleton>()->Cache.Erase(args...);
            }

            static void Clear() {
                return Singleton<TThreadSafeCacheSingleton>()->Cache.Clear();
            }

            TThreadSafeCacheSingleton()
                : Cache(Callbacks)
            {
            }

        private:
            TCallbacks Callbacks;
            typename TCallbacks::TOwner Cache;
        };

    private:
        TRWMutex Mutex;
        const ICallbacks& Callbacks;
        mutable TInternalCache Cache;
    };

    struct TLWHelper {
        template <class TValue>
        struct TConstWeighter {
            static int Weight(const TValue& /*value*/) {
                return 0;
            }
        };

        template <class TKey, class TValue>
        using TListType = TLWList<TKey, TValue, int, TConstWeighter<TValue>>;

        template <class TKey, class TValue, class... TArgs>
        using TCache = TThreadSafeCache<TKey, TValue, TListType, EGettersPromotionPolicy::Unpromoted, TArgs...>;
    };

    struct TLRUHelper {
        template <class TKey, class TValue>
        using TListType = TLRUList<TKey, TValue>;

        template <class TKey, class TValue, class... TArgs>
        using TCache = TThreadSafeCache<TKey, TValue, TListType, EGettersPromotionPolicy::Promoted, TArgs...>;
    };

}

template <class TKey, class TValue, class... TArgs>
using TThreadSafeCache = typename NPrivate::TLWHelper::template TCache<TKey, TValue, TArgs...>;

template <class TKey, class TValue, class... TArgs>
using TThreadSafeLRUCache = typename NPrivate::TLRUHelper::template TCache<TKey, TValue, TArgs...>;

