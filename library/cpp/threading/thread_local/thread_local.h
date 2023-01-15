#pragma once

#include <util/generic/hash.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/memory/pool.h>
#include <util/system/mutex.h>
#include <util/system/thread.h>

#include <library/cpp/threading/hot_swap/hot_swap.h>
#include <library/cpp/threading/skip_list/skiplist.h>

#include <array>
#include <atomic>
#include <thread>

namespace NThreading {

// TThreadLocalValue
//
// Safe RAII-friendly thread local storage without dirty hacks from util/system/tls
//
// Example 1:
//
// THolder<IThreadPool> pool = CreateThreadPool(threads);
// TThreadLocalValue<ui32> tls;
// for (ui32 i : xrange(threads)) {
//     pool->SafeAddFunc([&]) {
//         *tls->Get() = 1337;
//     }
// }
//
// Example 2:
//
// class TNoisy {
// public:
//     TNoisy(const char* name = "TNoisy")
//         : Name_{name} {
//         printf("%s::%s\n", Name_, Name_);
//     }
//
//     ~TNoisy() {
//         printf("%s::~%s\n", Name_, Name_);
//     }
// private:
//     const char* Name_;
// };
//
// class TWrapper {
// public:
//     TWrapper() {
//         Println(__PRETTY_FUNCTION__);
//     }
//
//     ~TWrapper() {
//         Println(__PRETTY_FUNCTION__);
//     }
//
//     void DoWork() {
//         ThreadLocal_->Get();
//     }
//
// private:
//     TNoisy Noisy_{"TWrapper"};
//     TThreadLocalValue<TNoisy> ThreadLocal_;
// };
//
// THolder<IThreadPool> pool = CreateThreadPool(3);
// {
//     TWrapper wrapper;
//     for (ui32 i : xrange(3)) {
//         pool->SafeAddFunc([&] {
//             wrapper.DoWork();
//         });
//     }
// }
//
// Will always print:
// TWrapper::TWrapper()
// TNoisy::TNoisy()
// TNoisy::TNoisy()
// TNoisy::TNoisy()
// TNoisy::~TNoisy()
// TNoisy::~TNoisy()
// TNoisy::~TNoisy()
// TWrapper::~TWrapper()
//

enum class EThreadLocalImpl {
    HotSwap,
    SkipList,
    ForwardList,
};

namespace NDetail {

template <typename T, EThreadLocalImpl Impl, size_t NumShards>
class TThreadLocalValueImpl;

} // namespace NDetail

inline constexpr size_t DefaultNumShards = 3;

template <typename T, EThreadLocalImpl Impl = EThreadLocalImpl::SkipList, size_t NumShards = DefaultNumShards>
class TThreadLocalValue : private TNonCopyable {
public:
    template <typename ...ConstructArgs>
    T& GetRef(ConstructArgs&& ...args) const {
        return *Get(std::forward<ConstructArgs>(args)...);
    }

    template <typename ...ConstructArgs>
    T* Get(ConstructArgs&& ...args) const {
        TThread::TId tid = TThread::CurrentThreadId();
        return Shards_[tid % NumShards].Get(tid, std::forward<ConstructArgs>(args)...);
    }

private:
    using TStorage = NDetail::TThreadLocalValueImpl<T, Impl, NumShards>;

    mutable std::array<TStorage, NumShards> Shards_;
};

namespace NDetail {

template <typename T, size_t NumShards>
class TThreadLocalValueImpl<T, EThreadLocalImpl::HotSwap, NumShards> {
private:
    class TStorage: public THashMap<TThread::TId, TAtomicSharedPtr<T>>, public TAtomicRefCount<TStorage> {
    };

public:
    TThreadLocalValueImpl() {
        Registered_.AtomicStore(new TStorage());
    }

    template <typename ...ConstructArgs>
    T* Get(TThread::TId tid, ConstructArgs&& ...args) {
        if (TIntrusivePtr<TStorage> state = Registered_.AtomicLoad(); TAtomicSharedPtr<T>* result = state->FindPtr(tid)) {
            return result->Get();
        } else {
            TAtomicSharedPtr<T> value = MakeAtomicShared<T>(std::forward<ConstructArgs>(args)...);
            with_lock(RegisterLock_) {
                TIntrusivePtr<TStorage> oldState = Registered_.AtomicLoad();
                THolder<TStorage> newState = MakeHolder<TStorage>(*oldState);
                (*newState)[tid] = value;
                Registered_.AtomicStore(newState.Release());
            }
            return value.Get();
        }
    }

private:
    THotSwap<TStorage> Registered_;
    TMutex RegisterLock_;
};

template <typename T, size_t NumShards>
class TThreadLocalValueImpl<T, EThreadLocalImpl::SkipList, NumShards> {
private:
    struct TNode {
        TThread::TId Key;
        THolder<T> Value;
    };

    struct TCompare {
        int operator()(const TNode& lhs, const TNode& rhs) const {
            return ::NThreading::TCompare<TThread::TId>{}(lhs.Key, rhs.Key);
        }
    };

public:
    TThreadLocalValueImpl()
        : ListPool_{InitialPoolSize()}
        , SkipList_{ListPool_}
    {}

    template <typename ...ConstructArgs>
    T* Get(TThread::TId tid, ConstructArgs&& ...args) {
        TNode key{tid, {}};
        auto iterator = SkipList_.SeekTo(key);
        if (iterator.IsValid() && iterator.GetValue().Key == key.Key) {
            return iterator.GetValue().Value.Get();
        }

        with_lock (RegisterLock_) {
            SkipList_.Insert({tid, MakeHolder<T>(std::forward<ConstructArgs>(args)...)});
        }
        iterator = SkipList_.SeekTo(key);
        return iterator.GetValue().Value.Get();
    }

private:
    static size_t InitialPoolSize() {
        return std::thread::hardware_concurrency() * (sizeof(T) + sizeof(TThread::TId) + sizeof(void*)) / NumShards;
    }

private:
    static inline constexpr size_t MaxHeight = 6;
    using TCustomSkipList = TSkipList<TNode, TCompare, TMemoryPool, TSizeCounter, MaxHeight>;

    TMemoryPool ListPool_;
    TCustomSkipList SkipList_;
    TAdaptiveLock RegisterLock_;
};

template <typename T, size_t NumShards>
class TThreadLocalValueImpl<T, EThreadLocalImpl::ForwardList, NumShards> {
private:
    struct TNode {
        TThread::TId Key = 0;
        T Value;
        TNode* Next = nullptr;
    };

public:
    TThreadLocalValueImpl()
        : Head_{nullptr}
        , Pool_{0}
    {}

    template <typename ...ConsturctArgs>
    T* Get(TThread::TId tid, ConsturctArgs&& ...args) {
        TNode* node = Head_.load(std::memory_order_relaxed);
        for (; node; node = node->Next) {
            if (node->Key == tid) {
                return &node->Value;
            }
        }

        TNode* newNode = AllocateNode(tid, node, std::forward<ConsturctArgs>(args)...);
        while (!Head_.compare_exchange_weak(node, newNode, std::memory_order_release, std::memory_order_relaxed)) {
            newNode->Next = node;
        }

        return &newNode->Value;
    }

    template <typename ...ConstructArgs>
    TNode* AllocateNode(TThread::TId tid, TNode* next, ConstructArgs&& ...args) {
        TNode* storage = nullptr;
        with_lock(PoolMutex_) {
            storage = Pool_.Allocate<TNode>();
        }
        new (storage) TNode{tid, T{std::forward<ConstructArgs>(args)...}, next};
        return storage;
    }

    ~TThreadLocalValueImpl() {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            TNode* next = nullptr;
            for (TNode* node = Head_.load(); node; node = next) {
                next = node->Next;
                node->~TNode();
            }
        }
    }

private:
    std::atomic<TNode*> Head_;
    TMemoryPool Pool_;
    TMutex PoolMutex_;
};

} // namespace NDetail

} // namespace NThreading
