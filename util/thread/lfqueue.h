#pragma once

#include "fwd.h"
#include "lfstack.h"

#include <util/generic/ptr.h>
#include <util/system/yassert.h>

#include <atomic>

struct TDefaultLFCounter {
    template <class T>
    void IncCount(const T& data) {
        (void)data;
    }
    template <class T>
    void DecCount(const T& data) {
        (void)data;
    }
};

// @brief lockfree queue
// @tparam T - the queue element, should be movable
// @tparam TCounter, a observer class to count number of items in queue
//                   be careful, IncCount and DecCount can be called on a moved object and
//                   it is TCounter class responsibility to check validity of passed object
template <class T, class TCounter>
class TLockFreeQueue: public TNonCopyable {
    struct TListNode {
        template <typename U>
        TListNode(U&& u, TListNode* next)
            : Next(next)
            , Data(std::forward<U>(u))
        {
        }

        template <typename U>
        explicit TListNode(U&& u)
            : Data(std::forward<U>(u))
        {
        }

        std::atomic<TListNode*> Next;
        T Data;
    };

    // using inheritance to be able to use 0 bytes for TCounter when we don't need one
    struct TRootNode: public TCounter {
        std::atomic<TListNode*> PushQueue = nullptr;
        std::atomic<TListNode*> PopQueue = nullptr;
        std::atomic<TListNode*> ToDelete = nullptr;
        std::atomic<TRootNode*> NextFree = nullptr;

        void CopyCounter(TRootNode* x) {
            *(TCounter*)this = *(TCounter*)x;
        }
    };

    static void EraseList(TListNode* n) {
        while (n) {
            TListNode* keepNext = n->Next.load(std::memory_order_acquire);
            delete n;
            n = keepNext;
        }
    }

    alignas(64) std::atomic<TRootNode*> JobQueue;
    alignas(64) std::atomic<size_t> FreememCounter;
    alignas(64) std::atomic<size_t> FreeingTaskCounter;
    alignas(64) std::atomic<TRootNode*> FreePtr;

    void TryToFreeAsyncMemory() {
        const auto keepCounter = FreeingTaskCounter.load();
        TRootNode* current = FreePtr.load(std::memory_order_acquire);
        if (current == nullptr)
            return;
        if (FreememCounter.load() == 1) {
            // we are the last thread, try to cleanup
            // check if another thread have cleaned up
            if (keepCounter != FreeingTaskCounter.load()) {
                return;
            }
            if (FreePtr.compare_exchange_strong(current, nullptr)) {
                // free list
                while (current) {
                    TRootNode* p = current->NextFree.load(std::memory_order_acquire);
                    EraseList(current->ToDelete.load(std::memory_order_acquire));
                    delete current;
                    current = p;
                }
                ++FreeingTaskCounter;
            }
        }
    }
    void AsyncRef() {
        ++FreememCounter;
    }
    void AsyncUnref() {
        TryToFreeAsyncMemory();
        --FreememCounter;
    }
    void AsyncDel(TRootNode* toDelete, TListNode* lst) {
        toDelete->ToDelete.store(lst, std::memory_order_release);
        for (auto freePtr = FreePtr.load();;) {
            toDelete->NextFree.store(freePtr, std::memory_order_release);
            if (FreePtr.compare_exchange_weak(freePtr, toDelete))
                break;
        }
    }
    void AsyncUnref(TRootNode* toDelete, TListNode* lst) {
        TryToFreeAsyncMemory();
        if (--FreememCounter == 0) {
            // no other operations in progress, can safely reclaim memory
            EraseList(lst);
            delete toDelete;
        } else {
            // Dequeue()s in progress, put node to free list
            AsyncDel(toDelete, lst);
        }
    }

    struct TListInvertor {
        TListNode* Copy;
        TListNode* Tail;
        TListNode* PrevFirst;

        TListInvertor()
            : Copy(nullptr)
            , Tail(nullptr)
            , PrevFirst(nullptr)
        {
        }
        ~TListInvertor() {
            EraseList(Copy);
        }
        void CopyWasUsed() {
            Copy = nullptr;
            Tail = nullptr;
            PrevFirst = nullptr;
        }
        void DoCopy(TListNode* ptr) {
            TListNode* newFirst = ptr;
            TListNode* newCopy = nullptr;
            TListNode* newTail = nullptr;
            while (ptr) {
                if (ptr == PrevFirst) {
                    // short cut, we have copied this part already
                    Tail->Next.store(newCopy, std::memory_order_release);
                    newCopy = Copy;
                    Copy = nullptr; // do not destroy prev try
                    if (!newTail)
                        newTail = Tail; // tried to invert same list
                    break;
                }
                TListNode* newElem = new TListNode(ptr->Data, newCopy);
                newCopy = newElem;
                ptr = ptr->Next.load(std::memory_order_acquire);
                if (!newTail)
                    newTail = newElem;
            }
            EraseList(Copy); // copy was useless
            Copy = newCopy;
            PrevFirst = newFirst;
            Tail = newTail;
        }
    };

    void EnqueueImpl(TListNode* head, TListNode* tail) {
        TRootNode* newRoot = new TRootNode;
        AsyncRef();
        newRoot->PushQueue.store(head, std::memory_order_release);
        for (TRootNode* curRoot = JobQueue.load(std::memory_order_acquire);;) {
            tail->Next.store(curRoot->PushQueue.load(std::memory_order_acquire), std::memory_order_release);
            newRoot->PopQueue.store(curRoot->PopQueue.load(std::memory_order_acquire), std::memory_order_release);
            newRoot->CopyCounter(curRoot);

            for (TListNode* node = head;; node = node->Next.load(std::memory_order_acquire)) {
                newRoot->IncCount(node->Data);
                if (node == tail)
                    break;
            }

            if (JobQueue.compare_exchange_weak(curRoot, newRoot)) {
                AsyncUnref(curRoot, nullptr);
                break;
            }
        }
    }

    template <typename TCollection>
    static void FillCollection(TListNode* lst, TCollection* res) {
        while (lst) {
            res->emplace_back(std::move(lst->Data));
            lst = lst->Next.load(std::memory_order_acquire);
        }
    }

    /** Traverses a given list simultaneously creating its inversed version.
     *  After that, fills a collection with a reversed version and returns the last visited lst's node.
     */
    template <typename TCollection>
    static TListNode* FillCollectionReverse(TListNode* lst, TCollection* res) {
        if (!lst) {
            return nullptr;
        }

        TListNode* newCopy = nullptr;
        do {
            TListNode* newElem = new TListNode(std::move(lst->Data), newCopy);
            newCopy = newElem;
            lst = lst->Next.load(std::memory_order_acquire);
        } while (lst);

        FillCollection(newCopy, res);
        EraseList(newCopy);

        return lst;
    }

public:
    TLockFreeQueue()
        : JobQueue(new TRootNode)
        , FreememCounter(0)
        , FreeingTaskCounter(0)
        , FreePtr(nullptr)
    {
    }
    ~TLockFreeQueue() {
        AsyncRef();
        AsyncUnref(); // should free FreeList
        EraseList(JobQueue.load(std::memory_order_relaxed)->PushQueue.load(std::memory_order_relaxed));
        EraseList(JobQueue.load(std::memory_order_relaxed)->PopQueue.load(std::memory_order_relaxed));
        delete JobQueue;
    }
    template <typename U>
    void Enqueue(U&& data) {
        TListNode* newNode = new TListNode(std::forward<U>(data));
        EnqueueImpl(newNode, newNode);
    }
    void Enqueue(T&& data) {
        TListNode* newNode = new TListNode(std::move(data));
        EnqueueImpl(newNode, newNode);
    }
    void Enqueue(const T& data) {
        TListNode* newNode = new TListNode(data);
        EnqueueImpl(newNode, newNode);
    }
    template <typename TCollection>
    void EnqueueAll(const TCollection& data) {
        EnqueueAll(data.begin(), data.end());
    }
    template <typename TIter>
    void EnqueueAll(TIter dataBegin, TIter dataEnd) {
        if (dataBegin == dataEnd)
            return;

        TIter i = dataBegin;
        TListNode* node = new TListNode(*i);
        TListNode* tail = node;

        for (++i; i != dataEnd; ++i) {
            TListNode* nextNode = node;
            node = new TListNode(*i, nextNode);
        }
        EnqueueImpl(node, tail);
    }
    bool Dequeue(T* data) {
        TRootNode* newRoot = nullptr;
        TListInvertor listInvertor;
        AsyncRef();
        for (TRootNode* curRoot = JobQueue.load(std::memory_order_acquire);;) {
            TListNode* tail = curRoot->PopQueue.load(std::memory_order_acquire);
            if (tail) {
                // has elems to pop
                if (!newRoot)
                    newRoot = new TRootNode;

                newRoot->PushQueue.store(curRoot->PushQueue.load(std::memory_order_acquire), std::memory_order_release);
                newRoot->PopQueue.store(tail->Next.load(std::memory_order_acquire), std::memory_order_release);
                newRoot->CopyCounter(curRoot);
                newRoot->DecCount(tail->Data);
                Y_ASSERT(curRoot->PopQueue.load() == tail);
                if (JobQueue.compare_exchange_weak(curRoot, newRoot)) {
                    *data = std::move(tail->Data);
                    tail->Next.store(nullptr, std::memory_order_release);
                    AsyncUnref(curRoot, tail);
                    return true;
                }
                continue;
            }
            if (curRoot->PushQueue.load(std::memory_order_acquire) == nullptr) {
                delete newRoot;
                AsyncUnref();
                return false; // no elems to pop
            }

            if (!newRoot)
                newRoot = new TRootNode;
            newRoot->PushQueue.store(nullptr, std::memory_order_release);
            listInvertor.DoCopy(curRoot->PushQueue.load(std::memory_order_acquire));
            newRoot->PopQueue.store(listInvertor.Copy, std::memory_order_release);
            newRoot->CopyCounter(curRoot);
            Y_ASSERT(curRoot->PopQueue.load() == nullptr);
            if (JobQueue.compare_exchange_weak(curRoot, newRoot)) {
                AsyncDel(curRoot, curRoot->PushQueue.load(std::memory_order_acquire));
                curRoot = newRoot;
                newRoot = nullptr;
                listInvertor.CopyWasUsed();
            } else {
                newRoot->PopQueue.store(nullptr, std::memory_order_release);
            }
        }
    }
    template <typename TCollection>
    void DequeueAll(TCollection* res) {
        AsyncRef();

        TRootNode* newRoot = new TRootNode;
        TRootNode* curRoot = JobQueue.load(std::memory_order_acquire);
        do {
        } while (!JobQueue.compare_exchange_weak(curRoot, newRoot));

        FillCollection(curRoot->PopQueue, res);

        TListNode* toDeleteHead = curRoot->PushQueue;
        TListNode* toDeleteTail = FillCollectionReverse(curRoot->PushQueue, res);
        curRoot->PushQueue.store(nullptr, std::memory_order_release);

        if (toDeleteTail) {
            toDeleteTail->Next.store(curRoot->PopQueue.load());
        } else {
            toDeleteTail = curRoot->PopQueue;
        }
        curRoot->PopQueue.store(nullptr, std::memory_order_release);

        AsyncUnref(curRoot, toDeleteHead);
    }
    bool IsEmpty() {
        AsyncRef();
        TRootNode* curRoot = JobQueue.load(std::memory_order_acquire);
        bool res = curRoot->PushQueue.load(std::memory_order_acquire) == nullptr &&
                   curRoot->PopQueue.load(std::memory_order_acquire) == nullptr;
        AsyncUnref();
        return res;
    }
    TCounter GetCounter() {
        AsyncRef();
        TRootNode* curRoot = JobQueue.load(std::memory_order_acquire);
        TCounter res = *(TCounter*)curRoot;
        AsyncUnref();
        return res;
    }
};

template <class T, class TCounter>
class TAutoLockFreeQueue {
public:
    using TRef = THolder<T>;

    inline ~TAutoLockFreeQueue() {
        TRef tmp;

        while (Dequeue(&tmp)) {
        }
    }

    inline bool Dequeue(TRef* t) {
        T* res = nullptr;

        if (Queue.Dequeue(&res)) {
            t->Reset(res);

            return true;
        }

        return false;
    }

    inline void Enqueue(TRef& t) {
        Queue.Enqueue(t.Get());
        Y_UNUSED(t.Release());
    }

    inline void Enqueue(TRef&& t) {
        Queue.Enqueue(t.Get());
        Y_UNUSED(t.Release());
    }

    inline bool IsEmpty() {
        return Queue.IsEmpty();
    }

    inline TCounter GetCounter() {
        return Queue.GetCounter();
    }

private:
    TLockFreeQueue<T*, TCounter> Queue;
};
