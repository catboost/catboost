#pragma once

#include "fwd.h"

#include <util/generic/ptr.h>
#include <util/system/atomic.h>
#include <util/system/yassert.h>
#include "lfstack.h"

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
//                   be carifull, IncCount and DecCount can be called on a moved object and
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

        TListNode* volatile Next;
        T Data;
    };

    // using inheritance to be able to use 0 bytes for TCounter when we don't need one
    struct TRootNode: public TCounter {
        TListNode* volatile PushQueue;
        TListNode* volatile PopQueue;
        TListNode* volatile ToDelete;
        TRootNode* volatile NextFree;

        TRootNode()
            : PushQueue(nullptr)
            , PopQueue(nullptr)
            , ToDelete(nullptr)
            , NextFree(nullptr)
        {
        }
        void CopyCounter(TRootNode* x) {
            *(TCounter*)this = *(TCounter*)x;
        }
    };

    static void EraseList(TListNode* n) {
        while (n) {
            TListNode* keepNext = AtomicGet(n->Next);
            delete n;
            n = keepNext;
        }
    }

    TRootNode* volatile JobQueue;
    volatile TAtomic FreememCounter;
    volatile TAtomic FreeingTaskCounter;
    TRootNode* volatile FreePtr;

    void TryToFreeAsyncMemory() {
        TAtomic keepCounter = AtomicAdd(FreeingTaskCounter, 0);
        TRootNode* current = AtomicGet(FreePtr);
        if (current == nullptr)
            return;
        if (AtomicAdd(FreememCounter, 0) == 1) {
            // we are the last thread, try to cleanup
            // check if another thread have cleaned up
            if (keepCounter != AtomicAdd(FreeingTaskCounter, 0)) {
                return;
            }
            if (AtomicCas(&FreePtr, (TRootNode*)nullptr, current)) {
                // free list
                while (current) {
                    TRootNode* p = AtomicGet(current->NextFree);
                    EraseList(AtomicGet(current->ToDelete));
                    delete current;
                    current = p;
                }
                AtomicAdd(FreeingTaskCounter, 1);
            }
        }
    }
    void AsyncRef() {
        AtomicAdd(FreememCounter, 1);
    }
    void AsyncUnref() {
        TryToFreeAsyncMemory();
        AtomicAdd(FreememCounter, -1);
    }
    void AsyncDel(TRootNode* toDelete, TListNode* lst) {
        AtomicSet(toDelete->ToDelete, lst);
        for (;;) {
            AtomicSet(toDelete->NextFree, AtomicGet(FreePtr));
            if (AtomicCas(&FreePtr, toDelete, AtomicGet(toDelete->NextFree)))
                break;
        }
    }
    void AsyncUnref(TRootNode* toDelete, TListNode* lst) {
        TryToFreeAsyncMemory();
        if (AtomicAdd(FreememCounter, -1) == 0) {
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
                    AtomicSet(Tail->Next, newCopy);
                    newCopy = Copy;
                    Copy = nullptr; // do not destroy prev try
                    if (!newTail)
                        newTail = Tail; // tried to invert same list
                    break;
                }
                TListNode* newElem = new TListNode(ptr->Data, newCopy);
                newCopy = newElem;
                ptr = AtomicGet(ptr->Next);
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
        AtomicSet(newRoot->PushQueue, head);
        for (;;) {
            TRootNode* curRoot = AtomicGet(JobQueue);
            AtomicSet(newRoot->PushQueue, head);
            AtomicSet(tail->Next, AtomicGet(curRoot->PushQueue));
            AtomicSet(newRoot->PopQueue, AtomicGet(curRoot->PopQueue));
            newRoot->CopyCounter(curRoot);

            for (TListNode* node = head;; node = AtomicGet(node->Next)) {
                newRoot->IncCount(node->Data);
                if (node == tail)
                    break;
            }

            if (AtomicCas(&JobQueue, newRoot, curRoot)) {
                AsyncUnref(curRoot, nullptr);
                break;
            }
        }
    }

    template <typename TCollection>
    static void FillCollection(TListNode* lst, TCollection* res) {
        while (lst) {
            res->emplace_back(std::move(lst->Data));
            lst = AtomicGet(lst->Next);
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
            lst = AtomicGet(lst->Next);
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
        EraseList(JobQueue->PushQueue);
        EraseList(JobQueue->PopQueue);
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
        TListNode* volatile node = new TListNode(*i);
        TListNode* volatile tail = node;

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
        for (;;) {
            TRootNode* curRoot = AtomicGet(JobQueue);
            TListNode* tail = AtomicGet(curRoot->PopQueue);
            if (tail) {
                // has elems to pop
                if (!newRoot)
                    newRoot = new TRootNode;

                AtomicSet(newRoot->PushQueue, AtomicGet(curRoot->PushQueue));
                AtomicSet(newRoot->PopQueue, AtomicGet(tail->Next));
                newRoot->CopyCounter(curRoot);
                newRoot->DecCount(tail->Data);
                Y_ASSERT(AtomicGet(curRoot->PopQueue) == tail);
                if (AtomicCas(&JobQueue, newRoot, curRoot)) {
                    *data = std::move(tail->Data);
                    AtomicSet(tail->Next, nullptr);
                    AsyncUnref(curRoot, tail);
                    return true;
                }
                continue;
            }
            if (AtomicGet(curRoot->PushQueue) == nullptr) {
                delete newRoot;
                AsyncUnref();
                return false; // no elems to pop
            }

            if (!newRoot)
                newRoot = new TRootNode;
            AtomicSet(newRoot->PushQueue, nullptr);
            listInvertor.DoCopy(AtomicGet(curRoot->PushQueue));
            newRoot->PopQueue = listInvertor.Copy;
            newRoot->CopyCounter(curRoot);
            Y_ASSERT(AtomicGet(curRoot->PopQueue) == nullptr);
            if (AtomicCas(&JobQueue, newRoot, curRoot)) {
                newRoot = nullptr;
                listInvertor.CopyWasUsed();
                AsyncDel(curRoot, AtomicGet(curRoot->PushQueue));
            } else {
                AtomicSet(newRoot->PopQueue, nullptr);
            }
        }
    }
    template <typename TCollection>
    void DequeueAll(TCollection* res) {
        AsyncRef();

        TRootNode* newRoot = new TRootNode;
        TRootNode* curRoot;
        do {
            curRoot = AtomicGet(JobQueue);
        } while (!AtomicCas(&JobQueue, newRoot, curRoot));

        FillCollection(curRoot->PopQueue, res);

        TListNode* toDeleteHead = curRoot->PushQueue;
        TListNode* toDeleteTail = FillCollectionReverse(curRoot->PushQueue, res);
        AtomicSet(curRoot->PushQueue, nullptr);

        if (toDeleteTail) {
            toDeleteTail->Next = curRoot->PopQueue;
        } else {
            toDeleteTail = curRoot->PopQueue;
        }
        AtomicSet(curRoot->PopQueue, nullptr);

        AsyncUnref(curRoot, toDeleteHead);
    }
    bool IsEmpty() {
        AsyncRef();
        TRootNode* curRoot = AtomicGet(JobQueue);
        bool res = AtomicGet(curRoot->PushQueue) == nullptr && AtomicGet(curRoot->PopQueue) == nullptr;
        AsyncUnref();
        return res;
    }
    TCounter GetCounter() {
        AsyncRef();
        TRootNode* curRoot = AtomicGet(JobQueue);
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
