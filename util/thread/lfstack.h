#pragma once

#include <util/generic/noncopyable.h>

#include <atomic>
#include <cstddef>
#include <utility>

//////////////////////////////
// lock free lifo stack
template <class T>
class TLockFreeStack: TNonCopyable {
    struct TNode {
        T Value;
        std::atomic<TNode*> Next;

        TNode() = default;

        template <class U>
        explicit TNode(U&& val)
            : Value(std::forward<U>(val))
            , Next(nullptr)
        {
        }
    };

    std::atomic<TNode*> Head = nullptr;
    std::atomic<TNode*> FreePtr = nullptr;
    std::atomic<size_t> DequeueCount = 0;

    void TryToFreeMemory() {
        TNode* current = FreePtr.load(std::memory_order_acquire);
        if (!current)
            return;
        if (DequeueCount.load() == 1) {
            // node current is in free list, we are the last thread so try to cleanup
            if (FreePtr.compare_exchange_strong(current, nullptr))
                EraseList(current);
        }
    }
    void EraseList(TNode* p) {
        while (p) {
            TNode* next = p->Next;
            delete p;
            p = next;
        }
    }
    void EnqueueImpl(TNode* head, TNode* tail) {
        auto headValue = Head.load(std::memory_order_acquire);
        for (;;) {
            tail->Next.store(headValue, std::memory_order_release);
            // NB. See https://en.cppreference.com/w/cpp/atomic/atomic/compare_exchange
            // The weak forms (1-2) of the functions are allowed to fail spuriously, that is,
            // act as if *this != expected even if they are equal.
            // When a compare-and-exchange is in a loop, the weak version will yield better
            // performance on some platforms.
            if (Head.compare_exchange_weak(headValue, head))
                break;
        }
    }
    template <class U>
    void EnqueueImpl(U&& u) {
        TNode* node = new TNode(std::forward<U>(u));
        EnqueueImpl(node, node);
    }

public:
    TLockFreeStack() = default;

    ~TLockFreeStack() {
        EraseList(Head.load());
        EraseList(FreePtr.load());
    }

    void Enqueue(const T& t) {
        EnqueueImpl(t);
    }

    void Enqueue(T&& t) {
        EnqueueImpl(std::move(t));
    }

    template <typename TCollection>
    void EnqueueAll(const TCollection& data) {
        EnqueueAll(data.begin(), data.end());
    }
    template <typename TIter>
    void EnqueueAll(TIter dataBegin, TIter dataEnd) {
        if (dataBegin == dataEnd) {
            return;
        }
        TIter i = dataBegin;
        TNode* node = new TNode(*i);
        TNode* tail = node;

        for (++i; i != dataEnd; ++i) {
            TNode* nextNode = node;
            node = new TNode(*i);
            node->Next.store(nextNode, std::memory_order_release);
        }
        EnqueueImpl(node, tail);
    }
    bool Dequeue(T* res) {
        ++DequeueCount;
        for (TNode* current = Head.load(std::memory_order_acquire); current;) {
            if (Head.compare_exchange_weak(current, current->Next.load(std::memory_order_acquire))) {
                *res = std::move(current->Value);
                // delete current; // ABA problem
                // even more complex node deletion
                TryToFreeMemory();
                if (--DequeueCount == 0) {
                    // no other Dequeue()s, can safely reclaim memory
                    delete current;
                } else {
                    // Dequeue()s in progress, put node to free list
                    for (TNode* freePtr = FreePtr.load(std::memory_order_acquire);;) {
                        current->Next.store(freePtr, std::memory_order_release);
                        if (FreePtr.compare_exchange_weak(freePtr, current))
                            break;
                    }
                }
                return true;
            }
        }
        TryToFreeMemory();
        --DequeueCount;
        return false;
    }
    // add all elements to *res
    // elements are returned in order of dequeue (top to bottom; see example in unittest)
    template <typename TCollection>
    void DequeueAll(TCollection* res) {
        ++DequeueCount;
        for (TNode* current = Head.load(std::memory_order_acquire); current;) {
            if (Head.compare_exchange_weak(current, nullptr)) {
                for (TNode* x = current; x;) {
                    res->push_back(std::move(x->Value));
                    x = x->Next;
                }
                // EraseList(current); // ABA problem
                // even more complex node deletion
                TryToFreeMemory();
                if (--DequeueCount == 0) {
                    // no other Dequeue()s, can safely reclaim memory
                    EraseList(current);
                } else {
                    // Dequeue()s in progress, add nodes list to free list
                    TNode* currentLast = current;
                    while (currentLast->Next) {
                        currentLast = currentLast->Next;
                    }
                    for (TNode* freePtr = FreePtr.load(std::memory_order_acquire);;) {
                        currentLast->Next.store(freePtr, std::memory_order_release);
                        if (FreePtr.compare_exchange_weak(freePtr, current))
                            break;
                    }
                }
                return;
            }
        }
        TryToFreeMemory();
        --DequeueCount;
    }
    bool DequeueSingleConsumer(T* res) {
        for (TNode* current = Head.load(std::memory_order_acquire); current;) {
            if (Head.compare_exchange_weak(current, current->Next)) {
                *res = std::move(current->Value);
                delete current; // with single consumer thread ABA does not happen
                return true;
            }
        }
        return false;
    }
    // add all elements to *res
    // elements are returned in order of dequeue (top to bottom; see example in unittest)
    template <typename TCollection>
    void DequeueAllSingleConsumer(TCollection* res) {
        for (TNode* head = Head.load(std::memory_order_acquire); head;) {
            if (Head.compare_exchange_weak(head, nullptr)) {
                for (TNode* x = head; x;) {
                    res->push_back(std::move(x->Value));
                    x = x->Next;
                }
                EraseList(head); // with single consumer thread ABA does not happen
                return;
            }
        }
    }
    bool IsEmpty() {
        return Head.load() == nullptr; // without lock, so result is approximate
    }
};
