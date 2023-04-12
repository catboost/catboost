#pragma once

#include "compare.h"

#include <util/generic/algorithm.h>
#include <util/generic/noncopyable.h>
#include <util/generic/typetraits.h>
#include <util/memory/pool.h>
#include <util/random/random.h>
#include <library/cpp/deprecated/atomic/atomic.h>

namespace NThreading {
    ////////////////////////////////////////////////////////////////////////////////

    class TNopCounter {
    protected:
        template <typename T>
        void OnInsert(const T&) {
        }

        template <typename T>
        void OnUpdate(const T&) {
        }

        void Reset() {
        }
    };

    ////////////////////////////////////////////////////////////////////////////////

    class TSizeCounter {
    private:
        size_t Size;

    public:
        TSizeCounter()
            : Size(0)
        {
        }

        size_t GetSize() const {
            return Size;
        }

    protected:
        template <typename T>
        void OnInsert(const T&) {
            ++Size;
        }

        template <typename T>
        void OnUpdate(const T&) {
        }

        void Reset() {
            Size = 0;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Append-only concurrent skip-list
    //
    // Readers do not require any synchronization.
    // Writers should be externally synchronized.
    // Nodes will be allocated using TMemoryPool instance.

    template <
        typename T,
        typename TComparer = TCompare<T>,
        typename TAllocator = TMemoryPool,
        typename TCounter = TSizeCounter,
        int MaxHeight = 12,
        int Branching = 4>
    class TSkipList: public TCounter, private TNonCopyable {
        class TNode {
        private:
            T Value;       // should be immutable after insert
            TNode* Next[]; // variable-size array maximum of MaxHeight values

        public:
            TNode(T&& value)
                : Value(std::move(value))
            {
                Y_UNUSED(Next);
            }

            const T& GetValue() const {
                return Value;
            }

            T& GetValue() {
                return Value;
            }

            TNode* GetNext(int height) const {
                return AtomicGet(Next[height]);
            }

            void Link(int height, TNode** prev) {
                for (int i = 0; i < height; ++i) {
                    Next[i] = prev[i]->Next[i];
                    AtomicSet(prev[i]->Next[i], this);
                }
            }
        };

    public:
        class TIterator {
        private:
            const TSkipList* List;
            const TNode* Node;

        public:
            TIterator()
                : List(nullptr)
                , Node(nullptr)
            {
            }

            TIterator(const TSkipList* list, const TNode* node)
                : List(list)
                , Node(node)
            {
            }

            TIterator(const TIterator& other)
                : List(other.List)
                , Node(other.Node)
            {
            }

            TIterator& operator=(const TIterator& other) {
                List = other.List;
                Node = other.Node;
                return *this;
            }

            void Next() {
                Node = Node ? Node->GetNext(0) : nullptr;
            }

            // much less efficient than Next as our list is single-linked
            void Prev() {
                if (Node) {
                    TNode* node = List->FindLessThan(Node->GetValue(), nullptr);
                    Node = (node != List->Head ? node : nullptr);
                }
            }

            void Reset() {
                Node = nullptr;
            }

            bool IsValid() const {
                return Node != nullptr;
            }

            const T& GetValue() const {
                Y_ASSERT(IsValid());
                return Node->GetValue();
            }
        };

    private:
        TAllocator& Allocator;
        TComparer Comparer;

        TNode* Head;
        TAtomic Height;
        TCounter Counter;

        TNode* Prev[MaxHeight];

        template <typename TValue>
        using TComparerReturnType = std::invoke_result_t<TComparer, const T&, const TValue&>;

    public:
        TSkipList(TAllocator& allocator, const TComparer& comparer = TComparer())
            : Allocator(allocator)
            , Comparer(comparer)
        {
            Init();
        }

        ~TSkipList() {
            CallDtors();
        }

        void Clear() {
            CallDtors();
            Allocator.ClearKeepFirstChunk();
            Init();
        }

        bool Insert(T value) {
            TNode* node = PrepareInsert(value);
            if (Y_UNLIKELY(node && Compare(node, value) == 0)) {
                // we do not allow duplicates
                return false;
            }
            node = DoInsert(std::move(value));
            TCounter::OnInsert(node->GetValue());
            return true;
        }

        template <typename TInsertAction, typename TUpdateAction>
        bool Insert(const T& value, TInsertAction insert, TUpdateAction update) {
            TNode* node = PrepareInsert(value);
            if (Y_UNLIKELY(node && Compare(node, value) == 0)) {
                if (update(node->GetValue())) {
                    TCounter::OnUpdate(node->GetValue());
                    return true;
                }
                // we do not allow duplicates
                return false;
            }
            node = DoInsert(insert(value));
            TCounter::OnInsert(node->GetValue());
            return true;
        }

        template <typename TValue>
        bool Contains(const TValue& value) const {
            TNode* node = FindGreaterThanOrEqual(value);
            return node && Compare(node, value) == 0;
        }

        TIterator SeekToFirst() const {
            return TIterator(this, FindFirst());
        }

        TIterator SeekToLast() const {
            TNode* last = FindLast();
            return TIterator(this, last != Head ? last : nullptr);
        }

        template <typename TValue>
        TIterator SeekTo(const TValue& value) const {
            return TIterator(this, FindGreaterThanOrEqual(value));
        }

    private:
        static int RandomHeight() {
            int height = 1;
            while (height < MaxHeight && (RandomNumber<unsigned int>() % Branching) == 0) {
                ++height;
            }
            return height;
        }

        void Init() {
            Head = AllocateRootNode();
            Height = 1;
            TCounter::Reset();

            for (int i = 0; i < MaxHeight; ++i) {
                Prev[i] = Head;
            }
        }

        void CallDtors() {
            if (!TTypeTraits<T>::IsPod) {
                // we should explicitly call destructors for our nodes
                TNode* node = Head->GetNext(0);
                while (node) {
                    TNode* next = node->GetNext(0);
                    node->~TNode();
                    node = next;
                }
            }
        }

        TNode* AllocateRootNode() {
            size_t size = sizeof(TNode) + sizeof(TNode*) * MaxHeight;
            void* buffer = Allocator.Allocate(size);
            memset(buffer, 0, size);
            return static_cast<TNode*>(buffer);
        }

        TNode* AllocateNode(T&& value, int height) {
            size_t size = sizeof(TNode) + sizeof(TNode*) * height;
            void* buffer = Allocator.Allocate(size);
            memset(buffer, 0, size);
            return new (buffer) TNode(std::move(value));
        }

        TNode* FindFirst() const {
            return Head->GetNext(0);
        }

        TNode* FindLast() const {
            TNode* node = Head;
            int height = AtomicGet(Height) - 1;

            while (true) {
                TNode* next = node->GetNext(height);
                if (next) {
                    node = next;
                    continue;
                }

                if (height) {
                    --height;
                } else {
                    return node;
                }
            }
        }

        template <typename TValue>
        TComparerReturnType<TValue> Compare(const TNode* node, const TValue& value) const {
            return Comparer(node->GetValue(), value);
        }

        template <typename TValue>
        TNode* FindLessThan(const TValue& value, TNode** links) const {
            TNode* node = Head;
            int height = AtomicGet(Height) - 1;

            TNode* prev = nullptr;
            while (true) {
                TNode* next = node->GetNext(height);
                if (next && next != prev) {
                    TComparerReturnType<TValue> cmp = Compare(next, value);
                    if (cmp < 0) {
                        node = next;
                        continue;
                    }
                }

                if (links) {
                    // collect links from upper levels
                    links[height] = node;
                }

                if (height) {
                    prev = next;
                    --height;
                } else {
                    return node;
                }
            }
        }

        template <typename TValue>
        TNode* FindGreaterThanOrEqual(const TValue& value) const {
            TNode* node = Head;
            int height = AtomicGet(Height) - 1;

            TNode* prev = nullptr;
            while (true) {
                TNode* next = node->GetNext(height);
                if (next && next != prev) {
                    TComparerReturnType<TValue> cmp = Compare(next, value);
                    if (cmp < 0) {
                        node = next;
                        continue;
                    }
                    if (cmp == 0) {
                        return next;
                    }
                }

                if (height) {
                    prev = next;
                    --height;
                } else {
                    return next;
                }
            }
        }

        TNode* PrepareInsert(const T& value) {
            TNode* prev = Prev[0];
            TNode* next = prev->GetNext(0);
            if ((prev == Head || Compare(prev, value) < 0) && (next == nullptr || Compare(next, value) >= 0)) {
                // avoid seek in case of sequential insert
            } else {
                prev = FindLessThan(value, Prev);
                next = prev->GetNext(0);
            }
            return next;
        }

        TNode* DoInsert(T&& value) {
            // choose level to place new node
            int currentHeight = AtomicGet(Height);
            int height = RandomHeight();
            if (height > currentHeight) {
                for (int i = currentHeight; i < height; ++i) {
                    // head should link to all levels
                    Prev[i] = Head;
                }
                AtomicSet(Height, height);
            }

            TNode* node = AllocateNode(std::move(value), height);
            node->Link(height, Prev);

            // keep last inserted node to optimize sequential inserts
            for (int i = 0; i < height; i++) {
                Prev[i] = node;
            }
            return node;
        }
    };

}
