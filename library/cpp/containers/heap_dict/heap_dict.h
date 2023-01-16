#pragma once

#include <util/generic/vector.h>
#include <util/generic/hash.h>
#include <util/generic/maybe.h>

template <class TKey,
          class TPriority,
          class TCompare = TLess<TPriority>,
          class THashFcn = THash<TKey>,
          class TEqualKey = TEqualTo<TKey>>
class THeapDict {
    using THeapItem = std::pair<TKey, TPriority>;

    template <class TValue>
    class TIteratorBase {
    public:
        TIteratorBase(THeapDict<TKey, TPriority, TCompare, THashFcn, TEqualKey>& heapDict, size_t position)
            : HeapDict(&heapDict)
            , Position(position)
        {
            HeapDict->ConsiderHeapifying();
        }

        TValue* operator->() const {
            return &HeapDict->Heap[Position];
        }

        TValue& operator*() const {
            return HeapDict->Heap[Position];
        }

        bool operator==(const TIteratorBase& other) const {
            return Position == other.Position;
        }

        bool operator!=(const TIteratorBase& other) const {
            return !(*this == other);
        }

    protected:
        THeapDict<TKey, TPriority, TCompare, THashFcn, TEqualKey>* HeapDict;
        size_t Position;
    };

    class TIterator: public TIteratorBase<THeapItem> {
        friend class THeapDict;
        using TBase = TIteratorBase<THeapItem>;
        using TBase::HeapDict;
        using TBase::Position;

    public:
        TIterator(THeapDict<TKey, TPriority, TCompare, THashFcn, TEqualKey>& heapDict, size_t position)
            : TBase(heapDict, position)
        {
        }

        THeapItem* operator->() const {
            HeapDict->SetModifiedPosition(Position);
            return TBase::operator->();
        }

        THeapItem& operator*() const {
            HeapDict->SetModifiedPosition(Position);
            return TBase::operator*();
        }
    };

    class TConstIterator: public TIteratorBase<const THeapItem> {
        friend class THeapDict;
        using TBase = TIteratorBase<const THeapItem>;
        using TBase::Position;

    public:
        TConstIterator(THeapDict<TKey, TPriority, TCompare, THashFcn, TEqualKey>& heapDict, size_t position)
            : TBase(heapDict, position)
        {
        }

        TConstIterator& operator++() {
            ++Position;
            return *this;
        }

        TConstIterator operator++(int) {
            auto copy(*this);
            ++Position;
            return copy;
        }

        TConstIterator& operator--() {
            --Position;
            return *this;
        }

        TConstIterator operator--(int) {
            auto copy(*this);
            --Position;
            return copy;
        }
    };

public:
    using value_type = THeapItem;
    using iterator = TIterator;
    using const_iterator = TConstIterator;

public:
    THeapDict() = default;

    THeapDict(TCompare compare, THashFcn hashFcn, TEqualKey equalKey)
        : Compare(compare)
        , PositionsInHeap(0, hashFcn, equalKey)
    {

    }

    THeapItem& top() {
        return *GetFromPosition(0);
    }

    void pop() {
        ConsiderHeapifying();
        PopFromPosition(0);
    }

    void erase(const TKey& key) {
        if (const size_t* position = FindPositionByKey(key)) {
            PopFromPosition(*position);
        }
    }

    void erase(const iterator& it) {
        Y_ASSERT((!ModifiedPosition || *ModifiedPosition == it.Position) && "Logic error");
        ModifiedPosition.Clear();
        PopFromPosition(it.Position);
    }

    void push(const TKey& key, const TPriority& priority) {
        const size_t* position = FindPositionByKey(key);
        if (!position) {
            PushUnique(key, priority);
        }
    }

    void insert(const value_type& item) {
        push(item.first, item.second);
    }

    TPriority& operator[](const TKey& key) {
        const size_t* position = FindPositionByKey(key);
        if (!position) {
            PushUnique(key, TPriority());
            position = FindPositionByKey(key);
        }
        auto heapIt = GetFromPosition(*position);
        return heapIt->second;
    }

    iterator find(const TKey& key) {
        const size_t* position = FindPositionByKey(key);
        if (!position) {
            return end();
        }
        return GetFromPosition(*position);
    }

    iterator begin() {
        return GetFromPosition(0);
    }

    const_iterator cbegin() {
        return GetConstFromPosition(0);
    }

    iterator end() {
        return GetFromPosition(Heap.size());
    }

    const_iterator cend() {
        return GetConstFromPosition(Heap.size());
    }

    size_t size() const {
        return Heap.size();
    }

    bool empty() const {
        return Heap.empty();
    }

private:
    static size_t GetLeftChild(size_t i) {
        return 2 * i + 1;
    }

    static size_t GetRightChild(size_t i) {
        return 2 * i + 2;
    }

    static size_t GetParent(size_t i) {
        return (i - 1) / 2;
    }

    void SwapInHeap(size_t i, size_t j) {
        DoSwap(Heap[i], Heap[j]);
        DoSwap(PositionsInHeap[Heap[i].first], PositionsInHeap[Heap[j].first]);
    }

    void SiftDown(size_t i) {
        for (;;) {
            size_t maxItem = i;
            for (size_t child : {GetLeftChild(i), GetRightChild(i)}) {
                if (child < Heap.size() && Compare(Heap[maxItem].second, Heap[child].second)) {
                    maxItem = child;
                }
            }
            if (maxItem == i) {
                break;
            }
            SwapInHeap(maxItem, i);
            i = maxItem;
        }
    }

    void SiftUp(size_t i) {
        for (; i > 0;) {
            size_t parent = GetParent(i);
            if (Compare(Heap[parent].second, Heap[i].second)) {
                SwapInHeap(parent, i);
                i = parent;
            } else {
                break;
            }
        }
    }

    void Heapify(size_t i) {
        SiftDown(i);
        SiftUp(i);
    }

    iterator GetFromPosition(size_t i) {
        return {*this, i};
    }

    const_iterator GetConstFromPosition(size_t i) {
        return {*this, i};
    }

    void PopFromPosition(size_t i) {
        SwapInHeap(i, Heap.size() - 1);
        PositionsInHeap.erase(Heap.back().first);
        Heap.pop_back();
        if (i != Heap.size()) {
            Heapify(i);
        }
    }

    void PushUnique(const TKey& key, const TPriority& priority) {
        size_t position = Heap.size();
        Heap.emplace_back(key, priority);
        PositionsInHeap[key] = position;
        SiftUp(position);
    }

    void ConsiderHeapifying() {
        if (!ModifiedPosition) {
            return;
        }
        Y_ASSERT(PositionsInHeap[Heap[*ModifiedPosition].first] == *ModifiedPosition && "It's forbidden to modify THeapItem::first (key of element in HeapDict)!");
        Heapify(*ModifiedPosition);
        ModifiedPosition.Clear();
    }

    void SetModifiedPosition(size_t i) {
        Y_ASSERT((!ModifiedPosition || *ModifiedPosition == i) && "Logic error");
        ModifiedPosition = i;
    }

    const size_t* FindPositionByKey(const TKey& key) {
        ConsiderHeapifying();
        return MapFindPtr(PositionsInHeap, key);
    }

private:
    TCompare Compare;
    THashMap<TKey, size_t, THashFcn, TEqualKey> PositionsInHeap;
    TVector<THeapItem> Heap;
    TMaybe<size_t> ModifiedPosition;
};
