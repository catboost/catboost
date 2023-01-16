#pragma once

#include <util/generic/algorithm.h>
#include <util/generic/ptr.h>
#include <util/generic/intrlist.h>
#include <util/generic/hash_set.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <utility>

template <class TValue>
struct TUniformSizeProvider {
    size_t operator()(const TValue&) {
        return 1;
    }
};

template <typename TKey, typename TValue, class TSizeProvider = TUniformSizeProvider<TValue>>
class TLRUList {
public:
    TLRUList(size_t maxSize, const TSizeProvider& sizeProvider = TSizeProvider())
        : List()
        , SizeProvider(sizeProvider)
        , ItemsAmount(0)
        , TotalSize(0)
        , MaxSize(maxSize)
    {
    }

public:
    struct TItem: public TIntrusiveListItem<TItem> {
        typedef TIntrusiveListItem<TItem> TBase;
        TItem(const TKey& key, const TValue& value = TValue())
            : TBase()
            , Key(key)
            , Value(value)
        {
        }

        TItem(const TItem& rhs)
            : TBase()
            , Key(rhs.Key)
            , Value(rhs.Value)
        {
        }

        bool operator<(const TItem& rhs) const {
            return Key < rhs.Key;
        }

        bool operator==(const TItem& rhs) const {
            return Key == rhs.Key;
        }

        TKey Key;
        TValue Value;

        struct THash {
            size_t operator()(const TItem& item) const {
                return ::THash<TKey>()(item.Key);
            }
        };
    };

public:
    TItem* Insert(TItem* item) {
        List.PushBack(item);
        ++ItemsAmount;
        TotalSize += SizeProvider(item->Value);

        return RemoveIfOverflown();
    }

    TItem* RemoveIfOverflown() {
        TItem* deleted = nullptr;
        if (TotalSize > MaxSize && ItemsAmount > 1) {
            deleted = GetOldest();
            Erase(deleted);
        }
        return deleted;
    }

    TItem* GetOldest() {
        typename TListType::TIterator it = List.Begin();
        Y_ASSERT(it != List.End());
        return &*it;
    }

    void Erase(TItem* item) {
        item->Unlink();
        --ItemsAmount;
        TotalSize -= SizeProvider(item->Value);
    }

    void Promote(TItem* item) {
        item->Unlink();
        List.PushBack(item);
    }

    size_t GetSize() const {
        return ItemsAmount;
    }

    size_t GetTotalSize() const {
        return TotalSize;
    }

    size_t GetMaxSize() const {
        return MaxSize;
    }

    // It does not remove current items if newSize is less than TotalSize.
    // Caller should use RemoveIfOverflown to clean up list in this case
    void SetMaxSize(size_t newSize) {
        MaxSize = newSize;
    }

private:
    typedef TIntrusiveList<TItem> TListType;
    TListType List;
    TSizeProvider SizeProvider;
    size_t ItemsAmount;
    size_t TotalSize;
    size_t MaxSize;
};

template <typename TKey, typename TValue>
class TLFUList {
public:
    TLFUList(size_t maxSize)
        : List()
        , ListSize(0)
        , MaxSize(maxSize)
    {
    }

    struct TItem: public TIntrusiveListItem<TItem> {
        typedef TIntrusiveListItem<TItem> TBase;
        TItem(const TKey& key, const TValue& value = TValue())
            : TBase()
            , Key(key)
            , Value(value)
            , Counter(0)
        {
        }

        TItem(const TItem& rhs)
            : TBase()
            , Key(rhs.Key)
            , Value(rhs.Value)
            , Counter(rhs.Counter)
        {
        }

        bool operator<(const TItem& rhs) const {
            return Key < rhs.Key;
        }

        bool operator==(const TItem& rhs) const {
            return Key == rhs.Key;
        }

        TKey Key;
        TValue Value;
        size_t Counter;

        struct THash {
            size_t operator()(const TItem& item) const {
                return ::THash<TKey>()(item.Key);
            }
        };
    };

public:
    TItem* Insert(TItem* item) {
        List.PushBack(item); // give a chance for promotion
        ++ListSize;

        return RemoveIfOverflown();
    }

    TItem* RemoveIfOverflown() {
        TItem* deleted = nullptr;
        if (ListSize > MaxSize) {
            deleted = GetLeastFrequentlyUsed();
            Erase(deleted);
        }
        return deleted;
    }

    TItem* GetLeastFrequentlyUsed() {
        typename TListType::TIterator it = List.Begin();
        Y_ASSERT(it != List.End());
        return &*it;
    }

    void Erase(TItem* item) {
        item->Unlink();
        --ListSize;
    }

    void Promote(TItem* item) {
        size_t counter = ++item->Counter;
        typename TListType::TIterator it = item;
        while (it != List.End() && counter >= it->Counter) {
            ++it;
        }
        item->LinkBefore(&*it);
    }

    size_t GetSize() const {
        return ListSize;
    }

    size_t GetMaxSize() const {
        return MaxSize;
    }

    // It does not remove current items if newSize is less than TotalSize.
    // Caller should use RemoveIfOverflown to clean up list in this case
    void SetMaxSize(size_t newSize) {
        MaxSize = newSize;
    }

private:
    typedef TIntrusiveList<TItem> TListType;
    TListType List;
    size_t ListSize;
    size_t MaxSize;
};

// Least Weighted list
// discards the least weighted items first
// doesn't support promotion
template <typename TKey, typename TValue, typename TWeight, typename TWeighter>
class TLWList {
public:
    TLWList(size_t maxSize)
        : Size(0)
        , MaxSize(maxSize)
    {
    }

    struct TItem {
        TItem(const TKey& key, const TValue& value = TValue())
            : Key(key)
            , Value(value)
            , Weight(TWeighter::Weight(value))
        {
        }

        TItem(const TItem& rhs)
            : Key(rhs.Key)
            , Value(rhs.Value)
            , Weight(rhs.Weight)
        {
        }

        bool operator<(const TItem& rhs) const {
            return Key < rhs.Key;
        }

        bool operator==(const TItem& rhs) const {
            return Key == rhs.Key;
        }

        TKey Key;
        TValue Value;
        TWeight Weight;

        struct THash {
            size_t operator()(const TItem& item) const {
                return ::THash<TKey>()(item.Key);
            }
        };
    };

    struct THeapComparator {
        bool operator()(TItem* const item1, TItem* const item2) const {
            return item1->Weight > item2->Weight;
        }
    };

public:
    TItem* Insert(TItem* item) {
        FixHeap();

        if (Size >= MaxSize && item->Weight < GetLightest()->Weight) {
            return item;
        }

        Heap.push_back(item);
        PushHeap(Heap.begin(), Heap.end(), THeapComparator());
        ++Size;

        return RemoveIfOverflown();
    }

    TItem* RemoveIfOverflown() {
        if (Size <= MaxSize) {
            return nullptr;
        }

        auto lightest = GetLightest();
        Erase(lightest);
        PopHeap(Heap.begin(), Heap.end(), THeapComparator());
        return lightest;
    }

    TItem* GetLightest() {
        FixHeap();

        Y_ASSERT(!Heap.empty());

        return Heap.front();
    }

    // This method doesn't remove items from the heap.
    // Erased items are stored in Removed set
    // and will be deleted on-access (using FixHeap method)
    void Erase(TItem* item) {
        Y_ASSERT(Size > 0);

        --Size;
        Removed.insert(item);
    }

    void Promote(TItem*) {
        // do nothing
    }

    [[nodiscard]] size_t GetSize() const {
        return Size;
    }

    size_t GetMaxSize() const {
        return MaxSize;
    }

    // It does not remove current items if newSize is less than TotalSize.
    // Caller should use RemoveIfOverflown to clean up list in this case
    void SetMaxSize(size_t newSize) {
        MaxSize = newSize;
    }

    void Clear() {
        Heap.clear();
        Removed.clear();
        Size = 0;
    }

private:
    // Physically remove erased elements from the heap
    void FixHeap() {
        if (Removed.empty()) {
            return;
        }

        Heap.erase(std::remove_if(Heap.begin(), Heap.end(), [this](TItem* item) {
                       return this->Removed.contains(item);
                   }),
                   Heap.end());
        MakeHeap(Heap.begin(), Heap.end(), THeapComparator());
        Removed.clear();
        Size = Heap.size();
    }

private:
    TVector<TItem*> Heap;
    THashSet<TItem*> Removed;

    size_t Size;
    size_t MaxSize;
};

template <typename TKey, typename TValue, typename TListType, typename TDeleter>
class TCache {
    typedef typename TListType::TItem TItem;
    typedef typename TItem::THash THash;
    typedef THashMultiSet<TItem, THash> TIndex;
    typedef typename TIndex::iterator TIndexIterator;
    typedef typename TIndex::const_iterator TIndexConstIterator;

public:
    class TIterator {
    public:
        explicit TIterator(const TIndexConstIterator& iter)
            : Iter(iter)
        {
        }

        TValue& operator*() {
            return const_cast<TValue&>(Iter->Value);
        }

        TValue* operator->() {
            return const_cast<TValue*>(&Iter->Value);
        }

        bool operator==(const TIterator& rhs) const {
            return Iter == rhs.Iter;
        }

        bool operator!=(const TIterator& rhs) const {
            return Iter != rhs.Iter;
        }

        TIterator& operator++() {
            ++Iter;
            return *this;
        }

        const TKey& Key() const {
            return Iter->Key;
        }

        const TValue& Value() const {
            return Iter->Value;
        }

        friend class TCache<TKey, TValue, TListType, TDeleter>;

    private:
        TIndexConstIterator Iter;
    };

    TCache(TListType&& list, bool multiValue = false)
        : Index()
        , List(std::move(list))
        , MultiValue(multiValue)
    {
    }

    ~TCache() {
        Clear();
    }

    size_t Size() const {
        return Index.size();
    }

    TIterator Begin() const {
        return TIterator(Index.begin());
    }

    TIterator End() const {
        return TIterator(Index.end());
    }

    TIterator Find(const TKey& key) {
        TIndexIterator it = Index.find(TItem(key));
        if (it != Index.end())
            List.Promote(const_cast<TItem*>(&*it));
        return TIterator(it);
    }

    TIterator FindWithoutPromote(const TKey& key) const {
        return TIterator(Index.find(TItem(key)));
    }

    // note: it shouldn't touch 'value' if it returns false.
    bool PickOut(const TKey& key, TValue* value) {
        Y_ASSERT(value);
        TIndexIterator it = Index.find(TItem(key));
        if (it == Index.end())
            return false;
        *value = it->Value;
        List.Erase(const_cast<TItem*>(&*it));
        Index.erase(it);
        Y_ASSERT(Index.size() == List.GetSize());
        return true;
    }

    bool Insert(const std::pair<TKey, TValue>& p) {
        return Insert(p.first, p.second);
    }

    bool Insert(const TKey& key, const TValue& value) {
        TItem tmpItem(key, value);
        if (!MultiValue && Index.find(tmpItem) != Index.end())
            return false;
        TIndexIterator it = Index.insert(tmpItem);

        TItem* insertedItem = const_cast<TItem*>(&*it);
        auto removedItem = List.Insert(insertedItem);
        auto insertedWasRemoved = removedItem == insertedItem;
        if (removedItem) {
            EraseFromIndex(removedItem);
            while ((removedItem = List.RemoveIfOverflown())) {
                insertedWasRemoved = insertedWasRemoved || insertedItem == removedItem;
                EraseFromIndex(removedItem);
            }
        }

        Y_ASSERT(Index.size() == List.GetSize());
        return !insertedWasRemoved;
    }

    void Update(const TKey& key, const TValue& value) {
        if (MultiValue)
            ythrow yexception() << "TCache: can't \"Update\" in multicache";
        TIterator it = Find(key);
        if (it != End()) {
            Erase(it);
        }
        Insert(key, value);

        Y_ASSERT(Index.size() == List.GetSize());
    }

    void Erase(TIterator it) {
        TItem* item = const_cast<TItem*>(&*it.Iter);
        List.Erase(item);
        TDeleter::Destroy(item->Value);
        Index.erase(it.Iter);

        Y_ASSERT(Index.size() == List.GetSize());
    }

    bool Empty() const {
        return Index.empty();
    }

    void Clear() {
        for (TIndexIterator it = Index.begin(); it != Index.end(); ++it) {
            TItem* item = const_cast<TItem*>(&*it);
            List.Erase(item);
            TDeleter::Destroy(item->Value);
        }
        Y_ASSERT(List.GetSize() == 0);
        Index.clear();
    }

    void SetMaxSize(size_t newSize) {
        List.SetMaxSize(newSize);

        TItem* removedItem = nullptr;
        while ((removedItem = List.RemoveIfOverflown())) {
            EraseFromIndex(removedItem);
        }
        Y_ASSERT(Index.size() == List.GetSize());
    }

    size_t GetMaxSize() const {
        return List.GetMaxSize();
    }

protected:
    TIndex Index;
    TListType List;
    bool MultiValue;

    TIterator FindByItem(TItem* item) {
        std::pair<TIndexIterator, TIndexIterator> p = Index.equal_range(*item);
        // we have to delete the exact unlinked item (there may be multiple items for one key)
        TIndexIterator it;
        for (it = p.first; it != p.second; ++it)
            if (&*it == item)
                break;
        return (it == p.second ? End() : TIterator(it));
    }

    void EraseFromIndex(TItem* item) {
        TDeleter::Destroy(item->Value);
        TIterator it = FindByItem(item);
        Y_ASSERT(it != End());
        Index.erase(it.Iter);
    }
};

struct TNoopDelete {
    template <class T>
    static inline void Destroy(const T&) noexcept {
    }
};

template <typename TKey, typename TValue, typename TDeleter = TNoopDelete, class TSizeProvider = TUniformSizeProvider<TValue>>
class TLRUCache: public TCache<TKey, TValue, TLRUList<TKey, TValue, TSizeProvider>, TDeleter> {
    using TListType = TLRUList<TKey, TValue, TSizeProvider>;
    typedef TCache<TKey, TValue, TListType, TDeleter> TBase;

public:
    TLRUCache(size_t maxSize, bool multiValue = false, const TSizeProvider& sizeProvider = TSizeProvider())
        : TBase(TListType(maxSize, sizeProvider), multiValue)
    {
    }

public:
    typedef typename TBase::TIterator TIterator;

    TValue& GetOldest() {
        return TBase::List.GetOldest()->Value;
    }

    TIterator FindOldest() {
        return TBase::Empty() ? TBase::End() : this->FindByItem(TBase::List.GetOldest());
    }

    size_t TotalSize() const {
        return TBase::List.GetTotalSize();
    }
};

template <typename TKey, typename TValue, typename TDeleter = TNoopDelete>
class TLFUCache: public TCache<TKey, TValue, TLFUList<TKey, TValue>, TDeleter> {
    typedef TCache<TKey, TValue, TLFUList<TKey, TValue>, TDeleter> TBase;
    using TListType = TLFUList<TKey, TValue>;

public:
    typedef typename TBase::TIterator TIterator;

    TLFUCache(size_t maxSize, bool multiValue = false)
        : TBase(TListType(maxSize), multiValue)
    {
    }

    TValue& GetLeastFrequentlyUsed() {
        return TBase::List.GetLeastFrequentlyUsed()->Value;
    }

    TIterator FindLeastFrequentlyUsed() {
        return TBase::Empty() ? TBase::End() : this->FindByItem(TBase::List.GetLeastFrequentlyUsed());
    }
};

// Least Weighted cache
// discards the least weighted items first
// doesn't support promotion
template <typename TKey, typename TValue, typename TWeight, typename TWeighter, typename TDeleter = TNoopDelete>
class TLWCache: public TCache<TKey, TValue, TLWList<TKey, TValue, TWeight, TWeighter>, TDeleter> {
    typedef TCache<TKey, TValue, TLWList<TKey, TValue, TWeight, TWeighter>, TDeleter> TBase;
    using TListType = TLWList<TKey, TValue, TWeight, TWeighter>;

public:
    typedef typename TBase::TIterator TIterator;

    TLWCache(size_t maxSize, bool multiValue = false)
        : TBase(TListType(maxSize), multiValue)
    {
    }

    TValue& GetLightest() {
        return TBase::List.GetLightest()->Value;
    }

    TIterator FindLightest() {
        return TBase::Empty() ? TBase::End() : this->FindByItem(TBase::List.GetLightest());
    }
};
