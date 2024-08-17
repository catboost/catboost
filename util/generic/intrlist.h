#pragma once

#include "utility.h"

#include <util/system/yassert.h>
#include <iterator>

struct TIntrusiveListDefaultTag {};

/*
 * two-way linked list
 */
template <class T, class Tag = TIntrusiveListDefaultTag>
class TIntrusiveListItem {
private:
    using TListItem = TIntrusiveListItem<T, Tag>;

public:
    inline TIntrusiveListItem() noexcept
        : Next_(this)
        , Prev_(Next_)
    {
    }

    inline ~TIntrusiveListItem() {
        Unlink();
    }

public:
    Y_PURE_FUNCTION inline bool Empty() const noexcept {
        return (Prev_ == this) && (Next_ == this);
    }

    inline void Unlink() noexcept {
        if (Empty()) {
            return;
        }

        Prev_->SetNext(Next_);
        Next_->SetPrev(Prev_);

        ResetItem();
    }

    inline void LinkBefore(TListItem* before) noexcept {
        Unlink();
        LinkBeforeNoUnlink(before);
    }

    inline void LinkBeforeNoUnlink(TListItem* before) noexcept {
        TListItem* const after = before->Prev();

        after->SetNext(this);
        SetPrev(after);
        SetNext(before);
        before->SetPrev(this);
    }

    inline void LinkBefore(TListItem& before) noexcept {
        LinkBefore(&before);
    }

    inline void LinkAfter(TListItem* after) noexcept {
        Unlink();
        LinkBeforeNoUnlink(after->Next());
    }

    inline void LinkAfter(TListItem& after) noexcept {
        LinkAfter(&after);
    }

public:
    inline TListItem* Prev() noexcept {
        return Prev_;
    }

    inline const TListItem* Prev() const noexcept {
        return Prev_;
    }

    inline TListItem* Next() noexcept {
        return Next_;
    }

    inline const TListItem* Next() const noexcept {
        return Next_;
    }

public:
    inline void SetNext(TListItem* item) noexcept {
        Next_ = item;
    }

    inline void SetPrev(TListItem* item) noexcept {
        Prev_ = item;
    }

public:
    inline T* Node() noexcept {
        return static_cast<T*>(this);
    }

    inline const T* Node() const noexcept {
        return static_cast<const T*>(this);
    }

public:
    // NB(arkady-e1ppa): These methods are used to implement
    // intrusive lock-free algorithms which want to natively
    // interact with TIntrusiveList.
    // Assume that if you've used MutableNext/MutablePrev
    // methods, you are not safe to use anything but
    // MutableNext/MutablePrev/ResetItem methods until
    // you call a ResetItem method.

    inline TListItem*& MutableNext() noexcept {
        return Next_;
    }

    inline TListItem*& MutablePrev() noexcept {
        return Prev_;
    }

    inline void ResetItem() noexcept {
        Next_ = this;
        Prev_ = Next_;
    }

private:
    inline TIntrusiveListItem(const TIntrusiveListItem&) = delete;
    inline TIntrusiveListItem& operator=(const TIntrusiveListItem&) = delete;

private:
    TListItem* Next_;
    TListItem* Prev_;
};

template <class T, class Tag>
class TIntrusiveList {
private:
    using TListItem = TIntrusiveListItem<T, Tag>;

    template <class TListItem, class TNode>
    class TIteratorBase {
    public:
        using TItem = TListItem;
        using TReference = TNode&;
        using TPointer = TNode*;

        using iterator_category = std::bidirectional_iterator_tag;
        using difference_type = ptrdiff_t;

        using value_type = TNode;
        using reference = TReference;
        using pointer = TPointer;

        inline TIteratorBase() noexcept
            : Item_(nullptr)
        {
        }

        template <class TListItem_, class TNode_>
        inline TIteratorBase(const TIteratorBase<TListItem_, TNode_>& right) noexcept
            : Item_(right.Item())
        {
        }

        inline TIteratorBase(TItem* item) noexcept
            : Item_(item)
        {
        }

        inline TItem* Item() const noexcept {
            return Item_;
        }

        inline void Next() noexcept {
            Item_ = Item_->Next();
        }

        inline void Prev() noexcept {
            Item_ = Item_->Prev();
        }

        template <class TListItem_, class TNode_>
        inline bool operator==(const TIteratorBase<TListItem_, TNode_>& right) const noexcept {
            return Item() == right.Item();
        }

        template <class TListItem_, class TNode_>
        inline bool operator!=(const TIteratorBase<TListItem_, TNode_>& right) const noexcept {
            return Item() != right.Item();
        }

        inline TIteratorBase& operator++() noexcept {
            Next();

            return *this;
        }

        inline TIteratorBase operator++(int) noexcept {
            TIteratorBase ret(*this);

            Next();

            return ret;
        }

        inline TIteratorBase& operator--() noexcept {
            Prev();

            return *this;
        }

        inline TIteratorBase operator--(int) noexcept {
            TIteratorBase ret(*this);

            Prev();

            return ret;
        }

        inline TReference operator*() const noexcept {
            return *Item_->Node();
        }

        inline TPointer operator->() const noexcept {
            return Item_->Node();
        }

    private:
        TItem* Item_;
    };

    template <class TIterator>
    class TReverseIteratorBase {
    public:
        using TItem = typename TIterator::TItem;
        using TReference = typename TIterator::TReference;
        using TPointer = typename TIterator::TPointer;

        using iterator_category = typename TIterator::iterator_category;
        using difference_type = typename TIterator::difference_type;

        using value_type = typename TIterator::value_type;
        using reference = typename TIterator::reference;
        using pointer = typename TIterator::pointer;

        inline TReverseIteratorBase() noexcept = default;

        template <class TIterator_>
        inline TReverseIteratorBase(const TReverseIteratorBase<TIterator_>& right) noexcept
            : Current_(right.Base())
        {
        }

        inline explicit TReverseIteratorBase(TIterator item) noexcept
            : Current_(item)
        {
        }

        inline TIterator Base() const noexcept {
            return Current_;
        }

        inline TItem* Item() const noexcept {
            TIterator ret = Current_;

            return (--ret).Item();
        }

        inline void Next() noexcept {
            Current_.Prev();
        }

        inline void Prev() noexcept {
            Current_.Next();
        }

        template <class TIterator_>
        inline bool operator==(const TReverseIteratorBase<TIterator_>& right) const noexcept {
            return Base() == right.Base();
        }

        template <class TIterator_>
        inline bool operator!=(const TReverseIteratorBase<TIterator_>& right) const noexcept {
            return Base() != right.Base();
        }

        inline TReverseIteratorBase& operator++() noexcept {
            Next();

            return *this;
        }

        inline TReverseIteratorBase operator++(int) noexcept {
            TReverseIteratorBase ret(*this);

            Next();

            return ret;
        }

        inline TReverseIteratorBase& operator--() noexcept {
            Prev();

            return *this;
        }

        inline TReverseIteratorBase operator--(int) noexcept {
            TReverseIteratorBase ret(*this);

            Prev();

            return ret;
        }

        inline TReference operator*() const noexcept {
            TIterator ret = Current_;

            return *--ret;
        }

        inline TPointer operator->() const noexcept {
            TIterator ret = Current_;

            return &*--ret;
        }

    private:
        TIterator Current_;
    };

public:
    using TIterator = TIteratorBase<TListItem, T>;
    using TConstIterator = TIteratorBase<const TListItem, const T>;

    using TReverseIterator = TReverseIteratorBase<TIterator>;
    using TConstReverseIterator = TReverseIteratorBase<TConstIterator>;

    using iterator = TIterator;
    using const_iterator = TConstIterator;

    using reverse_iterator = TReverseIterator;
    using const_reverse_iterator = TConstReverseIterator;

public:
    inline void Swap(TIntrusiveList& right) noexcept {
        TIntrusiveList temp;

        temp.Append(right);
        Y_ASSERT(right.Empty());
        right.Append(*this);
        Y_ASSERT(this->Empty());
        this->Append(temp);
        Y_ASSERT(temp.Empty());
    }

public:
    inline TIntrusiveList() noexcept = default;

    inline ~TIntrusiveList() = default;

    inline TIntrusiveList(TIntrusiveList&& right) noexcept {
        this->Swap(right);
    }

    inline TIntrusiveList& operator=(TIntrusiveList&& rhs) noexcept {
        this->Swap(rhs);
        return *this;
    }

    inline explicit operator bool() const noexcept {
        return !Empty();
    }

    Y_PURE_FUNCTION inline bool Empty() const noexcept {
        return End_.Empty();
    }

    inline size_t Size() const noexcept {
        return std::distance(Begin(), End());
    }

    inline void Remove(TListItem* item) noexcept {
        item->Unlink();
    }

    inline void Clear() noexcept {
        End_.Unlink();
    }

public:
    inline TIterator Begin() noexcept {
        return ++End();
    }

    inline TIterator End() noexcept {
        return TIterator(&End_);
    }

    inline TConstIterator Begin() const noexcept {
        return ++End();
    }

    inline TConstIterator End() const noexcept {
        return TConstIterator(&End_);
    }

    inline TReverseIterator RBegin() noexcept {
        return TReverseIterator(End());
    }

    inline TReverseIterator REnd() noexcept {
        return TReverseIterator(Begin());
    }

    inline TConstReverseIterator RBegin() const noexcept {
        return TConstReverseIterator(End());
    }

    inline TConstReverseIterator REnd() const noexcept {
        return TConstReverseIterator(Begin());
    }

    inline TConstIterator CBegin() const noexcept {
        return Begin();
    }

    inline TConstIterator CEnd() const noexcept {
        return End();
    }

    inline TConstReverseIterator CRBegin() const noexcept {
        return RBegin();
    }

    inline TConstReverseIterator CREnd() const noexcept {
        return REnd();
    }

public:
    inline iterator begin() noexcept {
        return Begin();
    }

    inline iterator end() noexcept {
        return End();
    }

    inline const_iterator begin() const noexcept {
        return Begin();
    }

    inline const_iterator end() const noexcept {
        return End();
    }

    inline reverse_iterator rbegin() noexcept {
        return RBegin();
    }

    inline reverse_iterator rend() noexcept {
        return REnd();
    }

    inline const_iterator cbegin() const noexcept {
        return CBegin();
    }

    inline const_iterator cend() const noexcept {
        return CEnd();
    }

    inline const_reverse_iterator crbegin() const noexcept {
        return CRBegin();
    }

    inline const_reverse_iterator crend() const noexcept {
        return CREnd();
    }

public:
    inline T* Back() noexcept {
        return End_.Prev()->Node();
    }

    inline T* Front() noexcept {
        return End_.Next()->Node();
    }

    inline const T* Back() const noexcept {
        return End_.Prev()->Node();
    }

    inline const T* Front() const noexcept {
        return End_.Next()->Node();
    }

    inline void PushBack(TListItem* item) noexcept {
        item->LinkBefore(End_);
    }

    inline void PushFront(TListItem* item) noexcept {
        item->LinkAfter(End_);
    }

    inline T* PopBack() noexcept {
        TListItem* const ret = End_.Prev();

        ret->Unlink();

        return ret->Node();
    }

    inline T* PopFront() noexcept {
        TListItem* const ret = End_.Next();

        ret->Unlink();

        return ret->Node();
    }

    inline void Append(TIntrusiveList& list) noexcept {
        Cut(list.Begin(), list.End(), End());
    }

    inline void Append(TIntrusiveList&& list) noexcept {
        Append(list);
    }

    inline static void Cut(TIterator begin, TIterator end, TIterator pasteBefore) noexcept {
        if (begin == end) {
            return;
        }

        TListItem* const cutFront = begin.Item();
        TListItem* const gapBack = end.Item();

        TListItem* const gapFront = cutFront->Prev();
        TListItem* const cutBack = gapBack->Prev();

        gapFront->SetNext(gapBack);
        gapBack->SetPrev(gapFront);

        TListItem* const pasteBack = pasteBefore.Item();
        TListItem* const pasteFront = pasteBack->Prev();

        pasteFront->SetNext(cutFront);
        cutFront->SetPrev(pasteFront);

        cutBack->SetNext(pasteBack);
        pasteBack->SetPrev(cutBack);
    }

public:
    template <class TFunctor>
    inline void ForEach(TFunctor&& functor) {
        TIterator i = Begin();

        while (i != End()) {
            functor(&*(i++));
        }
    }

    template <class TFunctor>
    inline void ForEach(TFunctor&& functor) const {
        TConstIterator i = Begin();

        while (i != End()) {
            functor(&*(i++));
        }
    }

    template <class TComparer>
    inline void QuickSort(TComparer&& comparer) {
        if (Begin() == End() || ++Begin() == End()) {
            return;
        }

        T* const pivot = PopFront();
        TIntrusiveList bigger;
        TIterator i = Begin();

        while (i != End()) {
            if (comparer(*pivot, *i)) {
                bigger.PushBack(&*i++);
            } else {
                ++i;
            }
        }

        this->QuickSort(comparer);
        bigger.QuickSort(comparer);

        PushBack(pivot);
        Append(bigger);
    }

private:
    inline TIntrusiveList(const TIntrusiveList&) = delete;
    inline TIntrusiveList& operator=(const TIntrusiveList&) = delete;

private:
    TListItem End_;
};

template <class T, class D, class Tag>
class TIntrusiveListWithAutoDelete: public TIntrusiveList<T, Tag> {
public:
    using TIterator = typename TIntrusiveList<T, Tag>::TIterator;
    using TConstIterator = typename TIntrusiveList<T, Tag>::TConstIterator;

    using TReverseIterator = typename TIntrusiveList<T, Tag>::TReverseIterator;
    using TConstReverseIterator = typename TIntrusiveList<T, Tag>::TConstReverseIterator;

    using iterator = TIterator;
    using const_iterator = TConstIterator;

    using reverse_iterator = TReverseIterator;
    using const_reverse_iterator = TConstReverseIterator;

public:
    inline TIntrusiveListWithAutoDelete() noexcept = default;

    inline TIntrusiveListWithAutoDelete(TIntrusiveListWithAutoDelete&& right) noexcept
        : TIntrusiveList<T, Tag>(std::move(right))
    {
    }

    inline ~TIntrusiveListWithAutoDelete() {
        this->Clear();
    }

    TIntrusiveListWithAutoDelete& operator=(TIntrusiveListWithAutoDelete&& rhs) noexcept {
        TIntrusiveList<T, Tag>::operator=(std::move(rhs));
        return *this;
    }

public:
    inline void Clear() noexcept {
        this->ForEach([](auto* item) {
            D::Destroy(item);
        });
    }

    inline static void Cut(TIterator begin, TIterator end) noexcept {
        TIntrusiveListWithAutoDelete<T, D, Tag> temp;
        Cut(begin, end, temp.End());
    }

    inline static void Cut(TIterator begin, TIterator end, TIterator pasteBefore) noexcept {
        TIntrusiveList<T, Tag>::Cut(begin, end, pasteBefore);
    }
};

/*
 * one-way linked list
 */
template <class T, class Tag = TIntrusiveListDefaultTag>
class TIntrusiveSListItem {
private:
    using TListItem = TIntrusiveSListItem<T, Tag>;

public:
    inline TIntrusiveSListItem() noexcept
        : Next_(nullptr)
    {
    }

    inline ~TIntrusiveSListItem() = default;

    inline bool IsEnd() const noexcept {
        return Next_ == nullptr;
    }

    inline TListItem* Next() noexcept {
        return Next_;
    }

    inline const TListItem* Next() const noexcept {
        return Next_;
    }

    inline void SetNext(TListItem* item) noexcept {
        Next_ = item;
    }

public:
    inline T* Node() noexcept {
        return static_cast<T*>(this);
    }

    inline const T* Node() const noexcept {
        return static_cast<const T*>(this);
    }

private:
    TListItem* Next_;
};

template <class T, class Tag>
class TIntrusiveSList {
private:
    using TListItem = TIntrusiveSListItem<T, Tag>;

public:
    template <class TListItem, class TNode>
    class TIteratorBase {
    public:
        using TItem = TListItem;
        using TReference = TNode&;
        using TPointer = TNode*;

        using difference_type = std::ptrdiff_t;
        using value_type = TNode;
        using pointer = TPointer;
        using reference = TReference;
        using iterator_category = std::forward_iterator_tag;

        inline TIteratorBase(TListItem* item) noexcept
            : Item_(item)
        {
        }

        inline void Next() noexcept {
            Item_ = Item_->Next();
        }

        inline bool operator==(const TIteratorBase& right) const noexcept {
            return Item_ == right.Item_;
        }

        inline bool operator!=(const TIteratorBase& right) const noexcept {
            return Item_ != right.Item_;
        }

        inline TIteratorBase& operator++() noexcept {
            Next();

            return *this;
        }

        inline TIteratorBase operator++(int) noexcept {
            TIteratorBase ret(*this);

            Next();

            return ret;
        }

        inline TNode& operator*() noexcept {
            return *Item_->Node();
        }

        inline TNode* operator->() noexcept {
            return Item_->Node();
        }

    private:
        TListItem* Item_;
    };

public:
    using TIterator = TIteratorBase<TListItem, T>;
    using TConstIterator = TIteratorBase<const TListItem, const T>;

    using iterator = TIterator;
    using const_iterator = TConstIterator;

public:
    inline TIntrusiveSList() noexcept
        : Begin_(nullptr)
    {
    }

    inline void Swap(TIntrusiveSList& right) noexcept {
        DoSwap(Begin_, right.Begin_);
    }

    inline explicit operator bool() const noexcept {
        return !Empty();
    }

    Y_PURE_FUNCTION inline bool Empty() const noexcept {
        return Begin_ == nullptr;
    }

    inline size_t Size() const noexcept {
        return std::distance(Begin(), End());
    }

    inline void Clear() noexcept {
        Begin_ = nullptr;
    }

    inline TIterator Begin() noexcept {
        return TIterator(Begin_);
    }

    inline TIterator End() noexcept {
        return TIterator(nullptr);
    }

    inline TConstIterator Begin() const noexcept {
        return TConstIterator(Begin_);
    }

    inline TConstIterator End() const noexcept {
        return TConstIterator(nullptr);
    }

    inline TConstIterator CBegin() const noexcept {
        return Begin();
    }

    inline TConstIterator CEnd() const noexcept {
        return End();
    }

    // compat methods
    inline iterator begin() noexcept {
        return Begin();
    }

    inline iterator end() noexcept {
        return End();
    }

    inline const_iterator begin() const noexcept {
        return Begin();
    }

    inline const_iterator end() const noexcept {
        return End();
    }

    inline const_iterator cbegin() const noexcept {
        return CBegin();
    }

    inline const_iterator cend() const noexcept {
        return CEnd();
    }

    inline T* Front() noexcept {
        Y_ASSERT(Begin_);
        return Begin_->Node();
    }

    inline const T* Front() const noexcept {
        Y_ASSERT(Begin_);
        return Begin_->Node();
    }

    inline void PushFront(TListItem* item) noexcept {
        item->SetNext(Begin_);
        Begin_ = item;
    }

    inline T* PopFront() noexcept {
        Y_ASSERT(Begin_);

        TListItem* const ret = Begin_;
        Begin_ = Begin_->Next();

        return ret->Node();
    }

    inline void Reverse() noexcept {
        TIntrusiveSList temp;

        while (!Empty()) {
            temp.PushFront(PopFront());
        }

        this->Swap(temp);
    }

    template <class TFunctor>
    inline void ForEach(TFunctor&& functor) const noexcept(noexcept(functor(std::declval<TListItem>().Node()))) {
        TListItem* i = Begin_;

        while (i) {
            TListItem* const next = i->Next();
            functor(i->Node());
            i = next;
        }
    }

private:
    TListItem* Begin_;
};
