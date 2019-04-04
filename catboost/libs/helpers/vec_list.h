#pragma once

#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/system/types.h>

#include <iterator>


/*
 * Single-linked list similar to std::forward_list but avoiding per-node allocations, storing all data
 * in a single TVector instead.
 *
 * The downsides are that there's no insert at an arbitrary position and that iterators are invalidated
 *  on push_back and erase.
 */
template <class T, class TSize = ui32>
class TVecList {
private:
    struct TNode {
        using value_type = T;

    public:
        TSize NextOffset;
        alignas(T) ui8 Data[sizeof(T)];
    };

private:
    // 0th element is a sentinel
    TVector<TNode> Data;
    TSize Size;
    TNode* EndPrev; // for end iterator

private:
    void Assign(const TVecList& rhs) {
        Size = rhs.Size;
        Data.yresize(Size + 1);

        Data.front().NextOffset = 1;
        size_t i = 1;
        for (const auto& element : rhs) {
            Data[i].NextOffset = 1;
            new(Data[i].Data) T(element);
            ++i;
        }
        EndPrev = &Data.back();
    }

    // w/o initializing new Data in node, do it after this call
    TNode* PushBackImpl() {
        TNode* newNode;
        if (EndPrev == &Data.back()) {
            Data.resize(Data.size() + 1);
            EndPrev = &Data.back() - 1;
            newNode = &Data.back();
            newNode->NextOffset = 1;
        } else {
            newNode = EndPrev + 1;
            newNode->NextOffset = EndPrev->NextOffset - 1;
        }
        EndPrev->NextOffset = 1;
        EndPrev = newNode;
        ++Size;
        return newNode;
    }

public:
    template <class TNode>
    class TIteratorBase {
    public:
        friend class TVecList;

        using iterator_category = std::forward_iterator_tag;
        using value_type = typename TNode::value_type;
        using difference_type = ptrdiff_t;
        using size_type = size_t;
        using reference = typename TNode::value_type&;
        using pointer = typename TNode::value_type*;

    private:
        TNode* Prev; // needed to adjust its' NextOffset field if Current is being deleted
    public:
        explicit TIteratorBase(TNode* prev = nullptr)
            : Prev(prev)
        {}

        reference operator*() const {
            return *(pointer)((Prev + Prev->NextOffset)->Data);
        }

        reference operator->() const {
            return *(pointer)((Prev + Prev->NextOffset)->Data);
        }

        TIteratorBase& operator++() {
            Prev += Prev->NextOffset;
            return *this;
        }

        TIteratorBase& operator++(int) {
            TIteratorBase result(Prev);
            operator++();
            return result;
        }

        bool operator==(const TIteratorBase& rhs) const {
            return Prev == rhs.Prev;
        }

        bool operator!=(const TIteratorBase& rhs) const {
            return Prev != rhs.Prev;
        }
    };

    using iterator = TIteratorBase<TNode>;
    using const_iterator = TIteratorBase<const TNode>;

public:
    TVecList()
        : Data(1)
        , Size(0)
        , EndPrev(&Data.front())
    {
        Data[0].NextOffset = 1;
    }

    explicit TVecList(TSize size, const T& init = T())
        : Data(size + 1)
        , Size(size)
        , EndPrev(&Data.front() + size)
    {
        Data[0].NextOffset = 1;
        for (auto i : xrange(TSize(1), size + 1)) {
            Data[i].NextOffset = 1;
            new(Data[i].Data) T(init);
        }
    }

    TVecList(const TVecList& rhs)
        : Size(0) // changed in Assign
        , EndPrev(nullptr) // changed in Assign
    {
        Assign(rhs);
    }

    TVecList(TVecList&& rhs)
        : Data(std::move(rhs.Data))
        , Size(rhs.Size)
        , EndPrev(rhs.EndPrev)
    {
        rhs.Size = 0;
    }

    ~TVecList() {
        if (Size) {
            for (auto& element : *this) {
                element.~T();
            }
        }
    }

    TVecList& operator=(const TVecList& rhs) {
        if (this != &rhs) {
            Assign(rhs);
        }
        return *this;
    }

    TVecList& operator=(TVecList&& rhs) {
        if (this != &rhs) {
            auto endPrevOffset = rhs.EndPrev - &rhs.Data.front();
            Data = std::move(rhs.Data);
            Size = rhs.Size;
            EndPrev = &Data.front() + endPrevOffset;
            rhs.Size = 0;
        }
        return *this;
    }

    TSize size() const {
        return Size;
    }

    bool empty() const {
        return size() == 0;
    }

    void reserve(TSize size) {
        Data.reserve(size + 1);
    }

    iterator begin() {
        return iterator(&Data.front());
    }

    iterator end() {
        return iterator(EndPrev);
    }

    const_iterator begin() const {
        return const_iterator(&Data.front());
    }

    const_iterator end() const {
        return const_iterator(EndPrev);
    }

    iterator erase(iterator it) {
        TNode* current = it.Prev + it.Prev->NextOffset;
        if (current == EndPrev) {
            EndPrev = it.Prev;
        }
        ((T*)current->Data)->~T();
        it.Prev->NextOffset += current->NextOffset;
        --Size;
        return it;
    }

    void push_back(const T& element) {
        TNode* newNode = PushBackImpl();
        new ((T*)newNode->Data) T(element);
    }

    void push_back(T&& element) {
        TNode* newNode = PushBackImpl();
        new ((T*)newNode->Data) T(std::move(element));
    }
};
