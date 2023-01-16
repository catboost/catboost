#pragma once

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class TItem>
struct TIntrusiveLinkedListNode
{
    TItem* Next = nullptr;
    TItem* Prev = nullptr;
};

template <class TItem, class TItemToNode>
class TIntrusiveLinkedList
{
public:
    explicit TIntrusiveLinkedList(TItemToNode itemToNode = TItemToNode());

    TItem* GetFront() const;
    TItem* GetBack() const;

    int GetSize() const;

    void PushFront(TItem* item);
    void PopFront();

    void PushBack(TItem* item);
    void PopBack();

    void Remove(TItem* item);

    void Clear();

private:
    const TItemToNode ItemToNode_;

    TItem* Front_ = nullptr;
    TItem* Back_ = nullptr;
    int Size_ = 0;

};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define INTRUSIVE_LINKED_LIST_INL_H_
#include "intrusive_linked_list-inl.h"
#undef INTRUSIVE_LINKED_LIST_INL_H_

