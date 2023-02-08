#pragma once
#ifndef INTRUSIVE_LINKED_LIST_INL_H_
#error "Direct inclusion of this file is not allowed, include intrusive_linked_list.h"
// For the sake of sane code completion.
#include "intrusive_linked_list.h"
#endif

#include <library/cpp/yt/assert/assert.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class TItem, class TItemToNode>
TIntrusiveLinkedList<TItem, TItemToNode>::TIntrusiveLinkedList(TItemToNode itemToNode)
    : ItemToNode_(itemToNode)
{ }

template <class TItem, class TItemToNode>
TItem* TIntrusiveLinkedList<TItem, TItemToNode>::GetFront() const
{
    return Front_;
}

template <class TItem, class TItemToNode>
TItem* TIntrusiveLinkedList<TItem, TItemToNode>::GetBack() const
{
    return Back_;
}

template <class TItem, class TItemToNode>
int TIntrusiveLinkedList<TItem, TItemToNode>::GetSize() const
{
    return Size_;
}

template <class TItem, class TItemToNode>
void TIntrusiveLinkedList<TItem, TItemToNode>::PushBack(TItem* item)
{
    auto* node = ItemToNode_(item);
    if (Back_) {
        ItemToNode_(Back_)->Next = item;
    } else {
        Front_ = item;
    }
    node->Next = nullptr;
    node->Prev = Back_;
    Back_ = item;
    ++Size_;
}

template <class TItem, class TItemToNode>
void TIntrusiveLinkedList<TItem, TItemToNode>::PopBack()
{
    Y_ASSERT(Back_);
    if (Front_ == Back_) {
        Front_ = Back_ = nullptr;
    } else {
        Back_ = ItemToNode_(Back_)->Prev;
        ItemToNode_(Back_)->Next = nullptr;
    }
    --Size_;
}

template <class TItem, class TItemToNode>
void TIntrusiveLinkedList<TItem, TItemToNode>::PushFront(TItem* item)
{
    auto* node = ItemToNode_(item);
    if (Front_) {
        ItemToNode_(Front_)->Prev = item;
    } else {
        Back_ = item;
    }
    node->Next = Front_;
    node->Prev = nullptr;
    Front_ = item;
    ++Size_;
}

template <class TItem, class TItemToNode>
void TIntrusiveLinkedList<TItem, TItemToNode>::PopFront()
{
    Y_ASSERT(Front_);
    if (Front_ == Back_) {
        Front_ = Back_ = nullptr;
    } else {
        Front_ = ItemToNode_(Front_)->Next;
        ItemToNode_(Front_)->Prev = nullptr;
    }
    --Size_;
}

template <class TItem, class TItemToNode>
void TIntrusiveLinkedList<TItem, TItemToNode>::Remove(TItem* item)
{
    YT_ASSERT(Front_);
    auto* node = ItemToNode_(item);
    if (node->Next) {
        ItemToNode_(node->Next)->Prev = node->Prev;
    }
    if (node->Prev) {
        ItemToNode_(node->Prev)->Next = node->Next;
    }
    if (Front_ == item) {
        Front_ = node->Next;
    }
    if (Back_ == item) {
        Back_ = node->Prev;
    }
    --Size_;
}

template <class TItem, class TItemToNode>
void TIntrusiveLinkedList<TItem, TItemToNode>::Clear()
{
    Front_ = Back_ = nullptr;
    Size_ = 0;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
