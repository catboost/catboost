#pragma once

#include <util/memory/smallobj.h>
#include "ptr.h"

template <class T>
class TFastQueue {
    struct THelper: public TObjectFromPool<THelper>, public TIntrusiveListItem<THelper> {
        inline THelper(const T& t)
            : Obj(t)
        {
        }

        T Obj;
    };

public:
    inline TFastQueue()
        : Pool_(TDefaultAllocator::Instance())
        , Size_(0)
    {
    }

    inline void Push(const T& t) {
        Queue_.PushFront(new (&Pool_) THelper(t));
        ++Size_;
    }

    inline T Pop() {
        Y_ASSERT(!this->Empty());

        THolder<THelper> tmp(Queue_.PopBack());
        --Size_;

        return tmp->Obj;
    }

    inline size_t Size() const noexcept {
        return Size_;
    }

    Y_PURE_FUNCTION inline bool Empty() const noexcept {
        return !this->Size();
    }

    inline explicit operator bool() const noexcept {
        return !this->Empty();
    }

private:
    typename THelper::TPool Pool_;
    TIntrusiveListWithAutoDelete<THelper, TDelete> Queue_;
    size_t Size_;
};
