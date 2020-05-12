#pragma once

#include <util/generic/ptr.h>
#include <util/generic/noncopyable.h>
#include <util/generic/utility.h>
#include <util/system/sys_alloc.h>

template <typename T>
class TArrayWithSizeHolder : TNonCopyable {
    typedef TArrayWithSizeHolder<T> TThis;

    T* Data;

public:
    TArrayWithSizeHolder()
        : Data(nullptr)
    {
    }

    ~TArrayWithSizeHolder() {
        if (!Data)
            return;
        for (size_t i = 0; i < Size(); ++i) {
            try {
                Data[i].~T();
            } catch (...) {
            }
        }
        y_deallocate(((size_t*)Data) - 1);
    }

    void Swap(TThis& copy) {
        DoSwap(Data, copy.Data);
    }

    void Resize(size_t newSize) {
        if (newSize == Size())
            return;
        TThis copy;
        copy.Data = (T*)(((size_t*)y_allocate(sizeof(size_t) + sizeof(T) * newSize)) + 1);
        // does not handle constructor exceptions properly
        for (size_t i = 0; i < Min(Size(), newSize); ++i) {
            new (copy.Data + i) T(Data[i]);
        }
        for (size_t i = Min(Size(), newSize); i < newSize; ++i) {
            new (copy.Data + i) T;
        }
        ((size_t*)copy.Data)[-1] = newSize;
        Swap(copy);
    }

    size_t Size() const {
        return Data ? ((size_t*)Data)[-1] : 0;
    }

    bool Empty() const {
        return Size() == 0;
    }

    T* Get() {
        return Data;
    }

    const T* Get() const {
        return Data;
    }
};
