#pragma once

#include <util/generic/vector.h>
#include <util/system/yassert.h>

template <typename T>
class TSimpleRingBuffer {
public:
    TSimpleRingBuffer(size_t maxSize)
        : MaxSize(maxSize)
    {
        Items.reserve(MaxSize);
    }

    TSimpleRingBuffer(const TSimpleRingBuffer&) = default;
    TSimpleRingBuffer(TSimpleRingBuffer&&) = default;

    TSimpleRingBuffer& operator=(const TSimpleRingBuffer&) = default;
    TSimpleRingBuffer& operator=(TSimpleRingBuffer&&) = default;

    // First available item
    size_t FirstIndex() const {
        return Begin;
    }

    size_t AvailSize() const {
        return Items.size();
    }

    // Total number of items inserted
    size_t TotalSize() const {
        return FirstIndex() + AvailSize();
    }

    bool IsAvail(size_t index) const {
        return index >= FirstIndex() && index < TotalSize();
    }

    const T& operator[](size_t index) const {
        Y_ASSERT(IsAvail(index));
        return Items[RealIndex(index)];
    }

    T& operator[](size_t index) {
        Y_ASSERT(IsAvail(index));
        return Items[RealIndex(index)];
    }

    void PushBack(const T& t) {
        if (Items.size() < MaxSize) {
            Items.push_back(t);
        } else {
            Items[RealIndex(Begin)] = t;
            Begin += 1;
        }
    }

    void Clear() {
        Items.clear();
        Begin = 0;
    }

private:
    size_t RealIndex(size_t index) const {
        return index % MaxSize;
    }

private:
    size_t MaxSize;
    size_t Begin = 0;
    TVector<T> Items;
};

template <typename T, size_t maxSize>
class TStaticRingBuffer: public TSimpleRingBuffer<T> {
public:
    TStaticRingBuffer()
        : TSimpleRingBuffer<T>(maxSize)
    {
    }
};
