#pragma once

#include <util/generic/bitops.h>
#include <util/generic/yexception.h>
#include <util/system/sys_alloc.h>

// vector that is 8 bytes when empty (TVector is 24 bytes)

template <typename T>
class TCompactVector {
private:
    typedef TCompactVector<T> TThis;

    // XXX: make header independent on T and introduce nullptr
    struct THeader {
        size_t Size;
        size_t Capacity;
    };

    T* Ptr;

    THeader* Header() {
        return ((THeader*)Ptr) - 1;
    }

    const THeader* Header() const {
        return ((THeader*)Ptr) - 1;
    }

    void destruct_at(size_t pos) {
        (*this)[pos].~T();
    }

public:
    using value_type = T;

    using TIterator = T*;
    using TConstIterator = const T*;

    using iterator = TIterator ;
    using const_iterator = TConstIterator;

    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    TCompactVector()
        : Ptr(nullptr)
    {
    }

    TCompactVector(const TThis& that)
        : Ptr(nullptr)
    {
        Reserve(that.Size());
        for (TConstIterator i = that.Begin(); i != that.End(); ++i) {
            PushBack(*i);
        }
    }

    TCompactVector(TThis&& that) noexcept
        : Ptr(nullptr)
    {
        Swap(that);
    }

    TCompactVector(std::initializer_list<T> init)
        : Ptr(nullptr)
    {
        Reserve(init.size());
        for (const T& val : init) {
            PushBack(val);
        }
    }

    template <class InputIterator>
    TCompactVector(InputIterator begin, InputIterator end)
        : Ptr(nullptr)
    {
        Reserve(std::distance(begin, end));

        for (auto it = begin; it != end; ++it) {
            push_back(*it);
        }
    }

    ~TCompactVector() {
        for (size_t i = 0; i < Size(); ++i) {
            try {
                destruct_at(i);
            } catch (...) {
            }
        }
        if (Ptr)
            y_deallocate(Header());
    }

    TThis& operator = (TThis&& that) noexcept {
        Swap(that);
        return *this;
    }

    TThis& operator = (const TThis& that) {
        if (Y_LIKELY(this != &that)) {
            TThis tmp(that);
            Swap(tmp);
        }
        return *this;
    }

    TThis& operator = (std::initializer_list<T> init) {
        TThis data(init);
        Swap(data);
        return *this;
    }

    bool operator==(const TCompactVector<T>& other) const {
        return size() == other.size() && std::equal(begin(), end(), other.begin());
    }

    explicit operator bool() const {
        return !empty();
    }

    TIterator Begin() {
        return Ptr;
    }

    TIterator End() {
        return Ptr + Size();
    }

    TConstIterator Begin() const {
        return Ptr;
    }

    TConstIterator End() const {
        return Ptr + Size();
    }

    iterator begin() {
        return Begin();
    }

    const_iterator begin() const {
        return Begin();
    }

    iterator end() {
        return End();
    }

    const_iterator end() const {
        return End();
    }

    reverse_iterator rbegin() {
        return std::make_reverse_iterator(end());
    }

    const_reverse_iterator rbegin() const {
        return std::make_reverse_iterator(end());
    }

    reverse_iterator rend() {
        return std::make_reverse_iterator(begin());
    }

    const_reverse_iterator rend() const {
        return std::make_reverse_iterator(begin());
    }

    value_type* data() {
        return Begin();
    }

    const value_type* data() const {
        return Begin();
    }

    void Swap(TThis& that) noexcept {
        DoSwap(Ptr, that.Ptr);
    }

    void Reserve(size_t newCapacity) {
        if (newCapacity <= Capacity()) {
        } else if (Ptr == nullptr) {
            constexpr size_t maxBlockSize = static_cast<size_t>(1) << (sizeof(size_t) * 8 - 1);
            constexpr size_t maxCapacity = (maxBlockSize - sizeof(THeader)) / sizeof(T);
            Y_ENSURE(newCapacity <= maxCapacity);

            const size_t requiredMemSize = sizeof(THeader) + newCapacity * sizeof(T);
            // most allocators operates pow-of-two memory blocks,
            // so we try to allocate such memory block to fully utilize its capacity
            const size_t memSizePowOf2 = FastClp2(requiredMemSize);
            const size_t realNewCapacity = (memSizePowOf2 - sizeof(THeader)) / sizeof(T);
            Y_ASSERT(realNewCapacity >= newCapacity);

            void* mem = ::y_allocate(memSizePowOf2);
            Ptr = (T*)(((THeader*)mem) + 1);
            Header()->Size = 0;
            Header()->Capacity = realNewCapacity;
        } else {
            TThis copy;
            copy.Reserve(newCapacity);
            for (TConstIterator it = Begin(); it != End(); ++it) {
                copy.PushBack(*it);
            }
            Swap(copy);
        }
    }

    void reserve(size_t newCapacity) {
        Reserve(newCapacity);
    }

    size_t Size() const {
        return Ptr ? Header()->Size : 0;
    }

    size_t size() const {
        return Size();
    }

    bool Empty() const {
        return Size() == 0;
    }

    bool empty() const {
        return Empty();
    }

    size_t Capacity() const {
        return Ptr ? Header()->Capacity : 0;
    }

    void PushBack(const T& elem) {
        EmplaceBack(elem);
    }

    void push_back(const T& elem) {
        PushBack(elem);
    }

    template <class... Args>
    T& EmplaceBack(Args&&... args) {
        Reserve(Size() + 1);
        auto* t = new (Ptr + Size()) T(std::forward<Args>(args)...);
        ++(Header()->Size);
        return *t;
    }

    template <class... Args>
    T& emplace_back(Args&&... args) {
        return EmplaceBack(std::forward<Args>(args)...);
    }

    T& Back() {
        return *(End() - 1);
    }

    const T& Back() const {
        return *(End() - 1);
    }

    T& back() {
        return Back();
    }

    const T& back() const {
        return Back();
    }

    TIterator Insert(TIterator pos, const T& elem) {
        Y_ASSERT(pos >= Begin());
        Y_ASSERT(pos <= End());

        size_t posn = pos - Begin();
        if (pos == End()) {
            PushBack(elem);
        } else {
            Y_ASSERT(Size() > 0);

            Reserve(Size() + 1);

            PushBack(*(End() - 1));

            for (size_t i = Size() - 2; i + 1 > posn; --i) {
                (*this)[i + 1] = (*this)[i];
            }

            (*this)[posn] = elem;
        }
        return Begin() + posn;
    }

    iterator insert(iterator pos, const T& elem) {
        return Insert(pos, elem);
    }

    void Clear() {
        TThis clean;
        Swap(clean);
    }

    void clear() {
        Clear();
    }

    void erase(iterator position) {
        Y_ENSURE(position >= begin() && position < end());
        std::move(position + 1, end(), position);
        destruct_at(Size() - 1);
        Header()->Size -= 1;
    }

    T& operator[](size_t index) {
        Y_ASSERT(index < Size());
        return Ptr[index];
    }

    const T& operator[](size_t index) const {
        Y_ASSERT(index < Size());
        return Ptr[index];
    }
};
