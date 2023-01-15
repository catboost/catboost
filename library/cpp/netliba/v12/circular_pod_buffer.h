#pragma once

#include <util/generic/utility.h>
#include <util/generic/vector.h>

namespace NNetliba_v12 {
    // CircularBuffer with fast insert/clear (not ctors/dtors actually called, only pointer arithmetics is used).
    template <class T>
    class TCircularPodBuffer {
#if (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
        static_assert(__is_pod(T), "expect __is_pod(T)");
#endif

    public:
        TCircularPodBuffer(const size_t maxSize)
            : Begin(0)
            , End(0)
            , IsFull(false)
        {
            Array.resize(maxSize);
        }

        size_t Size() const {
            return (Array.size() + End - Begin) % (Array.size() + !!IsFull);
        }
        size_t Capacity() const {
            return Array.capacity();
        }

        bool Empty() const {
            return Begin == End && !IsFull;
        }
        bool Full() const {
            return IsFull;
        }

        T* GetContinuousRegion(const size_t n) {
            T* result;

            if (GetTailSize() >= n) {
                result = &Array[End];
                End = (End + n) % Array.size(); // % for End + n == Array.size() only

            } else if (IsFull || End < Begin || n > Begin) {
                return nullptr;

            } else {
                result = &Array[0];
                End = n;
            }

            IsFull = Begin == End && (n > 0 || IsFull);
            return result;
        }
        T* PushBack(const T& t = T()) {
            T* result = GetContinuousRegion(1);
            if (result) {
                *result = t;
            }
            return result;
        }

        void EraseHead(size_t n) {
            n = Min(n, Size());
            Begin = (Begin + n) % Array.size();
            IsFull = IsFull && n == 0;
        }
        void EraseBefore(const T* ptr) {
            // t == Begin - should we delete everything or nothing?
            // Better for user to decide - to call Clear() or nothing.
            Y_ASSERT(ptr != &Array[Begin] && "Call Clear() or nothing");

            Y_ASSERT(IsInside<const T*>(ptr, &Array.front(), &Array.back() + 1));
            const size_t t = ptr - &Array[0];
            Y_ASSERT(!Empty() && (Full() || (Begin < End ? IsInside(t, Begin, End) : !IsInside(t, End, Begin))));

            IsFull = false;
            Begin = t;
        }

        void PopFront() {
            EraseHead(1);
        }
        void Clear() {
            Begin = End = IsFull = 0;
        }

        T& Front() {
            Y_ASSERT(!Empty());
            return Array[Begin];
        }
        T& Back() {
            Y_ASSERT(!Empty());
            return Array[(Array.size() + End - 1) % Array.size()];
        }

        T& operator[](const size_t i) {
            Y_ASSERT(i < Size());
            return Array[(Begin + i) % Array.size()];
        }
        const T& operator[](const size_t i) const {
            Y_ASSERT(i < Size());
            return Array[(Begin + i) % Array.size()];
        }

    private:
        template <class U>
        static bool IsInside(const U& val, const U& begin, const U& end) {
            return begin <= val && val < end;
        }

        // * - tail (from End to either array end or Begin)
        // ====E**B===
        // ====E,B====
        // ----B==E***
        // ----B,E****
        size_t GetTailSize() const {
            return End < Begin || IsFull ? Begin - End : Array.size() - End;
        }

        TVector<T> Array;
        size_t Begin;
        size_t End;
        bool IsFull;
    };
}
