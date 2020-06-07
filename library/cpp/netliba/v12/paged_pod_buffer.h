#pragma once

#include <util/generic/deque.h>
#include <util/generic/vector.h>

namespace NNetliba_v12 {
    // Similar to std::deque but allows to force storing interval of elements in continuous region of memory.
    template <class T>
    class TPagedPodBuffer {
#if (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
        static_assert(__is_pod(T), "expect __is_pod(T)");
#endif
        typedef TVector<T> TPage;

    public:
        TPagedPodBuffer(const size_t pageSize)
            : PageSize(pageSize)
        {
            AddNewPage();
        }

        T* PushBack(const T& t) {
            if (LastPage().size() == LastPage().capacity()) {
                AddNewPage();
            }
            LastPage().push_back(t);
            return &LastPage().back();
        }

        T* PushBackToContRegion(const T& t, T** regionBegin) {
            Y_ASSERT(!V.empty() && !LastPage().empty());

            // No need to move region which already was moved and don't want to move more than 1 page.
            if (!(&LastPage().front() <= *regionBegin && *regionBegin <= &LastPage().back())) {
                return nullptr;
            }

            // Oops, new element can't fit in same page with region.
            // Moving whole region [*regionBegin, LastPage.back()] to new page and appending new element.
            if (LastPage().size() == LastPage().capacity()) {
                const size_t regionLen = &LastPage().back() + 1 - *regionBegin;
                const size_t numElementsLeft = LastPage().size() - regionLen;

                AddNewPage();
                LastPage().yresize(regionLen);
                memcpy(&LastPage()[0], *regionBegin, regionLen * sizeof(T));
                V[V.size() - 2].yresize(numElementsLeft);

                *regionBegin = &LastPage()[0];
            }

            Y_ASSERT(LastPage().size() < LastPage().capacity());
            LastPage().push_back(t);
            return &LastPage().back();
        }

        void CleanupBefore(const T* regionBegin) {
            // We leave elements preceding regionBegin in its page, we just don't want to store 100% garbage pages.
            while (!V.empty() && !(&V.front().front() <= regionBegin && regionBegin <= &V.front().back())) {
                V.pop_front();
            }
        }

        void Clear() {
            V.resize(1);
            LastPage().clear();
            LastPage().reserve(PageSize);
        }

    private:
        void AddNewPage() {
            V.push_back(TPage());
            LastPage().reserve(PageSize);
        }

        TPage& LastPage() {
            return V.back();
        }

        TDeque<TPage> V;
        size_t PageSize;
    };
}
