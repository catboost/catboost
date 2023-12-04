#pragma once

#include "cstriter.h"

#include <util/generic/bitmap.h>

template <class TSetType>
class TStrSpnImpl {
public:
    inline TStrSpnImpl(const char* b, const char* e) {
        Init(b, e);
    }

    inline TStrSpnImpl(const char* s) {
        Init(s, TCStringEndIterator());
    }

    // FirstOf
    template <class It1, class It2>
    inline It1 FindFirstOf(It1 b, It2 e) const noexcept {
        return FindFirst<false>(b, e);
    }

    template <class It>
    inline It FindFirstOf(It s) const noexcept {
        return FindFirst<false>(s, TCStringEndIterator());
    }

    // FirstNotOf
    template <class It1, class It2>
    inline It1 FindFirstNotOf(It1 b, It2 e) const noexcept {
        return FindFirst<true>(b, e);
    }

    template <class It>
    inline It FindFirstNotOf(It s) const noexcept {
        return FindFirst<true>(s, TCStringEndIterator());
    }

    inline void Set(ui8 b) noexcept {
        S_.Set(b);
    }

private:
    template <bool Result, class It1, class It2>
    inline It1 FindFirst(It1 b, It2 e) const noexcept {
        while (b != e && (S_.Get((ui8)*b) == Result)) {
            ++b;
        }

        return b;
    }

    template <class It1, class It2>
    inline void Init(It1 b, It2 e) {
        while (b != e) {
            this->Set((ui8)*b++);
        }
    }

private:
    TSetType S_;
};

using TCompactStrSpn = TStrSpnImpl<TBitMap<256>>;
