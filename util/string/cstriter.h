#pragma once

struct TCStringEndIterator {
};

template <class It>
static inline bool operator==(It b, TCStringEndIterator) {
    return !*b;
}

template <class It>
static inline bool operator!=(It b, TCStringEndIterator) {
    return !!*b;
}
