#pragma once

#include "reader.h"
#include "writer.h"

template <bool>
struct TDefaultAhoCorasickG;

template <>
struct TDefaultAhoCorasickG<false> {
    typedef TDefaultMappedAhoCorasick T;
};

template <>
struct TDefaultAhoCorasickG<true> {
    typedef TDefaultAhoCorasickBuilder T;
};
