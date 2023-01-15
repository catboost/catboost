#pragma once

#include "packers.h"

#include <util/generic/array_ref.h>

// Stores an array of PODs in the trie (copying them with memcpy).
// Byte order and alignment are your problem.

template <class TRecord>
class TRegionPacker {
public:
    typedef TArrayRef<TRecord> TRecords;

    void UnpackLeaf(const char* p, TRecords& result) const {
        size_t len;
        NPackers::TIntegralPacker<size_t>().UnpackLeaf(p, len);
        size_t start = NPackers::TIntegralPacker<size_t>().SkipLeaf(p);
        result = TRecords((TRecord*)(p + start), len);
    }

    void PackLeaf(char* buf, const TRecords& data, size_t computedSize) const {
        size_t len = data.size();
        size_t lenChar = len * sizeof(TRecord);
        size_t start = computedSize - lenChar;
        NPackers::TIntegralPacker<size_t>().PackLeaf(buf, len, NPackers::TIntegralPacker<size_t>().MeasureLeaf(len));
        memcpy(buf + start, data.data(), lenChar);
    }

    size_t MeasureLeaf(const TRecords& data) const {
        size_t len = data.size();
        return NPackers::TIntegralPacker<size_t>().MeasureLeaf(len) + len * sizeof(TRecord);
    }

    size_t SkipLeaf(const char* p) const {
        size_t result = NPackers::TIntegralPacker<size_t>().SkipLeaf(p);
        size_t len;
        NPackers::TIntegralPacker<size_t>().UnpackLeaf(p, len);
        result += len * sizeof(TRecord);
        return result;
    }
};
