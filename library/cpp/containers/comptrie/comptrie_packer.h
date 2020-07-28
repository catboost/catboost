#pragma once

#include <library/cpp/packers/packers.h>

template <class T>
class TCompactTriePacker {
public:
    void UnpackLeaf(const char* p, T& t) const {
        NPackers::TPacker<T>().UnpackLeaf(p, t);
    }
    void PackLeaf(char* buffer, const T& data, size_t computedSize) const {
        NPackers::TPacker<T>().PackLeaf(buffer, data, computedSize);
    }
    size_t MeasureLeaf(const T& data) const {
        return NPackers::TPacker<T>().MeasureLeaf(data);
    }
    size_t SkipLeaf(const char* p) const // this function better be fast because it is very frequently used
    {
        return NPackers::TPacker<T>().SkipLeaf(p);
    }
};
