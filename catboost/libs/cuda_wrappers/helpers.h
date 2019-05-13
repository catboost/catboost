#pragma once

#include "cuda_vec.h"

#include <util/stream/file.h>

template <class T, EMemoryType Type>
inline void Dump(TCudaVec<T>& arr, TString name) {
    TVector<T> tmp(arr.Size());
    arr.Read(tmp);
    TOFStream out(name);
    for (auto val : tmp) {
        out << val << Endl;
    }
}

template <class T>
inline void DumpRef(TArrayRef<T>& arr, TString name) {
    TVector<T> tmp(arr.size());
    MemoryCopy<T>(MakeConstArrayRef(arr), tmp);
    TOFStream out(name);
    bool hasNan = false;
    for (auto val : tmp) {
        out << val << Endl;
        if (!std::isfinite(val)) {
            hasNan = true;
        }
    }
    CB_ENSURE(!hasNan);
}
