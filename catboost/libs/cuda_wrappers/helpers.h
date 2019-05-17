#pragma once

#include "cuda_vec.h"

#include <util/stream/file.h>

template <class T>
inline void Dump(TArrayRef<T>& arr, TString name) {
    using T_ = std::remove_const_t<T>;
    TVector<T_> tmp(arr.size());
    MemoryCopy<T_>(arr, tmp);
    TOFStream out(name);
    for (auto val : tmp) {
        out << val << Endl;
    }
}

template <class T>
inline void Dump(TCudaVec<T>& arr, TString name) {
    TVector<std::remove_const_t<T>> tmp(arr.Size());
    arr.Read(tmp);
    TOFStream out(name);
    for (auto val : tmp) {
        out << val << Endl;
    }
}

