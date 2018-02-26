#pragma once

#include "cuda_buffer.h"
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/buffer_resharding.h>

//read/write ignoring mapping.
template <class T, class TMapping, NCudaLib::EPtrType Type>
inline void Read(const TVector<TCudaBuffer<T, TMapping, Type>>& src, TVector<TVector<T>>& dst) {
    dst.resize(src.size());
    for (ui32 i = 0; i < dst.size(); ++i) {
        src[i].Read(dst[i]);
    }
};

template <class T, class TMapping, NCudaLib::EPtrType Type>
inline void Read(const TVector<TVector<TCudaBuffer<T, TMapping, Type>>>& src,
                 TVector<TVector<TVector<T>>>& dst) {
    dst.resize(src.size());
    for (ui32 i = 0; i < dst.size(); ++i) {
        Read(src[i], dst[i]);
    }
};

template <class T, class TMapping, NCudaLib::EPtrType Type>
inline void Write(const TVector<TVector<T>>& src,
                  TVector<TCudaBuffer<T, TMapping, Type>>& dst) {
    CB_ENSURE(dst.size() == src.size());

    for (ui32 i = 0; i < dst.size(); ++i) {
        CB_ENSURE(dst[i].GetObjectsSlice().Size() == src[i].size());
        dst[i].Write(src[i]);
    }
};

template <class T, class TMapping, NCudaLib::EPtrType Type>
inline void Write(const TVector<TVector<TVector<T>>>& src,
                  TVector<TVector<TCudaBuffer<T, TMapping, Type>>>& dst) {
    CB_ENSURE(dst.size() == src.size());

    for (ui32 i = 0; i < dst.size(); ++i) {
        CB_ENSURE(dst[i].size() == src[i].size());
        Write(src[i], dst[i]);
    }
};

template <class T,
          class TMapping,
          NCudaLib::EPtrType Type>
void ThroughHostBroadcast(const TVector<T>& values,
                          NCudaLib::TCudaBuffer<T, TMapping, Type>& dst,
                          ui32 stream = 0,
                          bool compress = false) {
    ui64 firstDevSize = dst.GetMapping().DeviceSlice(0).Size();
    for (ui32 dev = 1; dev < NCudaLib::GetCudaManager().GetDeviceCount(); ++dev) {
        CB_ENSURE(firstDevSize == dst.GetMapping().DeviceSlice(dev).Size());
    }
    NCudaLib::TCudaBuffer<T, NCudaLib::TSingleMapping, NCudaLib::EPtrType::CudaDevice> tmp;
    tmp.Reset(NCudaLib::TSingleMapping(0, firstDevSize));
    tmp.Write(values, stream);
    NCudaLib::Reshard(tmp, dst, stream, compress);
};
