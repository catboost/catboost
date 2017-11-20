#pragma once

#include "cuda_buffer.h"

//read/write ignoring mapping.
template<class T, class TMapping, NCudaLib::EPtrType Type>
inline void Read(const TVector<TCudaBuffer<T, TMapping, Type>>& src, TVector<TVector<T>>& dst)
{
    dst.resize(src.size());
    for (ui32 i = 0; i < dst.size(); ++i)
    {
        src[i].Read(dst[i]);
    }

};

template<class T, class TMapping, NCudaLib::EPtrType Type>
inline void Read(const TVector<TVector<TCudaBuffer<T, TMapping, Type>>>& src,
                 TVector<TVector<TVector<T>>>& dst)
{
    dst.resize(src.size());
    for (ui32 i = 0; i < dst.size(); ++i)
    {
        Read(src[i], dst[i]);
    }
};

template<class T, class TMapping, NCudaLib::EPtrType Type>
inline void Write(const TVector<TVector<T>>& src,
                  TVector<TCudaBuffer<T, TMapping, Type>>& dst)
{
    CB_ENSURE(dst.size() == src.size());

    for (ui32 i = 0; i < dst.size(); ++i)
    {
        CB_ENSURE(dst[i].GetObjectsSlice().Size() == src[i].size());
        dst[i].Write(src[i]);
    }

};

template<class T, class TMapping, NCudaLib::EPtrType Type>
inline void Write(const TVector<TVector<TVector<T>>>& src,
                  TVector<TVector<TCudaBuffer<T, TMapping, Type>>>& dst)
{
    CB_ENSURE(dst.size() == src.size());

    for (ui32 i = 0; i < dst.size(); ++i)
    {
        CB_ENSURE(dst[i].size() == src[i].size());
        Write(src[i], dst[i]);
    }
};

