#pragma once

#include "cuda_buffer.h"

//read/write ignoring mapping.
template<class T, class TMapping, NCudaLib::EPtrType Type>
inline void Read(const yvector<TCudaBuffer<T, TMapping, Type>>& src, yvector<yvector<T>>& dst)
{
    dst.resize(src.size());
    for (ui32 i = 0; i < dst.size(); ++i)
    {
        src[i].Read(dst[i]);
    }

};

template<class T, class TMapping, NCudaLib::EPtrType Type>
inline void Read(const yvector<yvector<TCudaBuffer<T, TMapping, Type>>>& src,
                 yvector<yvector<yvector<T>>>& dst)
{
    dst.resize(src.size());
    for (ui32 i = 0; i < dst.size(); ++i)
    {
        Read(src[i], dst[i]);
    }
};

template<class T, class TMapping, NCudaLib::EPtrType Type>
inline void Write(const yvector<yvector<T>>& src,
                  yvector<TCudaBuffer<T, TMapping, Type>>& dst)
{
    CB_ENSURE(dst.size() == src.size());

    for (ui32 i = 0; i < dst.size(); ++i)
    {
        CB_ENSURE(dst[i].GetObjectsSlice().Size() == src[i].size());
        dst[i].Write(src[i]);
    }

};

template<class T, class TMapping, NCudaLib::EPtrType Type>
inline void Write(const yvector<yvector<yvector<T>>>& src,
                  yvector<yvector<TCudaBuffer<T, TMapping, Type>>>& dst)
{
    CB_ENSURE(dst.size() == src.size());

    for (ui32 i = 0; i < dst.size(); ++i)
    {
        CB_ENSURE(dst[i].size() == src[i].size());
        Write(src[i], dst[i]);
    }
};

