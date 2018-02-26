#pragma once

#include "reduce_scatter.h"
#include "buffer_resharding.h"
#include <catboost/cuda/cuda_lib/cuda_buffer.h>

namespace NCudaLib {
    //reduce-scatter followed by resharding
    template <class T>
    inline void AllReduce(TStripeBuffer<T>& tmp,
                          TMirrorBuffer<T>& dst,
                          ui32 stream = 0,
                          bool compress = false) {
        TReducer<TStripeBuffer<T>> reducer(stream);
        reducer(tmp);
        dst.Reset(NCudaLib::TMirrorMapping(tmp.GetObjectsSlice().Size(), tmp.GetMapping().SingleObjectSize()));
        Reshard(tmp, dst, stream, compress);
    }

    template <class T>
    inline TVector<T> ReadReduce(const TStripeBuffer<T>& tmp,
                                 ui32 stream = 0) {
        auto objectsSlice = tmp.GetMapping().DeviceSlice(0);
        TVector<T> result;
        NCudaLib::TCudaBufferReader<TStripeBuffer<T>>(tmp)
            .SetFactorSlice(objectsSlice)
            .SetReadSlice(objectsSlice)
            .SetCustomReadingStream(stream)
            .ReadReduce(result);
        return result;
    }

    //reduce-scatter followed by resharding
    template <class T>
    inline void AllReduceThroughMaster(const TStripeBuffer<T>& tmp,
                                       TMirrorBuffer<T>& dst,
                                       ui32 stream = 0,
                                       bool compress = false) {
        Y_UNUSED(compress);
        auto objectsSlice = tmp.GetMapping().DeviceSlice(0);
        dst.Reset(NCudaLib::TMirrorMapping(objectsSlice.Size(), tmp.GetMapping().SingleObjectSize()));
        if (NCudaLib::GetCudaManager().GetDeviceCount() == 1) {
            //just copy it
            Reshard(tmp, dst, stream);
            //ensure implicit syncing consistency
            NCudaLib::GetCudaManager().WaitComplete();
        } else {
            TVector<T> result = ReadReduce(tmp, stream);
            dst.Write(result);
        }
    }

}
