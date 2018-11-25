#pragma once

#include <catboost/cuda/cuda_lib/fwd.h>

struct TDataPartition;

template <class TMapping>
void CastCopy(
    const NCudaLib::TCudaBuffer<float, TMapping>& input,
    NCudaLib::TCudaBuffer<double, TMapping>* output,
    ui32 streamId = 0);

template <class TMapping>
void ComputePartitionStats(
    const NCudaLib::TCudaBuffer<float, TMapping>& input,
    const NCudaLib::TCudaBuffer<TDataPartition, TMapping>& parts,
    const NCudaLib::TCudaBuffer<ui32, NCudaLib::TMirrorMapping>& partIds,
    NCudaLib::TCudaBuffer<double, TMapping>* output,
    ui32 streamId = 0);

template <class TMapping, class TFloat>
void ComputePartitionStats(
    const NCudaLib::TCudaBuffer<TFloat, TMapping>& input,
    const NCudaLib::TCudaBuffer<ui32, TMapping>& offsets,
    NCudaLib::TCudaBuffer<double, TMapping>* output,
    ui32 streamId = 0);
