#pragma once

#include <catboost/cuda/cuda_lib/fwd.h>

struct TDataPartition;

template <class TMapping>
void UpdatePartitionDimensions(
    const NCudaLib::TCudaBuffer<ui32, TMapping>& sortedBins,
    NCudaLib::TCudaBuffer<TDataPartition, TMapping>& parts,
    ui32 stream = 0);

template <class TMapping>
void UpdatePartitionOffsets(
    const NCudaLib::TCudaBuffer<ui32, TMapping>& sortedBins,
    NCudaLib::TCudaBuffer<ui32, TMapping>& offsets,
    ui32 stream = 0);

template <class TMapping, class TUi32, NCudaLib::EPtrType DstPtr>
void ComputeSegmentSizes(
    const NCudaLib::TCudaBuffer<TUi32, TMapping>& offsets,
    NCudaLib::TCudaBuffer<float, TMapping, DstPtr>& dst,
    ui32 stream = 0);

//
//template <class TMapping>
//inline void WriteInitPartitions(const TMapping& objectsMapping, TCudaBuffer<TDataPartition, TMapping>& parts) {
//
//    const int devId = GetSingleDevId(objectsMapping);
//    const TDataPartition initialPart(0, objectsMapping->SizeAt(devId));
//    TVector<TDataPartition> partitioningInitial;
//    partitioningInitial.push_back(initialPart);
//    parts.Write(partitioningInitial);
//}
