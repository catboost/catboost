#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/fwd.h>

template <class TMapping>
void PoissonBootstrap(
    NCudaLib::TCudaBuffer<ui64, TMapping>& seeds,
    NCudaLib::TCudaBuffer<float, TMapping>& weights,
    float lambda,
    ui32 stream = 0);

template <class TMapping>
void UniformBootstrap(
    NCudaLib::TCudaBuffer<ui64, TMapping>& seeds,
    NCudaLib::TCudaBuffer<float, TMapping>& weights,
    float takenFraction = 0.5,
    ui32 stream = 0);

template <class TMapping>
void BayesianBootstrap(
    NCudaLib::TCudaBuffer<ui64, TMapping>& seeds,
    NCudaLib::TCudaBuffer<float, TMapping>& weights,
    float temperature,
    ui32 stream = 0);

template <class TMapping>
void MvsBootstrapRadixSort(
    NCudaLib::TCudaBuffer<ui64, TMapping>& seeds,
    NCudaLib::TCudaBuffer<float, TMapping>& weights,
    const NCudaLib::TCudaBuffer<float, TMapping>& ders,
    float takenFraction,
    float lambda,
    ui32 stream = 0);

template <class TMapping>
TVector<float> CalculateMvsThreshold(
    NCudaLib::TCudaBuffer<float, TMapping>& candidates,
    float takenFraction,
    ui32 stream = 0);
