#pragma once

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
