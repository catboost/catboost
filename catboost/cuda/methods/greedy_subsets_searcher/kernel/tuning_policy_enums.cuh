#pragma once

#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>

/*
 * All routines here assume histograms are zeroed externally
 */

namespace NKernel {

    enum class ECIndexLoadType {
        Direct,
        Gather
    };

    enum class ELoadSize {
        OneElement,
        TwoElements,
        FourElements
    };

    template <ELoadSize Size>
    struct TLoadSize;

    template <>
    struct TLoadSize<ELoadSize::OneElement> {
        static constexpr int Size() {
            return 1;
        }
    };


    template <>
    struct TLoadSize<ELoadSize::TwoElements> {
        static constexpr int Size() {
            return 2;
        }
    };


    template <>
    struct TLoadSize<ELoadSize::FourElements> {
        static constexpr int Size() {
            return 4;
        }
    };



    template <ELoadSize Size>
    struct TLoadSizeHist2;

    template <>
    struct TLoadSizeHist2<ELoadSize::OneElement> {
        static constexpr int Size() {
            return 1;
        }
    };


    template <>
    struct TLoadSizeHist2<ELoadSize::TwoElements> {
        static constexpr int Size() {
            return 2;
        }
    };


    template <>
    struct TLoadSizeHist2<ELoadSize::FourElements> {
        static constexpr int Size() {
            #if __CUDA_ARCH__ < 700
            return 4;
            #else
            return 8;
            #endif
        }
    };

}
