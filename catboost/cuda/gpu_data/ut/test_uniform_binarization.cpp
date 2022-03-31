#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>

#include <catboost/cuda/cuda_util/gpu_random.h>
#include <catboost/cuda/gpu_data/kernels.h>
#include <catboost/cuda/gpu_data/kernel/binarize.cuh>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/algorithm.h>
#include <util/generic/ymath.h>

#include <iostream>

using namespace std;
using namespace NCudaLib;

Y_UNIT_TEST_SUITE(TGridBuilderPerftest) {
    Y_UNIT_TEST(TestUniformBuilder) {
        TSetLogging inThisScope(ELoggingLevel::Debug);
        auto stopCudaManagerGuard = StartCudaManager();
        {
            ui64 tries = 20;
            TRandom rand(0);
            TCudaProfiler& profiler = NCudaLib::GetCudaManager().GetProfiler();
            SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);

            NCatboostOptions::TBinarizationOptions binarizationDescription;
            binarizationDescription.BorderCount = 64;
            binarizationDescription.BorderSelectionType = EBorderSelectionType::Uniform;
            TCudaStream stream = GetStreamsProvider().RequestStream();
            for (ui32 size = 100; size < 1000000001; size *= 10) {
                TVector<float> vecCpu;
                for (ui64 i = 0; i < size; ++i) {
                    vecCpu.push_back(abs(rand.NextUniform()) + 0.01);
                }

                auto mapping = TSingleMapping(0, size);
                auto resultMapping = TSingleMapping(0, binarizationDescription.BorderCount + 1);
                auto cudaVec = TSingleBuffer<float>::Create(mapping);
                auto result = TSingleBuffer<float>::Create(resultMapping);
                cudaVec.Write(vecCpu);
                for (ui32 k = 0; k < tries; ++k) {
                    auto guard = profiler.Profile(TStringBuilder() << "Calculate Uniform grid for #" << size << " elements");
                    NKernel::ComputeUniformBorders(
                        cudaVec.At(0).Get(),
                        cudaVec.At(0).Size(),
                        result.At(0).Get(),
                        binarizationDescription.BorderCount,
                        stream
                    );
                    stream.Synchronize();
                }
            }
        }
    }
}
