#include <library/cpp/testing/unittest/registar.h>

#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/methods/leaves_estimation/leaves_estimation_helper.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/metrics/optimal_const_for_loss.h>

#include <util/system/info.h>

using namespace NCatboostCuda;

Y_UNIT_TEST_SUITE(TExactLeavesEstimationTest) {
    void FillFloatVector(TVector<float>& data, TRandom& random) {
        std::generate(data.begin(), data.end(), [&]() {
            return random.NextUniform();
        });
    }

    void FillBins(TVector<ui32>& data, ui32 numberOfLeaves, TRandom& random) {
        const int maxObjectsInBin = data.size() / 2;

        ui32 i = 0;
        while (i < data.size()) {
            // The minimum number of cells is 100 because binary search on the CPU
            // is only used when the number of objects is greater than 100.
            const int count = 100 + random.NextUniformL() % maxObjectsInBin;
            const int bin = random.NextUniformL() % numberOfLeaves;

            for (ui32 j = i; j < count + i && j < data.size(); ++j) {
                data[j] = bin;
            }

            i += count;
        }
    }

    template <class TKey>
    TCudaBuffer<TKey, NCudaLib::TSingleMapping> MakeCudaBuffer(const TVector<TKey>& source) {
        auto dataMapping = NCudaLib::TSingleMapping(0, source.size());
        auto buffer = TSingleBuffer<TKey>::Create(dataMapping);
        buffer.Write(source);

        return buffer;
    }

    NCatboostOptions::TLossDescription MakeLossDescription(const ELossFunction& lossFunction) {
        NCatboostOptions::TLossDescription lossDescription = NCatboostOptions::TLossDescription();
        lossDescription = lossDescription.CloneWithLossFunction(lossFunction);

        return std::move(lossDescription);
    }

    void CalculatePointOnCPU(TVector<float>& point,
                             const TVector<float>& values,
                             const TVector<float>& weights,
                             const TVector<ui32>& bins,
                             const NCatboostOptions::TLossDescription& lossDescription) {
        ui32 pointSize = point.size();

        TVector<TVector<float>> leavesValues(pointSize);
        TVector<TVector<float>> leavesWeights(pointSize);
        for (size_t index = 0; index < bins.size(); ++index) {
            leavesValues[bins[index]].push_back(values[index]);
            leavesWeights[bins[index]].push_back(weights[index]);
        }

        CB_ENSURE(leavesValues.size() == leavesWeights.size());
        for (ui32 leafNum = 0; leafNum < pointSize; ++leafNum) {
            Y_UNUSED(lossDescription);
            point[leafNum] = *NCB::CalcOneDimensionalOptimumConstApprox(lossDescription,
                                                                        leavesValues[leafNum],
                                                                        leavesWeights[leafNum]);
        }
    }

    void CalculatePointOnGPU(TVector<float>& point,
                             const TVector<float>& values,
                             const TVector<float>& weights,
                             const TVector<ui32>& bins,
                             const NCatboostOptions::TLossDescription& lossDescription,
                             ui32 binCount) {
        auto binsBuffer = MakeCudaBuffer<ui32>(bins);
        auto valuesBuffer = MakeCudaBuffer<float>(values);
        auto weightsBuffer = MakeCudaBuffer<float>(weights);

        ComputeExactApprox(binsBuffer,
                           valuesBuffer,
                           weightsBuffer,
                           binCount,
                           point,
                           lossDescription,
                           /*binarySearchIterations =*/ 100);
    }

    void RunTests(ui32 seed,
                  ui32 numberOfObjects,
                  ui32 numberOfLeaves,
                  const NCatboostOptions::TLossDescription& lossDescription) {
        TRandom random(seed);

        auto& localExecutor = NPar::LocalExecutor();
        const int cpuCount = NSystemInfo::CachedNumberOfCpus();
        if (localExecutor.GetThreadCount() < cpuCount) {
            const int threadsToRun = cpuCount - localExecutor.GetThreadCount() - 1;
            localExecutor.RunAdditionalThreads(threadsToRun);
        }

        //
        TVector<float> values(numberOfObjects);
        FillFloatVector(values, random);

        TVector<float> weights(numberOfObjects);
        FillFloatVector(weights, random);

        TVector<ui32> bins(numberOfObjects);
        FillBins(bins, numberOfLeaves, random);
        //

        auto stopCudaManagerGuard = StartCudaManager();
        {
            TVector<float> cpuPoint(numberOfLeaves);
            TVector<float> gpuPoint(numberOfLeaves);

            NCudaLib::TCudaProfiler &profiler = NCudaLib::GetCudaManager().GetProfiler();
            SetDefaultProfileMode(NCudaLib::EProfileMode::ImplicitLabelSync);
            {
                auto guardCpu = profiler.Profile(TStringBuilder() << "Computed approx on CPU for #" <<
                                                                  numberOfObjects << " objects and #" << numberOfLeaves
                                                                  << " leaves: ");
                CalculatePointOnCPU(cpuPoint, values, weights, bins, lossDescription);

                auto guardGpu = profiler.Profile(TStringBuilder() << "Computed approx on GPU for #" <<
                                                                  numberOfObjects << " objects and #" << numberOfLeaves
                                                                  << " leaves: ");
                CalculatePointOnGPU(gpuPoint, values, weights, bins, lossDescription, numberOfLeaves);
            }
            profiler.PrintInfo();


            UNIT_ASSERT(cpuPoint.size() == gpuPoint.size());
            constexpr double PRECISION = 1e-4;
            for (ui32 i = 0; i < cpuPoint.size(); ++i) {
                if (std::fabs(cpuPoint[i] - gpuPoint[i]) > PRECISION)
                    Cout << "Assertion has failed: cpuPoint[" << i << "] = " << cpuPoint[i] << ", but gpuPoint[" << i << "] = " << gpuPoint[i] << Endl;

                UNIT_ASSERT_DOUBLES_EQUAL(cpuPoint[i], gpuPoint[i], PRECISION);
            }
        }
    }

    // MAE
    Y_UNIT_TEST(TExactLeavesEstimationWithMAELossTest1) {
        RunTests(42, 100, 6, MakeLossDescription(ELossFunction::MAE));
    }

    Y_UNIT_TEST(TExactLeavesEstimationWithMAELossTest2) {
        RunTests(42, 512, 4, MakeLossDescription(ELossFunction::MAE));
    }

    Y_UNIT_TEST(TExactLeavesEstimationWithMAELossTest3) {
        RunTests(10, 1000, 1024, MakeLossDescription(ELossFunction::MAE));
    }

    Y_UNIT_TEST(TExactLeavesEstimationWithMAELossTest4) {
        RunTests(42, 8000, 32, MakeLossDescription(ELossFunction::MAE));
    }

    // Quantile
    Y_UNIT_TEST(TExactLeavesEstimationWithQuantileLossTest1) {
        RunTests(42, 1420, 8, MakeLossDescription(ELossFunction::Quantile));
    }

    Y_UNIT_TEST(TExactLeavesEstimationWithQuantileLossTest2) {
        RunTests(42, 1420, 128, MakeLossDescription(ELossFunction::Quantile));
    }

    Y_UNIT_TEST(TExactLeavesEstimationWithQuantileLossTest3) {
        RunTests(10, 20000, 6400, MakeLossDescription(ELossFunction::Quantile));
    }

    // MAPE
    Y_UNIT_TEST(TExactLeavesEstimationWithMAPELossTest1) {
        RunTests(77, 100, 2, MakeLossDescription(ELossFunction::MAPE));
    }

    Y_UNIT_TEST(TExactLeavesEstimationWithMAPELossTest2) {
        RunTests(77, 512, 4, MakeLossDescription(ELossFunction::MAPE));
    }

    Y_UNIT_TEST(TExactLeavesEstimationWithMAPELossTest3) {
        RunTests(77, 2000, 1024, MakeLossDescription(ELossFunction::MAPE));
    }

    Y_UNIT_TEST(TExactLeavesEstimationWithMAPELossTest4) {
        RunTests(77, 4096, 32, MakeLossDescription(ELossFunction::MAPE));
    }

    Y_UNIT_TEST(TExactLeavesEstimationWithMAPELossTest5) {
        RunTests(77, 1024, 8, MakeLossDescription(ELossFunction::MAPE));
    }

    Y_UNIT_TEST(TExactLeavesEstimationWithMAPELossTest6) {
        RunTests(77, 1330, 31, MakeLossDescription(ELossFunction::MAPE));
    }

    Y_UNIT_TEST(TExactLeavesEstimationWithMAPELossTest7) {
        RunTests(77, 1330, 4200, MakeLossDescription(ELossFunction::MAPE));
    }

    Y_UNIT_TEST(TExactLeavesEstimationWithMAPELossTest8) {
        RunTests(78, 30450, 7001, MakeLossDescription(ELossFunction::MAPE));
    }
}
