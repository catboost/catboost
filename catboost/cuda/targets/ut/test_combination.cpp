#include <library/cpp/testing/unittest/registar.h>
#include <catboost/cuda/ut_helpers/test_utils.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset_builder.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/gpu_random.h>
#include <catboost/libs/helpers/matrix.h>
#include <catboost/cuda/targets/combination_targets_impl.h>
#include <catboost/cuda/targets/pointwise_target_impl.h>
#include <catboost/cuda/targets/querywise_targets_impl.h>
#include <catboost/cuda/cuda_util/helpers.h>
#include <catboost/private/libs/quantization/grid_creator.h>

#include <util/system/info.h>
#include <util/generic/ymath.h>

using namespace std;
using namespace NCudaLib;
using namespace NCatboostCuda;

Y_UNIT_TEST_SUITE(TCombinationTargetTests) {
    void TestCombinationGradientImpl(ui32 seed, const TString& loss0, const TString& loss1) {
        auto stopCudaManagerGuard = StartCudaManager();

        TBinarizedPool pool;

        const ui32 binarization = 32;
        const ui32 catFeatures = 0;
        GenerateTestPool(pool, binarization, catFeatures, /*seed*/seed);

        SavePoolToFile(pool, "test-pool.txt");
        SavePoolCDToFile("test-pool.txt.cd", catFeatures);

        NCatboostOptions::TBinarizationOptions floatBinarization(EBorderSelectionType::GreedyLogSum, binarization);

        NCB::TTrainingDataProviderPtr dataProvider;
        THolder<TBinarizedFeaturesManager> featuresManager;
        NCB::TFeatureEstimators estimators;

        const NCatboostOptions::TCatFeatureParams catFeatureOptions(ETaskType::GPU);
        LoadTrainingData(
            NCB::TPathWithScheme("dsv://test-pool.txt"),
            NCB::TPathWithScheme("dsv://test-pool.txt.cd"),
            floatBinarization,
            catFeatureOptions,
            estimators,
            &dataProvider,
            &featuresManager);

        NCB::TOnCpuGridBuilderFactory gridBuilderFactory;

        {
            const auto dataProviderTarget = *dataProvider->TargetData->GetOneDimensionalTarget();

            featuresManager->SetTargetBorders(
                NCB::TBordersBuilder(
                    gridBuilderFactory,
                    dataProviderTarget)(floatBinarization));

            const auto& targetBorders = featuresManager->GetTargetBorders();
            UNIT_ASSERT_VALUES_EQUAL(targetBorders.size(), 4);
        }

        UNIT_ASSERT_VALUES_EQUAL(
            pool.NumFeatures + catFeatures,
            dataProvider->MetaInfo.FeaturesLayout->GetExternalFeatureCount());
        UNIT_ASSERT_VALUES_EQUAL(
            pool.NumSamples, dataProvider->GetObjectCount());

        TDocParallelDataSetBuilder dataSetsHolderBuilder(
            *featuresManager,
            *dataProvider,
            estimators);
        const TDocParallelDataSetsHolder dataSet = dataSetsHolderBuilder.BuildDataSet(/*permutationCount*/1, &NPar::LocalExecutor());

        const auto& dataSetRef = dataSet.GetDataSetForPermutation(0);

        TGpuAwareRandom random(seed);

        TVector<float> cursor(dataProvider->GetObjectCount());
        for (auto& element : cursor) {
            element = 2 * random.NextUniform();
        }
        auto gpuCursor = TStripeBuffer<float>::CopyMapping(dataSetRef.GetTarget().GetTargets());
        gpuCursor.Write(cursor);

        TPointwiseTargetsImpl<NCudaLib::TStripeMapping> rmseTarget(
            dataSetRef,
            random,
            NCatboostOptions::ParseLossDescription(loss0));

        auto gpuWeightedDer0 = TStripeBuffer<float>::CopyMapping(gpuCursor);
        auto gpuWeights0 = TStripeBuffer<float>::CopyMapping(gpuCursor);
        rmseTarget.GradientAt(gpuCursor.AsConstBuf(), gpuWeightedDer0, gpuWeights0);
        TVector<float> weightedDer0, weights0;
        gpuWeightedDer0.Read(weightedDer0);
        gpuWeights0.Read(weights0);

        TQuerywiseTargetsImpl<NCudaLib::TStripeMapping> queryRmseTarget(
            dataSetRef,
            random,
            NCatboostOptions::ParseLossDescription(loss1));

        auto gpuWeightedDer1 = TStripeBuffer<float>::CopyMapping(gpuCursor);
        auto gpuWeights1 = TStripeBuffer<float>::CopyMapping(gpuCursor);
        queryRmseTarget.GradientAt(gpuCursor.AsConstBuf(), gpuWeightedDer1, gpuWeights1);
        TVector<float> weightedDer1, weights1;
        gpuWeightedDer1.Read(weightedDer1);
        gpuWeights1.Read(weights1);

        const float w0 = 0.5, w1 = 0.5;
        TCombinationTargetsImpl<NCudaLib::TStripeMapping> combinationTarget(
            dataSetRef,
            random,
            NCatboostOptions::ParseLossDescription(
                "Combination:loss0=" + loss0 + ";weight0=" + ToString(w0) + ";"
                "loss1=" + loss1 + ";weight1=" + ToString(w1)));

        auto gpuWeightedDer2 = TStripeBuffer<float>::CopyMapping(gpuCursor);
        auto gpuWeights2 = TStripeBuffer<float>::CopyMapping(gpuCursor);
        combinationTarget.GradientAt(gpuCursor.AsConstBuf(), gpuWeightedDer2, gpuWeights2);
        TVector<float> weightedDer2, weights2;
        gpuWeightedDer2.Read(weightedDer2);
        gpuWeights2.Read(weights2);

        const double eps = 1e-5;
        ui32 errorCount = 0;
        for (ui32 idx : xrange(dataProvider->GetObjectCount())) {
            if (std::abs(weightedDer2[idx] - (weightedDer0[idx] * w0 + weightedDer1[idx] * w1)) >= eps) {
                Cout << idx << " " << weightedDer2[idx] << " " << weightedDer0[idx] << " " << weightedDer1[idx] << Endl;
                ++errorCount;
            }
        }
        Cout << "error count: " << errorCount << Endl;
        UNIT_ASSERT_EQUAL(errorCount, 0);
    }

    Y_UNIT_TEST(TestCombinationLoglossQuerySoftMax) {
        TestCombinationGradientImpl(/*seed*/0, "Logloss", "QuerySoftMax");
    }

    Y_UNIT_TEST(TestCombinationRMSEQueryRMSE) {
        TestCombinationGradientImpl(/*seed*/42, "RMSE", "QueryRMSE");
    }

    Y_UNIT_TEST(TestCombinationLoglossQuerySoftMaxBeta) {
        TestCombinationGradientImpl(/*seed*/96, "Logloss", "QuerySoftMax:beta=0.5");
    }
}
