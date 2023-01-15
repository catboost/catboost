#include <catboost/cuda/ut_helpers/test_utils.h>
#include <library/cpp/testing/unittest/registar.h>

#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/all_reduce.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset_builder.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset_builder.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>
#include <catboost/cuda/methods/histograms_helper.h>
#include <catboost/cuda/methods/oblivious_tree_structure_searcher.h>
#include <catboost/cuda/methods/pairwise_oblivious_trees/pairwise_optimization_subsets.h>
#include <catboost/cuda/methods/pairwise_oblivious_trees/pairwise_scores_calcer.h>

#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/helpers/matrix.h>
#include <catboost/private/libs/lapack/linear_system.h>
#include <catboost/private/libs/quantization/grid_creator.h>

#include <util/system/info.h>

using namespace std;
using namespace NCatboostCuda;

Y_UNIT_TEST_SUITE(TPairwiseHistogramTest) {
    inline ui32 GetMatrixOffset(ui32 row, ui32 col) {
        if (col <= row) {
            return row * (row + 1) / 2 + col;
        } else {
            return GetMatrixOffset(col, row);
        }
    }

    template <class T>
    inline void DumpVec(const T* data, ui32 size, const TString& message) {
        Cout << message << ": ";
        for (ui32 i = 0; i < size; ++i) {
            Cout << data[i] << " ";
        }
        Cout << Endl;
    }

    //warning: this split is for GPU histograms
    //in catBoost one-hot is bin == fold, but for pairwise mode it's easier to use bin != fold as 1
    inline bool SplitValue(ui32 bin, ui32 fold, bool isOneHot) {
        if (isOneHot) {
            return bin != fold;
        } else {
            return bin > fold;
        }
    }

    void inline CalcRefMatrices(EFeaturesGroupingPolicy policy,
                                const typename TSharedCompressedIndex<NCatboostCuda::TDocParallelLayout>::TCompressedDataSet& dataSet,
                                const TPairwiseOptimizationSubsets& subsets,
                                ui32 depth,
                                TVector<float>* matrices,
                                TVector<float>* vectors) {
        const ui32 numLeaves = 1ULL << depth;

        const ui32 matrixSize = 4 * numLeaves * numLeaves;
        const ui32 vectorSize = 2 * numLeaves;

        matrices->clear();
        vectors->clear();

        matrices->resize(dataSet.GetBinFeatures(policy).size() * matrixSize);
        vectors->resize(dataSet.GetBinFeatures(policy).size() * vectorSize);

        TVector<float>& binMatrices = *matrices;
        TVector<float>& binVectors = *vectors;

        const auto& cpuGrid = dataSet.GetCpuGrid(policy);
        const auto foldOffset = cpuGrid.ComputeFoldOffsets();
        const TNonDiagQuerywiseTargetDers& target = subsets.GetPairwiseTarget();
        auto& localExecutor = NPar::LocalExecutor();
        const int cpuCount = NSystemInfo::CachedNumberOfCpus();
        if (localExecutor.GetThreadCount() < cpuCount) {
            const int threadsToRun = cpuCount - localExecutor.GetThreadCount() - 1;
            localExecutor.RunAdditionalThreads(threadsToRun);
        }

        for (ui32 dev = 0; dev < GetDeviceCount(); ++dev) {
            const TSlice docSlice = target.Docs.GetMapping().DeviceSlice(dev);

            if (docSlice.Size() == 0) {
                continue;
            }

            TVector<ui32> compressedIndex;
            dataSet.GetCompressedIndex().DeviceView(dev).Read(compressedIndex);

            TVector<float> gradient;
            TVector<float> weights;
            TVector<ui32> docs;

            target.PointWeightedDer.DeviceView(dev).Read(gradient);
            weights.resize(gradient.size());

            if (target.PointDer2OrWeights.GetObjectsSlice().Size()) {
                target.PointDer2OrWeights.DeviceView(dev).Read(weights);
            }

            target.Docs.DeviceView(dev).Read(docs);

            TVector<TDataPartition> pointParts;
            subsets.GetPointPartitions().DeviceView(dev).Read(pointParts);

            TVector<uint2> pairs;
            TVector<float> pairWeights;

            target.Pairs.DeviceView(dev).Read(pairs);
            target.PairDer2OrWeights.DeviceView(dev).Read(pairWeights);

            TVector<TDataPartition> pairParts;
            subsets.GetPairPartitions().DeviceView(dev).Read(pairParts);

            auto computeFeatureHistograms = [&](ui32 f) {
                ui32 featureId = cpuGrid.FeatureIds[f];
                if (dataSet.GetTCFeature(featureId).IsEmpty(dev)) {
                    return;
                }
                const TCFeature feature = dataSet.GetTCFeature(featureId).At(dev);
                const ui32* cindexPtr = &compressedIndex[feature.Offset];

                const bool isOneHot = feature.OneHotFeature;

                for (ui32 binId = 0; binId < cpuGrid.Folds[f]; ++binId) {
                    const ui32 vecOffset = (foldOffset.at(featureId) + binId) * vectorSize;
                    const ui32 matrixOffset = (foldOffset.at(featureId) + binId) * matrixSize;

                    for (ui32 partId = 0; partId < numLeaves; ++partId) {
                        const TDataPartition& part = pointParts[partId];
                        for (ui32 i = 0; i < part.Size; i++) {
                            const ui32 doc = docs[part.Offset + i];
                            const ui32 bin = ((cindexPtr[doc] >> feature.Shift) & feature.Mask);
                            const ui32 leaf = 2 * partId + SplitValue(bin, binId, isOneHot);
                            const ui32 loadAddr = part.Offset + i;
                            CB_ENSURE(loadAddr < gradient.size(), loadAddr);
                            CB_ENSURE(loadAddr < weights.size(), loadAddr);
                            const ui32 writeVecAddr = vecOffset + leaf;
                            const ui32 writeMxAddr = matrixOffset + leaf * vectorSize + leaf;
                            CB_ENSURE(writeVecAddr < binVectors.size(), writeVecAddr << " " << binVectors.size());
                            CB_ENSURE(writeMxAddr < binMatrices.size(), writeMxAddr << " " << binMatrices.size());

                            binVectors[writeVecAddr] += gradient[loadAddr];
                            binMatrices[writeMxAddr] += weights[loadAddr];
                        }
                    }

                    CB_ENSURE(pairParts.size() == numLeaves * numLeaves);
                    for (ui32 p = 0; p < numLeaves * numLeaves; ++p) {
                        const int leafy = GetOddBits(p) * 2;
                        const int leafx = GetEvenBits(p) * 2;

                        for (ui32 i = 0; i < pairParts[p].Size; i++) {
                            const int idx = pairParts[p].Offset + i;
                            CB_ENSURE((ui32)idx < pairWeights.size());
                            const uint2 pair = pairs.at(idx);

                            const ui32 bin0 = ((cindexPtr[pair.x] >> feature.Shift) & feature.Mask);
                            const ui32 bin1 = ((cindexPtr[pair.y] >> feature.Shift) & feature.Mask);
                            const ui32 ci1 = leafy + SplitValue(bin0, binId, isOneHot);
                            const ui32 ci2 = leafx + SplitValue(bin1, binId, isOneHot);

                            binMatrices[matrixOffset + ci1 * vectorSize + ci1] += pairWeights[idx];
                            binMatrices[matrixOffset + ci2 * vectorSize + ci2] += pairWeights[idx];
                            binMatrices[matrixOffset + ci1 * vectorSize + ci2] -= pairWeights[idx];
                            binMatrices[matrixOffset + ci2 * vectorSize + ci1] -= pairWeights[idx];
                        }
                    }
                }
            };
            NPar::ParallelFor(localExecutor, 0, cpuGrid.FeatureIds.size(), computeFeatureHistograms);
        }
    }
    //
    void SolveCPU(const TVector<float>& linearSystems,
                  const TVector<float>& sqrtMatrices,
                  const ui32 rowSize,
                  const ui32 binFeatureCount,
                  bool removeLastRow,
                  TVector<float>* solutions,
                  TVector<float>* scores) {
        const ui32 sqrtSystemSize = rowSize * (rowSize + 1) / 2;
        const ui32 linSystemSize = sqrtSystemSize + rowSize;
        const ui32 vecSize = removeLastRow ? rowSize - 1 : rowSize;

        solutions->resize(rowSize * binFeatureCount);
        scores->resize(binFeatureCount);

        auto computeSolution = [&](int bf) {
            TVector<double> sqrtMatrix(vecSize * vecSize);
            TVector<double> solution(vecSize);

            for (ui32 y = 0; y < vecSize; y++) {
                for (ui32 x = 0; x <= y; x++) {
                    float val = sqrtMatrices[bf * sqrtSystemSize + GetMatrixOffset(y, x)];
                    sqrtMatrix[y * vecSize + x] = val;

                    if (x != y) {
                        sqrtMatrix[x * vecSize + y] = val;
                    }
                }
                solution[y] = linearSystems[bf * linSystemSize + rowSize * (rowSize + 1) / 2 + y];
            }

            SolveLinearSystemCholesky(&sqrtMatrix,
                                      &solution);

            TVector<float> solutionFloat;
            for (const auto& val : solution) {
                solutionFloat.push_back((float)val);
            }

            if (removeLastRow) {
                solutionFloat.push_back(0.0f);
                MakeZeroAverage(&solutionFloat);
            }

            float score = 0;
            const float* targetPtr = linearSystems.data() + bf * linSystemSize + rowSize * (rowSize + 1) / 2;
            for (ui32 i = 0; i < rowSize; ++i) {
                score += targetPtr[i] * solutionFloat[i];
            }

            {
                TVector<float> betaProj(rowSize);
                for (ui32 row = 0; row < rowSize; ++row) {
                    for (ui32 col = 0; col < rowSize; ++col) {
                        betaProj[row] += linearSystems[bf * linSystemSize + GetMatrixOffset(row, col)] * solutionFloat[col];
                    }
                }
                for (ui32 row = 0; row < rowSize; ++row) {
                    score -= 0.5 * betaProj[row] * solutionFloat[row];
                }
            }

            Copy(solutionFloat.begin(), solutionFloat.end(), solutions->begin() + bf * rowSize);
            (*scores)[bf] = score;
        };
        auto& localExecutor = NPar::LocalExecutor();

        NPar::ParallelFor(localExecutor, 0, binFeatureCount, computeSolution);
    };

    TVector<ui32> BuildBinFeaturesOrder(const TVector<TCBinFeature>& features) {
        TVector<ui32> order(features.size());
        Iota(order.begin(), order.end(), 0);
        Sort(order.begin(), order.end(), [&](ui32 i, ui32 j) {
            return features[i] < features[j];
        });
        return order;
    }

    void CheckLinearSystems(EFeaturesGroupingPolicy policy,
                            ui32 depth,
                            const TVector<TCBinFeature>& binFeatures,
                            const TVector<float>& gpuLinearSystems,
                            const TVector<TCBinFeature>& refBinFeatures,
                            const TVector<float>& refMatrices,
                            const TVector<float>& refVectors) {
        UNIT_ASSERT_EQUAL_C(binFeatures.size(), refBinFeatures.size(), "GPU " << binFeatures.size() << "; Ref: " << refBinFeatures.size());

        std::vector<ui32> orderGpu = BuildBinFeaturesOrder(binFeatures);
        std::vector<ui32> orderCpu = BuildBinFeaturesOrder(refBinFeatures);

        const ui32 rowSize = 1 << (depth + 1);
        const ui32 matrixSize = rowSize * rowSize;
        const ui32 linearSystemSize = rowSize * (rowSize + 1) / 2 + rowSize;

        for (ui32 i = 0; i < orderGpu.size(); ++i) {
            UNIT_ASSERT_EQUAL_C(binFeatures[orderGpu[i]].FeatureId, refBinFeatures[orderCpu[i]].FeatureId, binFeatures[orderGpu[i]].FeatureId << " " << refBinFeatures[orderCpu[i]].FeatureId);
            UNIT_ASSERT_EQUAL(binFeatures[orderGpu[i]].BinId, refBinFeatures[orderCpu[i]].BinId);

            const ui32 refVecOffset = orderCpu[i] * rowSize;
            const ui32 refMxOffset = orderCpu[i] * matrixSize;
            const ui32 gpuMxOffset = orderGpu[i] * linearSystemSize;
            const ui32 gpuVecOffset = orderGpu[i] * linearSystemSize + (rowSize * (rowSize + 1)) / 2;

            for (ui32 x = 0; x < rowSize; ++x) {
                if (std::abs(refVectors[refVecOffset + x] - gpuLinearSystems[gpuVecOffset + x]) > 1e-5) {
                    DumpVec(refVectors.data() + refVecOffset, rowSize, "reference");
                    DumpVec(gpuLinearSystems.data() + gpuVecOffset, rowSize, "gpu");
                }
                UNIT_ASSERT_DOUBLES_EQUAL_C(refVectors[refVecOffset + x], gpuLinearSystems[gpuVecOffset + x], 1e-5, "Policy " << policy << " rowSize " << rowSize << ": " << x << " " << i << " " << refVectors[refVecOffset + x] << " " << gpuLinearSystems[gpuVecOffset + x]);
            }

            for (ui32 row = 0; row < rowSize; ++row) {
                for (ui32 col = 0; col <= row; ++col) {
                    const float valGpu = gpuLinearSystems[gpuMxOffset + row * (row + 1) / 2 + col];
                    const float valCpu = refMatrices[refMxOffset + row * rowSize + col];

                    if (std::abs(valCpu - valGpu) > 1e-5) {
                        DumpVec(refMatrices.data() + refMxOffset, matrixSize + 20, "reference");
                        DumpVec(gpuLinearSystems.data() + gpuMxOffset, linearSystemSize, "gpu");
                    }

                    UNIT_ASSERT_DOUBLES_EQUAL_C(valCpu, valGpu, 1e-5,
                                                i << " " << valCpu << " " << valGpu << " " << policy);
                }
            }
        }
    }
    //
    void CheckResults(EFeaturesGroupingPolicy policy,
                      int currentDepth,
                      const typename TSharedCompressedIndex<TDocParallelLayout>::TCompressedDataSet& features,
                      const TPairwiseOptimizationSubsets& subsets,
                      const TBinaryFeatureSplitResults& result) {
        const bool removeLast = !subsets.GetPairwiseTarget().PointDer2OrWeights.GetObjectsSlice().Size();
        TVector<float> matrices;
        TVector<float> vectors;
        CalcRefMatrices(policy, features, subsets, currentDepth, &matrices, &vectors);

        TVector<float> computedSystems;
        TVector<TCBinFeature> binFeatures;

        result.LinearSystems->Read(computedSystems);
        result.BinFeatures.Read(binFeatures);

        CheckLinearSystems(policy,
                           currentDepth,
                           binFeatures,
                           computedSystems,
                           features.GetBinFeatures(policy),
                           matrices,
                           vectors);

        TVector<float> solutionsGpu;
        TVector<float> scoresGpu;

        result.Solutions.Read(solutionsGpu);
        result.Scores.Read(scoresGpu);

        TVector<float> sqrtMatrices;
        result.SqrtMatrices->Read(sqrtMatrices);

        TVector<float> solutionsCpu;
        TVector<float> scoresCpu;

        SolveCPU(computedSystems,
                 sqrtMatrices,
                 1u << (currentDepth + 1),
                 binFeatures.size(),
                 removeLast,
                 &solutionsCpu,
                 &scoresCpu);

        UNIT_ASSERT_EQUAL_C(scoresCpu.size(), scoresGpu.size(), scoresCpu.size() << " neq " << scoresGpu.size());
        UNIT_ASSERT_EQUAL_C(solutionsCpu.size(), solutionsGpu.size(), solutionsCpu.size() << " neq " << solutionsGpu.size());

        for (ui32 i = 0; i < solutionsCpu.size(); ++i) {
            if (std::abs(solutionsCpu[i] - solutionsGpu[i]) > 1e-5f || std::isnan(solutionsCpu[i]) || std::isnan(solutionsGpu[i])) {
                const ui32 rowSize = 1u << (currentDepth + 1);
                ui32 bf = i / (rowSize);
                const ui32 sqrtSystemSize = (rowSize * (rowSize + 1)) / 2;
                const ui32 systemSize = rowSize + sqrtSystemSize;
                DumpVec(computedSystems.data() + bf * systemSize, systemSize, "Linear system");
                DumpVec(sqrtMatrices.data() + bf * sqrtSystemSize, sqrtSystemSize, "Sqrt system");
                DumpVec(solutionsCpu.data() + bf * rowSize, rowSize, "CPU");
                DumpVec(solutionsGpu.data() + bf * rowSize, rowSize, "GPU");
            }
            UNIT_ASSERT_DOUBLES_EQUAL_C(solutionsCpu[i], solutionsGpu[i], 1e-5, i << " " << solutionsCpu[i] << " not equal to " << solutionsGpu[i] << " depth " << (1 << currentDepth));
        }
        Cout << "Solutions OK" << Endl;

        for (ui32 i = 0; i < scoresCpu.size(); ++i) {
            UNIT_ASSERT_DOUBLES_EQUAL_C(scoresCpu[i], scoresGpu[i], 1e-2, i << " " << scoresCpu[i] << " not equal to " << scoresGpu[i] << " depth " << (1 << currentDepth));
        }
        Cout << "Scores OK" << Endl;
    }

    TNonDiagQuerywiseTargetDers ComputeWeakTarget(TRandom & random,
                                                  NCudaLib::TStripeMapping samplesMapping,
                                                  bool useDiagDer2) {
        TNonDiagQuerywiseTargetDers target;
        target.PointWeightedDer.Reset(samplesMapping);

        TVector<uint2> pairs;
        TVector<float> pairTargets;
        TVector<float> pairWeights;
        TVector<float> pointwiseTarget;
        TVector<float> pointwiseWeights;

        pointwiseTarget.resize(samplesMapping.GetObjectsSlice().Size());
        pointwiseWeights.resize(samplesMapping.GetObjectsSlice().Size());

        auto pairsMapping = samplesMapping.Transform([&](const TSlice samples) -> ui64 {
            ui64 estimatedSize = samples.Size() + random.NextUniformL() % (samples.Size() * 10);
            ui64 currentPairCount = pairs.size();
            for (ui32 i = 0; i < estimatedSize; ++i) {
                uint2 pair;
                pair.x = random.NextUniformL() % samples.Size();
                pair.y = random.NextUniformL() % samples.Size();
                if (pair.x != pair.y) {
                    pairs.push_back(pair);
                    const float w = (random.NextUniformL() % 2) * 1.0 / 2;
                    const float t = (random.NextUniformL() % 8) * w / 8;
                    pairWeights.push_back(w);

                    pointwiseTarget[samples.Left + pair.x] += t;
                    pointwiseTarget[samples.Left + pair.y] -= t;
                }
            }

            if (useDiagDer2) {
                for (ui32 i = samples.Left; i < samples.Right; ++i) {
                    const float w = (random.NextUniformL() % 4) * 1.0 / 4;
                    const float t = (random.NextUniformL() % 8) * w / 8;

                    pointwiseTarget[i] += t;
                    pointwiseWeights[i] += w;
                }
            }
            return pairs.size() - currentPairCount;
        });

        target.Pairs.Reset(pairsMapping);
        target.PairDer2OrWeights.Reset(pairsMapping);

        target.Pairs.Write(pairs);
        target.PairDer2OrWeights.Write(pairWeights);

        target.Docs.Reset(samplesMapping);
        MakeSequence(target.Docs);
        target.PointWeightedDer.Reset(samplesMapping);
        target.PointWeightedDer.Write(pointwiseTarget);

        if (useDiagDer2) {
            target.PointDer2OrWeights.Reset(samplesMapping);
            target.PointDer2OrWeights.Write(pointwiseWeights);
        }
        return target;
    }

    //
    void TreeSearcherTest(const TDocParallelDataSet& dataSet,
                          const TBinarizedFeaturesManager& featuresManager,
                          ui32 maxDepth,
                          bool nzDer2) {
        TRandom random(10);
        auto samplesMapping = dataSet.GetSamplesMapping();

        TPairwiseOptimizationSubsets subsets(ComputeWeakTarget(random, samplesMapping, nzDer2),
                                             maxDepth);

        using TScoreCalcer = TPairwiseScoreCalcer;
        using TScoreCalcerPtr = THolder<TScoreCalcer>;

        TScoreCalcerPtr featuresScoreCalcer;
        TScoreCalcerPtr simpleCtrScoreCalcer;

        NCatboostOptions::TObliviousTreeLearnerOptions treeConfig(ETaskType::GPU);
        treeConfig.MaxDepth = maxDepth;
        treeConfig.L2Reg = 10.0;
        treeConfig.PairwiseNonDiagReg = 1;

        if (dataSet.HasFeatures()) {
            featuresScoreCalcer = MakeHolder<TScoreCalcer>(dataSet.GetFeatures(),
                                                   treeConfig,
                                                   subsets,
                                                   random,
                                                   true);
        }

        if (dataSet.HasPermutationDependentFeatures()) {
            simpleCtrScoreCalcer = MakeHolder<TScoreCalcer>(dataSet.GetPermutationFeatures(),
                                                    treeConfig,
                                                    subsets,
                                                    random,
                                                    true);
        }

        for (ui32 depth = 0; depth < maxDepth; ++depth) {
            TBinarySplit bestSplit;
            {
                if (featuresScoreCalcer) {
                    featuresScoreCalcer->Compute();
                }
                if (simpleCtrScoreCalcer) {
                    simpleCtrScoreCalcer->Compute();
                }
            }
            NCudaLib::GetCudaManager().Barrier();

            if (featuresScoreCalcer) {
                for (auto policy : GetEnumAllValues<NCatboostCuda::EFeaturesGroupingPolicy>()) {
                    if (featuresScoreCalcer->HasHelperForPolicy(policy)) {
                        const TBinaryFeatureSplitResults& results = featuresScoreCalcer->GetResultsForPolicy(
                            policy);
                        CheckResults(policy,
                                     depth,
                                     dataSet.GetFeatures(),
                                     subsets,
                                     results);
                    }
                }
            }

            if (simpleCtrScoreCalcer) {
                for (auto policy : GetEnumAllValues<NCatboostCuda::EFeaturesGroupingPolicy>()) {
                    if (simpleCtrScoreCalcer->HasHelperForPolicy(policy)) {
                        const TBinaryFeatureSplitResults& results = simpleCtrScoreCalcer->GetResultsForPolicy(policy);
                        CheckResults(policy,
                                     depth,
                                     dataSet.GetPermutationFeatures(),
                                     subsets,
                                     results);
                    }
                }
            }

            NCudaLib::GetCudaManager().WaitComplete();
            {
                auto featureIds = dataSet.GetFeatures().GetFeatures();
                auto localIdx = random.NextUniformL() % featureIds.size();
                bestSplit.FeatureId = featureIds[localIdx];
                bestSplit.BinIdx = random.NextUniformL() % featuresManager.GetBinCount(bestSplit.FeatureId);
                bestSplit.SplitType = featuresManager.IsCat(localIdx) ? EBinSplitType::TakeBin : EBinSplitType::TakeGreater;
            }

            if (((depth + 1) != treeConfig.MaxDepth)) {
                subsets.Split(dataSet.GetCompressedIndex().GetStorage(),
                              dataSet.GetTCFeature(bestSplit.FeatureId),
                              bestSplit.BinIdx);
            }
        }
    }

    void TestPairwiseHist(ui32 binarization,
                          ui32 oneHotLimit,
                          ui32 permutationCount,
                          ui32 maxDepth,
                          bool nzDiagWeights) {
        NCatboostOptions::TBinarizationOptions floatBinarization(EBorderSelectionType::GreedyLogSum,
                                                                 binarization);
        NCatboostOptions::TCatFeatureParams catFeatureParams(ETaskType::GPU);
        catFeatureParams.MaxTensorComplexity = 3;
        catFeatureParams.OneHotMaxSize = oneHotLimit;

        NCB::TTrainingDataProviderPtr dataProvider;
        THolder<TBinarizedFeaturesManager> featuresManager;
        NCB::TFeatureEstimators estimators;

        LoadTrainingData(NCB::TPathWithScheme("dsv://test-pool.txt"),
                         NCB::TPathWithScheme("dsv://test-pool.txt.cd"),
                         floatBinarization,
                         catFeatureParams,
                         estimators,
                         &dataProvider,
                         &featuresManager);

        NCB::TOnCpuGridBuilderFactory gridBuilderFactory;
        {
            NCatboostOptions::TBinarizationOptions bucketsBinarization(EBorderSelectionType::GreedyLogSum, binarization);
            NCatboostOptions::TBinarizationOptions freqBinarization(EBorderSelectionType::GreedyLogSum, binarization);

            TVector<TVector<float>> prior = {{0.5, 1.0}};
            NCatboostOptions::TCtrDescription bucketsCtr(ECtrType::Buckets, prior, bucketsBinarization);
            NCatboostOptions::TCtrDescription freqCtr(ECtrType::FeatureFreq, prior, freqBinarization);
            catFeatureParams.AddSimpleCtrDescription(bucketsCtr);
            catFeatureParams.AddSimpleCtrDescription(freqCtr);

            catFeatureParams.AddTreeCtrDescription(bucketsCtr);
            catFeatureParams.AddTreeCtrDescription(freqCtr);
        }


        TDocParallelDataSetBuilder dataSetsHolderBuilder(*featuresManager,
                                                         *dataProvider,
                                                         estimators,
                                                         nullptr);

        auto dataSet = dataSetsHolderBuilder.BuildDataSet(permutationCount, &NPar::LocalExecutor());

        for (ui32 i = 0; i < dataSet.PermutationsCount(); ++i) {
            TreeSearcherTest(dataSet.GetDataSetForPermutation(i), *featuresManager, maxDepth, nzDiagWeights);
        }
    }

    void RunTests(ui32 seed,
                  int oneHotLimit,
                  bool nzDiagWeights = false) {
        TRandom random(seed);
        TBinarizedPool pool;

        auto stopCudaManagerGuard = StartCudaManager();
        {
            for (ui32 bin : {2, 15, 20, 32, 60, 64, 128, 255}) {
                {
                    Cout << "Test bin count #" << bin << Endl;
                    const ui32 numCatFeatures = 32;
                    GenerateTestPool(pool, bin, numCatFeatures, random.NextUniformL());

                    SavePoolToFile(pool, "test-pool.txt");
                    SavePoolCDToFile("test-pool.txt.cd", numCatFeatures);

                    TestPairwiseHist(bin, oneHotLimit == -1 ? bin : oneHotLimit, 4, 7, nzDiagWeights);
                }
            }
        }
    }

    Y_UNIT_TEST(TestPairwiseHistWithoutOneHot) {
        RunTests(0, 0);
    }

    Y_UNIT_TEST(TestPairwiseHistWithOneHot) {
        RunTests(0, -1);
    }

    Y_UNIT_TEST(TestPairwiseHistPlusDiagDer2WithoutOneHot) {
        RunTests(0, 0, true);
    }

    Y_UNIT_TEST(TestPairwiseHistPlusDiagDer2WithOneHot) {
        RunTests(0, 6, true);
    }
}
