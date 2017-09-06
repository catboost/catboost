#include "model_calcer.h"
#include "static_ctr_provider.h"

#include <set>

size_t RepackLeaves(
    const yvector<yvector<yvector<double>>>& modelValues,
    yvector<yvector<double>>* result) {
    if (modelValues.empty()) {
        return 0;
    }
    size_t classCount = modelValues[0].size();
    for (auto& modelTree : modelValues) {
        CB_ENSURE(modelTree.size() == classCount);
        result->emplace_back(classCount * modelTree[0].size());
        auto& resultVec = result->back();
        for (size_t classId = 0; classId < classCount; ++classId) {
            for (size_t bucketId = 0; bucketId < modelTree[classId].size(); ++bucketId) {
                resultVec[bucketId * classCount + classId] = modelTree[classId][bucketId];
            }
        }
    }
    return classCount;
}


namespace NCatBoost {
    void TModelCalcer::InitBinTreesFromCoreModel(const TCoreModel& model) {
        const auto featureCount = model.CatFeatures.ysize() + model.Borders.ysize();
        CatFeatureFlatIndex = model.CatFeatures;
        Sort(CatFeatureFlatIndex);
        FloatFeatureFlatIndex.clear();
        int prevCatFeatureIdx = -1;
        for (auto catFeature : CatFeatureFlatIndex) {
            for (int i = prevCatFeatureIdx + 1; i < catFeature; ++i) {
                FloatFeatureFlatIndex.push_back(i);
            }
            prevCatFeatureIdx = catFeature;
        }

        for (int i = prevCatFeatureIdx + 1; i < featureCount; ++i) {
            FloatFeatureFlatIndex.push_back(i);
        }
        ModelClassCount = RepackLeaves(model.LeafValues, &LeafValues);
        yvector<TModelSplit> usedSplits;
        {
            std::set<TModelSplit> allSplits;
            std::set<TProjection> UniqueCtrProjections;
            for (const auto& treeStruct : model.TreeStruct) {
                for (const auto& split : treeStruct.SelectedSplits) {
                    allSplits.insert(split);
                    if (split.Type == ESplitType::OnlineCtr) {
                        auto& proj = split.OnlineCtr.Ctr.Projection;
                        for (const auto& binF : proj.BinFeatures) {
                            allSplits.insert(TModelSplit(binF));
                        }
                        for (const auto& oheFeature : proj.OneHotFeatures) {
                            allSplits.insert(TModelSplit(oheFeature));
                        }
                    }
                }
            }
            usedSplits.assign(allSplits.begin(), allSplits.end());
        }
        UsedBinaryFeaturesCount = usedSplits.size();
        yhash<TModelSplit, int> binFeatureIndexes;
        for (int i = 0; i < usedSplits.ysize(); ++i) {
            if (usedSplits[i].Type == ESplitType::OnlineCtr) {
                continue;
            }
            int binFeatureIdx = binFeatureIndexes.ysize();
            Y_ASSERT(!binFeatureIndexes.has(usedSplits[i]));
            binFeatureIndexes[usedSplits[i]] = binFeatureIdx;
        }
        for (int i = 0; i < usedSplits.ysize(); ++i) {
            if (usedSplits[i].Type != ESplitType::OnlineCtr) {
                continue;
            }
            int binFeatureIdx = binFeatureIndexes.ysize();
            Y_ASSERT(!binFeatureIndexes.has(usedSplits[i]));
            binFeatureIndexes[usedSplits[i]] = binFeatureIdx;
        }
        BinaryTrees.reserve(model.TreeStruct.size());
        Y_ASSERT(UsedBinaryFeaturesCount == binFeatureIndexes.size());
        for (const auto& treeStruct : model.TreeStruct) {
            yvector<int> binFeaturesTree;
            binFeaturesTree.reserve(treeStruct.SelectedSplits.size());
            for (const auto& split : treeStruct.SelectedSplits) {
                binFeaturesTree.push_back(binFeatureIndexes.at(split));
            }
            BinaryTrees.emplace_back(std::move(binFeaturesTree));
        }
        for (const auto& split : usedSplits) {
            if (split.Type == ESplitType::FloatFeature) {
                if (UsedFloatFeatures.empty() || UsedFloatFeatures.back().FeatureIndex != split.BinFeature.FloatFeature) {
                    UsedFloatFeatures.emplace_back();
                    UsedFloatFeatures.back().FeatureIndex = split.BinFeature.FloatFeature;
                }
                UsedFloatFeatures.back().Borders.push_back(
                    model.Borders[split.BinFeature.FloatFeature][split.BinFeature.SplitIdx]);
            } else if (split.Type == ESplitType::OneHotFeature) {
                if (UsedOHEFeatures.empty() || UsedOHEFeatures.back().CatFeatureIndex != split.OneHotFeature.CatFeatureIdx) {
                    UsedOHEFeatures.emplace_back();
                    UsedOHEFeatures.back().CatFeatureIndex = split.OneHotFeature.CatFeatureIdx;
                }
                UsedOHEFeatures.back().Values.push_back(split.OneHotFeature.Value);
            } else {
                if (UsedCtrFeatures.empty() || UsedCtrFeatures.back().Ctr != split.OnlineCtr.Ctr) {
                    UsedCtrFeatures.emplace_back();
                    UsedCtrFeatures.back().Ctr = split.OnlineCtr.Ctr;
                }
                UsedCtrFeatures.back().Borders.push_back(split.OnlineCtr.Border);
            }
        }
        // remap indexes in projections to be indexes in binarized array
        for (auto& ctrFeature : UsedCtrFeatures) {
            auto& proj = ctrFeature.Ctr.Projection;
            for (auto& binF : proj.BinFeatures) {
                auto ms = TModelSplit(binF);
                BinFeatureIndexes[ms] = binFeatureIndexes.at(ms);
            }
            for (auto& oheF : proj.OneHotFeatures) {
                auto ms = TModelSplit(oheF);
                BinFeatureIndexes[ms] = binFeatureIndexes.at(ms);
            }
            UsedModelCtrs.push_back(ctrFeature.Ctr);
        }
    }

    void TModelCalcer::InitFromFullModel(const TFullModel& fullModel) {
        InitBinTreesFromCoreModel(fullModel);
        CtrProvider.Reset(new TStaticCtrProvider(fullModel.CtrCalcerData));
    }

    void TModelCalcer::InitFromFullModel(TFullModel&& fullModel) {
        InitBinTreesFromCoreModel(fullModel);
        CtrProvider.Reset(new TStaticCtrProvider(std::move(fullModel.CtrCalcerData)));
    }

    void TModelCalcer::CalcFlat(const yvector<NArrayRef::TConstArrayRef<float>>& features,
                                size_t treeStart,
                                size_t treeEnd,
                                NArrayRef::TArrayRef<double> results) const {
        size_t blockSize;
        if (UsedCtrFeatures.empty()) {
            blockSize = 128;
        } else {
            blockSize = 4096;
        }
        blockSize = Min(blockSize, features.size());
        CB_ENSURE(results.size() == features.size() * ModelClassCount);
        std::fill(results.begin(), results.end(), 0.0);
        yvector<ui8> binFeatures(UsedBinaryFeaturesCount * blockSize);
        yvector<ui32> indexesVec(blockSize);
        yvector<int> transposedHash(blockSize * CatFeatureFlatIndex.size());
        yvector<float> ctrs(UsedModelCtrs.size() * blockSize);
        for (size_t blockStart = 0; blockStart < features.size(); blockStart += blockSize) {
            const auto docCountInBlock = Min(blockSize, features.size() - blockStart);
            BinarizeFeaturesFlat(features, blockStart, blockStart + docCountInBlock, binFeatures, transposedHash, ctrs);
            CalcTrees(blockStart, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, results);
        }
    }

    void TModelCalcer::Calc(const yvector<NArrayRef::TConstArrayRef<float>>& floatFeatures,
                            const yvector<NArrayRef::TConstArrayRef<int>>& catFeatures,
                            size_t treeStart,
                            size_t treeEnd,
                            NArrayRef::TArrayRef<double> results) const {
        size_t blockSize;
        if (UsedCtrFeatures.empty()) {
            blockSize = 128;
        } else {
            blockSize = 4096;
        }
        if (!floatFeatures.empty() && !catFeatures.empty()) {
            CB_ENSURE(catFeatures.size() == floatFeatures.size());
        }
        auto docCount = Max(catFeatures.size(), floatFeatures.size());
        blockSize = Min(blockSize, docCount);
        CB_ENSURE(results.size() == docCount * ModelClassCount);
        std::fill(results.begin(), results.end(), 0.0);
        yvector<ui8> binFeatures(UsedBinaryFeaturesCount * blockSize);
        yvector<ui32> indexesVec(blockSize);
        yvector<int> transposedHash(blockSize * CatFeatureFlatIndex.size());
        yvector<float> ctrs(UsedModelCtrs.size() * blockSize);
        for (size_t blockStart = 0; blockStart < docCount; blockStart += blockSize) {
            const auto docCountInBlock = Min(blockSize, docCount - blockStart);
            BinarizeFeatures(floatFeatures, catFeatures, blockStart, blockStart + docCountInBlock, binFeatures, transposedHash, ctrs);
            CalcTrees(blockStart, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, results);
        }
    }

    yvector<yvector<double>> TModelCalcer::CalcTreeIntervals(
        const yvector<NArrayRef::TConstArrayRef<float>>& floatFeatures,
        const yvector<NArrayRef::TConstArrayRef<int>>& catFeatures,
        size_t incrementStep) {
        size_t blockSize;
        if (UsedCtrFeatures.empty()) {
            blockSize = 128;
        } else {
            blockSize = 4096;
        }
        if (!floatFeatures.empty() && !catFeatures.empty()) {
            CB_ENSURE(catFeatures.size() == floatFeatures.size());
        }
        auto docCount = Max(catFeatures.size(), floatFeatures.size());
        blockSize = Min(blockSize, docCount);
        auto treeStepCount = (BinaryTrees.size() + incrementStep - 1) / incrementStep;
        yvector<yvector<double>> results(docCount, yvector<double>(treeStepCount));
        CB_ENSURE(ModelClassCount == 1);
        yvector<ui8> binFeatures(UsedBinaryFeaturesCount * blockSize);
        yvector<ui32> indexesVec(blockSize);
        yvector<int> transposedHash(blockSize * CatFeatureFlatIndex.size());
        yvector<float> ctrs(UsedModelCtrs.size() * blockSize);
        yvector<double> tmpResult(docCount);
        NArrayRef::TArrayRef<double> tmpResultRef(tmpResult);
        for (size_t blockStart = 0; blockStart < docCount; blockStart += blockSize) {
            const auto docCountInBlock = Min(blockSize, docCount - blockStart);
            BinarizeFeatures(floatFeatures, catFeatures, blockStart, blockStart + docCountInBlock, binFeatures, transposedHash, ctrs);
            for (size_t stepIdx = 0; stepIdx < treeStepCount; ++stepIdx) {
                CalcTrees(blockStart,
                          binFeatures,
                          docCountInBlock,
                          indexesVec,
                          stepIdx * incrementStep,
                          Min((stepIdx + 1) * incrementStep, BinaryTrees.size()),
                          tmpResultRef);
                for (size_t i = 0; i < docCountInBlock; ++i) {
                    results[blockStart + i][stepIdx] = tmpResult[i];
                }
            }
        }
        return results;
    }

    yvector<yvector<double>> TModelCalcer::CalcTreeIntervalsFlat(
        const yvector<NArrayRef::TConstArrayRef<float>>& mixedFeatures,
        size_t incrementStep) {
        size_t blockSize;
        if (UsedCtrFeatures.empty()) {
            blockSize = 128;
        } else {
            blockSize = 4096;
        }
        auto docCount = mixedFeatures.size();
        blockSize = Min(blockSize, docCount);
        auto treeStepCount = (BinaryTrees.size() + incrementStep - 1) / incrementStep;
        yvector<yvector<double>> results(docCount, yvector<double>(treeStepCount));
        CB_ENSURE(ModelClassCount == 1);
        yvector<ui8> binFeatures(UsedBinaryFeaturesCount * blockSize);
        yvector<ui32> indexesVec(blockSize);
        yvector<int> transposedHash(blockSize * CatFeatureFlatIndex.size());
        yvector<float> ctrs(UsedModelCtrs.size() * blockSize);
        yvector<double> tmpResult(docCount);
        NArrayRef::TArrayRef<double> tmpResultRef(tmpResult);
        for (size_t blockStart = 0; blockStart < docCount; blockStart += blockSize) {
            const auto docCountInBlock = Min(blockSize, docCount - blockStart);
            BinarizeFeaturesFlat(mixedFeatures, blockStart, blockStart + docCountInBlock, binFeatures, transposedHash, ctrs);
            for (size_t stepIdx = 0; stepIdx < treeStepCount; ++stepIdx) {
                CalcTrees(blockStart,
                          binFeatures,
                          docCountInBlock,
                          indexesVec,
                          stepIdx * incrementStep,
                          Min((stepIdx + 1) * incrementStep, BinaryTrees.size()),
                          tmpResultRef);
                for (size_t i = 0; i < docCountInBlock; ++i) {
                    results[blockStart + i][stepIdx] = tmpResult[i];
                }
            }
        }
        return results;
    }

    void TModelCalcer::Calc(const yvector<NArrayRef::TConstArrayRef<float>>& floatFeatures,
                            const yvector<yvector<TStringBuf>>& catFeatures, size_t treeStart, size_t treeEnd,
                            NArrayRef::TArrayRef<double> results) const {
        size_t blockSize;
        if (UsedCtrFeatures.empty()) {
            blockSize = 128;
        } else {
            blockSize = 4096;
        }
        if (!floatFeatures.empty() && !catFeatures.empty()) {
            CB_ENSURE(catFeatures.size() == floatFeatures.size());
        }
        auto docCount = Max(catFeatures.size(), floatFeatures.size());
        blockSize = Min(blockSize, docCount);
        CB_ENSURE(results.size() == docCount * ModelClassCount);
        std::fill(results.begin(), results.end(), 0.0);
        yvector<ui8> binFeatures(UsedBinaryFeaturesCount * blockSize);
        yvector<ui32> indexesVec(blockSize);
        yvector<int> transposedHash(blockSize * CatFeatureFlatIndex.size());
        yvector<float> ctrs(UsedModelCtrs.size() * blockSize);
        for (size_t blockStart = 0; blockStart < docCount; blockStart += blockSize) {
            const auto docCountInBlock = Min(blockSize, docCount - blockStart);
            BinarizeFeatures(floatFeatures, catFeatures, blockStart, blockStart + docCountInBlock, binFeatures, transposedHash, ctrs);
            CalcTrees(blockStart, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, results);
        }
    }
}
