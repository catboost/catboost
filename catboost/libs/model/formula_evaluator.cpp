#include <catboost/libs/model/formula_evaluator.h>
#include <catboost/libs/model/static_ctr_provider.h>

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
    void TFormulaEvaluator::InitBinTreesFromCoreModel(const TCoreModel& model) {
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
        ModelClassCount = static_cast<size_t>(model.ApproxDimension);
        auto realClassCount = RepackLeaves(model.LeafValues, &LeafValues);
        CB_ENSURE(realClassCount == ModelClassCount, "ApproxDimension != real model class count: " << realClassCount << " != " << ModelClassCount);
        yvector<TModelSplit> usedSplits;
        {
            std::set<TModelSplit> allSplits;
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
                    FloatFeaturesCount = Max(FloatFeaturesCount, (size_t)split.BinFeature.FloatFeature + 1);
                }
                UsedFloatFeatures.back().Borders.push_back(
                    model.Borders[split.BinFeature.FloatFeature][split.BinFeature.SplitIdx]);
            } else if (split.Type == ESplitType::OneHotFeature) {
                if (UsedOHEFeatures.empty() || UsedOHEFeatures.back().CatFeatureIndex != split.OneHotFeature.CatFeatureIdx) {
                    UsedOHEFeatures.emplace_back();
                    UsedOHEFeatures.back().CatFeatureIndex = split.OneHotFeature.CatFeatureIdx;
                    CatFeaturesCount = Max(CatFeaturesCount, (size_t)split.OneHotFeature.CatFeatureIdx + 1);
                }
                UsedOHEFeatures.back().Values.push_back(split.OneHotFeature.Value);
            } else {
                if (UsedCtrFeatures.empty() || UsedCtrFeatures.back().Ctr != split.OnlineCtr.Ctr) {
                    UsedCtrFeatures.emplace_back();
                    UsedCtrFeatures.back().Ctr = split.OnlineCtr.Ctr;
                    for (const auto& catFeatureIdx : split.OnlineCtr.Ctr.Projection.CatFeatures) {
                        CatFeaturesCount = Max(CatFeaturesCount, (size_t)catFeatureIdx + 1);
                    }
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

    void TFormulaEvaluator::InitFromFullModel(const TFullModel& fullModel) {
        InitBinTreesFromCoreModel(fullModel);
        CtrProvider.Reset(new TStaticCtrProvider(fullModel.CtrCalcerData));
    }

    void TFormulaEvaluator::InitFromFullModel(TFullModel&& fullModel) {
        InitBinTreesFromCoreModel(fullModel);
        CtrProvider.Reset(new TStaticCtrProvider(std::move(fullModel.CtrCalcerData)));
    }

    void TFormulaEvaluator::CalcFlat(const yvector<TConstArrayRef<float>>& features,
                                size_t treeStart,
                                size_t treeEnd,
                                TArrayRef<double> results) const {
        const auto expectedFlatVecSize = GetFlatFeatureVectorExpectedSize();
        for (const auto& flatFeaturesVec : features) {
            CB_ENSURE(flatFeaturesVec.size() >= expectedFlatVecSize,
                      "insufficient flat features vector size: " << flatFeaturesVec.size()
                                                                 << " expected: " << expectedFlatVecSize);
        }
        CalcGeneric(
            [&](const TFloatFeature& floatFeature, size_t index) {
                return features[index][FloatFeatureFlatIndex[floatFeature.FeatureIndex]];
            },
            [&](size_t catFeatureIdx, size_t index) {
                return ConvertFloatCatFeatureToIntHash(features[index][CatFeatureFlatIndex[catFeatureIdx]]);
            },
            features.size(),
            treeStart,
            treeEnd,
            results
        );
    }

    void TFormulaEvaluator::CalcFlatTransposed(const yvector<TConstArrayRef<float>>& transposedFeatures,
                                size_t treeStart,
                                size_t treeEnd,
                                TArrayRef<double> results) const {
        CB_ENSURE(!transposedFeatures.empty(), "Features should not be empty");
        CalcGeneric(
            [&](const TFloatFeature& floatFeature, size_t index) {
                return transposedFeatures[FloatFeatureFlatIndex[floatFeature.FeatureIndex]][index];
            },
            [&](size_t catFeatureIdx, size_t index) {
                return ConvertFloatCatFeatureToIntHash(transposedFeatures[CatFeatureFlatIndex[catFeatureIdx]][index]);
            },
            transposedFeatures[0].Size(),
            treeStart,
            treeEnd,
            results
        );
    }

    void TFormulaEvaluator::Calc(const yvector<TConstArrayRef<float>>& floatFeatures,
                            const yvector<TConstArrayRef<int>>& catFeatures,
                            size_t treeStart,
                            size_t treeEnd,
                            TArrayRef<double> results) const {
        if (!floatFeatures.empty() && !catFeatures.empty()) {
            CB_ENSURE(catFeatures.size() == floatFeatures.size());
        }
        for (const auto& floatFeaturesVec : floatFeatures) {
            CB_ENSURE(floatFeaturesVec.size() >= FloatFeaturesCount,
                      "insufficient float features vector size: " << floatFeaturesVec.size()
                                                                  << " expected: " << FloatFeaturesCount);
        }
        for (const auto& catFeaturesVec : catFeatures) {
            CB_ENSURE(catFeaturesVec.size() >= CatFeaturesCount,
                      "insufficient cat features vector size: " << catFeaturesVec.size()
                                                                << " expected: " << CatFeaturesCount);
        }
        CalcGeneric(
            [&](const TFloatFeature& floatFeature, size_t index) {
                return floatFeatures[index][floatFeature.FeatureIndex];
            },
            [&](size_t catFeatureIdx, size_t index) {
                return catFeatures[index][catFeatureIdx];
            },
            floatFeatures.size(),
            treeStart,
            treeEnd,
            results
        );
    }

    void TFormulaEvaluator::Calc(const yvector<TConstArrayRef<float>>& floatFeatures,
                            const yvector<yvector<TStringBuf>>& catFeatures, size_t treeStart, size_t treeEnd,
                            TArrayRef<double> results) const {
        if (!floatFeatures.empty() && !catFeatures.empty()) {
            CB_ENSURE(catFeatures.size() == floatFeatures.size());
        }
        for (const auto& floatFeaturesVec : floatFeatures) {
            CB_ENSURE(floatFeaturesVec.size() >= FloatFeaturesCount,
                      "insufficient float features vector size: " << floatFeaturesVec.size()
                                                                  << " expected: " << FloatFeaturesCount);
        }
        for (const auto& catFeaturesVec : catFeatures) {
            CB_ENSURE(catFeaturesVec.size() >= CatFeaturesCount,
                      "insufficient cat features vector size: " << catFeaturesVec.size()
                                                                << " expected: " << CatFeaturesCount);
        }
        CalcGeneric(
            [&](const TFloatFeature& floatFeature, size_t index) {
                return floatFeatures[index][floatFeature.FeatureIndex];
            },
            [&](size_t catFeatureIdx, size_t index) {
                return CalcCatFeatureHash(catFeatures[index][catFeatureIdx]);
            },
            floatFeatures.size(),
            treeStart,
            treeEnd,
            results
        );
    }

    yvector<yvector<double>> TFormulaEvaluator::CalcTreeIntervals(
        const yvector<TConstArrayRef<float>>& floatFeatures,
        const yvector<TConstArrayRef<int>>& catFeatures,
        size_t incrementStep) const {
        if (!floatFeatures.empty() && !catFeatures.empty()) {
            CB_ENSURE(catFeatures.size() == floatFeatures.size());
        }
        for (const auto& floatFeaturesVec : floatFeatures) {
            CB_ENSURE(floatFeaturesVec.size() >= FloatFeaturesCount,
                      "insufficient float features vector size: " << floatFeaturesVec.size()
                                                                  << " expected: " << FloatFeaturesCount);
        }
        for (const auto& catFeaturesVec : catFeatures) {
            CB_ENSURE(catFeaturesVec.size() >= CatFeaturesCount,
                      "insufficient cat features vector size: " << catFeaturesVec.size()
                                                                  << " expected: " << CatFeaturesCount);
        }
        return CalcTreeIntervalsGeneric(
            [&](const TFloatFeature& floatFeature, size_t index) {
                return floatFeatures[index][floatFeature.FeatureIndex];
            },
            [&](size_t catFeatureIdx, size_t index) {
                return catFeatures[index][catFeatureIdx];
            },
            floatFeatures.size(),
            incrementStep
        );
    }
    yvector<yvector<double>> TFormulaEvaluator::CalcTreeIntervalsFlat(
        const yvector<TConstArrayRef<float>>& features,
        size_t incrementStep) const {
        const auto expectedFlatVecSize = GetFlatFeatureVectorExpectedSize();
        for (const auto& flatFeaturesVec : features) {
            CB_ENSURE(flatFeaturesVec.size() >= expectedFlatVecSize,
                      "insufficient flat features vector size: " << flatFeaturesVec.size()
                                                                  << " expected: " << expectedFlatVecSize);
        }
        return CalcTreeIntervalsGeneric(
            [&](const TFloatFeature& floatFeature, size_t index) {
                return features[index][FloatFeatureFlatIndex[floatFeature.FeatureIndex]];
            },
            [&](size_t catFeatureIdx, size_t index) {
                return ConvertFloatCatFeatureToIntHash(features[index][CatFeatureFlatIndex[catFeatureIdx]]);
            },
            features.size(),
            incrementStep
        );
    }

    TFormulaEvaluator TFormulaEvaluator::CopyTreeRange(const size_t begin, const size_t end) const {
        TFormulaEvaluator result = *this;
        result.TruncateModel(begin, end);
        return result;
    }

    namespace {
        template<typename T>
        void TruncateVector(const size_t begin, const size_t end, yvector<T>* vector) {
            yvector<T> tmp;
            tmp.reserve(end - begin);
            for (auto iter = vector->begin() + begin; iter != vector->begin() + end; ++iter) {
                tmp.emplace_back(std::move(*iter));
            }
            vector->swap(tmp);
        }
    }

    void TFormulaEvaluator::TruncateModel(const size_t begin, const size_t end) {
        TruncateVector(begin, end, &BinaryTrees);
        TruncateVector(begin, end, &LeafValues);
    }
}
