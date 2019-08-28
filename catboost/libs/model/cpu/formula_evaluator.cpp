#include <catboost/libs/model/eval_processing.h>
#include <catboost/libs/model/model.h>

#include "evaluator.h"

namespace NCB::NModelEvaluation {
    namespace NDetail {
        template <typename TFloatFeatureAccessor, typename TCatFeatureAccessor>
        inline void CalcGeneric(
            const TObliviousTrees& trees,
            const TIntrusivePtr<ICtrProvider>& ctrProvider,
            TFloatFeatureAccessor floatFeatureAccessor,
            TCatFeatureAccessor catFeaturesAccessor,
            size_t docCount,
            size_t treeStart,
            size_t treeEnd,
            EPredictionType predictionType,
            TArrayRef<double> results,
            const NCB::NModelEvaluation::TFeatureLayout* featureInfo = nullptr
        ) {
            const size_t blockSize = Min(FORMULA_EVALUATION_BLOCK_SIZE, docCount);
            auto calcTrees = GetCalcTreesFunction(trees, blockSize);
            std::fill(results.begin(), results.end(), 0.0);
            if (trees.GetTreeCount() == 0) {
                return;
            }
            TVector<TCalcerIndexType> indexesVec(blockSize);
            TEvalResultProcessor resultProcessor(
                docCount,
                results,
                predictionType,
                trees.ApproxDimension,
                blockSize
            );
            ui32 blockId = 0;
            ProcessDocsInBlocks(
                trees,
                ctrProvider,
                floatFeatureAccessor,
                catFeaturesAccessor,
                docCount,
                blockSize,
                [&] (size_t docCountInBlock, const TCPUEvaluatorQuantizedData* quantizedData) {
                    auto blockResultsView = resultProcessor.GetViewForRawEvaluation(blockId);
                    calcTrees(
                        trees,
                        quantizedData,
                        docCountInBlock,
                        docCount == 1 ? nullptr : indexesVec.data(),
                        treeStart,
                        treeEnd,
                        blockResultsView.data()
                    );
                    resultProcessor.PostprocessBlock(blockId);
                    ++blockId;
                },
                featureInfo
            );
        }

        class TCpuEvaluator final : public IModelEvaluator {
        public:
            explicit TCpuEvaluator(const TFullModel& fullModel)
                : ObliviousTrees(fullModel.ObliviousTrees)
                , CtrProvider(fullModel.CtrProvider)
            {}

            void SetPredictionType(EPredictionType type) override {
                PredictionType = type;
            }

            EPredictionType GetPredictionType() const override {
                return PredictionType;
            }

            void SetFeatureLayout(const TFeatureLayout& featureLayout) override {
                ExtFeatureLayout = featureLayout;
            }

            size_t GetTreeCount() const {
                return ObliviousTrees->GetTreeCount();
            }

            TModelEvaluatorPtr Clone() const override {
                return new TCpuEvaluator(*this);
            }

            i32 GetApproxDimension() const override {
                return ObliviousTrees->ApproxDimension;
            }

            void SetProperty(const TStringBuf propName, const TStringBuf propValue) override {
                CB_ENSURE(false, "CPU evaluator don't have any properties. Got: " << propName);
                Y_UNUSED(propValue);
            }

            void CalcFlatTransposed(
                TConstArrayRef<TConstArrayRef<float>> transposedFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo
            ) const override {
                if (!featureInfo) {
                    featureInfo = ExtFeatureLayout.Get();
                }
                CB_ENSURE(
                    ObliviousTrees->GetFlatFeatureVectorExpectedSize() <= transposedFeatures.size(),
                    "Not enough features provided" << LabeledOutput(ObliviousTrees->GetFlatFeatureVectorExpectedSize(), transposedFeatures.size())
                );
                TMaybe<size_t> docCount;
                CB_ENSURE(!ObliviousTrees->FloatFeatures.empty() || !ObliviousTrees->CatFeatures.empty(),
                          "Both float features and categorical features information are empty");
                auto getPosition = [featureInfo] (const auto& feature) -> TFeaturePosition {
                    if (!featureInfo) {
                        return feature.Position;
                    } else {
                        return featureInfo->AdjustFeature(feature);
                    }
                };
                if (!ObliviousTrees->FloatFeatures.empty()) {
                    for (const auto& floatFeature : ObliviousTrees->FloatFeatures) {
                        if (floatFeature.UsedInModel()) {
                            docCount = transposedFeatures[getPosition(floatFeature).FlatIndex].size();
                            break;
                        }
                    }
                }
                if (!docCount.Defined() && !ObliviousTrees->CatFeatures.empty()) {
                    for (const auto& catFeature : ObliviousTrees->CatFeatures) {
                        if (catFeature.UsedInModel) {
                            docCount = transposedFeatures[getPosition(catFeature).FlatIndex].size();
                            break;
                        }
                    }
                }

                CB_ENSURE(docCount.Defined(), "couldn't determine document count, something went wrong");
                CalcGeneric(
                    *ObliviousTrees,
                    CtrProvider,
                    [&transposedFeatures](TFeaturePosition floatFeature, size_t index) -> float {
                        return transposedFeatures[floatFeature.FlatIndex][index];
                    },
                    [&transposedFeatures](TFeaturePosition catFeature, size_t index) -> int {
                        return ConvertFloatCatFeatureToIntHash(transposedFeatures[catFeature.FlatIndex][index]);
                    },
                    *docCount,
                    treeStart,
                    treeEnd,
                    PredictionType,
                    results,
                    featureInfo
                );
            }

            void CalcFlat(
                TConstArrayRef<TConstArrayRef<float>> features,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo
            ) const override {
                if (!featureInfo) {
                    featureInfo = ExtFeatureLayout.Get();
                }
                auto expectedFlatVecSize = ObliviousTrees->GetFlatFeatureVectorExpectedSize();
                if (featureInfo && featureInfo->FlatIndexes) {
                    CB_ENSURE(
                        featureInfo->FlatIndexes->size() >= expectedFlatVecSize,
                        "Feature layout FlatIndexes expected to be at least " << expectedFlatVecSize << " long"
                    );
                    expectedFlatVecSize = *MaxElement(featureInfo->FlatIndexes->begin(), featureInfo->FlatIndexes->end());
                }
                for (const auto& flatFeaturesVec : features) {
                    CB_ENSURE(
                        flatFeaturesVec.size() >= expectedFlatVecSize,
                        "insufficient flat features vector size: " << flatFeaturesVec.size() << " expected: " << expectedFlatVecSize
                    );
                }
                CalcGeneric(
                    *ObliviousTrees,
                    CtrProvider,
                    [&features](TFeaturePosition position, size_t index) -> float {
                        return features[index][position.FlatIndex];
                    },
                    [&features](TFeaturePosition position, size_t index) -> int {
                        return ConvertFloatCatFeatureToIntHash(features[index][position.FlatIndex]);
                    },
                    features.size(),
                    treeStart,
                    treeEnd,
                    PredictionType,
                    results,
                    featureInfo
                );
            }

            void CalcFlatSingle(
                TConstArrayRef<float> features,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo
            ) const override {
                if (!featureInfo) {
                    featureInfo = ExtFeatureLayout.Get();
                }
                CB_ENSURE(
                    ObliviousTrees->GetFlatFeatureVectorExpectedSize() <= features.size(),
                    "Not enough features provided"
                );
                CalcGeneric(
                    *ObliviousTrees,
                    CtrProvider,
                    [&features](TFeaturePosition position, size_t ) -> float {
                        return features[position.FlatIndex];
                    },
                    [&features](TFeaturePosition position, size_t ) -> int {
                        return ConvertFloatCatFeatureToIntHash(features[position.FlatIndex]);
                    },
                    1,
                    treeStart,
                    treeEnd,
                    PredictionType,
                    results,
                    featureInfo
                );
            }

            void Calc(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<int>> catFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo
            ) const override {
                if (!featureInfo) {
                    featureInfo = ExtFeatureLayout.Get();
                }
                ValidateInputFeatures(floatFeatures, catFeatures, featureInfo);
                const size_t docCount = Max(catFeatures.size(), floatFeatures.size());
                CalcGeneric(
                    *ObliviousTrees,
                    CtrProvider,
                    [&floatFeatures](TFeaturePosition position, size_t index) -> float {
                        return floatFeatures[index][position.Index];
                    },
                    [&catFeatures](TFeaturePosition position, size_t index) -> int {
                        return catFeatures[index][position.Index];
                    },
                    docCount,
                    treeStart,
                    treeEnd,
                    PredictionType,
                    results,
                    featureInfo
                );
            }

            void Calc(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> catFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo
            ) const override {
                if (!featureInfo) {
                    featureInfo = ExtFeatureLayout.Get();
                }
                ValidateInputFeatures(floatFeatures, catFeatures, featureInfo);
                const size_t docCount = Max(catFeatures.size(), floatFeatures.size());
                CalcGeneric(
                    *ObliviousTrees,
                    CtrProvider,
                    [&floatFeatures](TFeaturePosition position, size_t index) -> float {
                        return floatFeatures[index][position.Index];
                    },
                    [&catFeatures](TFeaturePosition position, size_t index) -> int {
                        return CalcCatFeatureHash(catFeatures[index][position.Index]);
                    },
                    docCount,
                    treeStart,
                    treeEnd,
                    PredictionType,
                    results,
                    featureInfo
                );
            }

            void CalcLeafIndexesSingle(
                TConstArrayRef<float> floatFeatures,
                TConstArrayRef<TStringBuf> catFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<ui32> indexes,
                const TFeatureLayout* featureInfo
            ) const override {
                if (!featureInfo) {
                    featureInfo = ExtFeatureLayout.Get();
                }
                ValidateInputFeatures<TConstArrayRef<TStringBuf>>({floatFeatures}, {catFeatures}, featureInfo);
                CalcLeafIndexesGeneric(
                    *ObliviousTrees,
                    CtrProvider,
                    [&floatFeatures](TFeaturePosition position, size_t) -> float {
                        return floatFeatures[position.Index];
                    },
                    [&catFeatures](TFeaturePosition position, size_t) -> int {
                        return CalcCatFeatureHash(catFeatures[position.Index]);
                    },
                    1,
                    treeStart,
                    treeEnd,
                    indexes,
                    featureInfo
                );
            }

            void CalcLeafIndexes(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> catFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<ui32> indexes,
                const TFeatureLayout* featureInfo
            ) const override {
                if (!featureInfo) {
                    featureInfo = ExtFeatureLayout.Get();
                }
                ValidateInputFeatures(floatFeatures, catFeatures, featureInfo);
                const size_t docCount = Max(catFeatures.size(), floatFeatures.size());
                CB_ENSURE(docCount * (treeEnd - treeStart) == indexes.size(), LabeledOutput(docCount * (treeEnd - treeStart), indexes.size()));
                CalcLeafIndexesGeneric(
                    *ObliviousTrees,
                    CtrProvider,
                    [&floatFeatures](TFeaturePosition position, size_t index) -> float {
                        return floatFeatures[index][position.Index];
                    },
                    [&catFeatures](TFeaturePosition position, size_t index) -> int {
                        return CalcCatFeatureHash(catFeatures[index][position.Index]);
                    },
                    docCount,
                    treeStart,
                    treeEnd,
                    indexes,
                    featureInfo
                );
            }
            void Calc(
                const IQuantizedData* quantizedFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results
            ) const override {
                const TCPUEvaluatorQuantizedData* cpuQuantizedFeatures = reinterpret_cast<const TCPUEvaluatorQuantizedData*>(quantizedFeatures);
                CB_ENSURE(cpuQuantizedFeatures != nullptr, "Expected pointer to TCPUEvaluatorQuantizedData");
                CB_ENSURE(
                    cpuQuantizedFeatures->BlockStride % ObliviousTrees->GetEffectiveBinaryFeaturesBucketsCount() == 0,
                    "Unexpected block stride: " << cpuQuantizedFeatures->BlockStride
                );
                CB_ENSURE(cpuQuantizedFeatures->BlocksCount * FORMULA_EVALUATION_BLOCK_SIZE >= cpuQuantizedFeatures->ObjectsCount);
                std::fill(results.begin(), results.end(), 0.0);
                auto subBlockSize = Min<size_t>(FORMULA_EVALUATION_BLOCK_SIZE, cpuQuantizedFeatures->ObjectsCount);
                auto calcFunction = GetCalcTreesFunction(
                    *ObliviousTrees,
                    subBlockSize,
                    false
                );
                CB_ENSURE(results.size() == ObliviousTrees->ApproxDimension * cpuQuantizedFeatures->ObjectsCount);
                TVector<TCalcerIndexType> indexesVec(subBlockSize);
                double* resultPtr = results.data();
                for (size_t blockId = 0; blockId < cpuQuantizedFeatures->BlocksCount; ++blockId) {
                    auto subBlock = cpuQuantizedFeatures->ExtractBlock(blockId);
                    calcFunction(
                        *ObliviousTrees, &subBlock,
                        subBlock.ObjectsCount,
                        indexesVec.data(),
                        treeStart,
                        treeEnd,
                        resultPtr
                    );
                    resultPtr += subBlock.GetObjectsCount() * ObliviousTrees->ApproxDimension;
                }
            }

            void CalcLeafIndexes(
                const IQuantizedData* quantizedFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<ui32> indexes
            ) const override {
                const TCPUEvaluatorQuantizedData* cpuQuantizedFeatures = reinterpret_cast<const TCPUEvaluatorQuantizedData*>(quantizedFeatures);
                CB_ENSURE(cpuQuantizedFeatures != nullptr, "Expected pointer to TCPUEvaluatorQuantizedData");
                CB_ENSURE(
                    cpuQuantizedFeatures->BlockStride % ObliviousTrees->GetEffectiveBinaryFeaturesBucketsCount() == 0,
                    "Unexpected block stride: " << cpuQuantizedFeatures->BlockStride
                );
                CB_ENSURE(cpuQuantizedFeatures->BlocksCount * FORMULA_EVALUATION_BLOCK_SIZE >= cpuQuantizedFeatures->ObjectsCount);
                auto calcFunction = GetCalcTreesFunction(
                    *ObliviousTrees,
                    Min<size_t>(FORMULA_EVALUATION_BLOCK_SIZE, cpuQuantizedFeatures->ObjectsCount),
                    /*calcIndexesOnly*/ true
                );
                size_t treeCount = treeEnd - treeStart;
                CB_ENSURE(indexes.size() == treeCount * cpuQuantizedFeatures->ObjectsCount);
                for (size_t blockId = 0; blockId < cpuQuantizedFeatures->BlocksCount; ++blockId) {
                    auto subBlock = cpuQuantizedFeatures->ExtractBlock(blockId);
                    calcFunction(
                        *ObliviousTrees,
                        &subBlock,
                        subBlock.ObjectsCount,
                        indexes.data() + blockId * FORMULA_EVALUATION_BLOCK_SIZE * treeCount,
                        treeStart,
                        treeEnd,
                        /*results*/ nullptr
                    );
                }
            }

        private:
            template <typename TCatFeatureContainer = TConstArrayRef<int>>
            void ValidateInputFeatures(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TCatFeatureContainer> catFeatures,
                const TFeatureLayout* featureInfo
            ) const {
                if (!floatFeatures.empty() && !catFeatures.empty()) {
                    CB_ENSURE(catFeatures.size() == floatFeatures.size());
                }
                CB_ENSURE(
                    ObliviousTrees->GetUsedFloatFeaturesCount() == 0 || !floatFeatures.empty(),
                    "Model has float features but no float features provided"
                );
                CB_ENSURE(
                    ObliviousTrees->GetUsedCatFeaturesCount() == 0 || !catFeatures.empty(),
                    "Model has categorical features but no categorical features provided"
                );
                size_t minimalSufficientFloatFeatureCount = ObliviousTrees->GetMinimalSufficientFloatFeaturesVectorSize();
                if (featureInfo && featureInfo->FloatFeatureIndexes.Defined()) {
                    CB_ENSURE(featureInfo->FloatFeatureIndexes->size() >= minimalSufficientFloatFeatureCount);
                    minimalSufficientFloatFeatureCount = *MaxElement(
                        featureInfo->FloatFeatureIndexes->begin(),
                        featureInfo->FloatFeatureIndexes->end()
                    );
                }
                for (const auto& floatFeaturesVec : floatFeatures) {
                    CB_ENSURE(
                        floatFeaturesVec.size() >= minimalSufficientFloatFeatureCount,
                        "insufficient float features vector size: " << floatFeaturesVec.size()
                        << " expected: " << minimalSufficientFloatFeatureCount
                    );
                }
                size_t minimalSufficientCatFeatureCount = ObliviousTrees->GetMinimalSufficientCatFeaturesVectorSize();
                if (featureInfo && featureInfo->CatFeatureIndexes.Defined()) {
                    CB_ENSURE(featureInfo->CatFeatureIndexes->size() >= minimalSufficientCatFeatureCount);
                    minimalSufficientCatFeatureCount = *MaxElement(
                        featureInfo->CatFeatureIndexes->begin(),
                        featureInfo->CatFeatureIndexes->end()
                    );
                }
                for (const auto& catFeaturesVec : catFeatures) {
                    CB_ENSURE(
                        catFeaturesVec.size() >= minimalSufficientCatFeatureCount,
                        "insufficient cat features vector size: " << catFeaturesVec.size()
                        << " expected: " << minimalSufficientCatFeatureCount
                    );
                }
            }
        private:
            TCOWTreeWrapper ObliviousTrees;
            const TIntrusivePtr<ICtrProvider> CtrProvider;
            EPredictionType PredictionType = EPredictionType::RawFormulaVal;
            TMaybe<TFeatureLayout> ExtFeatureLayout;
        };
    }

    TEvaluationBackendFactory::TRegistrator<NDetail::TCpuEvaluator> CPUEvaluationBackendRegistrator(EFormulaEvaluatorType::CPU);
}
