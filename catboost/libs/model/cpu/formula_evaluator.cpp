#include <catboost/libs/model/eval_processing.h>
#include <catboost/libs/model/model.h>

#include "evaluator.h"

namespace NCB::NModelEvaluation {
    namespace NDetail {
        template <typename TFloatFeatureAccessor, typename TCatFeatureAccessor, typename TTextFeatureAccessor>
        inline void CalcGeneric(
            const TModelTrees& trees,
            const TIntrusivePtr<ICtrProvider>& ctrProvider,
            const TIntrusivePtr<TTextProcessingCollection>& textProcessingCollection,
            TFloatFeatureAccessor floatFeatureAccessor,
            TCatFeatureAccessor catFeaturesAccessor,
            TTextFeatureAccessor textFeatureAccessor,
            size_t docCount,
            size_t treeStart,
            size_t treeEnd,
            EPredictionType predictionType,
            TArrayRef<double> results,
            const NCB::NModelEvaluation::TFeatureLayout* featureInfo = nullptr
        ) {
            const size_t blockSize = Min(FORMULA_EVALUATION_BLOCK_SIZE, docCount);
            auto calcTrees = GetCalcTreesFunction(trees, blockSize);
            if (trees.GetTreeCount() == 0) {
                Fill(results.begin(), results.end(), trees.GetScaleAndBias().Bias);
                return;
            }
            Fill(results.begin(), results.end(), 0.0);
            TVector<TCalcerIndexType> indexesVec(blockSize);
            TEvalResultProcessor resultProcessor(
                docCount,
                results,
                predictionType,
                trees.GetScaleAndBias(),
                trees.GetDimensionsCount(),
                blockSize
            );
            ui32 blockId = 0;
            ProcessDocsInBlocks(
                trees,
                ctrProvider,
                textProcessingCollection,
                floatFeatureAccessor,
                catFeaturesAccessor,
                textFeatureAccessor,
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
                    resultProcessor.PostprocessBlock(blockId, treeStart);
                    ++blockId;
                },
                featureInfo
            );
        }

        class TCpuEvaluator final : public IModelEvaluator {
        public:
            explicit TCpuEvaluator(const TFullModel& fullModel)
                : ModelTrees(fullModel.ModelTrees)
                , CtrProvider(fullModel.CtrProvider)
                , TextProcessingCollection(fullModel.TextProcessingCollection)
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
                return ModelTrees->GetTreeCount();
            }

            TModelEvaluatorPtr Clone() const override {
                return new TCpuEvaluator(*this);
            }

            i32 GetApproxDimension() const override {
                return ModelTrees->GetDimensionsCount();
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
                    ModelTrees->GetFlatFeatureVectorExpectedSize() <= transposedFeatures.size(),
                    "Not enough features provided" << LabeledOutput(ModelTrees->GetFlatFeatureVectorExpectedSize(), transposedFeatures.size())
                );
                TMaybe<size_t> docCount;
                CB_ENSURE(!ModelTrees->GetFloatFeatures().empty() || !ModelTrees->GetCatFeatures().empty(),
                          "Both float features and categorical features information are empty");
                auto getPosition = [featureInfo] (const auto& feature) -> TFeaturePosition {
                    if (!featureInfo) {
                        return feature.Position;
                    } else {
                        return featureInfo->GetRemappedPosition(feature);
                    }
                };
                if (!ModelTrees->GetFloatFeatures().empty()) {
                    for (const auto& floatFeature : ModelTrees->GetFloatFeatures()) {
                        if (floatFeature.UsedInModel()) {
                            docCount = transposedFeatures[getPosition(floatFeature).FlatIndex].size();
                            break;
                        }
                    }
                }
                if (!docCount.Defined() && !ModelTrees->GetCatFeatures().empty()) {
                    for (const auto& catFeature : ModelTrees->GetCatFeatures()) {
                        if (catFeature.UsedInModel()) {
                            docCount = transposedFeatures[getPosition(catFeature).FlatIndex].size();
                            break;
                        }
                    }
                }

                CB_ENSURE(docCount.Defined(), "couldn't determine document count, something went wrong");
                CalcGeneric(
                    *ModelTrees,
                    CtrProvider,
                    TextProcessingCollection,
                    [&transposedFeatures](TFeaturePosition floatFeature, size_t index) -> float {
                        return transposedFeatures[floatFeature.FlatIndex][index];
                    },
                    [&transposedFeatures](TFeaturePosition catFeature, size_t index) -> int {
                        return ConvertFloatCatFeatureToIntHash(transposedFeatures[catFeature.FlatIndex][index]);
                    },
                    TCpuEvaluator::TextFeatureAccessorStub,
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
                auto expectedFlatVecSize = ModelTrees->GetFlatFeatureVectorExpectedSize();
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
                    *ModelTrees,
                    CtrProvider,
                    TextProcessingCollection,
                    [&features](TFeaturePosition position, size_t index) -> float {
                        return features[index][position.FlatIndex];
                    },
                    [&features](TFeaturePosition position, size_t index) -> int {
                        return ConvertFloatCatFeatureToIntHash(features[index][position.FlatIndex]);
                    },
                    TCpuEvaluator::TextFeatureAccessorStub,
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
                    ModelTrees->GetFlatFeatureVectorExpectedSize() <= features.size(),
                    "Not enough features provided"
                );
                CalcGeneric(
                    *ModelTrees,
                    CtrProvider,
                    TextProcessingCollection,
                    [&features](TFeaturePosition position, size_t ) -> float {
                        return features[position.FlatIndex];
                    },
                    [&features](TFeaturePosition position, size_t ) -> int {
                        return ConvertFloatCatFeatureToIntHash(features[position.FlatIndex]);
                    },
                    TCpuEvaluator::TextFeatureAccessorStub,
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
                CB_ENSURE(
                    ModelTrees->GetTextFeatures().empty(),
                    "Model contains text features but they aren't provided"
                );
                Calc(
                    floatFeatures,
                    catFeatures,
                    {},
                    treeStart,
                    treeEnd,
                    results,
                    featureInfo
                );
            }

            void Calc(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<int>> catFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> textFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo
            ) const {
                if (!featureInfo) {
                    featureInfo = ExtFeatureLayout.Get();
                }
                ValidateInputFeatures(floatFeatures, catFeatures, textFeatures, featureInfo);
                const size_t docCount = Max(catFeatures.size(), floatFeatures.size());
                CalcGeneric(
                    *ModelTrees,
                    CtrProvider,
                    TextProcessingCollection,
                    [&floatFeatures](TFeaturePosition position, size_t index) -> float {
                        return floatFeatures[index][position.Index];
                    },
                    [&catFeatures](TFeaturePosition position, size_t index) -> int {
                        return catFeatures[index][position.Index];
                    },
                    [&textFeatures](TFeaturePosition position, size_t index) -> TStringBuf {
                        return textFeatures[index][position.Index];
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
                CB_ENSURE(
                    ModelTrees->GetTextFeatures().empty(),
                    "Model contains text features but they aren't provided"
                );
                Calc(
                    floatFeatures,
                    catFeatures,
                    {},
                    treeStart,
                    treeEnd,
                    results,
                    featureInfo
                );
            }

            void Calc(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> catFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> textFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo
            ) const {
                if (!featureInfo) {
                    featureInfo = ExtFeatureLayout.Get();
                }
                ValidateInputFeatures(floatFeatures, catFeatures, textFeatures, featureInfo);
                const size_t docCount = Max(catFeatures.size(), floatFeatures.size(), textFeatures.size());
                CalcGeneric(
                    *ModelTrees,
                    CtrProvider,
                    TextProcessingCollection,
                    [&floatFeatures](TFeaturePosition position, size_t index) -> float {
                        return floatFeatures[index][position.Index];
                    },
                    [&catFeatures](TFeaturePosition position, size_t index) -> int {
                        return CalcCatFeatureHash(catFeatures[index][position.Index]);
                    },
                    [&textFeatures](TFeaturePosition position, size_t index) -> TStringBuf {
                        return textFeatures[index][position.Index];
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
                ValidateInputFeatures<TConstArrayRef<TStringBuf>>({floatFeatures}, {catFeatures}, {}, featureInfo);
                CalcLeafIndexesGeneric(
                    *ModelTrees,
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
                ValidateInputFeatures(floatFeatures, catFeatures, {}, featureInfo);
                const size_t docCount = Max(catFeatures.size(), floatFeatures.size());
                CB_ENSURE(docCount * (treeEnd - treeStart) == indexes.size(), LabeledOutput(docCount * (treeEnd - treeStart), indexes.size()));
                CalcLeafIndexesGeneric(
                    *ModelTrees,
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
                if (ModelTrees->GetEffectiveBinaryFeaturesBucketsCount() != 0) {
                    CB_ENSURE(
                        cpuQuantizedFeatures->BlockStride % ModelTrees->GetEffectiveBinaryFeaturesBucketsCount() == 0,
                        "Unexpected block stride: " << cpuQuantizedFeatures->BlockStride
                        << " (EffectiveBinaryFeaturesBucketsCount == " << ModelTrees->GetEffectiveBinaryFeaturesBucketsCount() << " )"
                    );
                }
                CB_ENSURE(cpuQuantizedFeatures->BlocksCount * FORMULA_EVALUATION_BLOCK_SIZE >= cpuQuantizedFeatures->ObjectsCount);
                std::fill(results.begin(), results.end(), 0.0);
                auto subBlockSize = Min<size_t>(FORMULA_EVALUATION_BLOCK_SIZE, cpuQuantizedFeatures->ObjectsCount);
                auto calcFunction = GetCalcTreesFunction(
                    *ModelTrees,
                    subBlockSize,
                    false
                );
                CB_ENSURE(results.size() == ModelTrees->GetDimensionsCount() * cpuQuantizedFeatures->ObjectsCount);
                TVector<TCalcerIndexType> indexesVec(subBlockSize);
                double* resultPtr = results.data();
                for (size_t blockId = 0; blockId < cpuQuantizedFeatures->BlocksCount; ++blockId) {
                    auto subBlock = cpuQuantizedFeatures->ExtractBlock(blockId);
                    calcFunction(
                        *ModelTrees, &subBlock,
                        subBlock.ObjectsCount,
                        indexesVec.data(),
                        treeStart,
                        treeEnd,
                        resultPtr
                    );
                    size_t items = subBlock.GetObjectsCount() * ModelTrees->GetDimensionsCount();
                    ApplyScaleAndBias(ModelTrees->GetScaleAndBias(), TArrayRef<double>(resultPtr, resultPtr + items), treeStart);
                    resultPtr += items;
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
                if (ModelTrees->GetEffectiveBinaryFeaturesBucketsCount() != 0) {
                    CB_ENSURE(
                        cpuQuantizedFeatures->BlockStride % ModelTrees->GetEffectiveBinaryFeaturesBucketsCount() == 0,
                        "Unexpected block stride: " << cpuQuantizedFeatures->BlockStride
                        << " (EffectiveBinaryFeaturesBucketsCount == " << ModelTrees->GetEffectiveBinaryFeaturesBucketsCount() << " )"
                    );
                }
                CB_ENSURE(cpuQuantizedFeatures->BlocksCount * FORMULA_EVALUATION_BLOCK_SIZE >= cpuQuantizedFeatures->ObjectsCount);
                auto calcFunction = GetCalcTreesFunction(
                    *ModelTrees,
                    Min<size_t>(FORMULA_EVALUATION_BLOCK_SIZE, cpuQuantizedFeatures->ObjectsCount),
                    /*calcIndexesOnly*/ true
                );
                size_t treeCount = treeEnd - treeStart;
                CB_ENSURE(indexes.size() == treeCount * cpuQuantizedFeatures->ObjectsCount);
                TVector<TCalcerIndexType> tmpLeafIndexHolder;
                TCalcerIndexType* indexesWritePtr = indexes.data();
                for (size_t blockId = 0; blockId < cpuQuantizedFeatures->BlocksCount; ++blockId) {
                    auto subBlock = cpuQuantizedFeatures->ExtractBlock(blockId);
                    tmpLeafIndexHolder.yresize(subBlock.GetObjectsCount() * treeCount);
                    TCalcerIndexType* transposedLeafIndexesPtr = tmpLeafIndexHolder.data();
                    calcFunction(
                        *ModelTrees,
                        &subBlock,
                        subBlock.ObjectsCount,
                        transposedLeafIndexesPtr,
                        treeStart,
                        treeEnd,
                        /*results*/ nullptr
                    );
                    const size_t indexCountInBlock = subBlock.GetObjectsCount() * treeCount;
                    Transpose2DArray<TCalcerIndexType>(
                        {transposedLeafIndexesPtr, indexCountInBlock},
                        treeCount,
                        subBlock.GetObjectsCount(),
                        {indexesWritePtr, indexCountInBlock}
                    );
                    indexesWritePtr += indexCountInBlock;
                }
            }

        private:
            template <typename TCatFeatureContainer = TConstArrayRef<int>>
            void ValidateInputFeatures(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TCatFeatureContainer> catFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> textFeatures,
                const TFeatureLayout* featureInfo
            ) const {
                if (!floatFeatures.empty() && !catFeatures.empty()) {
                    CB_ENSURE(catFeatures.size() == floatFeatures.size());
                }
                CB_ENSURE(
                    ModelTrees->GetUsedFloatFeaturesCount() == 0 || !floatFeatures.empty(),
                    "Model has float features but no float features provided"
                );
                CB_ENSURE(
                    ModelTrees->GetUsedCatFeaturesCount() == 0 || !catFeatures.empty(),
                    "Model has categorical features but no categorical features provided"
                );
                CB_ENSURE(
                    ModelTrees->GetUsedTextFeaturesCount() == 0 || !textFeatures.empty(),
                    "Model has text features but no text features provided"
                );
                size_t minimalSufficientFloatFeatureCount = ModelTrees->GetMinimalSufficientFloatFeaturesVectorSize();
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
                size_t minimalSufficientCatFeatureCount = ModelTrees->GetMinimalSufficientCatFeaturesVectorSize();
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
                size_t minimalSufficientTextFeatureCount = ModelTrees->GetMinimalSufficientTextFeaturesVectorSize();
                if (featureInfo && featureInfo->TextFeatureIndexes.Defined()) {
                    CB_ENSURE(featureInfo->TextFeatureIndexes->size() >= minimalSufficientTextFeatureCount);
                    minimalSufficientTextFeatureCount = *MaxElement(
                        featureInfo->TextFeatureIndexes->begin(),
                        featureInfo->TextFeatureIndexes->end()
                    );
                }
                for (const auto& textFeaturesVec : textFeatures) {
                    CB_ENSURE(
                        textFeaturesVec.size() >= minimalSufficientTextFeatureCount,
                        "insufficient text features vector size: " << textFeaturesVec.size()
                        << " expected: " << minimalSufficientTextFeatureCount
                    );
                }
            }

            static TStringBuf TextFeatureAccessorStub(TFeaturePosition position, size_t index) {
                Y_UNUSED(position, index);
                CB_ENSURE(false, "This type of apply interface is not implemented with text features yet");
            }
        private:
            TCOWTreeWrapper ModelTrees;
            const TIntrusivePtr<ICtrProvider> CtrProvider;
            const TIntrusivePtr<TTextProcessingCollection> TextProcessingCollection;
            EPredictionType PredictionType = EPredictionType::RawFormulaVal;
            TMaybe<TFeatureLayout> ExtFeatureLayout;
        };
    }

    TEvaluationBackendFactory::TRegistrator<NDetail::TCpuEvaluator> CPUEvaluationBackendRegistrator(EFormulaEvaluatorType::CPU);
}
