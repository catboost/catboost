#include <catboost/libs/model/eval_processing.h>
#include <catboost/libs/model/model.h>

#include "evaluator.h"

namespace NCB::NModelEvaluation {
    namespace NDetail {
        template <typename TFloatFeatureAccessor, typename TCatFeatureAccessor,
                  typename TTextFeatureAccessor, typename TEmbeddingFeatureAccessor>
        inline void CalcGeneric(
            const TModelTrees& trees,
            const TModelTrees::TForApplyData& applyData,
            const TIntrusivePtr<ICtrProvider>& ctrProvider,
            const TIntrusivePtr<TTextProcessingCollection>& textProcessingCollection,
            const TIntrusivePtr<TEmbeddingProcessingCollection>& embeddingProcessingCollection,
            TFloatFeatureAccessor floatFeatureAccessor,
            TCatFeatureAccessor catFeaturesAccessor,
            TTextFeatureAccessor textFeatureAccessor,
            TEmbeddingFeatureAccessor embeddingFeatureAccessor,
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
                auto biasRef = trees.GetScaleAndBias().GetBiasRef();
                if (biasRef.size() == 1) {
                    Fill(results.begin(), results.end(), biasRef[0]);
                } else {
                    for (size_t idx = 0; idx < results.size();) {
                        for (size_t dim = 0; dim < biasRef.size(); ++dim, ++idx) {
                            results[idx] = biasRef[dim];
                        }
                    }
                }
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
                embeddingProcessingCollection,
                floatFeatureAccessor,
                catFeaturesAccessor,
                textFeatureAccessor,
                embeddingFeatureAccessor,
                docCount,
                blockSize,
                [&] (size_t docCountInBlock, const TCPUEvaluatorQuantizedData* quantizedData) {
                    auto blockResultsView = resultProcessor.GetViewForRawEvaluation(blockId);
                    calcTrees(
                        trees,
                        applyData,
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
                , ApplyData(ModelTrees->GetApplyData())
                , CtrProvider(fullModel.CtrProvider)
                , TextProcessingCollection(fullModel.TextProcessingCollection)
                , EmbeddingProcessingCollection(fullModel.EmbeddingProcessingCollection)
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

            size_t GetTreeCount() const override {
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
                    *ApplyData,
                    CtrProvider,
                    TextProcessingCollection,
                    EmbeddingProcessingCollection,
                    [&transposedFeatures](TFeaturePosition floatFeature, size_t index) -> float {
                        return transposedFeatures[floatFeature.FlatIndex][index];
                    },
                    [&transposedFeatures](TFeaturePosition catFeature, size_t index) -> int {
                        return ConvertFloatCatFeatureToIntHash(transposedFeatures[catFeature.FlatIndex][index]);
                    },
                    TCpuEvaluator::TextFeatureAccessorStub,
                    TCpuEvaluator::EmbeddingFeatureAccessorStub,
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
                    *ApplyData,
                    CtrProvider,
                    TextProcessingCollection,
                    EmbeddingProcessingCollection,
                    [&features](TFeaturePosition position, size_t index) -> float {
                        return features[index][position.FlatIndex];
                    },
                    [&features](TFeaturePosition position, size_t index) -> int {
                        return ConvertFloatCatFeatureToIntHash(features[index][position.FlatIndex]);
                    },
                    TCpuEvaluator::TextFeatureAccessorStub,
                    TCpuEvaluator::EmbeddingFeatureAccessorStub,
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
                    *ApplyData,
                    CtrProvider,
                    TextProcessingCollection,
                    EmbeddingProcessingCollection,
                    [&features](TFeaturePosition position, size_t ) -> float {
                        return features[position.FlatIndex];
                    },
                    [&features](TFeaturePosition position, size_t ) -> int {
                        return ConvertFloatCatFeatureToIntHash(features[position.FlatIndex]);
                    },
                    TCpuEvaluator::TextFeatureAccessorStub,
                    TCpuEvaluator::EmbeddingFeatureAccessorStub,
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
                    {},
                    treeStart,
                    treeEnd,
                    results,
                    featureInfo
                );
            }

            void CalcWithHashedCatAndTextAndEmbeddings(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<int>> catFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> textFeatures,
                TConstArrayRef<TConstArrayRef<TConstArrayRef<float>>> embeddingFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo
            ) const override {
                Calc(floatFeatures, catFeatures, textFeatures, embeddingFeatures, treeStart, treeEnd, results, featureInfo);
            }

            void Calc(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<int>> catFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> textFeatures,
                TConstArrayRef<TConstArrayRef<TConstArrayRef<float>>> embeddingFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo
            ) const {
                if (!featureInfo) {
                    featureInfo = ExtFeatureLayout.Get();
                }
                ValidateInputFeatures(floatFeatures, catFeatures, textFeatures, embeddingFeatures, featureInfo);
                const size_t docCount = Max(catFeatures.size(), floatFeatures.size(), textFeatures.size(), embeddingFeatures.size());
                CalcGeneric(
                    *ModelTrees,
                    *ApplyData,
                    CtrProvider,
                    TextProcessingCollection,
                    EmbeddingProcessingCollection,
                    [&floatFeatures](TFeaturePosition position, size_t index) -> float {
                        return floatFeatures[index][position.Index];
                    },
                    [&catFeatures](TFeaturePosition position, size_t index) -> int {
                        return catFeatures[index][position.Index];
                    },
                    [&textFeatures](TFeaturePosition position, size_t index) -> TStringBuf {
                        return textFeatures[index][position.Index];
                    },
                    [&embeddingFeatures](TFeaturePosition position, size_t index) -> TConstArrayRef<float> {
                        return embeddingFeatures[index][position.Index];
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
            ) const override {
                if (!featureInfo) {
                    featureInfo = ExtFeatureLayout.Get();
                }
                ValidateInputFeatures(floatFeatures, catFeatures, textFeatures, {}, featureInfo);
                const size_t docCount = Max(catFeatures.size(), floatFeatures.size(), textFeatures.size());
                CalcGeneric(
                    *ModelTrees,
                    *ApplyData,
                    CtrProvider,
                    TextProcessingCollection,
                    EmbeddingProcessingCollection,
                    [&floatFeatures](TFeaturePosition position, size_t index) -> float {
                        return floatFeatures[index][position.Index];
                    },
                    [&catFeatures](TFeaturePosition position, size_t index) -> int {
                        return CalcCatFeatureHash(catFeatures[index][position.Index]);
                    },
                    [&textFeatures](TFeaturePosition position, size_t index) -> TStringBuf {
                        return textFeatures[index][position.Index];
                    },
                    TCpuEvaluator::EmbeddingFeatureAccessorStub,
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
                TConstArrayRef<TConstArrayRef<TStringBuf>> textFeatures,
                TConstArrayRef<TConstArrayRef<TConstArrayRef<float>>> embeddingFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo
            ) const override {
                if (!featureInfo) {
                    featureInfo = ExtFeatureLayout.Get();
                }
                ValidateInputFeatures(floatFeatures, catFeatures, textFeatures, embeddingFeatures, featureInfo);
                const size_t docCount = Max(catFeatures.size(), floatFeatures.size(), textFeatures.size());
                CalcGeneric(
                    *ModelTrees,
                    *ApplyData,
                    CtrProvider,
                    TextProcessingCollection,
                    EmbeddingProcessingCollection,
                    [&floatFeatures](TFeaturePosition position, size_t index) -> float {
                        return floatFeatures[index][position.Index];
                    },
                    [&catFeatures](TFeaturePosition position, size_t index) -> int {
                        return CalcCatFeatureHash(catFeatures[index][position.Index]);
                    },
                    [&textFeatures](TFeaturePosition position, size_t index) -> TStringBuf {
                        return textFeatures[index][position.Index];
                    },
                    [&embeddingFeatures](TFeaturePosition position, size_t index) -> TConstArrayRef<float> {
                        return embeddingFeatures[index][position.Index];
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
                ValidateInputFeatures<TConstArrayRef<TStringBuf>>({floatFeatures}, {catFeatures}, {}, {}, featureInfo);
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
                ValidateInputFeatures(floatFeatures, catFeatures, {}, {}, featureInfo);
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
                        *ModelTrees,
                        *ApplyData,
                        &subBlock,
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
                        *ApplyData,
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

            void Quantize(
            TConstArrayRef<TConstArrayRef<float>> features,
            IQuantizedData* quantizedData
        ) const override {
            Y_UNUSED(features);
            Y_UNUSED(quantizedData);
            CB_ENSURE(false, "Unimplemented method called, please contact catboost developers via GitHub issue or in support chat");
        }

        private:
            template <typename TCatFeatureContainer = TConstArrayRef<int>>
            void ValidateInputFeatures(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TCatFeatureContainer> catFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> textFeatures,
                TConstArrayRef<TConstArrayRef<TConstArrayRef<float>>> embeddingFeatures,
                const TFeatureLayout* featureInfo
            ) const {
                if (!floatFeatures.empty() && !catFeatures.empty()) {
                    CB_ENSURE(catFeatures.size() == floatFeatures.size());
                }
                CB_ENSURE(
                    ApplyData->UsedFloatFeaturesCount == 0 || !floatFeatures.empty(),
                    "Model has float features but no float features provided"
                );
                CB_ENSURE(
                    ApplyData->UsedCatFeaturesCount == 0 || !catFeatures.empty(),
                    "Model has categorical features but no categorical features provided"
                );
                CB_ENSURE(
                    ApplyData->UsedTextFeaturesCount == 0 || !textFeatures.empty(),
                    "Model has text features but no text features provided"
                );
                CB_ENSURE(
                    ApplyData->UsedEmbeddingFeaturesCount == 0 || !embeddingFeatures.empty(),
                    "Model has embedding features but no embedding features provided"
                );
                size_t minimalSufficientFloatFeatureCount = ApplyData->MinimalSufficientFloatFeaturesVectorSize;
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
                size_t minimalSufficientCatFeatureCount = ApplyData->MinimalSufficientCatFeaturesVectorSize;
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
                size_t minimalSufficientTextFeatureCount = ApplyData->MinimalSufficientTextFeaturesVectorSize;
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
                size_t minimalSufficientEmbeddingFeatureCount = ApplyData->MinimalSufficientEmbeddingFeaturesVectorSize;
                if (featureInfo && featureInfo->EmbeddingFeatureIndexes.Defined()) {
                    CB_ENSURE(featureInfo->EmbeddingFeatureIndexes->size() >= minimalSufficientEmbeddingFeatureCount);
                    minimalSufficientEmbeddingFeatureCount = *MaxElement(
                        featureInfo->EmbeddingFeatureIndexes->begin(),
                        featureInfo->EmbeddingFeatureIndexes->end()
                    );
                }
                for (const auto& embeddingFeaturesVec : embeddingFeatures) {
                    CB_ENSURE(
                        embeddingFeaturesVec.size() >= minimalSufficientEmbeddingFeatureCount,
                        "insufficient embedding features vector size: " << embeddingFeaturesVec.size()
                        << " expected: " << minimalSufficientEmbeddingFeatureCount
                    );
                }
            }

            static TStringBuf TextFeatureAccessorStub(TFeaturePosition position, size_t index) {
                Y_UNUSED(position, index);
                CB_ENSURE(false, "This type of apply interface is not implemented with text features yet");
            }

            static TConstArrayRef<float> EmbeddingFeatureAccessorStub(TFeaturePosition position, size_t index) {
                Y_UNUSED(position, index);
                CB_ENSURE(false, "This type of apply interface is not implemented with embedding features yet");
            }
        private:
            TCOWTreeWrapper ModelTrees;
            TAtomicSharedPtr<TModelTrees::TForApplyData> ApplyData;
            const TIntrusivePtr<ICtrProvider> CtrProvider;
            const TIntrusivePtr<TTextProcessingCollection> TextProcessingCollection;
            const TIntrusivePtr<TEmbeddingProcessingCollection> EmbeddingProcessingCollection;
            EPredictionType PredictionType = EPredictionType::RawFormulaVal;
            TMaybe<TFeatureLayout> ExtFeatureLayout;
        };
    }

    TEvaluationBackendFactory::TRegistrator<NDetail::TCpuEvaluator> CPUEvaluationBackendRegistrator(EFormulaEvaluatorType::CPU);

    void* CPUEvaluationBackendRegistratorPointer = &CPUEvaluationBackendRegistrator;
}
