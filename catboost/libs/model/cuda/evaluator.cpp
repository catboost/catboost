#include <catboost/libs/model/evaluation_interface.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/scale_and_bias.h>
#include <catboost/libs/model/cpu/evaluator.h>

#include <catboost/libs/model/cuda/evaluator.cuh>

#include <util/generic/ymath.h>
#include <util/string/cast.h>
#include <util/system/hp_timer.h>

namespace NCB::NModelEvaluation {
    namespace NDetail {
        class TGpuEvaluator final : public IModelEvaluator {
        public:
            TGpuEvaluator(const TGpuEvaluator& other) = default;

            explicit TGpuEvaluator(const TFullModel& model)
                : ModelTrees(model.ModelTrees)
                , ApplyData(ModelTrees->GetApplyData())
            {
                CB_ENSURE(!model.HasCategoricalFeatures(), "Model contains categorical features, GPU evaluation impossible");
                CB_ENSURE(!model.HasTextFeatures(), "Model contains text features, GPU evaluation impossible");
                CB_ENSURE(!model.HasEmbeddingFeatures(), "Model contains embedding features, GPU evaluation impossible");
                CB_ENSURE(model.IsOblivious(), "Model is not oblivious, GPU evaluation impossible");

                // TODO(akhropov): Support Multidimensional models
                CB_ENSURE(ModelTrees->GetDimensionsCount() == 1, "Model is not one-dimensional, GPU evaluation is not supported yet");

                TVector<TGPURepackedBin> gpuBins;
                for (const TRepackedBin& cpuRepackedBin : ModelTrees->GetRepackedBins()) {
                    gpuBins.emplace_back(TGPURepackedBin{ static_cast<ui32>(cpuRepackedBin.FeatureIndex * WarpSize), cpuRepackedBin.SplitIdx, cpuRepackedBin.XorMask });
                }
                TVector<ui32> floatFeatureForBucketIdx(ApplyData->UsedFloatFeaturesCount, Max<ui32>());
                TVector<ui32> bordersOffsets(ApplyData->UsedFloatFeaturesCount, Max<ui32>());
                TVector<ui32> bordersCount(ApplyData->UsedFloatFeaturesCount, Max<ui32>());
                Ctx.GPUModelData.UsedInModel.resize(ApplyData->MinimalSufficientFloatFeaturesVectorSize, false);
                TVector<float> flatBordersVec;
                ui32 currentBinarizedBucket = 0;
                for (const TFloatFeature& floatFeature : ModelTrees->GetFloatFeatures()) {
                    if (!floatFeature.UsedInModel()) {
                        continue;
                    }
                    Ctx.GPUModelData.UsedInModel[floatFeature.Position.Index] = floatFeature.UsedInModel();
                    CB_ENSURE(floatFeature.Borders.size() > 0 && floatFeature.Borders.size() < MAX_VALUES_PER_BIN);
                    floatFeatureForBucketIdx[currentBinarizedBucket] = floatFeature.Position.Index;
                    bordersCount[currentBinarizedBucket] = floatFeature.Borders.size();
                    bordersOffsets[currentBinarizedBucket] = flatBordersVec.size();
                    flatBordersVec.insert(flatBordersVec.end(), floatFeature.Borders.begin(), floatFeature.Borders.end());
                    ++currentBinarizedBucket;
                }
                Ctx.GPUModelData.FloatFeatureForBucketIdx = TCudaVec<ui32>(floatFeatureForBucketIdx, EMemoryType::Device);
                Ctx.GPUModelData.TreeSplits = TCudaVec<TGPURepackedBin>(gpuBins, EMemoryType::Device);
                Ctx.GPUModelData.BordersOffsets = TCudaVec<ui32>(bordersOffsets, EMemoryType::Device);
                Ctx.GPUModelData.BordersCount = TCudaVec<ui32>(bordersCount, EMemoryType::Device);
                Ctx.GPUModelData.FlatBordersVector = TCudaVec<float>(flatBordersVec, EMemoryType::Device);

                Ctx.GPUModelData.TreeSizes = TCudaVec<ui32>(
                    TVector<ui32>(ModelTrees->GetModelTreeData()->GetTreeSizes().begin(), ModelTrees->GetModelTreeData()->GetTreeSizes().end()),
                    EMemoryType::Device
                );
                Ctx.GPUModelData.TreeStartOffsets = TCudaVec<ui32>(
                    TVector<ui32>(ModelTrees->GetModelTreeData()->GetTreeStartOffsets().begin(), ModelTrees->GetModelTreeData()->GetTreeStartOffsets().end()),
                    EMemoryType::Device
                );
                const auto& firstLeafOffsetRef = ApplyData->TreeFirstLeafOffsets;
                TVector<ui32> firstLeafOffset(firstLeafOffsetRef.begin(), firstLeafOffsetRef.end());
                Ctx.GPUModelData.TreeFirstLeafOffsets = TCudaVec<ui32>(firstLeafOffset, EMemoryType::Device);
                Ctx.GPUModelData.ModelLeafs = TCudaVec<TCudaEvaluatorLeafType>(
                    TVector<TCudaEvaluatorLeafType>(ModelTrees->GetModelTreeData()->GetLeafValues().begin(), ModelTrees->GetModelTreeData()->GetLeafValues().end()),
                    EMemoryType::Device
                );

                Ctx.GPUModelData.ApproxDimension = ModelTrees->GetDimensionsCount();

                const auto& scaleAndBias = ModelTrees->GetScaleAndBias();
                const auto& biasRef = scaleAndBias.GetBiasRef();
                Ctx.GPUModelData.Bias = TCudaVec<double>(
                    biasRef.empty() ? TVector<double>(Ctx.GPUModelData.ApproxDimension, 0.0) : biasRef,
                    EMemoryType::Device
                );
                Ctx.GPUModelData.Scale = scaleAndBias.Scale;

                Ctx.Stream = TCudaStream::NewStream();
            }

            void SetPredictionType(EPredictionType type) override {
                PredictionType = type;
            }

            void SetProperty(const TStringBuf propName, const TStringBuf propValue) override {
                {
                    Y_UNUSED(propValue);
                    CB_ENSURE(false, "GPU evaluator don't have property " << propName);
                }
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
                return new TGpuEvaluator(*this);
            }

            i32 GetApproxDimension() const override {
                return ModelTrees->GetDimensionsCount();
            }

            void CalcFlatTransposed(
                TConstArrayRef<TConstArrayRef<float>> transposedFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureLayout
            ) const override {
                CB_ENSURE(
                    ModelTrees->GetFlatFeatureVectorExpectedSize() <= transposedFeatures.size(),
                    "Not enough features provided"
                );
                CB_ENSURE(featureLayout == nullptr, "feature layout currenlty not supported");
                TMaybe<size_t> docCount;
                CB_ENSURE(!ModelTrees->GetFloatFeatures().empty() || !ModelTrees->GetCatFeatures().empty(),
                          "Both float features and categorical features information are empty");
                if (!ModelTrees->GetFloatFeatures().empty()) {
                    for (const auto& floatFeature : ModelTrees->GetFloatFeatures()) {
                        if (floatFeature.UsedInModel()) {
                            docCount = transposedFeatures[floatFeature.Position.FlatIndex].size();
                            break;
                        }
                    }
                }
                if (!docCount.Defined() && !ModelTrees->GetCatFeatures().empty()) {
                    for (const auto& catFeature : ModelTrees->GetCatFeatures()) {
                        if (catFeature.UsedInModel()) {
                            docCount = transposedFeatures[catFeature.Position.FlatIndex].size();
                            break;
                        }
                    }
                }

                CB_ENSURE(docCount.Defined(), "couldn't determine document count, something went wrong");
                const size_t stride = CeilDiv<size_t>(*docCount, 32) * 32;
                Ctx.EvalDataCache.PrepareCopyBufs(
                    ApplyData->MinimalSufficientFloatFeaturesVectorSize * stride,
                    *docCount
                );
                TGPUDataInput dataInput;
                dataInput.ObjectCount = *docCount;
                dataInput.FloatFeatureCount = transposedFeatures.size();
                dataInput.Stride = stride;
                dataInput.FlatFloatsVector = Ctx.EvalDataCache.CopyDataBufDevice.AsArrayRef();
                auto buf = Ctx.EvalDataCache.CopyDataBufHost.AsArrayRef();
                for (size_t featureId = 0; featureId < ApplyData->MinimalSufficientFloatFeaturesVectorSize; ++featureId) {
                    if (transposedFeatures[featureId].empty() || !Ctx.GPUModelData.UsedInModel[featureId]) {
                        continue;
                    }
                    memcpy(&buf[featureId * stride], transposedFeatures[featureId].data(), sizeof(float) * transposedFeatures[featureId].size());
                }
                MemoryCopyAsync<float>(MakeArrayRef(buf), Ctx.EvalDataCache.CopyDataBufDevice.AsArrayRef(), Ctx.Stream);
                Ctx.EvalData(dataInput, treeStart, treeEnd, results, PredictionType);
            }

            void CalcFlat(
                TConstArrayRef<TConstArrayRef<float>> features,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureLayout
            ) const override {
                CB_ENSURE(featureLayout == nullptr, "feature layout currently not supported");
                if (!featureLayout) {
                    featureLayout = ExtFeatureLayout.Get();
                }
                auto expectedFlatVecSize = ModelTrees->GetFlatFeatureVectorExpectedSize();
                if (featureLayout && featureLayout->FlatIndexes) {
                    CB_ENSURE(
                        featureLayout->FlatIndexes->size() >= expectedFlatVecSize,
                        "Feature layout FlatIndexes expected to be at least " << expectedFlatVecSize << " long"
                    );
                    expectedFlatVecSize = *MaxElement(featureLayout->FlatIndexes->begin(), featureLayout->FlatIndexes->end());
                }
                for (const auto& flatFeaturesVec : features) {
                    CB_ENSURE(
                        flatFeaturesVec.size() >= expectedFlatVecSize,
                        "insufficient flat features vector size: " << flatFeaturesVec.size() << " expected: " << expectedFlatVecSize
                    );
                }
                const size_t docCount = features.size();
                const size_t stride = CeilDiv<size_t>(expectedFlatVecSize, 32) * 32;

                TGPUDataInput dataInput;
                dataInput.FloatFeatureLayout = TGPUDataInput::EFeatureLayout::RowFirst;
                dataInput.ObjectCount = docCount;
                dataInput.FloatFeatureCount = expectedFlatVecSize;
                dataInput.Stride = stride;
                Ctx.EvalDataCache.PrepareCopyBufs(docCount * stride, docCount);
                dataInput.FlatFloatsVector = Ctx.EvalDataCache.CopyDataBufDevice.AsArrayRef();
                auto copyBufRef = Ctx.EvalDataCache.CopyDataBufHost.AsArrayRef();
                for (size_t docId = 0; docId < docCount; ++docId) {
                    memcpy(&copyBufRef[docId * stride], features[docId].data(), sizeof(float) * expectedFlatVecSize);
                }
                MemoryCopyAsync<float>(copyBufRef, Ctx.EvalDataCache.CopyDataBufDevice.AsArrayRef(), Ctx.Stream);
                Ctx.EvalData(dataInput, treeStart, treeEnd, results, PredictionType);
            }

            void CalcFlatSingle(
                TConstArrayRef<float> features,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureLayout
            ) const override {
                CalcFlat({ features }, treeStart, treeEnd, results, featureLayout);
            }

            void Calc(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<int>> catFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureLayout
            ) const override {
                ValidateInputFeatures(floatFeatures, catFeatures);
                CalcFlat(floatFeatures, treeStart, treeEnd, results, featureLayout);
            }

            void CalcWithHashedCatAndTextAndEmbeddings(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<int>> catFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> textFeatures,
                TConstArrayRef<TConstArrayRef<TConstArrayRef<float>>> embeddingFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureLayout
            ) const override {
                ValidateInputFeatures(floatFeatures, catFeatures);
                Y_UNUSED(textFeatures);
                Y_UNUSED(embeddingFeatures);
                CalcFlat(floatFeatures, treeStart, treeEnd, results, featureLayout);
            }

            void Calc(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> catFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureLayout
            ) const override {
                ValidateInputFeatures(floatFeatures, catFeatures);
                CalcFlat(floatFeatures, treeStart, treeEnd, results, featureLayout);
            }

            void Calc(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> catFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> textFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo = nullptr
            ) const override {
                ValidateInputFeatures(floatFeatures, catFeatures);
                Y_UNUSED(textFeatures);
                CalcFlat(floatFeatures, treeStart, treeEnd, results, featureInfo);
            }

            void Calc(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> catFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> textFeatures,
                TConstArrayRef<TConstArrayRef<TConstArrayRef<float>>> embeddingFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo = nullptr
            ) const override {
                ValidateInputFeatures(floatFeatures, catFeatures);
                Y_UNUSED(textFeatures);
                Y_UNUSED(embeddingFeatures);
                CalcFlat(floatFeatures, treeStart, treeEnd, results, featureInfo);
            }

            void Calc(
                const IQuantizedData* quantizedFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results
            ) const override {
                CB_ENSURE(quantizedFeatures != nullptr, "Got null quantizedFeatures");
                const TCudaQuantizedData* cudaQuantizedFeatures = dynamic_cast<const TCudaQuantizedData*>(quantizedFeatures);
                CB_ENSURE(cudaQuantizedFeatures != nullptr, "Got improperly typed quantized data");
                Ctx.EvalQuantizedData(cudaQuantizedFeatures, treeStart, treeEnd, results, PredictionType);
            }

            void CalcLeafIndexesSingle(
                TConstArrayRef<float> floatFeatures,
                TConstArrayRef<TStringBuf> catFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<ui32> indexes,
                const TFeatureLayout*
            ) const override {
                ValidateInputFeatures<TConstArrayRef<TStringBuf>>({floatFeatures}, {catFeatures});
                Y_UNUSED(treeStart);
                Y_UNUSED(treeEnd);
                Y_UNUSED(indexes);
                ythrow yexception() << "Unimplemented on GPU";
            }

            void CalcLeafIndexes(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> catFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<ui32> indexes,
                const TFeatureLayout*
            ) const override {
                ValidateInputFeatures(floatFeatures, catFeatures);
                Y_UNUSED(treeStart);
                Y_UNUSED(treeEnd);
                Y_UNUSED(indexes);
                ythrow yexception() << "Unimplemented on GPU";
            }

            void CalcLeafIndexes(
                const IQuantizedData* quantizedFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<ui32> indexes
            ) const override {
                Y_UNUSED(quantizedFeatures);
                Y_UNUSED(treeStart);
                Y_UNUSED(treeEnd);
                Y_UNUSED(indexes);
                ythrow yexception() << "Unimplemented on GPU";
            }

            void Quantize(
            TConstArrayRef<TConstArrayRef<float>> features,
            IQuantizedData* quantizedData
        ) const override {
            auto expectedFlatVecSize = ModelTrees->GetFlatFeatureVectorExpectedSize();
            const size_t docCount = features.size();
            const size_t stride = CeilDiv<size_t>(expectedFlatVecSize, 32) * 32;

            TGPUDataInput dataInput;
            dataInput.FloatFeatureLayout = TGPUDataInput::EFeatureLayout::RowFirst;
            dataInput.ObjectCount = docCount;
            dataInput.FloatFeatureCount = expectedFlatVecSize;
            dataInput.Stride = stride;
            Ctx.EvalDataCache.PrepareCopyBufs(docCount * stride, docCount);
            dataInput.FlatFloatsVector = Ctx.EvalDataCache.CopyDataBufDevice.AsArrayRef();
            auto copyBufRef = Ctx.EvalDataCache.CopyDataBufHost.AsArrayRef();
            for (size_t docId = 0; docId < docCount; ++docId) {
                memcpy(
                    &copyBufRef[docId * stride],
                    features[docId].data(),
                    sizeof(float) * expectedFlatVecSize
                );
            }
            MemoryCopyAsync<float>(copyBufRef, Ctx.EvalDataCache.CopyDataBufDevice.AsArrayRef(), Ctx.Stream);
            TCudaQuantizedData* cudaQuantizedData = reinterpret_cast<TCudaQuantizedData*>(quantizedData);
            Ctx.QuantizeData(dataInput, cudaQuantizedData);
        }
        private:
            template <typename TCatFeatureContainer = TConstArrayRef<int>>
            void ValidateInputFeatures(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TCatFeatureContainer> catFeatures
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
                for (const auto& floatFeaturesVec : floatFeatures) {
                    CB_ENSURE(
                        floatFeaturesVec.size() >= ApplyData->MinimalSufficientFloatFeaturesVectorSize,
                        "insufficient float features vector size: " << floatFeaturesVec.size() << " expected: " <<
                                                                    ApplyData->MinimalSufficientFloatFeaturesVectorSize
                    );
                }
                for (const auto& catFeaturesVec : catFeatures) {
                    CB_ENSURE(
                        catFeaturesVec.size() >= ApplyData->MinimalSufficientCatFeaturesVectorSize,
                        "insufficient cat features vector size: " << catFeaturesVec.size() << " expected: " <<
                                                                  ApplyData->MinimalSufficientCatFeaturesVectorSize
                    );
                }
            }

        private:
            EPredictionType PredictionType = EPredictionType::RawFormulaVal;
            TCOWTreeWrapper ModelTrees;
            TAtomicSharedPtr<TModelTrees::TForApplyData> ApplyData;
            TMaybe<TFeatureLayout> ExtFeatureLayout;
            TGPUCatboostEvaluationContext Ctx;
        };
    }

    TEvaluationBackendFactory::TRegistrator<NDetail::TGpuEvaluator> GPUEvaluationBackendRegistrator(EFormulaEvaluatorType::GPU);
}
