#include <catboost/libs/model/evaluation_interface.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/cpu/evaluator.h>

#include <catboost/libs/model/cuda/evaluator.cuh>

namespace NCB::NModelEvaluation {
    namespace NDetail {
        class TGpuEvaluator final : public IModelEvaluator {
        public:
            TGpuEvaluator(const TGpuEvaluator& other) = default;

            explicit TGpuEvaluator(const TFullModel& model)
                : ObliviousTrees(model.ObliviousTrees)
            {
                TVector<TGPURepackedBin> gpuBins;
                for (const TRepackedBin& cpuRepackedBin : ObliviousTrees->GetRepackedBins()) {
                    gpuBins.emplace_back(TGPURepackedBin{ static_cast<ui32>(cpuRepackedBin.FeatureIndex * WarpSize), cpuRepackedBin.SplitIdx, cpuRepackedBin.XorMask });
                }
                TVector<ui32> floatFeatureForBucketIdx(ObliviousTrees->GetUsedFloatFeaturesCount(), Max<ui32>());
                TVector<ui32> bordersOffsets(ObliviousTrees->GetUsedFloatFeaturesCount(), Max<ui32>());
                TVector<ui32> bordersCount(ObliviousTrees->GetUsedFloatFeaturesCount(), Max<ui32>());
                Ctx.GPUModelData.UsedInModel.resize(ObliviousTrees->GetMinimalSufficientFloatFeaturesVectorSize(), false);
                TVector<float> flatBordersVec;
                ui32 currentBinarizedBucket = 0;
                for (const TFloatFeature& floatFeature : ObliviousTrees->FloatFeatures) {
                    Ctx.GPUModelData.UsedInModel[floatFeature.Position.Index] = floatFeature.UsedInModel();
                    if (!floatFeature.UsedInModel()) {
                        continue;
                    }
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
                    TVector<ui32>(ObliviousTrees->TreeSizes.begin(), ObliviousTrees->TreeSizes.end()),
                    EMemoryType::Device
                );
                Ctx.GPUModelData.TreeStartOffsets = TCudaVec<ui32>(
                    TVector<ui32>(ObliviousTrees->TreeStartOffsets.begin(), ObliviousTrees->TreeStartOffsets.end()),
                    EMemoryType::Device
                );
                const auto& firstLeafOffsetRef = ObliviousTrees->GetFirstLeafOffsets();
                TVector<ui32> firstLeafOffset(firstLeafOffsetRef.begin(), firstLeafOffsetRef.end());
                Ctx.GPUModelData.TreeFirstLeafOffsets = TCudaVec<ui32>(firstLeafOffset, EMemoryType::Device);
                Ctx.GPUModelData.ModelLeafs = TCudaVec<TCudaEvaluatorLeafType>(
                    TVector<TCudaEvaluatorLeafType>(ObliviousTrees->LeafValues.begin(), ObliviousTrees->LeafValues.end()),
                    EMemoryType::Device
                );
                Ctx.Stream = TCudaStream::NewStream();
            }

            void SetPredictionType(EPredictionType type) override {
                Ctx.PredictionType = type;
            }

            EPredictionType GetPredictionType() const override {
                return Ctx.PredictionType;
            }

            void SetFeatureLayout(const TFeatureLayout& featureLayout) override {
                ExtFeatureLayout = featureLayout;
            }

            size_t GetTreeCount() const {
                return ObliviousTrees->GetTreeCount();
            }

            TModelEvaluatorPtr Clone() const override {
                return new TGpuEvaluator(*this);
            }

            i32 GetApproxDimension() const override {
                return ObliviousTrees->ApproxDimension;
            }

            void CalcFlatTransposed(
                TConstArrayRef<TConstArrayRef<float>> transposedFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureLayout
            ) const override {
                CB_ENSURE(
                    ObliviousTrees->GetFlatFeatureVectorExpectedSize() <= transposedFeatures.size(),
                    "Not enough features provided"
                );
                CB_ENSURE(featureLayout == nullptr, "feature layout currenlty not supported");
                TMaybe<size_t> docCount;
                CB_ENSURE(!ObliviousTrees->FloatFeatures.empty() || !ObliviousTrees->CatFeatures.empty(),
                          "Both float features and categorical features information are empty");
                if (!ObliviousTrees->FloatFeatures.empty()) {
                    for (const auto& floatFeature : ObliviousTrees->FloatFeatures) {
                        if (floatFeature.UsedInModel()) {
                            docCount = transposedFeatures[floatFeature.Position.FlatIndex].size();
                            break;
                        }
                    }
                }
                if (!docCount.Defined() && !ObliviousTrees->CatFeatures.empty()) {
                    for (const auto& catFeature : ObliviousTrees->CatFeatures) {
                        if (catFeature.UsedInModel) {
                            docCount = transposedFeatures[catFeature.Position.FlatIndex].size();
                            break;
                        }
                    }
                }

                CB_ENSURE(docCount.Defined(), "couldn't determine document count, something went wrong");
                const size_t FeatureStride = CeilDiv<size_t>(*docCount, 32) * 32;
                TGPUDataInput dataInput;
                dataInput.ObjectCount = *docCount;
                dataInput.FloatFeatureCount = transposedFeatures.size();
                dataInput.Stride = FeatureStride;
                dataInput.FlatFloatsVector = TCudaVec<float>(transposedFeatures.size() * FeatureStride, EMemoryType::Device);
                auto arrRef = dataInput.FlatFloatsVector.AsArrayRef();
                for (size_t featureId = 0; featureId < ObliviousTrees->GetMinimalSufficientFloatFeaturesVectorSize(); ++featureId) {
                    if (transposedFeatures[featureId].empty() || !Ctx.GPUModelData.UsedInModel[featureId]) {
                        continue;
                    }
                    MemoryCopyAsync(transposedFeatures[featureId], arrRef.Slice(featureId * FeatureStride, transposedFeatures[featureId].size()), Ctx.Stream);
                }
                TVector<TCudaEvaluatorLeafType> gpuD(results.size());
                Ctx.EvalData(dataInput, gpuD, treeStart, treeEnd);
            }

            void CalcFlat(
                TConstArrayRef<TConstArrayRef<float>> features,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureLayout
            ) const override {
                CB_ENSURE(featureLayout == nullptr, "feature layout currenlty not supported");
                if (!featureLayout) {
                    featureLayout = ExtFeatureLayout.Get();
                }
                auto expectedFlatVecSize = ObliviousTrees->GetFlatFeatureVectorExpectedSize();
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
                size_t docCount = features.size();

                const size_t FeatureStride = CeilDiv<size_t>(docCount, 32) * 32;
                TGPUDataInput dataInput;
                dataInput.ObjectCount = docCount;
                dataInput.FloatFeatureCount = features[0].size();
                dataInput.Stride = FeatureStride;
                dataInput.FlatFloatsVector = TCudaVec<float>(dataInput.FloatFeatureCount * FeatureStride, EMemoryType::Host);
                auto arrRef = dataInput.FlatFloatsVector.AsArrayRef();
                for (size_t featureId = 0; featureId < ObliviousTrees->GetMinimalSufficientFloatFeaturesVectorSize(); ++featureId) {
                    if (!Ctx.GPUModelData.UsedInModel[featureId]) {
                        continue;
                    }
                    auto target = arrRef.Slice(featureId * FeatureStride, docCount);
                    #pragma clang loop vectorize_width(16)
                    for (size_t i = 0; i < docCount; ++i) {
                        target[i] = features[i][featureId];
                    }
                }
                TVector<TCudaEvaluatorLeafType> gpuD(results.size());
                Ctx.EvalData(dataInput, gpuD, treeStart, treeEnd);
            }

            void CalcFlatSingle(
                TConstArrayRef<float> features,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout*
            ) const override {
                CB_ENSURE(
                    ObliviousTrees->GetFlatFeatureVectorExpectedSize() <= features.size(),
                    "Not enough features provided"
                );
                Y_UNUSED(treeStart);
                Y_UNUSED(treeEnd);
                Y_UNUSED(results);
                ythrow yexception() << "Unimplemented on GPU";
            }

            void Calc(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<int>> catFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout*
            ) const override {
                ValidateInputFeatures(floatFeatures, catFeatures);
                const size_t docCount = Max(catFeatures.size(), floatFeatures.size());
                Y_UNUSED(treeStart);
                Y_UNUSED(docCount);
                Y_UNUSED(treeEnd);
                Y_UNUSED(results);
                ythrow yexception() << "Unimplemented on GPU";
            }

            void Calc(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> catFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout*
            ) const override {
                ValidateInputFeatures(floatFeatures, catFeatures);
                const size_t docCount = Max(catFeatures.size(), floatFeatures.size());
                Y_UNUSED(treeStart);
                Y_UNUSED(docCount);
                Y_UNUSED(treeEnd);
                Y_UNUSED(results);
                ythrow yexception() << "Unimplemented on GPU";
            }

            void Calc(
                const IQuantizedData* quantizedFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results
            ) const override {
                Y_UNUSED(treeStart);
                Y_UNUSED(quantizedFeatures);
                Y_UNUSED(treeEnd);
                Y_UNUSED(results);
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
                    ObliviousTrees->GetUsedFloatFeaturesCount() == 0 || !floatFeatures.empty(),
                    "Model has float features but no float features provided"
                );
                CB_ENSURE(
                    ObliviousTrees->GetUsedCatFeaturesCount() == 0 || !catFeatures.empty(),
                    "Model has categorical features but no categorical features provided"
                );
                for (const auto& floatFeaturesVec : floatFeatures) {
                    CB_ENSURE(
                        floatFeaturesVec.size() >= ObliviousTrees->GetMinimalSufficientFloatFeaturesVectorSize(),
                        "insufficient float features vector size: " << floatFeaturesVec.size() << " expected: " <<
                                                                    ObliviousTrees->GetMinimalSufficientFloatFeaturesVectorSize()
                    );
                }
                for (const auto& catFeaturesVec : catFeatures) {
                    CB_ENSURE(
                        catFeaturesVec.size() >= ObliviousTrees->GetMinimalSufficientCatFeaturesVectorSize(),
                        "insufficient cat features vector size: " << catFeaturesVec.size() << " expected: " <<
                                                                  ObliviousTrees->GetMinimalSufficientCatFeaturesVectorSize()
                    );
                }
            }

        private:
            TCOWTreeWrapper ObliviousTrees;
            TMaybe<TFeatureLayout> ExtFeatureLayout;
            TGPUCatboostEvaluationContext Ctx;
        };
    }
    TModelEvaluatorPtr CreateGpuEvaluator(const TFullModel& model) {
        if (!CudaEvaluationPossible(model)) {
            CB_ENSURE(!model.HasCategoricalFeatures(), "Model contains categorical features, gpu evaluation impossible");
            CB_ENSURE(model.IsOblivious(), "Model is not oblivious, gpu evaluation impossible");
        }
        return new NDetail::TGpuEvaluator(model);
    }
    bool CudaEvaluationPossible(const TFullModel& model) {
        return !model.HasCategoricalFeatures() && model.IsOblivious();
    }
}
