#include <catboost/libs/model/evaluation_interface.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/scale_and_bias.h>
#include <catboost/libs/model/static_ctr_provider.h>
#include <catboost/libs/model/cpu/evaluator.h>

#include <catboost/libs/model/cuda/evaluator.cuh>

#include <util/generic/ymath.h>
#include <util/string/cast.h>
#include <util/system/hp_timer.h>

namespace NCB::NModelEvaluation {
    namespace NDetail {
        class TGpuEvaluator final : public IModelEvaluator, public IGpuModelEvaluator {
        public:
            TGpuEvaluator(const TGpuEvaluator& other) = default;

            explicit TGpuEvaluator(const TFullModel& model)
                : ModelTrees(model.ModelTrees)
                , ApplyData(ModelTrees->GetApplyData())
            {
                CB_ENSURE(!model.HasTextFeatures(), "Model contains text features, GPU evaluation impossible");
                CB_ENSURE(!model.HasEmbeddingFeatures(), "Model contains embedding features, GPU evaluation impossible");
                CB_ENSURE(model.IsOblivious(), "Model is not oblivious, GPU evaluation impossible");
                CB_ENSURE(ApplyData->UsedEstimatedFeaturesCount == 0, "Model contains estimated features, GPU evaluation impossible");
                CB_ENSURE(model.HasValidCtrProvider(), "Model contains CTR features but has no valid CTR provider");

                CB_ENSURE(ModelTrees->GetDimensionsCount() > 0, "Model has zero dimensions, GPU evaluation impossible");

                TVector<TGPURepackedBin> gpuBins;
                for (const TRepackedBin& cpuRepackedBin : ModelTrees->GetRepackedBins()) {
                    gpuBins.emplace_back(TGPURepackedBin{ static_cast<ui32>(cpuRepackedBin.FeatureIndex * WarpSize), cpuRepackedBin.SplitIdx, cpuRepackedBin.XorMask });
                }
                Ctx.GPUModelData.UsedInModel.resize(ApplyData->MinimalSufficientFloatFeaturesVectorSize, false);

                ui32 floatBucketCount = 0;
                for (const TFloatFeature& floatFeature : ModelTrees->GetFloatFeatures()) {
                    if (!floatFeature.UsedInModel()) {
                        continue;
                    }
                    floatBucketCount += (floatFeature.Borders.size() + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN;
                }

                TVector<ui32> floatFeatureForBucketIdx(floatBucketCount, Max<ui32>());
                TVector<ui32> bordersOffsets(floatBucketCount, Max<ui32>());
                TVector<ui32> bordersCount(floatBucketCount, Max<ui32>());
                TVector<float> flatBordersVec;
                ui32 currentBinarizedBucket = 0;
                for (const TFloatFeature& floatFeature : ModelTrees->GetFloatFeatures()) {
                    if (!floatFeature.UsedInModel()) {
                        continue;
                    }
                    Ctx.GPUModelData.UsedInModel[floatFeature.Position.Index] = floatFeature.UsedInModel();
                    CB_ENSURE(!floatFeature.Borders.empty(), "Expected non-empty borders for used float feature");

                    for (size_t blockStart = 0; blockStart < floatFeature.Borders.size(); blockStart += MAX_VALUES_PER_BIN) {
                        const size_t blockEnd = Min(blockStart + MAX_VALUES_PER_BIN, floatFeature.Borders.size());
                        floatFeatureForBucketIdx[currentBinarizedBucket] = floatFeature.Position.Index;
                        bordersCount[currentBinarizedBucket] = SafeIntegerCast<ui32>(blockEnd - blockStart);
                        bordersOffsets[currentBinarizedBucket] = SafeIntegerCast<ui32>(flatBordersVec.size());
                        flatBordersVec.insert(
                            flatBordersVec.end(),
                            floatFeature.Borders.begin() + blockStart,
                            floatFeature.Borders.begin() + blockEnd
                        );
                        ++currentBinarizedBucket;
                    }
                }
                CB_ENSURE(currentBinarizedBucket == floatBucketCount, "Float bucket count mismatch");
                Ctx.GPUModelData.FloatFeatureForBucketIdx = TCudaVec<ui32>(floatFeatureForBucketIdx, NCuda::EMemoryType::Device);
                Ctx.GPUModelData.TreeSplits = TCudaVec<TGPURepackedBin>(gpuBins, NCuda::EMemoryType::Device);
                Ctx.GPUModelData.BordersOffsets = TCudaVec<ui32>(bordersOffsets, NCuda::EMemoryType::Device);
                Ctx.GPUModelData.BordersCount = TCudaVec<ui32>(bordersCount, NCuda::EMemoryType::Device);
                Ctx.GPUModelData.FlatBordersVector = TCudaVec<float>(flatBordersVec, NCuda::EMemoryType::Device);
                Ctx.GPUModelData.FloatBucketCount = floatBucketCount;

                // Categorical feature packing (matches CPU apply quantization: only used cat features are packed).
                THashMap<int, int> catFeaturePackedIndex;
                int usedCatFeatureIdx = 0;
                for (const auto& catFeature : ModelTrees->GetCatFeatures()) {
                    if (!catFeature.UsedInModel()) {
                        continue;
                    }
                    catFeaturePackedIndex[catFeature.Position.Index] = usedCatFeatureIdx++;
                }

                TVector<ui32> oneHotCatFeaturePackedIdxForBucket;
                TVector<ui32> oneHotValuesOffsets;
                TVector<ui32> oneHotValuesCount;
                TVector<ui32> flatOneHotValues;

                for (const auto& oheFeature : ModelTrees->GetOneHotFeatures()) {
                    const auto it = catFeaturePackedIndex.find(oheFeature.CatFeatureIndex);
                    CB_ENSURE(it != catFeaturePackedIndex.end(), "OneHot feature refers to unknown cat feature");
                    const ui32 packedCatIdx = SafeIntegerCast<ui32>(it->second);

                    for (size_t blockStart = 0; blockStart < oheFeature.Values.size(); blockStart += MAX_VALUES_PER_BIN) {
                        const size_t blockEnd = Min(blockStart + MAX_VALUES_PER_BIN, oheFeature.Values.size());

                        oneHotCatFeaturePackedIdxForBucket.push_back(packedCatIdx);
                        oneHotValuesOffsets.push_back(SafeIntegerCast<ui32>(flatOneHotValues.size()));
                        oneHotValuesCount.push_back(SafeIntegerCast<ui32>(blockEnd - blockStart));

                        for (size_t i = blockStart; i < blockEnd; ++i) {
                            flatOneHotValues.push_back(static_cast<ui32>(oheFeature.Values[i]));
                        }
                    }
                }

                Ctx.GPUModelData.OneHotBucketCount = SafeIntegerCast<ui32>(oneHotCatFeaturePackedIdxForBucket.size());
                if (Ctx.GPUModelData.OneHotBucketCount > 0) {
                    Ctx.GPUModelData.OneHotCatFeaturePackedIdxForBucket = TCudaVec<ui32>(oneHotCatFeaturePackedIdxForBucket, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.OneHotValuesOffsets = TCudaVec<ui32>(oneHotValuesOffsets, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.OneHotValuesCount = TCudaVec<ui32>(oneHotValuesCount, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.FlatOneHotValues = TCudaVec<ui32>(flatOneHotValues, NCuda::EMemoryType::Device);
                }

                const auto& ctrFeatures = ModelTrees->GetCtrFeatures();
                Ctx.GPUModelData.CtrFeatureCount = SafeIntegerCast<ui32>(ctrFeatures.size());

                if (Ctx.GPUModelData.CtrFeatureCount > 0) {
                    const auto* staticCtrProvider = dynamic_cast<const TStaticCtrProvider*>(model.CtrProvider.Get());
                    CB_ENSURE(staticCtrProvider != nullptr, "CTR features require static CTR provider for GPU evaluation");

                    const auto usedCtrBases = ApplyData->GetUsedModelCtrBases();

                    THashMap<TModelCtrBase, ui32> ctrBaseToTableIdx;
                    TVector<TGPUCtrBucket> ctrIndexBuckets;
                    TVector<ui32> ctrTableBucketsOffsets;
                    TVector<ui32> ctrTableBucketsCount;
                    TVector<ui8> ctrTableDataKind;
                    TVector<ui32> ctrTableMeanOffsets;
                    TVector<ui32> ctrTableMeanCount;
                    TVector<TGPUCtrMeanHistory> ctrMeanData;
                    TVector<ui32> ctrTableIntOffsets;
                    TVector<ui32> ctrTableIntCount;
                    TVector<int> ctrIntData;
                    TVector<int> ctrTableCounterDenom;
                    TVector<int> ctrTableTargetClassesCount;

                    ctrTableBucketsOffsets.reserve(usedCtrBases.size());
                    ctrTableBucketsCount.reserve(usedCtrBases.size());
                    ctrTableDataKind.reserve(usedCtrBases.size());
                    ctrTableMeanOffsets.reserve(usedCtrBases.size());
                    ctrTableMeanCount.reserve(usedCtrBases.size());
                    ctrTableIntOffsets.reserve(usedCtrBases.size());
                    ctrTableIntCount.reserve(usedCtrBases.size());
                    ctrTableCounterDenom.reserve(usedCtrBases.size());
                    ctrTableTargetClassesCount.reserve(usedCtrBases.size());

                    ui32 tableIdx = 0;
                    for (const auto& base : usedCtrBases) {
                        const auto it = staticCtrProvider->CtrData.LearnCtrs.find(base);
                        CB_ENSURE(it != staticCtrProvider->CtrData.LearnCtrs.end(), "CTR table missing in static provider");
                        const TCtrValueTable& table = it->second;

                        const auto bucketsView = table.GetIndexHashViewer().GetBuckets();
                        ctrTableBucketsOffsets.push_back(SafeIntegerCast<ui32>(ctrIndexBuckets.size()));
                        ctrTableBucketsCount.push_back(SafeIntegerCast<ui32>(bucketsView.size()));
                        for (const auto& b : bucketsView) {
                            ctrIndexBuckets.push_back(TGPUCtrBucket{b.Hash, b.IndexValue});
                        }

                        ctrTableCounterDenom.push_back(table.CounterDenominator);
                        ctrTableTargetClassesCount.push_back(table.TargetClassesCount);

                        if (base.CtrType == ECtrType::BinarizedTargetMeanValue || base.CtrType == ECtrType::FloatTargetMeanValue) {
                            ctrTableDataKind.push_back(0);
                            ctrTableMeanOffsets.push_back(SafeIntegerCast<ui32>(ctrMeanData.size()));
                            const auto meanRef = table.GetTypedArrayRefForBlobData<TCtrMeanHistory>();
                            ctrTableMeanCount.push_back(SafeIntegerCast<ui32>(meanRef.size()));
                            for (const auto& m : meanRef) {
                                ctrMeanData.push_back(TGPUCtrMeanHistory{m.Sum, SafeIntegerCast<i32>(m.Count)});
                            }
                            ctrTableIntOffsets.push_back(0);
                            ctrTableIntCount.push_back(0);
                        } else {
                            ctrTableDataKind.push_back(1);
                            ctrTableIntOffsets.push_back(SafeIntegerCast<ui32>(ctrIntData.size()));
                            const auto intRef = table.GetTypedArrayRefForBlobData<int>();
                            ctrTableIntCount.push_back(SafeIntegerCast<ui32>(intRef.size()));
                            ctrIntData.insert(ctrIntData.end(), intRef.begin(), intRef.end());
                            ctrTableMeanOffsets.push_back(0);
                            ctrTableMeanCount.push_back(0);
                        }

                        ctrBaseToTableIdx[base] = tableIdx++;
                    }

                    Ctx.GPUModelData.CtrIndexBuckets = TCudaVec<TGPUCtrBucket>(ctrIndexBuckets, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrTableBucketsOffsets = TCudaVec<ui32>(ctrTableBucketsOffsets, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrTableBucketsCount = TCudaVec<ui32>(ctrTableBucketsCount, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrTableDataKind = TCudaVec<ui8>(ctrTableDataKind, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrTableMeanOffsets = TCudaVec<ui32>(ctrTableMeanOffsets, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrTableMeanCount = TCudaVec<ui32>(ctrTableMeanCount, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrMeanData = TCudaVec<TGPUCtrMeanHistory>(ctrMeanData, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrTableIntOffsets = TCudaVec<ui32>(ctrTableIntOffsets, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrTableIntCount = TCudaVec<ui32>(ctrTableIntCount, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrIntData = TCudaVec<int>(ctrIntData, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrTableCounterDenom = TCudaVec<int>(ctrTableCounterDenom, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrTableTargetClassesCount = TCudaVec<int>(ctrTableTargetClassesCount, NCuda::EMemoryType::Device);

                    TVector<ui32> ctrFeatureTableIdx;
                    TVector<ui8> ctrFeatureType;
                    TVector<ui32> ctrFeatureTargetBorderIdx;
                    TVector<float> ctrPriorNum;
                    TVector<float> ctrPriorDenom;
                    TVector<float> ctrShift;
                    TVector<float> ctrScale;

                    TVector<ui32> ctrFeatureCatOffsets;
                    TVector<ui32> ctrFeatureCatCount;
                    TVector<ui32> ctrProjCatPackedIdx;

                    TVector<ui32> ctrFeatureFloatOffsets;
                    TVector<ui32> ctrFeatureFloatCount;
                    TVector<TGPUCtrFloatSplit> ctrProjFloatSplits;

                    TVector<ui32> ctrFeatureOneHotOffsets;
                    TVector<ui32> ctrFeatureOneHotCount;
                    TVector<TGPUCtrOneHotSplit> ctrProjOneHotSplits;

                    ctrFeatureTableIdx.reserve(ctrFeatures.size());
                    ctrFeatureType.reserve(ctrFeatures.size());
                    ctrFeatureTargetBorderIdx.reserve(ctrFeatures.size());
                    ctrPriorNum.reserve(ctrFeatures.size());
                    ctrPriorDenom.reserve(ctrFeatures.size());
                    ctrShift.reserve(ctrFeatures.size());
                    ctrScale.reserve(ctrFeatures.size());
                    ctrFeatureCatOffsets.reserve(ctrFeatures.size());
                    ctrFeatureCatCount.reserve(ctrFeatures.size());
                    ctrFeatureFloatOffsets.reserve(ctrFeatures.size());
                    ctrFeatureFloatCount.reserve(ctrFeatures.size());
                    ctrFeatureOneHotOffsets.reserve(ctrFeatures.size());
                    ctrFeatureOneHotCount.reserve(ctrFeatures.size());

                    for (const auto& ctrFeature : ctrFeatures) {
                        const auto& ctr = ctrFeature.Ctr;
                        const auto baseIt = ctrBaseToTableIdx.find(ctr.Base);
                        CB_ENSURE(baseIt != ctrBaseToTableIdx.end(), "CTR base not found in uploaded tables");
                        ctrFeatureTableIdx.push_back(baseIt->second);
                        ctrFeatureType.push_back(static_cast<ui8>(ctr.Base.CtrType));
                        ctrFeatureTargetBorderIdx.push_back(SafeIntegerCast<ui32>(ctr.TargetBorderIdx));
                        ctrPriorNum.push_back(ctr.PriorNum);
                        ctrPriorDenom.push_back(ctr.PriorDenom);
                        ctrShift.push_back(ctr.Shift);
                        ctrScale.push_back(ctr.Scale);

                        const auto& proj = ctr.Base.Projection;

                        ctrFeatureCatOffsets.push_back(SafeIntegerCast<ui32>(ctrProjCatPackedIdx.size()));
                        ctrFeatureCatCount.push_back(SafeIntegerCast<ui32>(proj.CatFeatures.size()));
                        for (const int catIdx : proj.CatFeatures) {
                            const auto it = catFeaturePackedIndex.find(catIdx);
                            CB_ENSURE(it != catFeaturePackedIndex.end(), "CTR projection refers to unknown cat feature");
                            ctrProjCatPackedIdx.push_back(SafeIntegerCast<ui32>(it->second));
                        }

                        ctrFeatureFloatOffsets.push_back(SafeIntegerCast<ui32>(ctrProjFloatSplits.size()));
                        ctrFeatureFloatCount.push_back(SafeIntegerCast<ui32>(proj.BinFeatures.size()));
                        for (const auto& fs : proj.BinFeatures) {
                            ctrProjFloatSplits.push_back(TGPUCtrFloatSplit{SafeIntegerCast<ui32>(fs.FloatFeature), fs.Split});
                        }

                        ctrFeatureOneHotOffsets.push_back(SafeIntegerCast<ui32>(ctrProjOneHotSplits.size()));
                        ctrFeatureOneHotCount.push_back(SafeIntegerCast<ui32>(proj.OneHotFeatures.size()));
                        for (const auto& oh : proj.OneHotFeatures) {
                            const auto it = catFeaturePackedIndex.find(oh.CatFeatureIdx);
                            CB_ENSURE(it != catFeaturePackedIndex.end(), "CTR projection refers to unknown cat feature");
                            ctrProjOneHotSplits.push_back(TGPUCtrOneHotSplit{SafeIntegerCast<ui32>(it->second), static_cast<ui32>(oh.Value)});
                        }
                    }

                    Ctx.GPUModelData.CtrFeatureTableIdx = TCudaVec<ui32>(ctrFeatureTableIdx, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrFeatureType = TCudaVec<ui8>(ctrFeatureType, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrFeatureTargetBorderIdx = TCudaVec<ui32>(ctrFeatureTargetBorderIdx, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrFeaturePriorNum = TCudaVec<float>(ctrPriorNum, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrFeaturePriorDenom = TCudaVec<float>(ctrPriorDenom, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrFeatureShift = TCudaVec<float>(ctrShift, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrFeatureScale = TCudaVec<float>(ctrScale, NCuda::EMemoryType::Device);

                    Ctx.GPUModelData.CtrFeatureCatOffsets = TCudaVec<ui32>(ctrFeatureCatOffsets, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrFeatureCatCount = TCudaVec<ui32>(ctrFeatureCatCount, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrProjCatFeaturePackedIdx = TCudaVec<ui32>(ctrProjCatPackedIdx, NCuda::EMemoryType::Device);

                    Ctx.GPUModelData.CtrFeatureFloatOffsets = TCudaVec<ui32>(ctrFeatureFloatOffsets, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrFeatureFloatCount = TCudaVec<ui32>(ctrFeatureFloatCount, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrProjFloatSplits = TCudaVec<TGPUCtrFloatSplit>(ctrProjFloatSplits, NCuda::EMemoryType::Device);

                    Ctx.GPUModelData.CtrFeatureOneHotOffsets = TCudaVec<ui32>(ctrFeatureOneHotOffsets, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrFeatureOneHotCount = TCudaVec<ui32>(ctrFeatureOneHotCount, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrProjOneHotSplits = TCudaVec<TGPUCtrOneHotSplit>(ctrProjOneHotSplits, NCuda::EMemoryType::Device);

                    TVector<ui32> ctrFeatureBucketOffsets;
                    TVector<ui32> ctrBordersOffsets;
                    TVector<ui32> ctrBordersCount;
                    TVector<float> flatCtrBorders;

                    ctrFeatureBucketOffsets.reserve(ctrFeatures.size() + 1);
                    ui32 ctrBucketCount = 0;
                    for (const auto& ctrFeature : ctrFeatures) {
                        ctrFeatureBucketOffsets.push_back(ctrBucketCount);
                        for (size_t blockStart = 0; blockStart < ctrFeature.Borders.size(); blockStart += MAX_VALUES_PER_BIN) {
                            const size_t blockEnd = Min(blockStart + MAX_VALUES_PER_BIN, ctrFeature.Borders.size());
                            ctrBordersOffsets.push_back(SafeIntegerCast<ui32>(flatCtrBorders.size()));
                            ctrBordersCount.push_back(SafeIntegerCast<ui32>(blockEnd - blockStart));
                            flatCtrBorders.insert(
                                flatCtrBorders.end(),
                                ctrFeature.Borders.begin() + blockStart,
                                ctrFeature.Borders.begin() + blockEnd
                            );
                            ++ctrBucketCount;
                        }
                    }
                    ctrFeatureBucketOffsets.push_back(ctrBucketCount);

                    Ctx.GPUModelData.CtrBucketCount = ctrBucketCount;
                    Ctx.GPUModelData.CtrFeatureBucketOffsets = std::move(ctrFeatureBucketOffsets);
                    Ctx.GPUModelData.CtrBordersOffsets = TCudaVec<ui32>(ctrBordersOffsets, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.CtrBordersCount = TCudaVec<ui32>(ctrBordersCount, NCuda::EMemoryType::Device);
                    Ctx.GPUModelData.FlatCtrBordersVector = TCudaVec<float>(flatCtrBorders, NCuda::EMemoryType::Device);
                } else {
                    Ctx.GPUModelData.CtrBucketCount = 0;
                    Ctx.GPUModelData.CtrFeatureBucketOffsets.clear();
                }

                Ctx.GPUModelData.BucketsCount =
                    Ctx.GPUModelData.FloatBucketCount + Ctx.GPUModelData.OneHotBucketCount + Ctx.GPUModelData.CtrBucketCount;

                Ctx.GPUModelData.TreeSizes = TCudaVec<ui32>(
                    TVector<ui32>(ModelTrees->GetModelTreeData()->GetTreeSizes().begin(), ModelTrees->GetModelTreeData()->GetTreeSizes().end()),
                    NCuda::EMemoryType::Device
                );
                Ctx.GPUModelData.TreeStartOffsets = TCudaVec<ui32>(
                    TVector<ui32>(ModelTrees->GetModelTreeData()->GetTreeStartOffsets().begin(), ModelTrees->GetModelTreeData()->GetTreeStartOffsets().end()),
                    NCuda::EMemoryType::Device
                );
                const auto& firstLeafOffsetRef = ApplyData->TreeFirstLeafOffsets;
                TVector<ui32> firstLeafOffset(firstLeafOffsetRef.begin(), firstLeafOffsetRef.end());
                Ctx.GPUModelData.TreeFirstLeafOffsets = TCudaVec<ui32>(firstLeafOffset, NCuda::EMemoryType::Device);
                Ctx.GPUModelData.ModelLeafs = TCudaVec<TCudaEvaluatorLeafType>(
                    TVector<TCudaEvaluatorLeafType>(ModelTrees->GetModelTreeData()->GetLeafValues().begin(), ModelTrees->GetModelTreeData()->GetLeafValues().end()),
                    NCuda::EMemoryType::Device
                );

                Ctx.GPUModelData.ApproxDimension = ModelTrees->GetDimensionsCount();

                const auto& scaleAndBias = ModelTrees->GetScaleAndBias();
                const auto& biasRef = scaleAndBias.GetBiasRef();
                Ctx.GPUModelData.Bias = TCudaVec<double>(
                    biasRef.empty() ? TVector<double>(Ctx.GPUModelData.ApproxDimension, 0.0) : biasRef,
                    NCuda::EMemoryType::Device
                );
                Ctx.GPUModelData.Scale = scaleAndBias.Scale;

                Ctx.Stream = TCudaStream::ZeroStream();
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
                    *docCount * Ctx.GPUModelData.ApproxDimension
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
                NCuda::MemoryCopyAsync<float>(MakeArrayRef(buf), Ctx.EvalDataCache.CopyDataBufDevice.AsArrayRef(), Ctx.Stream);
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
                Ctx.EvalDataCache.PrepareCopyBufs(docCount * stride, docCount * Ctx.GPUModelData.ApproxDimension);
                dataInput.FlatFloatsVector = Ctx.EvalDataCache.CopyDataBufDevice.AsArrayRef();
                auto copyBufRef = Ctx.EvalDataCache.CopyDataBufHost.AsArrayRef();
                for (size_t docId = 0; docId < docCount; ++docId) {
                    memcpy(&copyBufRef[docId * stride], features[docId].data(), sizeof(float) * expectedFlatVecSize);
                }
                NCuda::MemoryCopyAsync<float>(copyBufRef, Ctx.EvalDataCache.CopyDataBufDevice.AsArrayRef(), Ctx.Stream);
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

            void CalcOnDevice(
                const TGPUDataInput& dataInput,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results
            ) const override {
                if (ApplyData->UsedCatFeaturesCount > 0) {
                    CB_ENSURE(
                        dataInput.CatFeatureCount >= ApplyData->UsedCatFeaturesCount,
                        "Insufficient categorical features provided for GPU evaluation"
                    );
                }
                Ctx.EvalData(dataInput, treeStart, treeEnd, results, PredictionType);
            }

            void CalcLeafIndexesOnDevice(
                const TGPUDataInput& dataInput,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<TCalcerIndexType> indexes
            ) const override {
                if (ApplyData->UsedCatFeaturesCount > 0) {
                    CB_ENSURE(
                        dataInput.CatFeatureCount >= ApplyData->UsedCatFeaturesCount,
                        "Insufficient categorical features provided for GPU evaluation"
                    );
                }
                Ctx.CalcLeafIndexesOnDevice(dataInput, treeStart, treeEnd, indexes);
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
            Ctx.EvalDataCache.PrepareCopyBufs(docCount * stride, docCount * Ctx.GPUModelData.ApproxDimension);
            dataInput.FlatFloatsVector = Ctx.EvalDataCache.CopyDataBufDevice.AsArrayRef();
            auto copyBufRef = Ctx.EvalDataCache.CopyDataBufHost.AsArrayRef();
            for (size_t docId = 0; docId < docCount; ++docId) {
                memcpy(
                    &copyBufRef[docId * stride],
                    features[docId].data(),
                    sizeof(float) * expectedFlatVecSize
                );
            }
            NCuda::MemoryCopyAsync<float>(copyBufRef, Ctx.EvalDataCache.CopyDataBufDevice.AsArrayRef(), Ctx.Stream);
            TCudaQuantizedData* cudaQuantizedData = static_cast<TCudaQuantizedData*>(quantizedData);
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
