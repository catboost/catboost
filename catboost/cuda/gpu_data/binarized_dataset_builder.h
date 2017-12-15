#pragma once

#include "binarized_dataset.h"
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/transform.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/cuda_util/helpers.h>
#include <catboost/cuda/utils/cpu_random.h>
#include <util/random/shuffle.h>
namespace NCatboostCuda {
    template <class TLayout>
    class TFeaturesSplitter {
    public:
        struct TFeaturesSplit {
            TVector<ui32> FeatureIds;
            TLayout Layout;
        };

        struct TPolicySplit {
            TVector<ui32> PolicyFeatures;
            TVector<ui32> RestFeatures;
        };

        template <class TGridPolicy>
        static TPolicySplit ExtractFeaturesForPolicy(const TBinarizedFeaturesManager& featuresManager,
                                                     const TVector<ui32>& ids) {
            TPolicySplit result;

            for (ui32 id : ids) {
                if (featuresManager.IsCat(id) && !featuresManager.UseForOneHotEncoding(id)) {
                    result.RestFeatures.push_back(id);
                    continue;
                }

                if (TGridPolicy::CanProceed(featuresManager.GetBinCount(id))) {
                    result.PolicyFeatures.push_back(id);
                } else {
                    result.RestFeatures.push_back(id);
                }
            }

            return result;
        };

        static TFeaturesSplit Split(const TBinarizedFeaturesManager& grid,
                                    const TVector<ui32>& featureIds,
                                    bool shuffleFeatures = true) {
            TFeaturesSplit split;

            split.FeatureIds = TVector<ui32>(featureIds.begin(), featureIds.end());
            TRandom rand(0);
            Shuffle(split.FeatureIds.begin(), split.FeatureIds.end(), rand);
            split.Layout = NCudaLib::TStripeMapping::SplitBetweenDevices(featureIds.size());

            if (shuffleFeatures) {
                for (ui32 dev : split.Layout.NonEmptyDevices()) {
                    TSlice devSlice = split.Layout.DeviceSlice(dev);
                    std::sort(split.FeatureIds.begin() + devSlice.Left, split.FeatureIds.begin() + devSlice.Right,
                              [&](ui32 left, ui32 right) -> bool {
                                  return grid.GetBinCount(left) < grid.GetBinCount(right);
                              });
                }
            }
            return split;
        }
    };

    template <class TBinarizedDataSet>
    class TGpuBinarizedDataSetBuilderHelper: public TNonCopyable {
    public:
        using TFeaturesMapping = typename TBinarizedDataSet::TFeaturesMapping;
        using TSampleMapping = typename TBinarizedDataSet::TSampleMapping;
        using TGridPolicy = typename TBinarizedDataSet::TGridPolicy;
        using TDataSet = TBinarizedDataSet;

        static void UpdateFoldOffsets(TDataSet& dataSet) {
            dataSet.Grid.GetMapping().Apply([&](const TSlice& devFeaturesSlice) -> ui64 {
                ui32 firstBinOffset = 0;
                for (ui32 i = 0; i < devFeaturesSlice.Size(); ++i) {
                    dataSet.HostFeatures[devFeaturesSlice.Left +
                                         i]
                        .FirstFoldIndex = firstBinOffset;
                    firstBinOffset += dataSet.HostFeatures[devFeaturesSlice.Left +
                                                           i]
                                          .Folds;
                }
                return firstBinOffset;
            });
        }

        static void InitTCFeatures(TDataSet& dataSet) {
            dataSet.HostFeatures.resize(dataSet.Grid.GetObjectsSlice().Size());

            dataSet.Grid.GetMapping().Apply([&](const TSlice& devFeaturesSlice) {
                using THelper = TCompressedIndexHelper<TGridPolicy>;
                const ui32 featuresPerInt = THelper::FeaturesPerInt();
                const ui32 mask = THelper::Mask();
                const ui32 maxFolds = THelper::MaxFolds();

                for (ui32 i = 0; i < devFeaturesSlice.Size(); ++i) {
                    const ui32 feature = (ui32)devFeaturesSlice.Left + i;
                    const ui32 shift = THelper::Shift(i);
                    dataSet.HostFeatures[feature] = {i / featuresPerInt,
                                                     mask,
                                                     shift,
                                                     0,
                                                     maxFolds,
                                                     i,
                                                     feature,
                                                     false};
                }
            });
        }

        static void WriteGridToDevice(TDataSet& dataSet) {
            dataSet.Grid.Write(dataSet.HostFeatures);
        }

        static TVector<TVector<TSlice>> Reset(TDataSet& dataSet,
                                              const TFeaturesMapping& featuresMapping,
                                              const TSampleMapping& docsMapping) {
            dataSet.Grid.Reset(featuresMapping);
            dataSet.DocsMapping = docsMapping;

            InitTCFeatures(dataSet);

            ui64 cursor = 0;
            TVector<TSlice> slices;

            TVector<TVector<TSlice>> groupedFeatureSlices;
            const auto deviceCount = GetDeviceCount();
            groupedFeatureSlices.resize(deviceCount);

            for (ui32 dev = 0; dev < deviceCount; ++dev) {
                const ui64 deviceDocs = docsMapping.DeviceSlice(dev).Size();
                const TSlice featuresSlice = featuresMapping.DeviceSlice(dev);

                const ui32 devFeatures = (const ui32)featuresSlice.Size();
                const ui32 featuresPerInt = sizeof(ui32) * TGridPolicy::FeaturesPerByte();
                const ui64 deviceSize = NHelpers::CeilDivide(devFeatures, featuresPerInt) * deviceDocs;

                for (ui32 feature = 0; feature < devFeatures; feature += featuresPerInt) {
                    const auto featureIdx = featuresSlice.Left + feature;
                    const ui64 offset = cursor + dataSet.HostFeatures[featureIdx].Offset * deviceDocs;

                    for (ui32 i = 0; i < featuresPerInt; ++i) {
                        if (feature + i < devFeatures) {
                            CB_ENSURE(dataSet.HostFeatures[featureIdx].Offset ==
                                      dataSet.HostFeatures[featureIdx + i].Offset);
                        }
                    }
                    groupedFeatureSlices[dev].push_back(TSlice(offset, offset + deviceDocs));
                }
                slices.push_back(TSlice(cursor, cursor + deviceSize));
                cursor += slices.back().Size();
            }

            const auto mapping = TFeaturesMapping(std::move(slices));
            dataSet.CompressedIndex.Reset(mapping);
            FillBuffer(dataSet.CompressedIndex, static_cast<ui32>(0));
            return groupedFeatureSlices;
        }

        static void BuildInverseIndex(TDataSet& dataSet) {
            for (ui32 i = 0; i < dataSet.FeatureIds.size(); ++i) {
                dataSet.LocalFeatureIndex[dataSet.FeatureIds[i]] = i;
            }
        }

        static void BuildBinaryFeatures(TDataSet& dataSet) {
            dataSet.HostBinFeatures.clear();

            auto mapping = dataSet.GetGrid().GetMapping().Transform([&](const TSlice& slice) -> ui64 {
                ui32 cursor = 0;
                for (ui32 f = slice.Left; f < slice.Right; ++f) {
                    auto feature = dataSet.HostFeatures[f];
                    TCBinFeature binFeatureBase;
                    binFeatureBase.FeatureId = feature.Index;

                    for (ui32 fold = 0;
                         fold < feature.Folds; ++fold) {
                        TCBinFeature binFeature = binFeatureBase;
                        binFeature.BinId = fold;
                        dataSet.HostBinFeatures.push_back(
                            binFeature);
                        cursor++;
                    }
                }
                return cursor;
            });
            dataSet.BinaryFeatures.Reset(mapping);
            dataSet.BinaryFeatures.Write(dataSet.HostBinFeatures);
        }
    };

    template <class TGridPolicy_,
              class TLayoutPolicy = TCatBoostPoolLayout>
    class TGpuBinarizedDataSetBuilder: public TNonCopyable {
    public:
        using TFeaturesMapping = typename TLayoutPolicy::TFeaturesMapping;
        using TSampleMapping = typename TLayoutPolicy::TSampleMapping;
        using TGridPolicy = TGridPolicy_;
        using TDataSet = TGpuBinarizedDataSet<TGridPolicy, TLayoutPolicy>;

        TGpuBinarizedDataSetBuilder(TFeaturesMapping& featuresMapping,
                                    TSampleMapping& docsMapping,
                                    TVector<ui32>* gatherIndex = nullptr)
            : GatherIndex(gatherIndex)
        {
            GroupedFeatureSlices = TGpuBinarizedDataSetBuilderHelper<TDataSet>::Reset(DataSet,
                                                                                      featuresMapping,
                                                                                      docsMapping);
            TempIndex.resize(NCudaLib::GetCudaManager().GetDeviceCount());
        }

        TGpuBinarizedDataSetBuilder& SetFeatureIds(const TVector<ui32>& featureIds) {
            CB_ENSURE(featureIds.size() == DataSet.Grid.GetObjectsSlice().Size());
            DataSet.FeatureIds = featureIds;
            TGpuBinarizedDataSetBuilderHelper<TDataSet>::BuildInverseIndex(DataSet);
            FeaturesAreSet = true;
            return *this;
        }

        TGpuBinarizedDataSetBuilder& UseForOneHotIds(const TVector<ui32>& featureIds) {
            for (auto featureId : featureIds) {
                const ui32 localId = DataSet.LocalFeatureIndex[featureId];
                DataSet.HostFeatures[localId].OneHotFeature = true;
            }
            return *this;
        }

        TGpuBinarizedDataSetBuilder& Write(ui32 featureManagerFeatureId,
                                           const ui32 binCount,
                                           const TVector<ui32>& bins) {
            CB_ENSURE(FeaturesAreSet, "Set features first");
            const auto& docsMapping = DataSet.DocsMapping;
            CB_ENSURE(bins.size() == docsMapping.GetObjectsSlice().Size());
            CB_ENSURE(binCount > 1, "Feature is empty");
            const ui32 featureId = DataSet.LocalFeatureIndex[featureManagerFeatureId];
            CB_ENSURE(!SeenFeatures.has(featureId), "Error: can't write feature twice");

            TCFeature& feature = DataSet.HostFeatures[featureId];

            auto& featuresMapping = DataSet.Grid.GetMapping();

            for (auto& dev : docsMapping.NonEmptyDevices()) {
                auto devSlice = docsMapping.DeviceSlice(dev);
                TSlice featureSlice = featuresMapping.DeviceSlice(dev);

                if (featureSlice.Contains(featureId)) {
                    auto& index = GetForGroupedFeature(feature, dev);

                    if (GatherIndex) {
                        TVector<ui32> gatherBins(bins.size());
                        for (ui32 i = 0; i < bins.size(); ++i) {
                            gatherBins[i] = bins[(*GatherIndex)[i]];
                        }
                        index.WriteToTempIndex(feature, ~gatherBins + devSlice.Left);
                    } else {
                        index.WriteToTempIndex(feature, ~bins + devSlice.Left);
                    }
                }
            }
            feature.Folds = (feature.OneHotFeature && binCount != 2) ? binCount : (binCount - 1);
            SeenFeatures.insert(featureId);
            return *this;
        }

        TDataSet Finish() {
            CB_ENSURE(!BuildIsDone, "Build could be finished only once");
            const ui32 featureCount = DataSet.Grid.GetMapping().GetObjectsSlice().Size();
            MATRIXNET_INFO_LOG << "Host features " << DataSet.HostFeatures.size() << Endl;

            for (ui32 f = 0; f < featureCount; ++f) {
                CB_ENSURE(SeenFeatures.count(f), "Unseen feature #" << DataSet.FeatureIds[f]);
            }

            Flush();
            TGpuBinarizedDataSetBuilderHelper<TDataSet>::UpdateFoldOffsets(DataSet);
            TGpuBinarizedDataSetBuilderHelper<TDataSet>::WriteGridToDevice(DataSet);

            TGpuBinarizedDataSetBuilderHelper<TDataSet>::BuildInverseIndex(DataSet);
            TGpuBinarizedDataSetBuilderHelper<TDataSet>::BuildBinaryFeatures(DataSet);

            BuildIsDone = true;
            MATRIXNET_INFO_LOG << "Build dataSet with feature count #" << featureCount << " and doc count #"
                               << DataSet.GetDocumentsMapping().GetObjectsSlice().Size() << Endl;
            MATRIXNET_INFO_LOG
                << "Features per ui32 " << TGridPolicy::TGridPolicy::FeaturesPerByte() * sizeof(ui32) << Endl;
            const auto& mapping = DataSet.GetCompressedIndex().GetMapping();
            for (auto dev : mapping.NonEmptyDevices()) {
                MATRIXNET_INFO_LOG
                    << "Memory usage for " << DataSet.GetGrid().GetMapping().DeviceSlice(dev).Size() << " features at #"
                    << dev << " " << mapping.MemoryUsageAt(dev) * sizeof(ui32) * 1.0 / 1024 / 1024 << " MB" << Endl;
            }
            return std::move(DataSet);
        };

    private:
        void Flush() {
            for (auto& tempIndex : TempIndex) {
                WriteIndex(tempIndex);
            }
        }

        struct TTempIndex {
            TCFeature Feature = {(ui32)-1, (ui32)-1, (ui32)-1, (ui32)-1, (ui32)-1, (ui32)-1, (ui32)-1, false};
            TVector<ui32> Data;
            TSlice Slice;
            bool Synced = true;

            void WriteToTempIndex(const TCFeature& feature,
                                  const ui32* bins) {
                CB_ENSURE(feature.Offset == Feature.Offset);

                Feature = feature;

                for (ui32 i = 0; i < Data.size(); ++i) {
                    CB_ENSURE((bins[i] & feature.Mask) == bins[i]);
                    CB_ENSURE((bins[i] & 255) == bins[i]);
                    Data[i] |= bins[i] << Feature.Shift;
                }

                Synced = false;
            }
        };

        TGpuBinarizedDataSetBuilder& WriteIndex(TTempIndex& index) {
            if (!index.Synced) {
                DataSet.CompressedIndex
                    .SliceView(index.Slice)
                    .Write(index.Data);

                index.Synced = true;
            }
            return *this;
        }

        TGpuBinarizedDataSetBuilder& ReadIndex(TTempIndex& index) {
            if (!index.Synced) {
                DataSet.CompressedIndex
                    .CreateReader()
                    .SetReadSlice(index.Slice)
                    .Read(index.Data);
                index.Synced = true;
            }

            return *this;
        }

        TTempIndex& GetForGroupedFeature(const TCFeature& feature,
                                         ui32 device) {
            auto& index = TempIndex[device];
            if (feature.Offset != index.Feature.Offset) {
                WriteIndex(index);

                index.Feature = feature;
                index.Slice = GroupedFeatureSlices[device][feature.Offset];
                index.Synced = false;

                ReadIndex(index);
            }
            return index;
        }

        TSet<ui32> SeenFeatures;
        TVector<TVector<TSlice>> GroupedFeatureSlices;
        TVector<TTempIndex> TempIndex;

        TVector<ui32>* GatherIndex = nullptr;

        bool BuildIsDone = false;
        bool FeaturesAreSet = false;
        TDataSet DataSet;
    };
}
