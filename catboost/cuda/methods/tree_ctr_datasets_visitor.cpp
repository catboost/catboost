#include "tree_ctr_datasets_visitor.h"
#include "pointwise_scores_calcer.h"

namespace NCatboostCuda {
    void TTreeCtrDataSetVisitor::Accept(const TTreeCtrDataSet& ctrDataSet,
                                        const TMirrorBuffer<const TPartitionStatistics>& partStats,
                                        const TMirrorBuffer<ui32>& ctrDataSetInverseIndices,
                                        const TMirrorBuffer<ui32>& subsetDocs,
                                        const TMirrorBuffer<const float>& featureWeights,
                                        double scoreBeforeSplit,
                                        ui32 maxUniqueValues,
                                        float modelSizeReg) {
        {
            auto cacheIds = GetCtrsBordersToCacheIds(ctrDataSet.GetCtrs());
            if (cacheIds.size()) {
                CacheCtrBorders(ctrDataSet.ReadBorders(cacheIds));
            }
        }

        using TScoreCalcer = TScoresCalcerOnCompressedDataSet<TSingleDevLayout>;
        auto& scoreHelper = *ctrDataSet.GetCacheHolder().Cache(ctrDataSet, 0, [&]() -> THolder<TScoreCalcer> {
            return MakeHolder<TScoreCalcer>(ctrDataSet.GetCompressedDataSet(),
                                    TreeConfig,
                                    FoldCount);
        });
        const ui32 devId = ctrDataSet.GetDeviceId();
        const ui64 taskSeed = Seeds[devId] + ctrDataSet.GetBaseTensor().GetHash();

        scoreHelper.SubmitCompute(Subsets.DeviceView(devId),
                                  subsetDocs.DeviceView(devId));

        scoreHelper.ComputeOptimalSplit(partStats.AsConstBuf(),
                                        ctrDataSet.GetCtrWeights(maxUniqueValues, modelSizeReg).AsConstBuf(),
                                        featureWeights,
                                        scoreBeforeSplit,
                                        ScoreStdDev,
                                        taskSeed);

        UpdateBestSplit(ctrDataSet,
                        ctrDataSetInverseIndices,
                        scoreHelper.ReadOptimalSplit());
    }

    TBestSplitProperties TTreeCtrDataSetVisitor::CreateBestSplitProperties() {
        EnsureHasBestProps();

        if (!FeaturesManager.IsKnown(BestCtr)) {
            TVector<float> borders(BestBorders[BestDevice].begin(),
                                   BestBorders[BestDevice].end());
            FeaturesManager.AddCtr(BestCtr,
                                   std::move(borders));
        }

        TBestSplitProperties splitProperties;
        splitProperties.FeatureId = FeaturesManager.GetId(BestCtr);
        splitProperties.BinId = BestBin;
        splitProperties.Score = static_cast<float>(BestScore);
        return splitProperties;
    }

    void TTreeCtrDataSetVisitor::UpdateBestSplit(const TTreeCtrDataSet& dataSet, const TMirrorBuffer<ui32>& inverseIndices,
                                                 const TBestSplitProperties& bestSplitProperties) {
        const ui32 dev = dataSet.GetCompressedDataSet().GetSamplesMapping().GetDeviceId();

        { //we don't need complex logic here. this should be pretty fast
            bool shouldReturn = false;
            with_lock (Lock) {
                if (bestSplitProperties.Gain < BestGain) {
                    BestScore = bestSplitProperties.Score;
                    BestGain = bestSplitProperties.Gain;
                    BestBin = bestSplitProperties.BinId;
                    BestDevice = dev;
                    BestCtr = dataSet.GetCtrs()[bestSplitProperties.FeatureId];
                } else {
                    shouldReturn = true;
                }
            }
            if (shouldReturn) {
                return;
            }
        }

        {
            const ui32 featureId = bestSplitProperties.FeatureId;
            const ui32 binId = bestSplitProperties.BinId;

            const auto& ctr = dataSet.GetCtrs()[featureId];
            const ui32 compressedSize = CompressedSize<ui64>(static_cast<ui32>(inverseIndices.GetObjectsSlice().Size()), 2);
            BestSplits[dev].Reset(NCudaLib::TSingleMapping(dev, compressedSize));
            const auto devInverseIndices = inverseIndices.ConstDeviceView(dev);
            auto& binarizedDataSet = dataSet.GetCompressedDataSet();

            CreateCompressedSplit(binarizedDataSet,
                                  binarizedDataSet.GetTCFeature(featureId),
                                  binId,
                                  BestSplits[dev],
                                  &devInverseIndices);

            if (!FeaturesManager.IsKnown(ctr)) {
                BestBorders[dev] = dataSet.ReadBorders(featureId);
            }
        }
    }

    void TTreeCtrDataSetVisitor::CacheCtrBorders(const TMap<TCtr, TVector<float>>& bordersMap) {
        for (auto& entry : bordersMap) {
            if (!FeaturesManager.IsKnown(entry.first)) {
                TVector<float> borders(entry.second.begin(), entry.second.end());
                with_lock (Lock) {
                    Y_ASSERT(!FeaturesManager.IsKnown(entry.first)); //we can't add one ctr from 2 different threads.
                    FeaturesManager.AddCtr(entry.first, std::move(borders));
                }
            }
        }
    }

    TVector<ui32> TTreeCtrDataSetVisitor::GetCtrsBordersToCacheIds(const TVector<TCtr>& ctrs) {
        TVector<ui32> result;
        for (ui32 i = 0; i < ctrs.size(); ++i) {
            const auto& ctr = ctrs.at(i);
            if (IsNeedToCacheBorders(ctr)) {
                result.push_back(i);
            }
        }
        return result;
    }

    bool TTreeCtrDataSetVisitor::IsNeedToCacheBorders(const TCtr& ctr) {
        return ctr.FeatureTensor.GetSplits().size() == 0 &&
               ctr.FeatureTensor.GetCatFeatures().size() < TreeConfig.MaxCtrComplexityForBordersCaching;
    }

    TSingleBuffer<const ui64> TTreeCtrDataSetVisitor::GetBestSplitBits() const {
        EnsureHasBestProps();
        return BestSplits[BestDevice].ConstCopyView();
    }

    TTreeCtrDataSetVisitor::TTreeCtrDataSetVisitor(TBinarizedFeaturesManager& featuresManager, const ui32 foldCount,
                                                   const NCatboostOptions::TObliviousTreeLearnerOptions& treeConfig,
                                                   const TOptimizationSubsets<NCudaLib::TMirrorMapping>& subsets)
        : FeaturesManager(featuresManager)
        , FoldCount(foldCount)
        , TreeConfig(treeConfig)
        , Subsets(subsets)
        , BestScore(std::numeric_limits<double>::infinity())
        , BestGain(std::numeric_limits<double>::infinity())
        , BestBin(-1)
        , BestDevice(-1)
        , BestBorders(NCudaLib::GetCudaManager().GetDeviceCount())
        , BestSplits(NCudaLib::GetCudaManager().GetDeviceCount())
        , Seeds(NCudaLib::GetCudaManager().GetDeviceCount(), 0)
    {
    }

    TTreeCtrDataSetVisitor& TTreeCtrDataSetVisitor::SetBestGain(double gain) {
        BestGain = gain;
        return *this;
    }

    TTreeCtrDataSetVisitor& TTreeCtrDataSetVisitor::SetScoreStdDevAndSeed(double scoreStdDev, ui64 seed) {
        ScoreStdDev = scoreStdDev;
        TRandom random(seed);
        for (ui32 i = 0; i < Seeds.size(); ++i) {
            Seeds[i] = random.NextUniformL();
        }
        return *this;
    }
}
