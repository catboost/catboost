#pragma once

#include "meta_l2.h"
#include "feature_metadata.h"

#include <algorithm>
#include <cstdint>
#include <functional>

namespace NCB {
    inline TVector<TMetalMetaL2DataSet> MakeMetalMetaL2StaticDataSets(
            const TMetalStaticFeatureMetadata& metadata, ui32 permutations, bool dependentFeatures) {
        // Even an empty independent grid has a compressed dataset and consumes
        // its score seed. The dependent dataset exists only for P>1.
        TVector<TMetalMetaL2DataSet> result(1 + ui32(permutations > 1 && dependentFeatures));
        for (const auto& feature : metadata.RsmFeatures) {
            if (!feature.FoldCount) continue;
            const ui32 dataset = result.size() > 1 && feature.PermutationDependent ? 1 : 0;
            const ui32 policy = feature.FoldCount <= 1 ? 0 : feature.FoldCount <= 15 ? 1 : 2;
            result[dataset].PolicyMask |= 1u << policy;
            for (ui32 runtimeFeature : feature.RuntimeFeatures)
                result[dataset].Features.push_back({runtimeFeature, EMetalMetaL2Policy(policy)});
        }
        return result;
    }

    class TMetalMetaL2Context {
    public:
        using TSeedProvider = std::function<TVector<ui64>(ui32 offset, ui32 count)>;
        // Each dynamic dataset's incoming ScoreSeed is its original base
        // tensor hash; this context adds the visitor's local device seed.
        using TDynamicProvider = std::function<TVector<TMetalMetaL2DataSet>()>;

        TMetalMetaL2Context(double exponent, double frequency, TVector<TMetalMetaL2DataSet> datasets)
            : Exponent(exponent), Frequency(frequency), StaticDataSets(std::move(datasets))
        {
            CB_ENSURE(!StaticDataSets.empty() && StaticDataSets.size() <= 2,
                "Metal meta-L2 requires independent and optional dependent datasets");
        }

        void SetSeedProvider(TSeedProvider provider) { SeedProvider = std::move(provider); }
        void SetDynamicProvider(TDynamicProvider provider) { DynamicProvider = std::move(provider); }
        void BeginTree() { Offset = 0; Error.clear(); }
        ui32 GetDrawCount() const { return Offset; }
        const TString& GetError() const { return Error; }
        ui32 GetStaticDataSetCount() const { return StaticDataSets.size(); }

        static int Callback(void* context, uint32_t featureCount, uint8_t* output) noexcept {
            auto& self = *static_cast<TMetalMetaL2Context*>(context);
            try {
                CB_ENSURE(output && self.SeedProvider, "Metal meta-L2 callback is missing output or score seeds");
                auto datasets = self.StaticDataSets;
                auto dynamic = self.DynamicProvider ? self.DynamicProvider() : TVector<TMetalMetaL2DataSet>();
                const ui32 count = datasets.size() + ui32(!dynamic.empty());
                const auto seeds = self.SeedProvider(self.Offset, count);
                CB_ENSURE(seeds.size() == count, "Metal meta-L2 score seed count differs from active datasets");
                for (ui32 i = 0; i < datasets.size(); ++i) datasets[i].ScoreSeed = seeds[i];
                for (auto& pack : dynamic) {
                    pack.ScoreSeed = MetalMetaL2TreeDataSetSeed(seeds.back(), 0, pack.ScoreSeed);
                    datasets.push_back(std::move(pack));
                }
                const auto choices = BuildMetalMetaL2FeatureChoices(featureCount, self.Exponent, self.Frequency, datasets);
                std::copy(choices.begin(), choices.end(), output);
                self.Offset += count;
                return 0;
            } catch (const std::exception& error) {
                self.Error = error.what();
                return 1;
            } catch (...) {
                self.Error = "Unknown Metal meta-L2 host callback failure";
                return 1;
            }
        }

    private:
        const double Exponent, Frequency;
        const TVector<TMetalMetaL2DataSet> StaticDataSets;
        TSeedProvider SeedProvider;
        TDynamicProvider DynamicProvider;
        ui32 Offset = 0;
        TString Error;
    };
}
