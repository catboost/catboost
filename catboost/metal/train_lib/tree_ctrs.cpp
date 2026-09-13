#include "tree_ctrs.h"
#include "tree_ctr_tensors.h"

#include <catboost/libs/data/objects.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/metal/native/metal_projection.h>
#include <catboost/private/libs/algo/helpers.h>
#include <catboost/private/libs/ctr_description/ctr_config.h>
#include <catboost/private/libs/quantization/utils.h>

#include <util/generic/map.h>
#include <util/generic/set.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace NCB {
namespace {
    constexpr ui64 MaxInputBytes = ui64(1) << 30;
    constexpr ui32 SnapshotVersion = 2;

    struct TVariantState {
        TFeatureCombination Projection;
        ui32 ConfigIndex = 0;
        ui32 History = 0;
        TCtrFeature Feature;
        Y_SAVELOAD_DEFINE(Projection, ConfigIndex, History, Feature);
    };

    // Standard TModelCtr drops the training binarization ID. Keep TCtrConfig
    // in this identity so equal model CTRs with different grids stay distinct.
    using TConfigIdentity = std::pair<TFeatureCombination, TCtrConfig>;

    struct TProjectionInput {
        TVector<ui32> Categories;
        TVector<ui8> Bins;
        TVector<ui8> Types;
        TVector<ui32> Features;
        TVector<ui32> Thresholds;
    };

    struct TProjectionGroups {
        TVector<ui64> Hashes;
        TVector<ui32> Bins;
        TVector<ui32> Rows;
    };

    struct TFinalStatistics {
        TVector<ui32> Counts;
        TVector<TVector<float>> Sums;
    };

    template <class TStats>
    void AddStats(const TStats& source, CBMCtrStats* destination) {
        destination->kernel_dispatches += source.kernel_dispatches;
        destination->gpu_seconds += source.gpu_seconds;
        std::memcpy(destination->device_name, source.device_name, sizeof(source.device_name));
    }

    void AddTable(const TFeatureCombination& projection, ECtrType type,
                  TConstArrayRef<ui64> hashes, const TFinalStatistics& statistics,
                  ui32 rows, TStaticCtrProvider* provider) {
        TCtrValueTable table;
        table.ModelCtrBase.Projection = projection;
        table.ModelCtrBase.CtrType = type;
        auto indexBuilder = table.GetIndexHashBuilder(hashes.size());
        if (type == ECtrType::FeatureFreq) {
            auto values = table.AllocateBlobAndGetArrayRef<int>(hashes.size());
            table.CounterDenominator = rows;
            for (ui32 category = 0; category < hashes.size(); ++category)
                values[indexBuilder.AddIndex(hashes[category])] = statistics.Counts[category];
        } else if (type == ECtrType::FloatTargetMeanValue) {
            auto values = table.AllocateBlobAndGetArrayRef<TCtrMeanHistory>(hashes.size());
            for (ui32 category = 0; category < hashes.size(); ++category) {
                auto& value = values[indexBuilder.AddIndex(hashes[category])];
                value.Sum = statistics.Sums[0][category];
                value.Count = statistics.Counts[category];
            }
        } else {
            const ui32 classes = statistics.Sums.size() + (type == ECtrType::Borders);
            table.TargetClassesCount = classes;
            auto values = table.AllocateBlobAndGetArrayRef<int>(hashes.size() * classes);
            for (ui32 category = 0; category < hashes.size(); ++category) {
                const ui32 index = indexBuilder.AddIndex(hashes[category]);
                for (ui32 targetClass = 0; targetClass < classes; ++targetClass) {
                    float value;
                    if (type == ECtrType::Buckets) {
                        value = statistics.Sums[targetClass][category];
                    } else {
                        const float previous = targetClass ? statistics.Sums[targetClass - 1][category]
                                                           : statistics.Counts[category];
                        const float next = targetClass < statistics.Sums.size()
                            ? statistics.Sums[targetClass][category] : 0.0f;
                        value = previous - next;
                    }
                    CB_ENSURE(value >= 0 && value <= statistics.Counts[category] && std::floor(value) == value,
                              "Metal compound CTR class counts are invalid");
                    values[index * classes + targetClass] = static_cast<int>(value);
                }
            }
        }
        // These are already complete 64-bit model projection hashes. Applying
        // the single-category hash adapter here would corrupt held-out lookup.
        provider->AddCtrCalcerData(std::move(table));
    }
}

class TMetalTreeCtrFeatures::TImpl {
public:
    TImpl(const TTrainingDataProvider& data,
          const NCatboostOptions::TCatBoostOptions& options,
          NPar::ILocalExecutor* executor, ui32 firstFeature,
          ui32 learnAndFirstEvalRows, TIntrusivePtr<TStaticCtrProvider> provider,
          TConstArrayRef<TVector<ui32>> historyOrders)
        : Objects(*data.ObjectsData)
        , Executor(executor)
        , Rows(data.GetObjectCount())
        , FirstFeature(firstFeature)
        , MaximumUniqueValues(Max(Rows, learnAndFirstEvalRows))
        , RandomSeed(static_cast<ui32>(options.RandomSeed.Get()))
        , BorderCacheComplexity(options.ObliviousTreeOptions->MaxCtrComplexityForBordersCaching.Get())
        , Provider(provider ? std::move(provider) : MakeIntrusive<TStaticCtrProvider>())
        , FloatFeatures(CreateFloatFeatures(*Objects.GetFeaturesLayout(), *Objects.GetQuantizedFeaturesInfo()))
        , CatFeatures(CreateCatFeatures(*Objects.GetFeaturesLayout()))
    {
        CB_ENSURE(Rows && Rows <= (1u << 24), "Metal tree CTR row count is invalid");
        const auto& categorical = options.CatFeatureParams.Get();
        CB_ENSURE(categorical.MaxTensorComplexity > 1, "Metal tree CTRs require max_ctr_complexity > 1");
        CB_ENSURE(categorical.CtrHistoryUnit == ECtrHistoryUnit::Sample,
                  "Metal tree CTRs require sample-level histories");
        CB_ENSURE(categorical.CounterCalcMethod == ECounterCalc::SkipTest,
                  "Metal tree CTRs currently require counter_calc_method='SkipTest'");
        const auto target = data.TargetData->GetTarget();
        CB_ENSURE(target && target->size() == 1 && (*target)[0].size() == Rows,
                  "Metal tree CTRs require one scalar target per row");
        Targets.assign((*target)[0].begin(), (*target)[0].end());
        const auto borders = BuildBorders(Targets, RandomSeed, categorical.TargetBinarization.Get());
        CB_ENSURE(borders.size() <= 255, "Metal tree CTR targets require at most 255 borders");
        TargetBorderCount = borders.size();
        const auto targetBins = BinarizeLine<ui8>(Targets, ENanMode::Forbidden, borders);
        BinTargets.assign(targetBins.begin(), targetBins.end());
        MakeConfigs(categorical.CombinationCtrs.Get());

        TVector<int> eligible;
        const auto& layout = *Objects.GetFeaturesLayout();
        for (const auto& feature : CatFeatures) {
            if (!layout.GetExternalFeatureMetaInfo(feature.Position.FlatIndex).IsAvailable) continue;
            const ui32 count = Objects.GetQuantizedFeaturesInfo()->GetUniqueValuesCounts(
                TCatFeatureIdx(feature.Position.Index)).OnAll;
            if (!(count > 1 && count <= categorical.OneHotMaxSize)) eligible.push_back(feature.Position.Index);
        }
        Scheduler = MakeHolder<TMetalTreeCtrTensorScheduler>(eligible, categorical.MaxTensorComplexity.Get());
        // Preflight the caller's views before making potentially large copies.
        const ui64 bytesPerHistory = ui64(Rows) * sizeof(ui32) + sizeof(TVector<ui32>);
        CB_ENSURE(historyOrders.size() <= MaxInputBytes / bytesPerHistory,
                  "Metal tree CTR history orders exceed the 1 GiB limit");
        for (const auto& order : historyOrders)
            CB_ENSURE(order.size() == Rows, "Metal tree CTR history row count differs");
        HistoryOrders.assign(historyOrders.begin(), historyOrders.end());
        if (HistoryOrders.empty()) {
            HistoryOrders.emplace_back(Rows);
            std::iota(HistoryOrders[0].begin(), HistoryOrders[0].end(), 0);
        }
        for (const auto& order : HistoryOrders) {
            CB_ENSURE(order.size() == Rows, "Metal tree CTR history row count differs");
            TVector<bool> seen(Rows, false);
            for (ui32 row : order) {
                CB_ENSURE(row < Rows && !seen[row], "Metal tree CTR history must be a source-row permutation");
                seen[row] = true;
            }
            HistoryHashes.push_back(VecCityHash(order));
        }
    }

    void MakeConfigs(const TVector<NCatboostOptions::TCtrDescription>& descriptions) {
        TSet<TCtrConfig> unique;
        for (const auto& description : descriptions) {
            const ECtrType type = description.Type;
            CB_ENSURE(type == ECtrType::Borders || type == ECtrType::Buckets ||
                      type == ECtrType::FloatTargetMeanValue || type == ECtrType::FeatureFreq,
                      "Metal tree CTRs support Borders, Buckets, FloatTargetMeanValue, and FeatureFreq");
            CB_ENSURE(description.PriorEstimation == EPriorEstimation::No,
                      "Metal tree CTR prior estimation is not yet supported");
            const auto& binarization = description.GetCtrBinarization();
            CB_ENSURE(binarization.BorderCount <= 255, "Metal tree CTRs require at most 255 feature borders");
            auto found = std::find(Binarizations.begin(), Binarizations.end(), binarization);
            const ui32 binarizationId = found - Binarizations.begin();
            if (found == Binarizations.end()) Binarizations.push_back(binarization);
            const ui32 count = type == ECtrType::Borders ? TargetBorderCount
                : type == ECtrType::Buckets ? TargetBorderCount + 1 : 1;
            CB_ENSURE(!description.GetPriors().empty(), "Metal tree CTR configurations require a prior");
            for (const auto& prior : description.GetPriors()) {
                CB_ENSURE(prior.size() >= 1 && prior.size() <= 2 && std::isfinite(prior[0]) &&
                          (prior.size() == 1 || (std::isfinite(prior[1]) && prior[1] > 0)),
                          "Metal tree CTR priors require a finite numerator and positive denominator");
                for (ui32 param = 0; param < count; ++param) {
                    if (type == ECtrType::Buckets && count == 2 && param == 0) continue;
                    TCtrConfig config;
                    config.Type = type;
                    config.ParamId = param;
                    config.Prior = {prior[0], prior.size() == 2 ? prior[1] : 1.0f};
                    config.CtrBinarizationConfigId = binarizationId;
                    unique.insert(std::move(config));
                }
            }
        }
        // CUDA CreateTreeCtrConfigs orders by type and the shared TCtrConfig
        // comparator, and deduplicates identical descriptions before scoring.
        Configs.assign(unique.begin(), unique.end());
    }

    const TMetalCategoryColumn& Category(int index) {
        auto found = Categories.find(index);
        if (found != Categories.end()) return found->second;
        auto feature = std::find_if(CatFeatures.begin(), CatFeatures.end(),
                                   [&](const auto& f) { return f.Position.Index == index; });
        CB_ENSURE(feature != CatFeatures.end(), "Metal tree CTR categorical index is invalid");
        return Categories.emplace(index, ReadMetalCategoryColumn(Objects, *feature, Executor)).first->second;
    }

    TProjectionInput ProjectionInput(const TFeatureCombination& projection) {
        TProjectionInput result;
        const ui64 inputBytes = ui64(projection.CatFeatures.size()) * Rows * sizeof(ui32) +
            ui64(projection.BinFeatures.size() + projection.OneHotFeatures.size()) * Rows;
        CB_ENSURE(inputBytes <= MaxInputBytes, "Metal tree CTR projection input exceeds the 1 GiB limit");
        for (int index : projection.CatFeatures) {
            const auto& column = Category(index);
            result.Types.push_back(0);
            result.Features.push_back(result.Categories.size() / Rows);
            result.Thresholds.push_back(0);
            for (ui32 row = 0; row < Rows; ++row) result.Categories.push_back(column.Hashes[column.Bins[row]]);
        }
        for (const auto& split : projection.BinFeatures) {
            auto feature = std::find_if(FloatFeatures.begin(), FloatFeatures.end(),
                                       [&](const auto& f) { return f.Position.Index == split.FloatFeature; });
            CB_ENSURE(feature != FloatFeatures.end(), "Metal tree CTR numeric index is invalid");
            const auto border = std::lower_bound(feature->Borders.begin(), feature->Borders.end(), split.Split);
            CB_ENSURE(border != feature->Borders.end() && *border == split.Split,
                      "Metal tree CTR predicate is absent from Pool numeric borders");
            const auto holder = Objects.GetFloatFeature(split.FloatFeature);
            CB_ENSURE(holder, "Metal tree CTR numeric values are unavailable");
            const auto values = (*holder)->ExtractValues<ui16>(Executor);
            CB_ENSURE(values.size() == Rows && feature->Borders.size() <= 255,
                      "Metal tree CTR numeric bins are invalid");
            result.Types.push_back(1);
            result.Features.push_back(result.Bins.size() / Rows);
            result.Thresholds.push_back(border - feature->Borders.begin());
            for (ui16 value : values) {
                CB_ENSURE(value <= feature->Borders.size(), "Metal tree CTR numeric bin is out of range");
                result.Bins.push_back(static_cast<ui8>(value));
            }
        }
        for (const auto& split : projection.OneHotFeatures) {
            const auto& column = Category(split.CatFeatureIdx);
            const auto value = std::lower_bound(column.Hashes.begin(), column.Hashes.end(), static_cast<ui32>(split.Value));
            CB_ENSURE(column.Hashes.size() <= 255 && value != column.Hashes.end() &&
                      *value == static_cast<ui32>(split.Value), "Metal tree CTR one-hot predicate is unavailable");
            result.Types.push_back(2);
            result.Features.push_back(result.Bins.size() / Rows);
            result.Thresholds.push_back(value - column.Hashes.begin());
            result.Bins.insert(result.Bins.end(), column.Bins.begin(), column.Bins.end());
        }
        return result;
    }

    TProjectionGroups Group(const TProjectionInput& input, ui32 permutation, CBMCtrStats* stats) {
        CBMProjectionParams params{Rows, static_cast<ui32>(input.Categories.size() / Rows),
                                  static_cast<ui32>(input.Bins.size() / Rows), static_cast<ui32>(input.Types.size())};
        TVector<uint64_t> rowHashes(Rows), sortedHashes(Rows);
        TProjectionGroups result;
        result.Rows.resize(Rows);
        result.Bins.resize(Rows);
        CBMSortStats currentStats = {};
        char error[2048] = {};
        CB_ENSURE(cbm_projection_group(&params, input.Categories.data(), input.Bins.data(),
            input.Types.data(), input.Features.data(), input.Thresholds.data(), HistoryOrders[permutation].data(),
            rowHashes.data(), sortedHashes.data(), result.Rows.data(), &currentStats, error, sizeof(error)) == 0,
            "Metal tree CTR grouping failed: " << error);
        AddStats(currentStats, stats);
        for (ui32 row = 0; row < Rows; ++row) {
            if (!row || sortedHashes[row] != sortedHashes[row - 1]) result.Hashes.push_back(sortedHashes[row]);
            result.Bins[row] = result.Hashes.size() - 1;
        }
        return result;
    }

    ui32 UniqueUpperBound(const TFeatureCombination& projection) {
        ui64 count = 1;
        auto multiply = [&](ui64 factor) { count = Min<ui64>(MaximumUniqueValues, count * factor); };
        for (int index : projection.CatFeatures) multiply(Category(index).UniqueValuesOnAll);
        for (size_t i = 0; i < projection.BinFeatures.size() + projection.OneHotFeatures.size(); ++i) multiply(2);
        return static_cast<ui32>(count);
    }

    bool EagerlyRegisters(const TFeatureCombination& projection) const {
        return projection.BinFeatures.empty() && projection.OneHotFeatures.empty() &&
            projection.CatFeatures.size() < BorderCacheComplexity;
    }

    TConfigIdentity Identity(const TVariantState& state) const {
        CB_ENSURE(state.ConfigIndex < Configs.size(), "Metal tree CTR variant config is out of range");
        return {state.Projection, Configs[state.ConfigIndex]};
    }

    void MarkSelected(ui32 absoluteFeature) {
        if (absoluteFeature < FirstFeature) return;
        const ui32 local = absoluteFeature - FirstFeature;
        CB_ENSURE(local < Registry.size(), "Metal selected an invalid tree CTR feature");
        const auto [found, inserted] = KnownFeatures.emplace(Identity(Registry[local]), absoluteFeature);
        CB_ENSURE(inserted || found->second == absoluteFeature,
                  "Metal selected an expired variant of an already registered tree CTR");
    }

    TVector<ui32> GetRegisteredFeatures() const {
        TVector<ui32> result;
        result.reserve(KnownFeatures.size());
        for (const auto& entry : KnownFeatures) result.push_back(entry.second);
        std::sort(result.begin(), result.end());
        return result;
    }

    // Append missing config variants in config order, preserving the original
    // absolute ID of every existing column. Histories for scoring use one grid
    // selected here; complete inference tables remain independent of history.
    void Generate(const TFeatureCombination& projection, TConstArrayRef<ui32> configIndices,
                  ui32 borderPermutation, TMetalTreeCtrBatch* batch,
                  TConstArrayRef<TVariantState> saved = {}) {
        CB_ENSURE(borderPermutation < HistoryOrders.size(), "Metal tree CTR border permutation is out of range");
        CB_ENSURE(saved.empty() || saved.size() == configIndices.size(), "Saved Metal tree CTR variant count differs");
        CB_ENSURE((ui64(FirstFeature) + Splits.size() + configIndices.size()) * Rows * HistoryOrders.size() <= MaxInputBytes,
                  "Metal dynamic feature banks exceed the experimental 1 GiB limit");
        TVector<int> requested(Configs.size(), -1);
        for (ui32 index = 0; index < configIndices.size(); ++index) {
            const ui32 config = configIndices[index];
            CB_ENSURE(config < Configs.size() && requested[config] < 0 &&
                      (!index || configIndices[index - 1] < config), "Metal tree CTR config order is invalid");
            const TConfigIdentity identity{projection, Configs[config]};
            const auto found = Variants.find(identity);
            CB_ENSURE(found == Variants.end() || !found->second.contains(borderPermutation),
                      "Metal tree CTR variant was registered twice");
            requested[config] = index;
        }
        const auto input = ProjectionInput(projection);
        const ui32 localStart = Splits.size();
        TVector<TVariantState> states(configIndices.size());
        const ui32 uniqueValues = UniqueUpperBound(projection);
        const bool buildTables = !TablesBuilt.contains(projection);
        TMap<ECtrType, TFinalStatistics> finalStatistics;
        TVector<ui64> tableHashes;
        TVector<ui32> permutationOrder{borderPermutation};
        for (ui32 p = 0; p < HistoryOrders.size(); ++p) if (p != borderPermutation) permutationOrder.push_back(p);
        for (ui32 permutation : permutationOrder) {
            const auto groups = Group(input, permutation, &batch->Stats);
            if (permutation == borderPermutation) tableHashes = groups.Hashes;
            else CB_ENSURE(groups.Hashes == tableHashes, "Metal tree CTR projection keys differ across histories");
            for (ui32 configIndex = 0; configIndex < Configs.size(); ++configIndex) {
                const int request = requested[configIndex];
                if (request < 0 && (permutation || !buildTables)) continue;
                const auto& config = Configs[configIndex];
                const ui32 nativeType = config.Type == ECtrType::Borders ? 0 : config.Type == ECtrType::Buckets ? 1
                    : config.Type == ECtrType::FloatTargetMeanValue ? 2 : 3;
                CBMCtrParams params{Rows, static_cast<ui32>(groups.Hashes.size()), nativeType, config.ParamId,
                                    config.Prior[0], config.Prior[1]};
                TVector<float> values(Rows), sums(groups.Hashes.size());
                TVector<ui32> counts(groups.Hashes.size());
                CBMCtrStats stats = {};
                char error[2048] = {};
                const float* targets = config.Type == ECtrType::FloatTargetMeanValue ? Targets.data() : BinTargets.data();
                CB_ENSURE(cbm_compute_ctrs(&params, groups.Bins.data(), groups.Rows.data(), targets,
                    values.data(), sums.data(), counts.data(), &stats, error, sizeof(error)) == 0,
                    "Metal tree CTR computation failed: " << error);
                AddStats(stats, &batch->Stats);
                if (!permutation && buildTables) {
                    auto& final = finalStatistics[config.Type];
                    const ui32 count = config.Type == ECtrType::Borders ? TargetBorderCount
                        : config.Type == ECtrType::Buckets ? TargetBorderCount + 1 : 1;
                    CB_ENSURE(ui64(count + 1) * groups.Hashes.size() * sizeof(float) <= MaxInputBytes,
                              "Metal compound CTR statistics exceed the 1 GiB limit");
                    final.Sums.resize(count);
                    final.Sums[config.ParamId] = std::move(sums);
                    final.Counts = std::move(counts);
                }
                if (request < 0) continue;
                auto& state = states[request];
                if (permutation == borderPermutation) {
                    state.Projection = projection;
                    state.ConfigIndex = configIndex;
                    state.History = borderPermutation;
                    auto& feature = state.Feature;
                    feature.Ctr.Base.Projection = projection;
                    feature.Ctr.Base.CtrType = config.Type;
                    feature.Ctr.TargetBorderIdx = config.ParamId;
                    feature.Ctr.PriorNum = config.Prior[0];
                    feature.Ctr.PriorDenom = config.Prior[1];
                    if (!saved.empty()) {
                        const auto& previous = saved[request];
                        CB_ENSURE(previous.Projection == projection && previous.ConfigIndex == configIndex &&
                                  previous.History == borderPermutation && previous.Feature.Ctr == feature.Ctr,
                                  "Saved Metal tree CTR descriptors differ from current options");
                        feature.Borders = previous.Feature.Borders;
                        CB_ENSURE(feature.Borders.size() <= 255 &&
                                  std::is_sorted(feature.Borders.begin(), feature.Borders.end()) &&
                                  std::adjacent_find(feature.Borders.begin(), feature.Borders.end()) == feature.Borders.end(),
                                  "Saved Metal tree CTR borders are invalid");
                        for (float border : feature.Borders)
                            CB_ENSURE(std::isfinite(border), "Saved Metal tree CTR border is nonfinite");
                    } else {
                        feature.Borders = BuildBorders(values, RandomSeed, Binarizations[config.CtrBinarizationConfigId]);
                    }
                    const ui32 absoluteFeature = FirstFeature + Splits.size();
                    const ui32 batchFeature = absoluteFeature - batch->FirstFeature;
                    TVector<TModelSplit> splits;
                    for (ui32 border = 0; border < feature.Borders.size(); ++border) {
                        splits.emplace_back(TModelCtrSplit{feature.Ctr, feature.Borders[border]});
                        batch->CandidateFeatures.push_back(batchFeature);
                        batch->CandidateBins.push_back(border);
                        batch->CandidateTypes.push_back(0);
                    }
                    batch->CtrUniqueValues.push_back(uniqueValues);
                    batch->RegisteredCtrFlags.push_back(EagerlyRegisters(projection));
                    batch->BinsPerFeature = Max<ui32>(batch->BinsPerFeature, feature.Borders.size() + 1);
                    Splits.push_back(std::move(splits));
                }
                const auto bins = BinarizeLine<ui8>(values, ENanMode::Forbidden, state.Feature.Borders);
                batch->PermutationBins[permutation].insert(batch->PermutationBins[permutation].end(), bins.begin(), bins.end());
            }
        }
        for (auto& [type, statistics] : finalStatistics) {
            if (type == ECtrType::Buckets && TargetBorderCount == 1) {
                statistics.Sums[0].resize(statistics.Counts.size());
                for (ui32 category = 0; category < statistics.Counts.size(); ++category)
                    statistics.Sums[0][category] = statistics.Counts[category] - statistics.Sums[1][category];
            }
            AddTable(projection, type, tableHashes, statistics, Rows, Provider.Get());
        }
        TablesBuilt.insert(projection);
        CB_ENSURE(Splits.size() - localStart == states.size(), "Metal tree CTR feature registration differs");
        for (auto& state : states) {
            const ui32 feature = FirstFeature + Registry.size();
            const auto identity = Identity(state);
            Variants[identity].emplace(borderPermutation, feature);
            Registry.push_back(std::move(state));
            if (EagerlyRegisters(projection)) MarkSelected(feature);
        }
    }

    TVector<ui32> ResolveTensor(const TFeatureCombination& projection, ui32 history,
                              TMetalTreeCtrBatch* batch) {
        TVector<ui32> result(Configs.size());
        TVector<ui32> missing;
        const ui32 firstNew = FirstFeature + Splits.size();
        for (ui32 config = 0; config < Configs.size(); ++config) {
            const TConfigIdentity identity{projection, Configs[config]};
            const auto known = KnownFeatures.find(identity);
            if (known != KnownFeatures.end()) {
                result[config] = known->second;
                continue;
            }
            const auto variants = Variants.find(identity);
            if (variants != Variants.end()) {
                const auto existing = variants->second.find(history);
                if (existing != variants->second.end()) {
                    result[config] = existing->second;
                    continue;
                }
            }
            result[config] = firstNew + missing.size();
            missing.push_back(config);
        }
        if (!missing.empty()) Generate(projection, missing, history, batch);
        return result;
    }

    void BeginTree() {
        Scheduler->BeginTree();
        TreeFeatures.clear();
    }

    TMetalTreeCtrBatch NewBatch() const {
        TMetalTreeCtrBatch result;
        result.FirstFeature = FirstFeature + Splits.size();
        result.PermutationBins.resize(HistoryOrders.size());
        return result;
    }

    TMetalTreeCtrBatch AddSplit(const TModelSplit& split, ui32 borderPermutation) {
        CB_ENSURE(borderPermutation < HistoryOrders.size(), "Metal tree CTR border permutation is out of range");
        Scheduler->AddSplit(split);
        auto batch = NewBatch();
        for (const auto& projection : Scheduler->GetActiveTensors()) {
            if (!TreeFeatures.contains(projection))
                TreeFeatures.emplace(projection, ResolveTensor(projection, borderPermutation, &batch));
            const auto& features = TreeFeatures.at(projection);
            batch.ActiveFeatures.insert(batch.ActiveFeatures.end(), features.begin(), features.end());
        }
        std::sort(batch.ActiveFeatures.begin(), batch.ActiveFeatures.end());
        return batch;
    }

    TVector<TVector<ui32>> BinarizationState() const {
        TVector<TVector<ui32>> result;
        for (const auto& options : Binarizations)
            result.push_back({static_cast<ui32>(options.BorderSelectionType.Get()), options.BorderCount.Get(),
                              static_cast<ui32>(options.NanMode.Get()), options.MaxSubsetSizeForBuildBorders.Get()});
        return result;
    }

    const TQuantizedObjectsDataProvider& Objects;
    NPar::ILocalExecutor* Executor;
    ui32 Rows;
    ui32 FirstFeature;
    ui32 MaximumUniqueValues;
    ui32 RandomSeed;
    ui32 BorderCacheComplexity;
    ui32 TargetBorderCount;
    TIntrusivePtr<TStaticCtrProvider> Provider;
    TVector<TFloatFeature> FloatFeatures;
    TVector<TCatFeature> CatFeatures;
    TMap<int, TMetalCategoryColumn> Categories;
    TVector<float> Targets;
    TVector<float> BinTargets;
    TVector<TVector<ui32>> HistoryOrders;
    TVector<ui64> HistoryHashes;
    TVector<TCtrConfig> Configs;
    TVector<NCatboostOptions::TBinarizationOptions> Binarizations;
    THolder<TMetalTreeCtrTensorScheduler> Scheduler;
    TMap<TFeatureCombination, TVector<ui32>> TreeFeatures;
    TMap<TConfigIdentity, TMap<ui32, ui32>> Variants;
    TMap<TConfigIdentity, ui32> KnownFeatures;
    TSet<TFeatureCombination> TablesBuilt;
    TVector<TVector<TModelSplit>> Splits;
    TVector<TVariantState> Registry;
};

TMetalTreeCtrFeatures::TMetalTreeCtrFeatures(const TTrainingDataProvider& data,
        const NCatboostOptions::TCatBoostOptions& options, NPar::ILocalExecutor* executor,
        ui32 firstFeature, ui32 learnAndFirstEvalRows, TIntrusivePtr<TStaticCtrProvider> provider,
        TConstArrayRef<TVector<ui32>> historyOrders)
    : Impl(MakeHolder<TImpl>(data, options, executor, firstFeature, learnAndFirstEvalRows,
                            std::move(provider), historyOrders))
{}

TMetalTreeCtrFeatures::~TMetalTreeCtrFeatures() = default;
void TMetalTreeCtrFeatures::BeginTree() { Impl->BeginTree(); }
void TMetalTreeCtrFeatures::MarkSelected(ui32 absoluteFeature) { Impl->MarkSelected(absoluteFeature); }
TVector<ui32> TMetalTreeCtrFeatures::GetRegisteredFeatures() const { return Impl->GetRegisteredFeatures(); }
TMetalTreeCtrBatch TMetalTreeCtrFeatures::AddSplit(const TModelSplit& split, ui32 borderPermutation) {
    return Impl->AddSplit(split, borderPermutation);
}
ui32 TMetalTreeCtrFeatures::GetFeatureCount() const { return Impl->Splits.size(); }
ui32 TMetalTreeCtrFeatures::GetPermutationCount() const { return Impl->HistoryOrders.size(); }
TIntrusivePtr<TStaticCtrProvider> TMetalTreeCtrFeatures::GetCtrProvider() const { return Impl->Provider; }

const TModelSplit& TMetalTreeCtrFeatures::GetSplit(ui32 feature, ui32 bin) const {
    CB_ENSURE(feature >= Impl->FirstFeature && feature - Impl->FirstFeature < Impl->Splits.size(),
              "Metal returned an invalid tree CTR feature");
    const auto& splits = Impl->Splits[feature - Impl->FirstFeature];
    CB_ENSURE(bin < splits.size(), "Metal returned an invalid tree CTR border");
    return splits[bin];
}

void TMetalTreeCtrFeatures::Save(IOutputStream* output) const {
    ::SaveMany(output, SnapshotVersion, Impl->FirstFeature, Impl->Rows,
               static_cast<ui32>(Impl->HistoryOrders.size()), Impl->HistoryHashes,
               Impl->Configs, Impl->BinarizationState(), Impl->BorderCacheComplexity,
               Impl->Registry, Impl->GetRegisteredFeatures());
}

TMetalTreeCtrBatch TMetalTreeCtrFeatures::Restore(IInputStream* input) {
    CB_ENSURE(Impl->Registry.empty() && Impl->Splits.empty(), "Metal tree CTR restore requires a fresh helper");
    ui32 version, firstFeature, rows, permutations, borderCacheComplexity;
    TVector<TCtrConfig> configs;
    TVector<ui64> historyHashes;
    TVector<TVector<ui32>> binarizations;
    TVector<TVariantState> registry;
    TVector<ui32> knownFeatures;
    ::LoadMany(input, version, firstFeature, rows, permutations, historyHashes, configs, binarizations,
               borderCacheComplexity, registry, knownFeatures);
    CB_ENSURE(version == SnapshotVersion && firstFeature == Impl->FirstFeature && rows == Impl->Rows &&
              permutations == Impl->HistoryOrders.size() && historyHashes == Impl->HistoryHashes &&
              configs == Impl->Configs && binarizations == Impl->BinarizationState() &&
              borderCacheComplexity == Impl->BorderCacheComplexity,
              "Saved Metal tree CTR options or feature layout differ");
    TMap<TConfigIdentity, ui32> known;
    for (ui32 feature : knownFeatures) {
        CB_ENSURE(feature >= firstFeature && feature - firstFeature < registry.size(),
                  "Saved Metal tree CTR registered feature is invalid");
        CB_ENSURE(known.emplace(Impl->Identity(registry[feature - firstFeature]), feature).second,
                  "Saved Metal tree CTR registers multiple variants of one config");
    }
    auto batch = Impl->NewBatch();
    for (ui32 start = 0; start < registry.size();) {
        const auto& first = registry[start];
        ui32 end = start + 1;
        while (end < registry.size() && registry[end].Projection == first.Projection &&
               registry[end].History == first.History && registry[end - 1].ConfigIndex < registry[end].ConfigIndex)
            ++end;
        TVector<ui32> indices;
        for (ui32 i = start; i < end; ++i) indices.push_back(registry[i].ConfigIndex);
        Impl->Generate(first.Projection, indices, first.History, &batch,
                       TConstArrayRef<TVariantState>(registry).Slice(start, end - start));
        start = end;
    }
    Impl->KnownFeatures = std::move(known);
    for (ui32 index = 0; index < Impl->Registry.size(); ++index) {
        const auto& state = Impl->Registry[index];
        const auto found = Impl->KnownFeatures.find(Impl->Identity(state));
        batch.RegisteredCtrFlags[index] = found != Impl->KnownFeatures.end() && found->second == firstFeature + index;
        CB_ENSURE(!Impl->EagerlyRegisters(state.Projection) || found != Impl->KnownFeatures.end(),
                  "Saved Metal tree CTR eager registration is missing");
    }
    Impl->BeginTree();
    return batch;
}
}
