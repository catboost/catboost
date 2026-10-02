// Native counterpart of CUDA's category perfect-hash remapping, single-feature
// CTR generation, and final CTR model conversion. All per-row histories and
// category reductions run in Metal; shared CatBoost code builds borders/models.
#include "categorical.h"
#include "tree_ctr_permutations.h"

#include <catboost/cuda/ctrs/prior_estimator.h>
#include <catboost/libs/data/objects.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/helpers/checksum.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/hash.h>
#include <catboost/metal/native/metal_sort.h>
#include <catboost/private/libs/algo/helpers.h>
#include <catboost/private/libs/quantization/utils.h>

#include <util/generic/algorithm.h>
#include <util/random/shuffle.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace NCB {
namespace {
    constexpr ui64 MaxInputBytes = ui64(1) << 30;

    void AppendFeature(TConstArrayRef<ui8> bins, TVector<TModelSplit>&& splits,
                       ui8 type, ui32 binCount, TMetalCategoricalData* result,
                       ui32 ctrUniqueValues = 0, const TModelCtr* modelCtr = nullptr,
                       const NCatboostOptions::TBinarizationOptions* ctrBinarization = nullptr,
                       ui32 uniqueValuesOnAll = 0, ui32 catFeatureIndex = ui32(-1)) {
        CB_ENSURE(ui64(result->Bins.size()) + bins.size() <= MaxInputBytes,
                  "Metal categorical input exceeds the experimental 1 GiB limit");
        const ui32 feature = result->SplitCandidates.size();
        result->Bins.insert(result->Bins.end(), bins.begin(), bins.end());
        for (ui32 bin = 0; bin < splits.size(); ++bin) {
            result->CandidateFeatures.push_back(feature);
            result->CandidateBins.push_back(bin);
            result->CandidateTypes.push_back(type);
        }
        result->SplitCandidates.push_back(std::move(splits));
        result->ColumnModelCtrs.emplace_back();
        if (modelCtr) result->ColumnModelCtrs.back() = *modelCtr;
        result->ColumnCtrBinarizations.emplace_back();
        if (ctrBinarization) result->ColumnCtrBinarizations.back() = *ctrBinarization;
        result->ColumnUniqueValuesOnAll.push_back(uniqueValuesOnAll);
        result->ColumnCatFeatureIndices.push_back(catFeatureIndex);
        result->CtrUniqueValues.push_back(ctrUniqueValues);
        result->BinsPerFeature = Max(result->BinsPerFeature, binCount);
    }

    using TCategoryColumn = TMetalCategoryColumn;

    TCategoryColumn ReadCategoryColumn(const TQuantizedObjectsDataProvider& objects,
                                       const TCatFeature& feature, ui32 rows,
                                       NPar::ILocalExecutor* executor) {
        const auto holder = objects.GetCatFeature(feature.Position.Index);
        CB_ENSURE(holder, "An available categorical feature has no quantized values");
        const auto perfectBins = (*holder)->ExtractValues<ui32>(executor);
        CB_ENSURE(perfectBins.size() == rows, "Categorical feature row count mismatch");
        const auto& perfectHash = objects.GetQuantizedFeaturesInfo()->GetCategoricalFeaturesPerfectHash(
            TCatFeatureIdx(feature.Position.Index));
        TVector<ui32> hashesByPerfectBin(perfectHash.GetSize());
        TVector<bool> present(perfectHash.GetSize(), false);
        auto save = [&](ui32 hash, ui32 bin) {
            CB_ENSURE(bin < hashesByPerfectBin.size(), "Categorical perfect-hash bin is out of range");
            hashesByPerfectBin[bin] = hash;
            present[bin] = true;
        };
        for (const auto& [hash, value] : perfectHash.Map) save(hash, value.Value);
        if (perfectHash.DefaultMap) save(perfectHash.DefaultMap->SrcValue,
                                         perfectHash.DefaultMap->DstValueWithCount.Value);
        TCategoryColumn result;
        result.UniqueValuesOnAll = objects.GetQuantizedFeaturesInfo()->GetUniqueValuesCounts(
            TCatFeatureIdx(feature.Position.Index)).OnAll;
        TVector<bool> used(perfectHash.GetSize(), false);
        for (ui32 row = 0; row < rows; ++row) {
            CB_ENSURE(perfectBins[row] < present.size() && present[perfectBins[row]],
                      "Categorical perfect-hash dictionary is missing a training value");
            used[perfectBins[row]] = true;
        }
        // The shared perfect-hash dictionary is already ordered by original
        // hash. Reuse that order instead of sorting every training row again.
        TVector<ui32> usedPerfectBins;
        for (const auto& [hash, value] : perfectHash.Map) {
            Y_UNUSED(hash);
            if (used[value.Value]) usedPerfectBins.push_back(value.Value);
        }
        if (perfectHash.DefaultMap && used[perfectHash.DefaultMap->DstValueWithCount.Value]) {
            const ui32 bin = perfectHash.DefaultMap->DstValueWithCount.Value;
            const auto position = std::lower_bound(usedPerfectBins.begin(), usedPerfectBins.end(), bin,
                [&](ui32 left, ui32 right) { return hashesByPerfectBin[left] < hashesByPerfectBin[right]; });
            usedPerfectBins.insert(position, bin);
        }
        TVector<ui32> denseBins(perfectHash.GetSize());
        for (ui32 bin : usedPerfectBins) {
            denseBins[bin] = result.Hashes.size();
            result.Hashes.push_back(hashesByPerfectBin[bin]);
        }
        result.Bins.resize(rows);
        for (ui32 row = 0; row < rows; ++row) {
            result.Bins[row] = denseBins[perfectBins[row]];
        }
        return result;
    }

    TVector<ui32> MakeHistoryOrder(const TTrainingDataProvider& data, ui32 permutation) {
        const ui32 rows = data.GetObjectCount();
        TVector<ui32> result(rows);
        std::iota(result.begin(), result.end(), 0);
        if (!permutation) return result;
        // CUDA doc_parallel_dataset_builder uses GetPermutation(data, index)
        // with the default block size of one. This index-based seed has uint32
        // arithmetic; user random_seed belongs to shared Pool preprocessing.
        const ui32 seed = 1664525u * permutation + 1013904223u + 1u;
        TRandom random(seed);
        random.Advance(10);
        if (!data.MetaInfo.HasGroupId || data.ObjectsGrouping->IsTrivial()) {
            ::Shuffle(result.begin(), result.end(), random);
        } else {
            const auto groups = data.ObjectsGrouping->GetNonTrivialGroups();
            TVector<ui32> order(groups.size());
            std::iota(order.begin(), order.end(), 0);
            ::Shuffle(order.begin(), order.end(), random);
            ui32 cursor = 0;
            for (ui32 group : order)
                for (ui32 row = groups[group].Begin; row < groups[group].End; ++row)
                    result[cursor++] = row;
        }
        return result;
    }

    ui64 ModelCategoryHash(ui32 hash) {
        return CalcHash(0, static_cast<ui64>(static_cast<i32>(hash)));
    }

    TCategoryColumn JoinCategoryColumns(const TCategoryColumn& learn, const TCategoryColumn& evaluation) {
        CB_ENSURE(ui64(learn.Bins.size()) + evaluation.Bins.size() <= (1u << 24),
                  "Metal Full CTR learn+eval rows exceed the 2^24 row limit");
        TCategoryColumn result;
        result.Hashes = learn.Hashes;
        result.Hashes.insert(result.Hashes.end(), evaluation.Hashes.begin(), evaluation.Hashes.end());
        SortUnique(result.Hashes);
        result.UniqueValuesOnAll = Max(learn.UniqueValuesOnAll, evaluation.UniqueValuesOnAll);
        result.Bins.reserve(learn.Bins.size() + evaluation.Bins.size());
        for (const auto* column : {&learn, &evaluation}) {
            for (ui32 bin : column->Bins) {
                const auto found = std::lower_bound(result.Hashes.begin(), result.Hashes.end(), column->Hashes[bin]);
                result.Bins.push_back(found - result.Hashes.begin());
            }
        }
        return result;
    }

    void AddFinalTable(const TModelCtrBase& base, const TCategoryColumn& column,
                       TConstArrayRef<ui32> counts, const TVector<TVector<float>>& sums,
                       TIntrusivePtr<TStaticCtrProvider>* provider) {
        TCtrValueTable table;
        table.ModelCtrBase = base;
        auto indexBuilder = table.GetIndexHashBuilder(column.Hashes.size());
        if (base.CtrType == ECtrType::FeatureFreq) {
            auto values = table.AllocateBlobAndGetArrayRef<int>(column.Hashes.size());
            table.CounterDenominator = column.Bins.size();
            for (ui32 category = 0; category < column.Hashes.size(); ++category)
                values[indexBuilder.AddIndex(ModelCategoryHash(column.Hashes[category]))] = counts[category];
        } else if (base.CtrType == ECtrType::FloatTargetMeanValue) {
            auto values = table.AllocateBlobAndGetArrayRef<TCtrMeanHistory>(column.Hashes.size());
            for (ui32 category = 0; category < column.Hashes.size(); ++category) {
                auto& value = values[indexBuilder.AddIndex(ModelCategoryHash(column.Hashes[category]))];
                value.Sum = sums[0][category];
                value.Count = counts[category];
            }
        } else {
            const ui32 classes = sums.size() + (base.CtrType == ECtrType::Borders);
            table.TargetClassesCount = classes;
            auto values = table.AllocateBlobAndGetArrayRef<int>(column.Hashes.size() * classes);
            for (ui32 category = 0; category < column.Hashes.size(); ++category) {
                const ui32 index = indexBuilder.AddIndex(ModelCategoryHash(column.Hashes[category]));
                for (ui32 targetClass = 0; targetClass < classes; ++targetClass) {
                    float value;
                    if (base.CtrType == ECtrType::Buckets) {
                        value = sums[targetClass][category];
                    } else {
                        const float previous = targetClass ? sums[targetClass - 1][category] : counts[category];
                        const float next = targetClass < sums.size() ? sums[targetClass][category] : 0.0f;
                        value = previous - next;
                    }
                    CB_ENSURE(value >= 0 && value <= counts[category] && std::floor(value) == value,
                              "Metal CTR class counts are invalid");
                    values[index * classes + targetClass] = static_cast<int>(value);
                }
            }
        }
        if (!*provider) *provider = MakeIntrusive<TStaticCtrProvider>();
        (*provider)->AddCtrCalcerData(std::move(table));
    }

    void AddCtrFeatures(const TCatFeature& feature, const TCategoryColumn& column,
                         TConstArrayRef<float> targets, const TVector<ui32>& historyOrder,
                         TConstArrayRef<ui32> groupIds,
                         const NCatboostOptions::TCatBoostOptions& options,
                         const TVector<NCatboostOptions::TCtrDescription>& descriptions,
                         const TMetalCategoricalData* gridReference,
                         TMetalCategoricalData* result,
                         const TCategoryColumn* fullColumn) {
        const ui32 rows = column.Bins.size();
        TVector<ui32> historyBins(rows), sortedBins(rows), indices(rows);
        for (ui32 row = 0; row < rows; ++row) historyBins[row] = column.Bins[historyOrder[row]];
        CBMSortStats sortStats = {};
        char sortError[2048] = {};
        CB_ENSURE(cbm_sort_u32(historyBins.data(), historyOrder.data(), rows,
            sortedBins.data(), indices.data(), &sortStats, sortError, sizeof(sortError)) == 0,
            "Metal categorical sorting failed: " << sortError);
        result->Stats.kernel_dispatches += sortStats.kernel_dispatches;
        result->Stats.gpu_seconds += sortStats.gpu_seconds;
        std::memcpy(result->Stats.device_name, sortStats.device_name, sizeof(sortStats.device_name));
        TVector<ui32> fullSortedBins, fullIndices;
        TVector<float> fullTargets;
        if (fullColumn) {
            const ui32 fullRows = fullColumn->Bins.size();
            TVector<ui32> fullOrder(fullRows);
            std::iota(fullOrder.begin(), fullOrder.end(), 0);
            fullSortedBins.resize(fullRows);
            fullIndices.resize(fullRows);
            fullTargets.resize(fullRows, 0);
            CB_ENSURE(cbm_sort_u32(fullColumn->Bins.data(), fullOrder.data(), fullRows,
                fullSortedBins.data(), fullIndices.data(), &sortStats, sortError, sizeof(sortError)) == 0,
                "Metal Full categorical sorting failed: " << sortError);
            result->Stats.kernel_dispatches += sortStats.kernel_dispatches;
            result->Stats.gpu_seconds += sortStats.gpu_seconds;
        }
        const auto targetBorders = BuildBorders(targets, static_cast<ui32>(options.RandomSeed.Get()),
                                                options.CatFeatureParams->TargetBinarization.Get());
        CB_ENSURE(targetBorders.size() <= 255, "Metal CTR targets require at most 255 borders");
        const auto targetBins = BinarizeLine<ui8>(targets, ENanMode::Forbidden, targetBorders);
        TVector<float> binTargets(targetBins.begin(), targetBins.end());
        const auto runtimeFeatureStart = result->SplitCandidates.size();
        for (const auto& description : descriptions) {
            const ECtrType type = description.Type;
            CB_ENSURE(type == ECtrType::Borders || type == ECtrType::Buckets ||
                      type == ECtrType::FloatTargetMeanValue || type == ECtrType::FeatureFreq,
                      "Metal supports Borders, Buckets, FloatTargetMeanValue, and FeatureFreq CTRs");
            // EstimateMetalCtrPriors resolves the complete-learn prior before
            // any permutation is built. Like CUDA, it retains BetaPrior in the
            // option metadata; multi-border targets keep their configured prior.
            CB_ENSURE(description.PriorEstimation == EPriorEstimation::No ||
                      (description.PriorEstimation == EPriorEstimation::BetaPrior && type == ECtrType::Borders),
                      "Metal automatic prior estimation supports simple Borders CTRs only");
            CB_ENSURE(description.CtrBinarization->BorderCount <= 255,
                      "Metal supports at most 255 borders per CTR feature");
            const ui32 paramsCount = type == ECtrType::Borders ? targetBorders.size()
                : type == ECtrType::Buckets ? targetBorders.size() + 1 : 1;
            if (!paramsCount) continue;
            result->HasPermutationDependentCtrs |= type != ECtrType::FeatureFreq;
            CB_ENSURE(ui64(column.Hashes.size()) * (paramsCount + 1) * sizeof(float) <= MaxInputBytes,
                      "Metal final CTR statistics exceed the experimental 1 GiB table limit");
            const auto& priors = description.GetPriors();
            CB_ENSURE(!priors.empty(), "Metal CTR configuration requires at least one prior");
            TModelCtrBase base;
            base.Projection.CatFeatures = {feature.Position.Index};
            base.CtrType = type;
            TVector<TVector<float>> finalSums(paramsCount);
            TVector<ui32> counts(column.Hashes.size());
            TVector<ui32> fullCounts;
            for (ui32 param = 0; param < paramsCount; ++param) {
                // CUDA omits bucket zero for binary targets because bucket
                // one contains the equivalent complementary information.
                if (type == ECtrType::Buckets && paramsCount == 2 && param == 0) continue;
                for (const auto& prior : priors) {
                    CB_ENSURE(prior.size() >= 1 && prior.size() <= 2 && std::isfinite(prior[0]) &&
                              (prior.size() == 1 || (std::isfinite(prior[1]) && prior[1] > 0)),
                              "Metal CTR priors require a finite numerator and positive denominator");
                    const float priorNum = prior[0], priorDenom = prior.size() == 2 ? prior[1] : 1.0f;
                    const ui32 nativeType = type == ECtrType::Borders ? 0 : type == ECtrType::Buckets ? 1
                        : type == ECtrType::FloatTargetMeanValue ? 2 : 3;
                    CBMCtrParams params{rows, static_cast<ui32>(column.Hashes.size()), nativeType, param,
                                        priorNum, priorDenom};
                    TVector<float> values(rows), sums(column.Hashes.size());
                    CBMCtrStats stats = {};
                    char error[2048] = {};
                    const float* nativeTargets = type == ECtrType::FloatTargetMeanValue ? targets.data() : binTargets.data();
                    CB_ENSURE(cbm_compute_ctrs_grouped(&params, sortedBins.data(), indices.data(), nativeTargets,
                        groupIds.empty() ? nullptr : groupIds.data(),
                        values.data(), sums.data(), counts.data(), &stats, error, sizeof(error)) == 0,
                        "Metal categorical statistics failed: " << error);
                    result->Stats.kernel_dispatches += stats.kernel_dispatches;
                    result->Stats.gpu_seconds += stats.gpu_seconds;
                    std::memcpy(result->Stats.device_name, stats.device_name, sizeof(stats.device_name));
                    finalSums[param] = std::move(sums);
                    if (type == ECtrType::FeatureFreq && fullColumn) {
                        // Only precomputed FeatureFreq consults Full. Its
                        // unweighted GPU count includes evaluation rows and
                        // evaluation-only categories, independently of labels,
                        // object/group weights and history permutations.
                        params.rows = fullColumn->Bins.size();
                        params.categories = fullColumn->Hashes.size();
                        values.resize(params.rows);
                        TVector<float> fullSums(params.categories);
                        fullCounts.resize(params.categories);
                        CB_ENSURE(cbm_compute_ctrs(&params, fullSortedBins.data(), fullIndices.data(), fullTargets.data(),
                            values.data(), fullSums.data(), fullCounts.data(), &stats, error, sizeof(error)) == 0,
                            "Metal Full categorical statistics failed: " << error);
                        result->Stats.kernel_dispatches += stats.kernel_dispatches;
                        result->Stats.gpu_seconds += stats.gpu_seconds;
                        // CUDA's border builder sees only the learn slice.
                        values.resize(rows);
                    }
                    TModelCtr ctr;
                    ctr.Base = base;
                    ctr.TargetBorderIdx = param;
                    ctr.PriorNum = priorNum;
                    ctr.PriorDenom = priorDenom;
                    TVector<float> borders;
                    if (gridReference) {
                        const ui32 featureIndex = result->SplitCandidates.size();
                        CB_ENSURE(featureIndex < gridReference->SplitCandidates.size(),
                                  "Metal CTR permutation feature grids differ");
                        for (const auto& split : gridReference->SplitCandidates[featureIndex]) {
                            CB_ENSURE(split.Type == ESplitType::OnlineCtr && split.OnlineCtr.Ctr == ctr,
                                      "Metal CTR permutation descriptors differ");
                            borders.push_back(split.OnlineCtr.Border);
                        }
                    } else {
                        borders = BuildBorders(values, static_cast<ui32>(options.RandomSeed.Get()),
                                               description.GetCtrBinarization());
                        // CUDA's CTR border builder retains a .5 candidate
                        // even for a constant column. New RSM sampling must
                        // keep that packed-grid entry without drawing again.
                        // Preserve accepted rsm=1 grids byte for byte.
                        if (options.ObliviousTreeOptions->Rsm < 1 && borders.empty()) {
                            borders.push_back(.5f);
                        }
                    }
                    TVector<TModelSplit> splits;
                    for (float border : borders) splits.emplace_back(TModelCtrSplit{ctr, border});
                    AppendFeature(BinarizeLine<ui8>(values, ENanMode::Forbidden, borders),
                                  std::move(splits), 0, borders.size() + 1, result, column.UniqueValuesOnAll,
                                  &ctr, &description.GetCtrBinarization(), column.UniqueValuesOnAll, feature.Position.Index);
                }
            }
            if (type == ECtrType::Buckets && paramsCount == 2) {
                finalSums[0].resize(counts.size());
                for (ui32 category = 0; category < counts.size(); ++category)
                    finalSums[0][category] = counts[category] - finalSums[1][category];
            }
            // CUDA writes test/full-learn tables while processing permutation
            // zero. Other permutations share those inference statistics.
            if (!gridReference) {
                if (type == ECtrType::FeatureFreq && fullColumn) {
                    AddFinalTable(base, *fullColumn, fullCounts, {}, &result->CtrProvider);
                    AddFinalTable(base, column, counts, finalSums, &result->FinalCounterProvider);
                } else {
                    AddFinalTable(base, column, counts, finalSums, &result->CtrProvider);
                }
            }
        }
        if (result->SplitCandidates.size() == runtimeFeatureStart) {
            TVector<ui8> bins(rows, 0);
            AppendFeature(bins, {}, 0, 1, result, 0, nullptr, nullptr, column.UniqueValuesOnAll, feature.Position.Index);
        }
    }
}

TMetalCategoryColumn ReadMetalCategoryColumn(const TQuantizedObjectsDataProvider& objects,
                                            const TCatFeature& feature,
                                            NPar::ILocalExecutor* executor) {
    return ReadCategoryColumn(objects, feature, objects.GetObjectCount(), executor);
}

void EstimateMetalCtrPriors(const TTrainingDataProvider& data,
                           NCatboostOptions::TCatBoostOptions* options) {
    CB_ENSURE(options, "Metal CTR prior estimation requires training options");
    auto& categorical = options->CatFeatureParams.Get();
    const auto needsEstimation = [](const TVector<NCatboostOptions::TCtrDescription>& descriptions) {
        return std::any_of(descriptions.begin(), descriptions.end(), [](const auto& description) {
            return description.PriorEstimation != EPriorEstimation::No;
        });
    };
    const bool estimateSimple = needsEstimation(categorical.SimpleCtrs.Get());
    const bool estimatePerFeature = std::any_of(categorical.PerFeatureCtrs->begin(),
        categorical.PerFeatureCtrs->end(), [&](const auto& item) { return needsEstimation(item.second); });
    const auto& layout = *data.ObjectsData->GetFeaturesLayout();
    if ((!estimateSimple && !estimatePerFeature) || !layout.GetCatFeatureCount()) return;
    const auto target = data.TargetData->GetTarget();
    CB_ENSURE(target && target->size() == 1 && (*target)[0].size() == data.GetObjectCount(),
              "Metal CTR prior estimation requires one scalar learn target per row");
    const ui32 rows = data.GetObjectCount();
    CB_ENSURE(rows > 0 && rows <= (1u << 24), "Metal CTR prior estimation row count is invalid");
    // Prepare the actual shared target grid before calling the estimator.
    // Its inputs are learn labels and original quantized category bins, never
    // an exclusive CTR history, a row permutation or effective sample weights.
    const auto borders = BuildBorders((*target)[0], static_cast<ui32>(options->RandomSeed.Get()),
                                      categorical.TargetBinarization.Get());
    if (borders.size() > 1) return; // CUDA leaves existing priors unchanged here.
    const auto classes = BinarizeLine<ui8>((*target)[0], ENanMode::Forbidden, borders);
    auto& perFeature = categorical.PerFeatureCtrs.Get();
    for (ui32 index = 0; index < layout.GetCatFeatureCount(); ++index) {
        if (!layout.GetInternalFeatureMetaInfo(index, EFeatureType::Categorical).IsAvailable) continue;
        const ui32 flatIndex = layout.GetExternalFeatureIdx(index, EFeatureType::Categorical);
        auto found = perFeature.find(flatIndex);
        if (found == perFeature.end()) {
            if (!estimateSimple) continue;
            found = perFeature.emplace(flatIndex, categorical.SimpleCtrs.Get()).first;
        }
        auto& descriptions = found->second;
        if (!needsEstimation(descriptions)) continue;
        TMaybe<TBetaPriorEstimator::TBetaPrior> estimated;
        for (auto& description : descriptions) {
            if (description.Type == ECtrType::Borders && categorical.TargetBinarization->BorderCount == 1u) {
                if (!estimated) {
                    const auto holder = data.ObjectsData->GetCatFeature(index);
                    CB_ENSURE(holder, "Metal prior estimation category values are unavailable");
                    const auto& values = **holder;
                    const ui32 unique = data.ObjectsData->GetQuantizedFeaturesInfo()->GetUniqueValuesCounts(
                        TCatFeatureIdx(index)).OnAll;
                    CB_ENSURE(ui64(unique) * 2 * sizeof(double) <= MaxInputBytes,
                              "Metal prior estimator category statistics exceed the experimental 1 GiB limit");
                    estimated = TBetaPriorEstimator::EstimateBetaPrior(
                        classes.data(), values.GetBlockIterator(), values.GetSize(), unique);
                    CB_ENSURE(std::isfinite(estimated->Alpha) && std::isfinite(estimated->Beta) &&
                              estimated->Alpha > 0 && estimated->Beta > 0,
                              "Metal CTR prior estimation produced invalid Beta parameters");
                }
                // When any description requests estimation, CUDA replaces
                // every Borders prior on this feature, including No siblings.
                description.Priors = {{static_cast<float>(estimated->Alpha),
                                       static_cast<float>(estimated->Alpha + estimated->Beta)}};
            } else {
                CB_ENSURE(description.PriorEstimation == EPriorEstimation::No,
                          "Metal auto prior estimation requires simple Borders CTRs and ctr_target_border_count=1");
            }
        }
    }
}

const TModelSplit& TMetalCategoricalData::GetSplit(ui32 feature, ui32 bin, ui8 type) const {
    CB_ENSURE(feature < SplitCandidates.size() && bin < SplitCandidates[feature].size(),
              "Metal returned an invalid categorical/CTR split");
    const auto& split = SplitCandidates[feature][bin];
    CB_ENSURE(type == (split.Type == ESplitType::OneHotFeature ? 1 : 0),
              "Metal returned a categorical split with the wrong comparison type");
    return split;
}

ui32 CalcMetalCategoricalChecksum(const TQuantizedObjectsDataProvider& objects,
                                  NPar::ILocalExecutor* executor) {
    const auto& layout = *objects.GetFeaturesLayout();
    ui32 checksum = UpdateCheckSum(0, layout.GetCatFeatureCount());
    checksum = UpdateCheckSum(checksum, objects.GetObjectCount());
    const auto features = CreateCatFeatures(layout);
    for (const auto& feature : features) {
        const bool available = layout.GetExternalFeatureMetaInfo(feature.Position.FlatIndex).IsAvailable;
        checksum = UpdateCheckSum(checksum, feature.Position.FlatIndex);
        checksum = UpdateCheckSum(checksum, TStringBuf(feature.FeatureId));
        checksum = UpdateCheckSum(checksum, available);
        if (!available) continue;
        const auto column = ReadCategoryColumn(objects, feature, objects.GetObjectCount(), executor);
        TVector<ui32> rowHashes(column.Bins.size());
        for (ui32 row = 0; row < column.Bins.size(); ++row)
            rowHashes[row] = column.Hashes[column.Bins[row]];
        checksum = UpdateCheckSum(checksum, rowHashes);
    }
    return checksum;
}

namespace {
TMetalCategoricalData PrepareCategoricalPermutation(const TTrainingDataProvider& data,
                                                    const NCatboostOptions::TCatBoostOptions& options,
                                                    NPar::ILocalExecutor* executor,
                                                    ui32 permutation,
                                                    const TMetalCategoricalData* gridReference,
                                                    const TTrainingDataProvider* evaluation) {
    TMetalCategoricalData result;
    const auto& objects = *data.ObjectsData;
    const auto& layout = *objects.GetFeaturesLayout();
    result.AllCatFeatures = CreateCatFeatures(layout);
    if (result.AllCatFeatures.empty()) return result;
    const ui32 rows = data.GetObjectCount();
    CB_ENSURE(rows > 0 && rows <= (1u << 24), "Metal categorical training row count is invalid");
    const auto& categoricalOptions = options.CatFeatureParams.Get();
    // Native training uses all uint8 values as known-category bins. Unseen
    // inference categories are handled by the shared hashed model reader,
    // rather than a reserved training bin.
    CB_ENSURE(categoricalOptions.OneHotMaxSize <= 256, "Metal currently supports one_hot_max_size up to 256");
    const auto target = data.TargetData->GetTarget();
    CB_ENSURE(!target || target->size() == 1,
              "Metal categorical training currently requires scalar targets");
    TVector<ui32> historyOrder;
    TVector<ui32> groupIds;
    for (const auto& feature : result.AllCatFeatures) {
        if (!layout.GetExternalFeatureMetaInfo(feature.Position.FlatIndex).IsAvailable) continue;
        const auto column = ReadCategoryColumn(objects, feature, rows, executor);
        // CUDA UseForOneHotEncoding uses OnAll, so eval-only categories can
        // move a feature above the one-hot threshold before training begins.
        const bool ordered = options.BoostingOptions->BoostingType == EBoostingType::Ordered;
        const bool featureParallel = options.BoostingOptions->DataPartitionType == EDataPartitionType::FeatureParallel;
        // Keep the preceding simple-only Ordered grid layout for snapshot
        // compatibility. Compound FeatureParallel follows CUDA's strict >1
        // one-hot cardinality rule, shared by the tree tensor scheduler.
        const bool legacyOrderedConstant = ordered && categoricalOptions.MaxTensorComplexity <= 1;
        if ((column.UniqueValuesOnAll > 1 || legacyOrderedConstant) && column.UniqueValuesOnAll <= categoricalOptions.OneHotMaxSize) {
            TVector<ui8> bins(column.Bins.begin(), column.Bins.end());
            TVector<TModelSplit> splits;
            if (column.Hashes.size() > 1) {
                for (ui32 hash : column.Hashes)
                    splits.emplace_back(TOneHotSplit{feature.Position.Index, static_cast<i32>(hash)});
            }
            AppendFeature(bins, std::move(splits), 1, Max<ui32>(1, column.Hashes.size()), &result,
                          0, nullptr, nullptr, column.UniqueValuesOnAll, feature.Position.Index);
        } else {
            CB_ENSURE(target && target->size() == 1 && (*target)[0].size() == rows,
                      "Metal categorical CTR training requires one scalar target per row");
            const auto targets = (*target)[0];
            if (historyOrder.empty()) {
                const auto& block = options.BoostingOptions->PermutationBlockSize;
                historyOrder = featureParallel ? MakeMetalFeatureParallelHistoryOrder(data, permutation,
                    GetMetalFeatureParallelBlockSize(rows, block.IsSet() ? block.Get() : 64)) : MakeHistoryOrder(data, permutation);
            }
            CB_ENSURE(categoricalOptions.MaxTensorComplexity <= 1 || featureParallel,
                      "Metal compound CTRs require data_partition='FeatureParallel'");
            if (categoricalOptions.CtrHistoryUnit == ECtrHistoryUnit::Group &&
                !data.ObjectsGrouping->IsTrivial() && groupIds.empty()) {
                // CUDA BuildCtrTarget assigns query ordinals in original row
                // order. Do not truncate external uint64 group hashes.
                groupIds.resize(rows);
                for (ui32 group = 0; group < data.ObjectsGrouping->GetGroupCount(); ++group) {
                    const auto bounds = data.ObjectsGrouping->GetGroup(group);
                    std::fill(groupIds.begin() + bounds.Begin, groupIds.begin() + bounds.End, group);
                }
            }
            const auto perFeature = categoricalOptions.PerFeatureCtrs->find(feature.Position.FlatIndex);
            const auto& descriptions = perFeature == categoricalOptions.PerFeatureCtrs->end()
                ? categoricalOptions.SimpleCtrs.Get() : perFeature->second;
            CB_ENSURE(!descriptions.empty(), "Metal high-cardinality categories require simple_ctr configurations");
            TMaybe<TCategoryColumn> fullColumn;
            if (categoricalOptions.CounterCalcMethod == ECounterCalc::Full && evaluation &&
                std::any_of(descriptions.begin(), descriptions.end(), [](const auto& description) {
                    return description.Type == ECtrType::FeatureFreq;
                })) {
                fullColumn = JoinCategoryColumns(column, ReadMetalCategoryColumn(*evaluation->ObjectsData, feature, executor));
            }
            AddCtrFeatures(feature, column, targets, historyOrder, groupIds, options, descriptions, gridReference, &result,
                           fullColumn ? &*fullColumn : nullptr);
        }
    }
    return result;
}
}

TConstArrayRef<ui8> TMetalCategoricalData::GetPermutationBins(ui32 permutation) const {
    CB_ENSURE(permutation < GetPermutationCount(), "Metal CTR permutation index is out of range");
    return permutation ? AdditionalPermutationBins[permutation - 1] : Bins;
}

TMetalCategoricalData PrepareMetalCategoricalData(const TTrainingDataProvider& data,
                                                 const NCatboostOptions::TCatBoostOptions& options,
                                                 NPar::ILocalExecutor* executor,
                                                 const TTrainingDataProvider* evaluation) {
    auto result = PrepareCategoricalPermutation(data, options, executor, 0, nullptr, evaluation);
    CB_ENSURE(!result.CtrProvider || options.BoostingOptions->PermutationCount <= 1,
              "Metal currently supports one CTR permutation; set permutation_count=1");
    return result;
}

TMetalCategoricalData PrepareMetalCategoricalPermutations(const TTrainingDataProvider& data,
                                                         const NCatboostOptions::TCatBoostOptions& options,
                                                         NPar::ILocalExecutor* executor,
                                                         const TTrainingDataProvider* evaluation) {
    auto result = PrepareCategoricalPermutation(data, options, executor, 0, nullptr, evaluation);
    // As in CUDA Plain boosting, numeric/one-hot-only training needs just one
    // dataset. has_time also disables the internal category permutations.
    const bool featureParallel = options.BoostingOptions->DataPartitionType == EDataPartitionType::FeatureParallel;
    const bool preserveOrder = featureParallel ? data.ObjectsData->GetOrder() == EObjectsOrder::Ordered :
                                        options.DataProcessingOptions->HasTimeFlag.Get();
    const ui32 permutationCount = !result.CtrProvider || preserveOrder
        ? 1 : options.BoostingOptions->PermutationCount.Get();
    CB_ENSURE(permutationCount >= 1, "Metal CTR permutation count must be positive");
    CB_ENSURE(ui64(result.Bins.size()) * permutationCount <= MaxInputBytes,
              "Metal categorical permutation inputs exceed the experimental 1 GiB limit");
    result.AdditionalPermutationBins.reserve(permutationCount - 1);
    for (ui32 permutation = 1; permutation < permutationCount; ++permutation) {
        auto current = PrepareCategoricalPermutation(data, options, executor, permutation, &result, evaluation);
        CB_ENSURE(current.SplitCandidates == result.SplitCandidates && current.Bins.size() == result.Bins.size() &&
                  current.CtrUniqueValues == result.CtrUniqueValues &&
                  current.ColumnModelCtrs == result.ColumnModelCtrs &&
                  current.ColumnCtrBinarizations == result.ColumnCtrBinarizations &&
                  current.ColumnUniqueValuesOnAll == result.ColumnUniqueValuesOnAll &&
                  current.ColumnCatFeatureIndices == result.ColumnCatFeatureIndices,
                  "Metal CTR permutations require identical feature columns and split grids");
        result.Stats.kernel_dispatches += current.Stats.kernel_dispatches;
        result.Stats.gpu_seconds += current.Stats.gpu_seconds;
        result.AdditionalPermutationBins.push_back(std::move(current.Bins));
    }
    return result;
}

void FinalizeMetalCounterTables(const TMetalCategoricalData& data, TFullModel* model) {
    if (!data.FinalCounterProvider) return;
    CB_ENSURE(model && model->CtrProvider, "Metal final counter tables require a model CTR provider");
    for (const auto& [base, table] : data.FinalCounterProvider->CtrData.LearnCtrs) {
        Y_UNUSED(base);
        model->CtrProvider->AddCtrCalcerData(TCtrValueTable(table));
    }
}
}
