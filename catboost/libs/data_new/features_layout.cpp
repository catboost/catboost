#include "features_layout.h"

#include "util.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/quantization_schema/schema.h>

#include <util/generic/algorithm.h>
#include <util/generic/scope.h>
#include <util/generic/xrange.h>
#include <util/string/join.h>

#include <tuple>


using namespace NCB;

TQuantizedFeatureIndexRemapper::TQuantizedFeatureIndexRemapper(
    const ui32 indexOffset,
    const ui32 binCount)
    : IndexOffset_(indexOffset)
    , BinCount_(binCount)
{
}

bool TFeatureMetaInfo::operator==(const TFeatureMetaInfo& rhs) const {
    return std::tie(Type, Name, IsIgnored, IsAvailable) ==
        std::tie(rhs.Type, rhs.Name, rhs.IsIgnored, rhs.IsAvailable);
}

TFeaturesLayout::TFeaturesLayout(const ui32 featureCount)
    : TFeaturesLayout(featureCount, {}, {}, nullptr)
{}

TFeaturesLayout::TFeaturesLayout(
    const ui32 featureCount,
    const TVector<ui32>& catFeatureIndices,
    const TVector<TString>& featureId,
    const TPoolQuantizationSchema* const schema)
{
    CheckDataSize(featureId.size(), (size_t)featureCount, "feature Ids", true, "feature count");

    ExternalIdxToMetaInfo.reserve((size_t)featureCount);
    for (auto externalFeatureIdx : xrange(featureCount)) {
        // cat feature will be set later
        ExternalIdxToMetaInfo.emplace_back(
            EFeatureType::Float,
            !featureId.empty() ? featureId[externalFeatureIdx] : ""
        );
    }
    for (auto catFeatureExternalIdx : catFeatureIndices) {
        CB_ENSURE(
            catFeatureExternalIdx < featureCount,
            "Cat feature index (" << catFeatureExternalIdx << ") is out of valid range [0,"
            << featureCount << ')'
        );
        ExternalIdxToMetaInfo[catFeatureExternalIdx].Type = EFeatureType::Categorical;
    }

    ExternalIdxRemappers.resize(featureCount);
    if (schema) {
        for (size_t i = 0, iEnd = schema->FeatureIndices.size(); i < iEnd; ++i) {
            const auto flatFeatureIdx = schema->FeatureIndices[i];
            const auto binCount = schema->Borders[i].size() + 1;
            ExternalIdxRemappers[flatFeatureIdx] = TQuantizedFeatureIndexRemapper(
                ExternalIdxToMetaInfo.size(),
                binCount);
            const auto artificialFeatureCount = ExternalIdxRemappers[flatFeatureIdx].GetArtificialFeatureCount();
            for (ui32 j = 0; j < artificialFeatureCount; ++j) {
                const auto& original = ExternalIdxToMetaInfo[flatFeatureIdx];
                ExternalIdxToMetaInfo.emplace_back(
                    original.Type,
                    Join("", original.Name, "_part_", j));
            }
        }
    }

    for (auto externalFeatureIdx : xrange(ExternalIdxToMetaInfo.size())) {
        if (ExternalIdxToMetaInfo[externalFeatureIdx].Type == EFeatureType::Float) {
            FeatureExternalIdxToInternalIdx.push_back((ui32)FloatFeatureInternalIdxToExternalIdx.size());
            FloatFeatureInternalIdxToExternalIdx.push_back(externalFeatureIdx);
        } else {
            FeatureExternalIdxToInternalIdx.push_back((ui32)CatFeatureInternalIdxToExternalIdx.size());
            CatFeatureInternalIdxToExternalIdx.push_back(externalFeatureIdx);
        }
    }
}

TFeaturesLayout::TFeaturesLayout(
    const TVector<TFloatFeature>& floatFeatures,
    const TVector<TCatFeature>& catFeatures)
{
    TFeatureMetaInfo defaultIgnoredMetaInfo(EFeatureType::Float, TString(), true);
    const ui32 internalOrExternalIndexPlaceholder = Max<ui32>();
    for (const TFloatFeature& floatFeature : floatFeatures) {
        CB_ENSURE(floatFeature.FlatFeatureIndex >= 0, "floatFeature.FlatFeatureIndex is negative");
        CB_ENSURE(floatFeature.FeatureIndex >= 0, "floatFeature.FeatureIndex is negative");
        if ((size_t)floatFeature.FlatFeatureIndex >= ExternalIdxToMetaInfo.size()) {
            CB_ENSURE(
                (size_t)floatFeature.FlatFeatureIndex < (size_t)Max<ui32>(),
                "floatFeature.FlatFeatureIndex is greater than maximum allowed index: " << (Max<ui32>() - 1)
            );
            ExternalIdxToMetaInfo.resize(floatFeature.FlatFeatureIndex + 1, defaultIgnoredMetaInfo);
            FeatureExternalIdxToInternalIdx.resize(floatFeature.FlatFeatureIndex + 1, internalOrExternalIndexPlaceholder);
        }
        ExternalIdxToMetaInfo[floatFeature.FlatFeatureIndex] =
            TFeatureMetaInfo(EFeatureType::Float, floatFeature.FeatureId);
        FeatureExternalIdxToInternalIdx[floatFeature.FlatFeatureIndex] = floatFeature.FeatureIndex;
        if ((size_t)floatFeature.FeatureIndex >= FloatFeatureInternalIdxToExternalIdx.size()) {
            FloatFeatureInternalIdxToExternalIdx.resize((size_t)floatFeature.FeatureIndex + 1, internalOrExternalIndexPlaceholder);
        }
        FloatFeatureInternalIdxToExternalIdx[floatFeature.FeatureIndex] = floatFeature.FlatFeatureIndex;
    }

    for (const TCatFeature& catFeature : catFeatures) {
        CB_ENSURE(catFeature.FlatFeatureIndex >= 0, "catFeature.FlatFeatureIndex is negative");
        CB_ENSURE(catFeature.FeatureIndex >= 0, "catFeature.FeatureIndex is negative");
        if ((size_t)catFeature.FlatFeatureIndex >= ExternalIdxToMetaInfo.size()) {
            CB_ENSURE(
                (size_t)catFeature.FlatFeatureIndex < (size_t)Max<ui32>(),
                "catFeature.FlatFeatureIndex is greater than maximum allowed index: " << (Max<ui32>() - 1)
            );
            ExternalIdxToMetaInfo.resize(catFeature.FlatFeatureIndex + 1, defaultIgnoredMetaInfo);
            FeatureExternalIdxToInternalIdx.resize(catFeature.FlatFeatureIndex + 1, internalOrExternalIndexPlaceholder);
        }
        ExternalIdxToMetaInfo[catFeature.FlatFeatureIndex] =
            TFeatureMetaInfo(EFeatureType::Categorical, catFeature.FeatureId);
        FeatureExternalIdxToInternalIdx[catFeature.FlatFeatureIndex] = catFeature.FeatureIndex;
        if ((size_t)catFeature.FeatureIndex >= CatFeatureInternalIdxToExternalIdx.size()) {
            CatFeatureInternalIdxToExternalIdx.resize((size_t)catFeature.FeatureIndex + 1, internalOrExternalIndexPlaceholder);
        }
        CatFeatureInternalIdxToExternalIdx[catFeature.FeatureIndex] = catFeature.FlatFeatureIndex;
    }

    ExternalIdxRemappers.resize(ExternalIdxToMetaInfo.size());
    for (const auto& floatFeature : floatFeatures) {
        const auto flatFeatureIdx = floatFeature.FlatFeatureIndex;
        const auto binCount = floatFeature.Borders.size() + 1;
        ExternalIdxRemappers[flatFeatureIdx] = TQuantizedFeatureIndexRemapper(
            ExternalIdxToMetaInfo.size(),
            binCount);
        const auto artificialFeatureCount = ExternalIdxRemappers[flatFeatureIdx].GetArtificialFeatureCount();
        for (ui32 j = 0; j < artificialFeatureCount; ++j) {
            const auto original = ExternalIdxToMetaInfo[flatFeatureIdx];
            FloatFeatureInternalIdxToExternalIdx.push_back(
                ExternalIdxToMetaInfo.size());
            ExternalIdxToMetaInfo.emplace_back(
                original.Type,
                Join("", original.Name, "_part_", j));
        }
    }
}

const TFeatureMetaInfo& TFeaturesLayout::GetInternalFeatureMetaInfo(
    ui32 internalFeatureIdx,
    EFeatureType type) const
{
    return ExternalIdxToMetaInfo[GetExternalFeatureIdx(internalFeatureIdx, type)];
}

TConstArrayRef<TFeatureMetaInfo> TFeaturesLayout::GetExternalFeaturesMetaInfo() const {
    return ExternalIdxToMetaInfo;
}

TString TFeaturesLayout::GetExternalFeatureDescription(ui32 internalFeatureIdx, EFeatureType type) const {
    return ExternalIdxToMetaInfo[GetExternalFeatureIdx(internalFeatureIdx, type)].Name;
}

ui32 TFeaturesLayout::GetExternalFeatureIdx(ui32 internalFeatureIdx, EFeatureType type) const {
    if (type == EFeatureType::Float) {
        return FloatFeatureInternalIdxToExternalIdx[internalFeatureIdx];
    } else {
        return CatFeatureInternalIdxToExternalIdx[internalFeatureIdx];
    }
}

ui32 TFeaturesLayout::GetInternalFeatureIdx(ui32 externalFeatureIdx) const {
    Y_ASSERT(IsCorrectExternalFeatureIdx(externalFeatureIdx));
    return FeatureExternalIdxToInternalIdx[externalFeatureIdx];
}

EFeatureType TFeaturesLayout::GetExternalFeatureType(ui32 externalFeatureIdx) const {
    Y_ASSERT(IsCorrectExternalFeatureIdx(externalFeatureIdx));
    return ExternalIdxToMetaInfo[externalFeatureIdx].Type;
}

bool TFeaturesLayout::IsCorrectExternalFeatureIdx(ui32 externalFeatureIdx) const {
    return (size_t)externalFeatureIdx < ExternalIdxToMetaInfo.size();
}

bool TFeaturesLayout::IsCorrectInternalFeatureIdx(ui32 internalFeatureIdx, EFeatureType type) const {
    if (type == EFeatureType::Float) {
        return (size_t)internalFeatureIdx < FloatFeatureInternalIdxToExternalIdx.size();
    } else {
        return (size_t)internalFeatureIdx < CatFeatureInternalIdxToExternalIdx.size();
    }
}

bool TFeaturesLayout::IsCorrectExternalFeatureIdxAndType(ui32 externalFeatureIdx, EFeatureType type) const {
    if ((size_t)externalFeatureIdx >= ExternalIdxToMetaInfo.size()) {
        return false;
    }
    return ExternalIdxToMetaInfo[externalFeatureIdx].Type == type;
}

ui32 TFeaturesLayout::GetFloatFeatureCount() const {
    // cast is safe because of size invariant established in constructors
    return (ui32)FloatFeatureInternalIdxToExternalIdx.size();
}

ui32 TFeaturesLayout::GetCatFeatureCount() const {
    // cast is safe because of size invariant established in constructors
    return (ui32)CatFeatureInternalIdxToExternalIdx.size();
}

ui32 TFeaturesLayout::GetExternalFeatureCount() const {
    // cast is safe because of size invariant established in constructors
    return (ui32)ExternalIdxToMetaInfo.size();
}

ui32 TFeaturesLayout::GetFeatureCount(EFeatureType type) const {
    if (type == EFeatureType::Float) {
        return GetFloatFeatureCount();
    } else {
        return GetCatFeatureCount();
    }
}

void TFeaturesLayout::IgnoreExternalFeature(ui32 externalFeatureIdx) {
    const auto nonArtificialFeatureCount = ExternalIdxRemappers.size();
    if (externalFeatureIdx >= nonArtificialFeatureCount) {
        return;
    }

    auto& metaInfo = ExternalIdxToMetaInfo[externalFeatureIdx];
    metaInfo.IsIgnored = true;
    metaInfo.IsAvailable = false;

    const auto& remapper = ExternalIdxRemappers[externalFeatureIdx];
    const auto artificialFeatureCount = remapper.GetArtificialFeatureCount();
    if (!artificialFeatureCount) {
        return;
    }

    const auto indexOffset = remapper.GetIdxOffset();
    for (ui32 i = 0; i < artificialFeatureCount; ++i) {
        ExternalIdxToMetaInfo[indexOffset + i].IsIgnored = true;
        ExternalIdxToMetaInfo[indexOffset + i].IsAvailable = false;
    }
}

void TFeaturesLayout::IgnoreExternalFeatures(TConstArrayRef<ui32> ignoredFeatures) {
    for (auto ignoredFeature : ignoredFeatures) {
        IgnoreExternalFeature(ignoredFeature);
    }
}

TConstArrayRef<ui32> TFeaturesLayout::GetCatFeatureInternalIdxToExternalIdx() const {
    return CatFeatureInternalIdxToExternalIdx;
}


bool TFeaturesLayout::operator==(const TFeaturesLayout& rhs) const {
    return std::tie(
            ExternalIdxToMetaInfo,
            FeatureExternalIdxToInternalIdx,
            CatFeatureInternalIdxToExternalIdx,
            FloatFeatureInternalIdxToExternalIdx
        ) == std::tie(
            rhs.ExternalIdxToMetaInfo,
            rhs.FeatureExternalIdxToInternalIdx,
            rhs.CatFeatureInternalIdxToExternalIdx,
            rhs.FloatFeatureInternalIdxToExternalIdx
        );
}


TVector<TString> TFeaturesLayout::GetExternalFeatureIds() const {
    TVector<TString> result;
    result.reserve(ExternalIdxToMetaInfo.size());
    for (const auto& metaInfo : ExternalIdxToMetaInfo) {
        result.push_back(metaInfo.Name);
    }
    return result;
}

void TFeaturesLayout::SetExternalFeatureIds(TConstArrayRef<TString> featureIds) {
    CheckDataSize(featureIds.size(), ExternalIdxToMetaInfo.size(), "feature names", false, "feature count");
    for (auto i : xrange(ExternalIdxToMetaInfo.size())) {
        ExternalIdxToMetaInfo[i].Name = featureIds[i];
    }
}

bool TFeaturesLayout::HasAvailableAndNotIgnoredFeatures() const {
    for (const auto& metaInfo : ExternalIdxToMetaInfo) {
        if (metaInfo.IsAvailable && !metaInfo.IsIgnored) {
            return true;
        }
    }
    return false;
}

TQuantizedFeatureIndexRemapper
TFeaturesLayout::GetQuantizedFeatureIndexRemapper(ui32 externalFeatureIdx) const {
    return ExternalIdxRemappers[externalFeatureIdx];
}


void NCB::CheckCompatibleForApply(
    const TFeaturesLayout& learnFeaturesLayout,
    const TFeaturesLayout& applyFeaturesLayout,
    const TString& applyDataName
) {
    auto learnFeaturesMetaInfo = learnFeaturesLayout.GetExternalFeaturesMetaInfo();
    auto applyFeaturesMetaInfo = applyFeaturesLayout.GetExternalFeaturesMetaInfo();

    auto featuresIntersectionSize = Min(learnFeaturesMetaInfo.size(), applyFeaturesMetaInfo.size());

    size_t i = 0;
    for (; i < featuresIntersectionSize; ++i) {
        const auto& learnFeatureMetaInfo = learnFeaturesMetaInfo[i];
        const auto& applyFeatureMetaInfo = applyFeaturesMetaInfo[i];

        if (!learnFeatureMetaInfo.IsAvailable || learnFeatureMetaInfo.IsIgnored) {
            continue;
        }

        CB_ENSURE(
            applyFeatureMetaInfo.IsAvailable,
            "Feature #" << i
            << " is used in training data, but not available in " << applyDataName
        );
        CB_ENSURE(
            !applyFeatureMetaInfo.IsIgnored,
            "Feature #" << i
            << " is used in training data, but is ignored in " << applyDataName
        );
        CB_ENSURE(
            learnFeatureMetaInfo.Type == applyFeatureMetaInfo.Type,
            "Feature #" << i << " has type " << learnFeatureMetaInfo.Type << " in training data, but "
            << applyFeatureMetaInfo.Type << " type in " << applyDataName
        );
        CB_ENSURE(
            !learnFeatureMetaInfo.Name || !applyFeatureMetaInfo.Name ||
            (learnFeatureMetaInfo.Name == applyFeatureMetaInfo.Name),
            "Feature #" << i << " has name " << learnFeatureMetaInfo.Type << " in training data, but "
            << applyFeatureMetaInfo.Type << " name in " << applyDataName
        );
    }
    for (; i < learnFeaturesMetaInfo.size(); ++i) {
        CB_ENSURE(
            !learnFeaturesMetaInfo[i].IsAvailable || learnFeaturesMetaInfo[i].IsIgnored,
            "Feature #" << i
            << " is used in training data, but not available in " << applyDataName
        );
    }
}
