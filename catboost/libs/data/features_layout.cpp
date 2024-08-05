#include "features_layout.h"

#include "util.h"
#include "graph.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/json_helpers.h>
#include <catboost/private/libs/quantization_schema/schema.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/scope.h>
#include <util/generic/xrange.h>
#include <util/generic/hash_set.h>
#include <util/string/join.h>

#include <algorithm>
#include <tuple>


using namespace NCB;

bool TFeatureMetaInfo::EqualTo(const TFeatureMetaInfo& rhs, bool ignoreSparsity) const {
    if (!ignoreSparsity && (IsSparse != rhs.IsSparse)) {
        return false;
    }
    return std::tie(Type, Name, IsIgnored, IsAvailable) ==
        std::tie(rhs.Type, rhs.Name, rhs.IsIgnored, rhs.IsAvailable);
}

TFeatureMetaInfo::operator NJson::TJsonValue() const {
    NJson::TJsonValue result(NJson::JSON_MAP);
    result.InsertValue("Type"sv, ToString(Type));
    result.InsertValue("Name"sv, Name);
    result.InsertValue("IsSparse"sv, IsSparse);
    result.InsertValue("IsIgnored"sv, IsIgnored);
    result.InsertValue("IsAvailable"sv, IsAvailable);
    return result;
}

TFeaturesLayout::TFeaturesLayout(const ui32 featureCount)
    : TFeaturesLayout(featureCount, {}, {}, {}, {}, /*graph*/ false)
{}

TFeaturesLayout::TFeaturesLayout(
    const ui32 featureCount,
    const TVector<ui32>& catFeatureIndices,
    const TVector<ui32>& textFeatureIndices,
    const TVector<ui32>& embeddingFeatureIndices,
    const TVector<TString>& featureId,
    bool hasGraph,
    const THashMap<TString, TTagDescription>& featureTags,
    bool allFeaturesAreSparse)
{
    CheckDataSize(featureId.size(), (size_t)featureCount, "feature Ids", true, "feature count");

    HasGraph = hasGraph;

    ExternalIdxToMetaInfo.reserve((size_t)featureCount);
    for (auto externalFeatureIdx : xrange(featureCount)) {
        // cat feature will be set later
        ExternalIdxToMetaInfo.emplace_back(
            EFeatureType::Float,
            !featureId.empty() ? featureId[externalFeatureIdx] : "",
            allFeaturesAreSparse
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
    for (auto textFeatureExternalIdx : textFeatureIndices) {
        CB_ENSURE(
            textFeatureExternalIdx < featureCount,
            "Text feature index (" << textFeatureExternalIdx << ") is out of valid range [0,"
            << featureCount << ')'
        );
        ExternalIdxToMetaInfo[textFeatureExternalIdx].Type = EFeatureType::Text;
    }
    for (auto embeddingFeatureExternalIdx : embeddingFeatureIndices) {
        CB_ENSURE(
            embeddingFeatureExternalIdx < featureCount,
            "Embedding feature index (" << embeddingFeatureExternalIdx << ") is out of valid range [0,"
            << featureCount << ')'
        );
        ExternalIdxToMetaInfo[embeddingFeatureExternalIdx].Type = EFeatureType::Embedding;
    }

    for (auto externalFeatureIdx : xrange(ExternalIdxToMetaInfo.size())) {
        switch (ExternalIdxToMetaInfo[externalFeatureIdx].Type) {
            case EFeatureType::Float: {
                FeatureExternalIdxToInternalIdx.push_back((ui32)FloatFeatureInternalIdxToExternalIdx.size());
                FloatFeatureInternalIdxToExternalIdx.push_back(externalFeatureIdx);
                break;
            }
            case EFeatureType::Categorical: {
                FeatureExternalIdxToInternalIdx.push_back((ui32)CatFeatureInternalIdxToExternalIdx.size());
                CatFeatureInternalIdxToExternalIdx.push_back(externalFeatureIdx);
                break;
            }
            case EFeatureType::Text: {
                FeatureExternalIdxToInternalIdx.push_back((ui32)TextFeatureInternalIdxToExternalIdx.size());
                TextFeatureInternalIdxToExternalIdx.push_back(externalFeatureIdx);
                break;
            }
            case EFeatureType::Embedding: {
                FeatureExternalIdxToInternalIdx.push_back((ui32)EmbeddingFeatureInternalIdxToExternalIdx.size());
                EmbeddingFeatureInternalIdxToExternalIdx.push_back(externalFeatureIdx);
                break;
            }
        }
    }

    THashSet<TString> featureNames;
    for (const TString& name : featureId) {
        if (!name.empty()) {
            CB_ENSURE(
                !featureNames.contains(name),
                "All feature names should be different, but '" << name << "' used more than once."
            );
            featureNames.insert(name);
        }
    }

    for (const auto& [tag, description] : featureTags) {
        for (auto featureIdx : description.Features) {
            CB_ENSURE(
                featureIdx < featureCount,
                "Feature index (" << featureIdx << ") from tag #" << tag
                << " is out of valid range [0," << featureCount << ")"
            );
        }
        TagToExternalIndices[tag] = description.Features;
    }
}

TFeaturesLayout::TFeaturesLayout(
    const TVector<TFloatFeature>& floatFeatures,
    const TVector<TCatFeature>& catFeatures)
{
    UpdateFeaturesMetaInfo(MakeConstArrayRef(floatFeatures), EFeatureType::Float);
    UpdateFeaturesMetaInfo(MakeConstArrayRef(catFeatures), EFeatureType::Categorical);
}

TFeaturesLayout::TFeaturesLayout(
    TConstArrayRef<TFloatFeature> floatFeatures,
    TConstArrayRef<TCatFeature> catFeatures,
    TConstArrayRef<TTextFeature> textFeatures,
    TConstArrayRef<TEmbeddingFeature> embeddingFeatures)
{
    UpdateFeaturesMetaInfo(floatFeatures, EFeatureType::Float);
    UpdateFeaturesMetaInfo(catFeatures, EFeatureType::Categorical);
    UpdateFeaturesMetaInfo(textFeatures, EFeatureType::Text);
    UpdateFeaturesMetaInfo(embeddingFeatures, EFeatureType::Embedding);
}

TFeaturesLayoutPtr TFeaturesLayout::CreateFeaturesLayout(
    TConstArrayRef<TColumn> columns,
    TMaybe<const TVector<TString>*> featureNames,
    TMaybe<const THashMap<TString, TTagDescription>*> featureTags,
    bool hasGraph
) {
    TVector<TString> finalFeatureNames;
    if (featureNames) {
        finalFeatureNames = **featureNames;
    }
    TVector<ui32> catFeatureIndices;
    TVector<ui32> textFeatureIndices;
    TVector<ui32> embeddingFeatureIndices;

    ui32 featureIdx = 0;

    auto processFeatureColumn = [&](const auto& column) {
        if (!featureNames) {
            finalFeatureNames.push_back(column.Id);
        }
        if ((column.Type == EColumn::Categ) || (column.Type == EColumn::HashedCateg)) {
            catFeatureIndices.push_back(featureIdx);
        } else if (column.Type == EColumn::Text) {
            textFeatureIndices.push_back(featureIdx);
        } else if (column.Type == EColumn::NumVector) {
            embeddingFeatureIndices.push_back(featureIdx);
        }
        ++featureIdx;
    };

    for (const auto& column : columns) {
        if (IsFactorColumn(column.Type)) {
            processFeatureColumn(column);
        } else if (column.Type == EColumn::Features) {
            for (const auto& subColumn : column.SubColumns) {
                if (IsFactorColumn(subColumn.Type)) {
                    processFeatureColumn(subColumn);
                } else {
                    CB_ENSURE(false, "Non-feature sub column in Features column");
                }
            }
        }
    }
    return MakeIntrusive<TFeaturesLayout>(
        featureIdx,
        catFeatureIndices,
        textFeatureIndices,
        embeddingFeatureIndices,
        finalFeatureNames,
        hasGraph,
        featureTags.Defined()
            ? **featureTags
            : THashMap<TString, TTagDescription>{});
}

TFeaturesLayout::TFeaturesLayout(TVector<TFeatureMetaInfo>* data) { // 'data' is moved into
    Init(data);
}

void TFeaturesLayout::Init(TVector<TFeatureMetaInfo>* data) { // 'data' is moved into
    for (auto& featureMetaInfo : *data) {
        AddFeature(std::move(featureMetaInfo));
    }
    data->clear();
}


TFeaturesLayout::operator NJson::TJsonValue() const {
    NJson::TJsonValue result(NJson::JSON_MAP);

    NJson::TJsonValue features(NJson::JSON_ARRAY);
    for (const auto& featureMetaInfo : ExternalIdxToMetaInfo) {
        features.AppendValue(featureMetaInfo);
    }
    result.InsertValue("Features"sv, std::move(features));

    if (!TagToExternalIndices.empty()) {
        NJson::TJsonValue tags(NJson::JSON_MAP);
        for (const auto& [name, value] : TagToExternalIndices) {
            tags.InsertValue(name, VectorToJson(value));
        }
        result.InsertValue("Tags"sv, std::move(tags));
    }

    return result;
}


const TFeatureMetaInfo& TFeaturesLayout::GetInternalFeatureMetaInfo(
    ui32 internalFeatureIdx,
    EFeatureType type) const
{
    return ExternalIdxToMetaInfo[GetExternalFeatureIdx(internalFeatureIdx, type)];
}

const TFeatureMetaInfo& TFeaturesLayout::GetExternalFeatureMetaInfo(ui32 externalFeatureIdx) const
{
    Y_ASSERT(IsCorrectExternalFeatureIdx(externalFeatureIdx));
    return ExternalIdxToMetaInfo[externalFeatureIdx];
}

TConstArrayRef<TFeatureMetaInfo> TFeaturesLayout::GetExternalFeaturesMetaInfo() const noexcept {
    return ExternalIdxToMetaInfo;
}

TString TFeaturesLayout::GetExternalFeatureDescription(ui32 internalFeatureIdx, EFeatureType type) const {
    return ExternalIdxToMetaInfo[GetExternalFeatureIdx(internalFeatureIdx, type)].Name;
}

ui32 TFeaturesLayout::GetInternalFeatureIdx(ui32 externalFeatureIdx) const {
    Y_ASSERT(IsCorrectExternalFeatureIdx(externalFeatureIdx));
    return FeatureExternalIdxToInternalIdx[externalFeatureIdx];
}

EFeatureType TFeaturesLayout::GetExternalFeatureType(ui32 externalFeatureIdx) const {
    Y_ASSERT(IsCorrectExternalFeatureIdx(externalFeatureIdx));
    return ExternalIdxToMetaInfo[externalFeatureIdx].Type;
}

bool TFeaturesLayout::IsCorrectExternalFeatureIdx(ui32 externalFeatureIdx) const noexcept {
    return (size_t)externalFeatureIdx < ExternalIdxToMetaInfo.size();
}

bool TFeaturesLayout::IsCorrectInternalFeatureIdx(ui32 internalFeatureIdx, EFeatureType type) const noexcept {
    switch (type) {
        case EFeatureType::Float:
            return (size_t)internalFeatureIdx < FloatFeatureInternalIdxToExternalIdx.size();
        case EFeatureType::Categorical:
            return (size_t)internalFeatureIdx < CatFeatureInternalIdxToExternalIdx.size();
        case EFeatureType::Text:
            return (size_t)internalFeatureIdx < TextFeatureInternalIdxToExternalIdx.size();
        case EFeatureType::Embedding:
            return (size_t)internalFeatureIdx < EmbeddingFeatureInternalIdxToExternalIdx.size();
    }
    Y_UNREACHABLE();
}

bool TFeaturesLayout::IsCorrectExternalFeatureIdxAndType(ui32 externalFeatureIdx, EFeatureType type) const noexcept {
    if ((size_t)externalFeatureIdx >= ExternalIdxToMetaInfo.size()) {
        return false;
    }
    return ExternalIdxToMetaInfo[externalFeatureIdx].Type == type;
}

ui32 TFeaturesLayout::GetFloatFeatureCount() const noexcept {
    // cast is safe because of size invariant established in constructors
    return (ui32)FloatFeatureInternalIdxToExternalIdx.size();
}

ui32 TFeaturesLayout::GetFloatAggregatedFeatureCount() const noexcept {
    if (!HasGraph) {
        return 0;
    }
    ui32 size = 0;
    for (auto& meta: ExternalIdxToMetaInfo) {
        if (meta.Type == EFeatureType::Float && !meta.IsSparse && !meta.IsAggregated) {
            size++;
        }
    }
    return size * NCB::kFloatAggregationFeaturesCount;
}

ui32 TFeaturesLayout::GetAggregatedFeatureCount(EFeatureType type) const noexcept {
    switch (type) {
        case EFeatureType::Float:
            return GetFloatAggregatedFeatureCount();
        default:
            return 0;
    }
}

ui32 TFeaturesLayout::GetCatFeatureCount() const noexcept {
    // cast is safe because of size invariant established in constructors
    return (ui32)CatFeatureInternalIdxToExternalIdx.size();
}

ui32 TFeaturesLayout::GetTextFeatureCount() const noexcept {
    // cast is safe because of size invariant established in constructors
    return (ui32)TextFeatureInternalIdxToExternalIdx.size();
}

ui32 TFeaturesLayout::GetEmbeddingFeatureCount() const noexcept {
    // cast is safe because of size invariant established in constructors
    return (ui32)EmbeddingFeatureInternalIdxToExternalIdx.size();
}

ui32 TFeaturesLayout::GetExternalFeatureCount() const noexcept {
    // cast is safe because of size invariant established in constructors
    return (ui32)ExternalIdxToMetaInfo.size();
}

ui32 TFeaturesLayout::GetFeatureCount(EFeatureType type) const noexcept {
    switch (type) {
        case EFeatureType::Float:
            return GetFloatFeatureCount();
        case EFeatureType::Categorical:
            return GetCatFeatureCount();
        case EFeatureType::Text:
            return GetTextFeatureCount();
        case EFeatureType::Embedding:
            return GetEmbeddingFeatureCount();
    }
    Y_UNREACHABLE();
}

bool TFeaturesLayout::HasSparseFeatures(bool checkOnlyAvailable) const noexcept {
    return FindIf(
        ExternalIdxToMetaInfo,
        [=] (const TFeatureMetaInfo& metaInfo) {
            return (!checkOnlyAvailable || metaInfo.IsAvailable) && metaInfo.IsSparse;
        }
    );
}

void TFeaturesLayout::IgnoreExternalFeature(ui32 externalFeatureIdx) noexcept {
    if (externalFeatureIdx >= ExternalIdxToMetaInfo.size()) {
        return;
    }

    auto& metaInfo = ExternalIdxToMetaInfo[externalFeatureIdx];
    metaInfo.IsIgnored = true;
    metaInfo.IsAvailable = false;
}

void TFeaturesLayout::IgnoreExternalFeatures(TConstArrayRef<ui32> ignoredFeatures) noexcept {
    for (auto ignoredFeature : ignoredFeatures) {
        IgnoreExternalFeature(ignoredFeature);
    }
}

TConstArrayRef<ui32> TFeaturesLayout::GetFloatFeatureInternalIdxToExternalIdx() const noexcept {
    return FloatFeatureInternalIdxToExternalIdx;
}

TConstArrayRef<ui32> TFeaturesLayout::GetCatFeatureInternalIdxToExternalIdx() const noexcept {
    return CatFeatureInternalIdxToExternalIdx;
}

TConstArrayRef<ui32> TFeaturesLayout::GetTextFeatureInternalIdxToExternalIdx() const noexcept {
    return TextFeatureInternalIdxToExternalIdx;
}

TConstArrayRef<ui32> TFeaturesLayout::GetEmbeddingFeatureInternalIdxToExternalIdx() const noexcept {
    return EmbeddingFeatureInternalIdxToExternalIdx;
}


const THashMap<TString, TVector<ui32>>& TFeaturesLayout::GetTagToExternalIndices() const noexcept {
    return TagToExternalIndices;
}


bool TFeaturesLayout::EqualTo(const TFeaturesLayout& rhs, bool ignoreSparsity) const {
    if (ExternalIdxToMetaInfo.size() != rhs.ExternalIdxToMetaInfo.size()) {
        return false;
    }
    for (auto i : xrange(ExternalIdxToMetaInfo.size())) {
        if (!ExternalIdxToMetaInfo[i].EqualTo(rhs.ExternalIdxToMetaInfo[i], ignoreSparsity)) {
            return false;
        }
    }

    return std::tie(
            FeatureExternalIdxToInternalIdx,
            CatFeatureInternalIdxToExternalIdx,
            FloatFeatureInternalIdxToExternalIdx,
            TextFeatureInternalIdxToExternalIdx,
            EmbeddingFeatureInternalIdxToExternalIdx
        ) == std::tie(
            rhs.FeatureExternalIdxToInternalIdx,
            rhs.CatFeatureInternalIdxToExternalIdx,
            rhs.FloatFeatureInternalIdxToExternalIdx,
            rhs.TextFeatureInternalIdxToExternalIdx,
            rhs.EmbeddingFeatureInternalIdxToExternalIdx
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

bool TFeaturesLayout::HasAvailableAndNotIgnoredFeatures() const noexcept {
    for (const auto& metaInfo : ExternalIdxToMetaInfo) {
        if (metaInfo.IsAvailable && !metaInfo.IsIgnored) {
            return true;
        }
    }
    return false;
}

void TFeaturesLayout::AddFeature(TFeatureMetaInfo&& featureMetaInfo) {
    const ui32 externalIdx = SafeIntegerCast<ui32>(ExternalIdxToMetaInfo.size());
    switch (featureMetaInfo.Type) {
        case EFeatureType::Float:
            FeatureExternalIdxToInternalIdx.push_back(
                SafeIntegerCast<ui32>(FloatFeatureInternalIdxToExternalIdx.size())
            );
            FloatFeatureInternalIdxToExternalIdx.push_back(externalIdx);
            break;
        case EFeatureType::Categorical:
            FeatureExternalIdxToInternalIdx.push_back(
                SafeIntegerCast<ui32>(CatFeatureInternalIdxToExternalIdx.size())
            );
            CatFeatureInternalIdxToExternalIdx.push_back(externalIdx);
            break;
        case EFeatureType::Text:
            FeatureExternalIdxToInternalIdx.push_back(
                SafeIntegerCast<ui32>(TextFeatureInternalIdxToExternalIdx.size())
            );
            TextFeatureInternalIdxToExternalIdx.push_back(externalIdx);
            break;
        case EFeatureType::Embedding:
            FeatureExternalIdxToInternalIdx.push_back(
                SafeIntegerCast<ui32>(EmbeddingFeatureInternalIdxToExternalIdx.size())
            );
            EmbeddingFeatureInternalIdxToExternalIdx.push_back(externalIdx);
            break;
        default:
            CB_ENSURE(false, "Unexpected feature type");
    }
    ExternalIdxToMetaInfo.push_back(std::move(featureMetaInfo));
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
            "Feature #" << i << " has '" << learnFeatureMetaInfo.Type << "' type in training data, but '"
            << applyFeatureMetaInfo.Type << "' type in " << applyDataName
        );
        CB_ENSURE(
            !learnFeatureMetaInfo.Name || !applyFeatureMetaInfo.Name ||
            (learnFeatureMetaInfo.Name == applyFeatureMetaInfo.Name),
            "Feature #" << i << " has '" << learnFeatureMetaInfo.Name << "' name in training data, but '"
            << applyFeatureMetaInfo.Name << "' name in " << applyDataName
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

void NCB::CheckCompatibleForQuantize(
    const TFeaturesLayout& dataFeaturesLayout,
    const TFeaturesLayout& quantizedFeaturesLayout,
    const TString& dataName
) {
    auto dataFeaturesMetaInfo = dataFeaturesLayout.GetExternalFeaturesMetaInfo();
    auto quantizedFeaturesMetaInfo = quantizedFeaturesLayout.GetExternalFeaturesMetaInfo();

    auto featuresIntersectionSize = Min(dataFeaturesMetaInfo.size(), quantizedFeaturesMetaInfo.size());

    size_t i = 0;
    for (; i < featuresIntersectionSize; ++i) {
        const auto& dataFeatureMetaInfo = dataFeaturesMetaInfo[i];
        const auto& quantizedFeatureMetaInfo = quantizedFeaturesMetaInfo[i];

        if (!dataFeatureMetaInfo.IsAvailable || dataFeatureMetaInfo.IsIgnored) {
            continue;
        }

        // ignored and not available features in quantizedFeaturesData are ok - it means they are constant

        CB_ENSURE(
            dataFeatureMetaInfo.Type == quantizedFeatureMetaInfo.Type,
            "Feature #" << i << " has '" << quantizedFeatureMetaInfo.Type << "' type in quantized info, but '"
            << dataFeatureMetaInfo.Type << "' type in " << dataName
        );
        CB_ENSURE(
            !dataFeatureMetaInfo.Name || !quantizedFeatureMetaInfo.Name ||
            (dataFeatureMetaInfo.Name == quantizedFeatureMetaInfo.Name),
            "Feature #" << i << " has '" << quantizedFeatureMetaInfo.Name << "' name in quantized info, but '"
            << dataFeatureMetaInfo.Name << "' name in " << dataName
        );
    }
    for (; i < dataFeaturesMetaInfo.size(); ++i) {
        CB_ENSURE(
            !dataFeaturesMetaInfo[i].IsAvailable || dataFeaturesMetaInfo[i].IsIgnored,
            "Feature #" << i
            << " is used in " << dataName << ", but not available in quantized info";
        );
    }
}
