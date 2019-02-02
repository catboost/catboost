#include "dsv_parser.h"
#include "features_layout.h"
#include "loader.h"
#include "visitor.h"

#include <catboost/libs/column_description/column.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/scope.h>
#include <util/stream/labeled.h>
#include <util/string/cast.h>
#include <util/string/escape.h>
#include <util/string/iterator.h>

NCB::TDsvLineParser::TDsvLineParser(
    char delimiter,
    TConstArrayRef<TColumn> columnDescriptions,
    TConstArrayRef<bool> featureIgnored,
    const TFeaturesLayout* featuresLayout,
    TArrayRef<float> numericFeaturesBuffer,
    TArrayRef<ui32> categoricalFeaturesBuffer,
    IRawObjectsOrderDataVisitor* visitor)
    : Delimiter_(delimiter)
    , ColumnDescriptions_(columnDescriptions)
    , FeatureIgnored_(featureIgnored)
    , FeaturesLayout_(featuresLayout)
    , NumericFeaturesBuffer_(numericFeaturesBuffer)
    , CategoricalFeaturesBuffer_(categoricalFeaturesBuffer)
    , Visitor_(visitor) {
}

TCatBoostException NCB::TDsvLineParser::MakeException(const TErrorContext& ctx) {
    return TCatBoostException() << LabeledOutput(
        ctx.Type,
        EscapeC(ctx.Token),
        ctx.ColumnIdx,
        ctx.FlatFeatureIdx,
        ctx.ColumnType);
}

TMaybe<NCB::TDsvLineParser::TErrorContext> NCB::TDsvLineParser::HandleToken(
    const TStringBuf token,
    const ui32 inBlockIdx,
    const ui32 columnIdx,
    ui32* const flatFeatureIdxPtr,
    ui32* const baselineIdxPtr)
{
    auto& flatFeatureIdx = *flatFeatureIdxPtr;
    auto& baselineIdx = *baselineIdxPtr;
    const auto columnType = ColumnDescriptions_[columnIdx].Type;

    if (token.empty()) {
        const auto isNonEmptyColumnType = EColumn::Label == columnType ||
            EColumn::Weight == columnType ||
            EColumn::GroupId == columnType ||
            EColumn::GroupWeight == columnType ||
            EColumn::SubgroupId == columnType ||
            EColumn::Baseline == columnType ||
            EColumn::Timestamp == columnType;
        if (isNonEmptyColumnType) {
            return TErrorContext{EErrorType::EmptyToken, {}, columnIdx, {}, columnType};
        }
    }

    // We don't check if `flatFeatureIdx` is in a valid range because we expect
    // `ColumnDescriptions_` and `FeatureIgnored_` to be consistent

    switch (columnType) {
        case EColumn::Categ: {
            if (!FeatureIgnored_[flatFeatureIdx]) {
                const ui32 catFeatureIdx = FeaturesLayout_->GetInternalFeatureIdx(flatFeatureIdx);
                CategoricalFeaturesBuffer_[catFeatureIdx] = Visitor_->GetCatFeatureValue(flatFeatureIdx, token);
            }
            ++flatFeatureIdx;
            break;
        } case EColumn::Num: {
            if (!FeatureIgnored_[flatFeatureIdx]) {
                const auto numFeatureIdx = FeaturesLayout_->GetInternalFeatureIdx(flatFeatureIdx);
                if (!TryParseFloatFeatureValue(token, &NumericFeaturesBuffer_[numFeatureIdx])) {
                    return TErrorContext{
                        EErrorType::FailedToParseNumericFeature,
                        TString(token),
                        columnIdx,
                        flatFeatureIdx,
                        columnType};
                }
            }
            ++flatFeatureIdx;
            break;
        } case EColumn::Weight: {
            if (float weight; TryFromString(token, weight)) {
                Visitor_->AddWeight(inBlockIdx, weight);
            } else {
                return TErrorContext{EErrorType::FailedToParseFloat, TString(token), columnIdx, {}, columnType};
            }
            break;
        } case EColumn::Label: {
            Visitor_->AddTarget(inBlockIdx, TString(token));
            break;
        } case EColumn::GroupId: {
            Visitor_->AddGroupId(inBlockIdx, CalcGroupIdFor(token));
            break;
        } case EColumn::GroupWeight: {
            if (float groupWeight; TryFromString(token, groupWeight)) {
                Visitor_->AddGroupWeight(inBlockIdx, groupWeight);
            } else {
                return TErrorContext{EErrorType::FailedToParseFloat, TString(token), columnIdx, {}, columnType};
            }
            break;
        } case EColumn::SubgroupId: {
            Visitor_->AddSubgroupId(inBlockIdx, CalcSubgroupIdFor(token));
            break;
        } case EColumn::Baseline: {
            if (float baseline; TryFromString(token, baseline)) {
                Visitor_->AddBaseline(inBlockIdx, baselineIdx, baseline);
            } else {
                return TErrorContext{EErrorType::FailedToParseFloat, TString(token), columnIdx, {}, columnType};
            }
            ++baselineIdx;
            break;
        } case EColumn::Timestamp: {
            if (ui64 timestamp; TryFromString(token, timestamp)) {
                Visitor_->AddTimestamp(inBlockIdx, timestamp);
            } else {
                return TErrorContext{EErrorType::FailedToParseFloat, TString(token), columnIdx, {}, columnType};
            }
            break;
        } case EColumn::Auxiliary:
          case EColumn::SampleId: {
            break;
        } case EColumn::Sparse:
          case EColumn::Prediction: {
            return TErrorContext{EErrorType::ColumnTypeIsNotSupported, TString(token), columnIdx, {}, columnType};
        }
    }

    return {};
}

TMaybe<NCB::TDsvLineParser::TErrorContext> NCB::TDsvLineParser::Parse(
    const TStringBuf line,
    const ui32 inBlockIdx)
{
    ui32 flatFeatureIdx = 0;
    ui32 baselineIdx = 0;
    ui32 columnIdx = 0;
    for (const TStringBuf token : StringSplitter(line).Split(Delimiter_)) {
        Y_DEFER { ++columnIdx; };

        if (columnIdx >= ColumnDescriptions_.size()) {
            return TErrorContext{EErrorType::TooManyColumns, {}, columnIdx, {}, {}};
        }

        if (auto errorContext = HandleToken(token, inBlockIdx, columnIdx, &flatFeatureIdx, &baselineIdx)) {
            return errorContext;
        }
    }

    if (columnIdx != ColumnDescriptions_.size()) {
        return TErrorContext{EErrorType::NotEnoughColumns, {}, columnIdx, {}, {}};
    }

    if (NumericFeaturesBuffer_) {
        Visitor_->AddAllFloatFeatures(inBlockIdx, NumericFeaturesBuffer_);
    }

    if (CategoricalFeaturesBuffer_) {
        Visitor_->AddAllCatFeatures(inBlockIdx, CategoricalFeaturesBuffer_);
    }

    return {};
}
