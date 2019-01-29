#include "data_provider.h"

#include <catboost/libs/column_description/cd_parser.h>
#include <catboost/libs/column_description/column.h>
#include <catboost/libs/data_new/data_provider_builders.h>
#include <catboost/libs/data_new/dsv_parser.h>
#include <catboost/libs/data_new/features_layout.h>
#include <catboost/libs/data_new/loader.h>
#include <catboost/libs/data_new/meta_info.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/algorithm.h>
#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>
#include <util/stream/mem.h>
#include <util/string/iterator.h>

static TVector<TColumn> MakeCd(const TStringBuf str, const ui32 columnCount) {
    TMemoryInput input(str.data(), str.size());
    return ReadCD(&input, TCdParserDefaults(EColumn::Num, static_cast<int>(columnCount)));
}

static TVector<TColumn> MakeDefaultColumnsDescription(const ui32 columnsCount) {
    TVector<TColumn> result(columnsCount, {EColumn::Num, TString()});
    result.front().Type = EColumn::Label;
    return result;
}

static TStringBuf StripLeft(TStringBuf s, const char what) noexcept {
    while (!s.empty() && s.front() == what) {
        s = {s.data() + 1, s.size() - 1};
    }
    return s;
}

static TStringBuf StripRight(TStringBuf s, const char what) noexcept {
    while (!s.empty() && s.back() == what) {
        s = {s.data(), s.size() - 1};
    }
    return s;
}

static ui32 GetColumnCount(const TStringBuf dataset, const NCB::TMakeDataProviderFromTextOptions& opts) {
    auto firstLine = StringSplitter(dataset).Split(opts.LineSeparator).begin()->Token();
    if (opts.StripSpacesLeft) {
        firstLine = StripLeft(firstLine, ' ');
    }

    if (opts.StripSpacesRight) {
        firstLine = StripRight(firstLine, ' ');
    }

    return StringSplitter(firstLine).Split(opts.Delimiter).Count();
}

namespace {
    template <EColumn... types>
    struct TDoesTypeMatch {
        bool operator() (const TColumn& description) const {
            bool match = false;
            int dummy[] = {(match |= description.Type == types, 0)... };
            Y_UNUSED(dummy);
            return match;
        }
    };
}

TVector<ui32> GetCategoricalFeatureIndices(const TConstArrayRef<TColumn> columnsDescription) {
    TVector<ui32> catFeatureIndices;
    ui32 flatFeatureIdx = 0;
    for (const auto& columnDescription : columnsDescription) {
        if (EColumn::Categ == columnDescription.Type) {
            catFeatureIndices.push_back(flatFeatureIdx);
            ++flatFeatureIdx;
        }
    }
    return catFeatureIndices;
}

TVector<TString> GetFeatureIds(const TConstArrayRef<TColumn> columnsDescription) {
    TVector<TString> featureIds;
    ui32 flatFeatureIdx = 0;
    for (const auto& columnDescription : columnsDescription) {
        if (EColumn::Num == columnDescription.Type || EColumn::Categ == columnDescription.Type) {
            featureIds.push_back(columnDescription.Id);
            ++flatFeatureIdx;
        }
    }
    return featureIds;
}

NCB::TDataMetaInfo MakeMetaInfo(const TConstArrayRef<TColumn> columnsDescription) {
    NCB::TDataMetaInfo metaInfo;
    metaInfo.HasTarget = CountIf(columnsDescription, TDoesTypeMatch<EColumn::Label>());
    metaInfo.BaselineCount = CountIf(columnsDescription, TDoesTypeMatch<EColumn::Baseline>());
    metaInfo.HasGroupId = CountIf(columnsDescription, TDoesTypeMatch<EColumn::GroupId>());
    metaInfo.HasGroupWeight = CountIf(columnsDescription, TDoesTypeMatch<EColumn::GroupWeight>());
    metaInfo.HasSubgroupIds = CountIf(columnsDescription, TDoesTypeMatch<EColumn::SubgroupId>());
    metaInfo.HasWeights = CountIf(columnsDescription, TDoesTypeMatch<EColumn::Weight>());
    metaInfo.HasTimestamp = CountIf(columnsDescription, TDoesTypeMatch<EColumn::Timestamp>());

    // TODO(yazevnul): support pairs
    metaInfo.HasPairs = false;

    metaInfo.ColumnsInfo = NCB::TDataColumnsMetaInfo();
    metaInfo.ColumnsInfo->Columns.assign(
        columnsDescription.begin(),
        columnsDescription.end());

    metaInfo.FeaturesLayout = MakeIntrusive<NCB::TFeaturesLayout>(
        static_cast<ui32>(CountIf(columnsDescription, TDoesTypeMatch<EColumn::Num, EColumn::Categ>())),
        GetCategoricalFeatureIndices(columnsDescription),
        GetFeatureIds(columnsDescription),
        nullptr);

    return metaInfo;
}

NCB::TDataProviderPtr NCB::MakeDataProviderFromText(
    const TStringBuf columnsDescriptionStr,
    TStringBuf datasetStr,
    const TMakeDataProviderFromTextOptions& opts)
{
    const auto isEmptyDataset = static_cast<size_t>(
        Count(datasetStr, opts.Delimiter) + Count(datasetStr, opts.LineSeparator) + Count(datasetStr, ' ')) ==
        datasetStr.size();
    CB_ENSURE(!isEmptyDataset, "dataset can't be empty");

    datasetStr = StripLeft(StripRight(datasetStr, ' '), ' ');
    datasetStr = StripLeft(StripRight(datasetStr, opts.LineSeparator), opts.LineSeparator);

    const ui32 columnCount = GetColumnCount(datasetStr, opts);
    const ui32 objectCount = StringSplitter(datasetStr).Split(opts.LineSeparator).Count();
    const auto columnsDescription = columnsDescriptionStr
        ? MakeCd(columnsDescriptionStr, columnCount)
        : MakeDefaultColumnsDescription(columnCount);
    const auto metaInfo = MakeMetaInfo(columnsDescription);

    TVector<float> numFeaturesBuffer(CountIf(columnsDescription, TDoesTypeMatch<EColumn::Num>()));
    TVector<ui32> catFeaturesBuffer(CountIf(columnsDescription, TDoesTypeMatch<EColumn::Categ>()));
    TVector<bool> featureIgnored(CountIf(columnsDescription, TDoesTypeMatch<EColumn::Num, EColumn::Categ>()));
    auto provider = NCB::CreateDataProvider<IRawObjectsOrderDataVisitor>([&](IRawObjectsOrderDataVisitor* const visitor) {
        visitor->Start(false, metaInfo, objectCount, NCB::EObjectsOrder::Undefined, {});
        visitor->StartNextBlock(objectCount);

        TDsvLineParser parser(
            opts.Delimiter,
            columnsDescription,
            featureIgnored,
            metaInfo.FeaturesLayout.Get(),
            numFeaturesBuffer,
            catFeaturesBuffer,
            visitor);
        ui32 lineIdx = 0;
        TMaybe<TDsvLineParser::TErrorContext> errorContext;
        for (TStringBuf line : StringSplitter(datasetStr).Split(opts.LineSeparator)) {
            if (opts.StripSpacesLeft) {
                line = StripLeft(line, ' ');
            }

            if (opts.StripSpacesRight) {
                line = StripRight(line, ' ');
            }

            if (auto errCtx = parser.Parse(line, lineIdx)) {
                errorContext = std::move(errCtx);
                break;
            }
            ++lineIdx;
        }

        if (errorContext) {
            ythrow TDsvLineParser::MakeException(errorContext.GetRef()) << ' ' << LabeledOutput(lineIdx);
        }

        visitor->Finish();
    });

    return provider;
}
