#include "cb_dsv_loader.h"
#include "dsv_parser.h"

#include <catboost/libs/column_description/cd_parser.h>
#include <catboost/libs/data_util/exists_checker.h>
#include <catboost/libs/helpers/mem_usage.h>

#include <library/object_factory/object_factory.h>

#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>
#include <util/generic/vector.h>
#include <util/stream/file.h>
#include <util/string/iterator.h>
#include <util/string/split.h>
#include <util/system/types.h>


namespace NCB {

    TCBDsvDataLoader::TCBDsvDataLoader(TDatasetLoaderPullArgs&& args)
        : TCBDsvDataLoader(
            TLineDataLoaderPushArgs {
                GetLineDataReader(args.PoolPath, args.CommonArgs.PoolFormat),
                std::move(args.CommonArgs)
            }
        )
    {
    }

    TCBDsvDataLoader::TCBDsvDataLoader(TLineDataLoaderPushArgs&& args)
        : TAsyncProcDataLoaderBase<TString>(std::move(args.CommonArgs))
        , FieldDelimiter(Args.PoolFormat.Delimiter)
        , LineDataReader(std::move(args.Reader))
    {
        CB_ENSURE(!Args.PairsFilePath.Inited() || CheckExists(Args.PairsFilePath),
                  "TCBDsvDataLoader:PairsFilePath does not exist");
        CB_ENSURE(!Args.GroupWeightsFilePath.Inited() || CheckExists(Args.GroupWeightsFilePath),
                  "TCBDsvDataLoader:GroupWeightsFilePath does not exist");

        TMaybe<TString> header = LineDataReader->GetHeader();
        TMaybe<TVector<TString>> headerColumns;
        if (header) {
            headerColumns = TVector<TString>(StringSplitter(*header).Split(FieldDelimiter));
        }

        TString firstLine;
        CB_ENSURE(LineDataReader->ReadLine(&firstLine), "TCBDsvDataLoader: no data rows in pool");
        const ui32 columnsCount = StringSplitter(firstLine).Split(FieldDelimiter).Count();

        auto columnsDescription = TDataColumnsMetaInfo{ CreateColumnsDescription(columnsCount) };
        auto featureIds = columnsDescription.GenerateFeatureIds(headerColumns);

        DataMetaInfo = TDataMetaInfo(
            std::move(columnsDescription),
            Args.GroupWeightsFilePath.Inited(),
            Args.PairsFilePath.Inited(),
            &featureIds
        );

        AsyncRowProcessor.AddFirstLine(std::move(firstLine));

        ProcessIgnoredFeaturesList(Args.IgnoredFeatures, &DataMetaInfo, &FeatureIgnored);

        AsyncRowProcessor.ReadBlockAsync(GetReadFunc());
    }

    TVector<TColumn> TCBDsvDataLoader::CreateColumnsDescription(ui32 columnsCount) {
        return Args.CdProvider->GetColumnsDescription(columnsCount);
    }


    void TCBDsvDataLoader::StartBuilder(bool inBlock,
                                          ui32 objectCount, ui32 /*offset*/,
                                          IRawObjectsOrderDataVisitor* visitor)
    {
        visitor->Start(inBlock, DataMetaInfo, objectCount, Args.ObjectsOrder, {});
    }


    void TCBDsvDataLoader::ProcessBlock(IRawObjectsOrderDataVisitor* visitor) {
        visitor->StartNextBlock(AsyncRowProcessor.GetParseBufferSize());

        auto parseBlock = [&](TString& line, int inBlockIdx) {
            const auto* const featuresLayout = DataMetaInfo.FeaturesLayout.Get();

            TVector<float> floatFeatures;
            floatFeatures.yresize(featuresLayout->GetFloatFeatureCount());

            TVector<ui32> catFeatures;
            catFeatures.yresize(featuresLayout->GetCatFeatureCount());

            TDsvLineParser parser(
                FieldDelimiter,
                DataMetaInfo.ColumnsInfo->Columns,
                FeatureIgnored,
                featuresLayout,
                floatFeatures,
                catFeatures,
                visitor);

            if (const auto errCtx = parser.Parse(line, inBlockIdx)) {
                const auto lineIdx = AsyncRowProcessor.GetLinesProcessed() + inBlockIdx + 1;
                ythrow TDsvLineParser::MakeException(errCtx.GetRef()) << "; " << LabeledOutput(lineIdx);
            }
        };

        AsyncRowProcessor.ProcessBlock(parseBlock);
    }

    namespace {
        TDatasetLoaderFactory::TRegistrator<TCBDsvDataLoader> DefDataLoaderReg("");
        TDatasetLoaderFactory::TRegistrator<TCBDsvDataLoader> CBDsvDataLoaderReg("dsv");
    }
}

