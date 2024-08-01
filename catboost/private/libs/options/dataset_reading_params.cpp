#include "dataset_reading_params.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/data_util/exists_checker.h>

#include <library/cpp/getopt/small/last_getopt.h>

#include <util/generic/strbuf.h>

using namespace NCB;


void NCatboostOptions::TDatasetReadingBaseParams::BindParserOpts(NLastGetopt::TOpts* parser) {
    BindColumnarPoolFormatParams(parser, &ColumnarPoolFormatParams);

    parser->AddLongOption("feature-names-path", "Path to feature names data")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([&](const TStringBuf& pathWithScheme) {
            FeatureNamesPath = TPathWithScheme(pathWithScheme, "dsv");
        });
    parser->AddLongOption("pool-metainfo-path", "Path to JSON file with additional dataset meta information")
        .RequiredArgument("PATH")
        .Handler1T<TStringBuf>([&](const TStringBuf& pathWithScheme) {
            PoolMetaInfoPath = TPathWithScheme(pathWithScheme);
        });
}


void NCatboostOptions::TSingleDatasetReadingParams::BindParserOpts(NLastGetopt::TOpts* parser) {
    NCatboostOptions::TDatasetReadingBaseParams::BindParserOpts(parser);

    parser->AddLongOption("input-path", "input path")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([&](const TStringBuf& pathWithScheme) {
            PoolPath = TPathWithScheme(pathWithScheme, "dsv");
        });
}

void NCatboostOptions::TSingleDatasetReadingParams::ValidatePoolParams() const {
    NCatboostOptions::ValidatePoolParams(PoolPath, ColumnarPoolFormatParams);
}


void NCatboostOptions::TDatasetReadingParams::BindParserOpts(NLastGetopt::TOpts* parser) {
    NCatboostOptions::TSingleDatasetReadingParams::BindParserOpts(parser);

    parser->AddLongOption("input-pairs", "Path to pairs data")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([&](const TStringBuf& pathWithScheme) {
            PairsFilePath = TPathWithScheme(pathWithScheme, "dsv-flat");
        });

    parser->AddLongOption("input-graph", "Path to graph data")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([&](const TStringBuf& pathWithScheme) {
            GraphFilePath = TPathWithScheme(pathWithScheme, "dsv-flat");
        });
}

void NCatboostOptions::TDatasetReadingParams::ValidatePoolParams() const {
    NCatboostOptions::TSingleDatasetReadingParams::ValidatePoolParams();
}

void NCatboostOptions::BindColumnarPoolFormatParams(
    NLastGetopt::TOpts* parser,
    NCatboostOptions::TColumnarPoolFormatParams* columnarPoolFormatParams) {

    parser->AddLongOption("column-description", "[for columnar formats] column description file path")
        .AddLongName("cd")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([columnarPoolFormatParams](const TStringBuf& str) {
            columnarPoolFormatParams->CdFilePath = TPathWithScheme(str, "file");
        });

    parser->AddLongOption("delimiter",
                          "[for dsv format] Learning and training sets delimiter (single char, '<tab>' by default)")
        .RequiredArgument("SYMBOL")
        .Handler1T<TString>([columnarPoolFormatParams](const TString& oneChar) {
            CB_ENSURE(oneChar.size() == 1, "only single char delimiters supported");
            columnarPoolFormatParams->DsvFormat.Delimiter = oneChar[0];
        });

    parser->AddLongOption("has-header", "[for dsv format] Read first line as header")
        .NoArgument()
        .StoreValue(&columnarPoolFormatParams->DsvFormat.HasHeader, true);
    parser->AddLongOption("ignore-csv-quoting")
        .NoArgument()
        .StoreValue(&columnarPoolFormatParams->DsvFormat.IgnoreCsvQuoting, true);
}
