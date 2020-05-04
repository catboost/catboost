#include "dataset_reading_params.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/data_util/exists_checker.h>

#include <library/getopt/small/last_getopt.h>

#include <util/generic/strbuf.h>

using namespace NCB;

void NCatboostOptions::TDatasetReadingParams::BindParserOpts(NLastGetopt::TOpts* parser) {
    BindColumnarPoolFormatParams(parser, &ColumnarPoolFormatParams);
    parser->AddLongOption("input-path", "input path")
        .DefaultValue("input.tsv")
        .Handler1T<TStringBuf>([&](const TStringBuf& pathWithScheme) {
            PoolPath = TPathWithScheme(pathWithScheme, "dsv");
        });
    parser->AddLongOption("input-pairs", "PATH")
        .Handler1T<TStringBuf>([&](const TStringBuf& pathWithScheme) {
            PairsFilePath = TPathWithScheme(pathWithScheme, "dsv");
        });
    parser->AddLongOption("feature-names-path", "PATH")
        .Handler1T<TStringBuf>([&](const TStringBuf& pathWithScheme) {
            FeatureNamesPath = TPathWithScheme(pathWithScheme, "dsv");
        });
}

void NCatboostOptions::TDatasetReadingParams::ValidatePoolParams() const {
    NCatboostOptions::ValidatePoolParams(PoolPath, ColumnarPoolFormatParams);
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
