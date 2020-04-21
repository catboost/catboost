#include "analytical_mode_params.h"

#include <catboost/libs/helpers/exception.h>

#include <library/getopt/small/last_getopt.h>

#include <util/generic/strbuf.h>
#include <util/generic/serialized_enum.h>
#include <util/string/split.h>
#include <util/string/vector.h>
#include <util/system/info.h>


using namespace NCB;

TString NCB::BuildModelFormatHelpMessage() {
    return TString::Join(
        "Alters format of output file for the model. ",
        "Supported values {", GetEnumAllNames<EModelType>(), "} ",
        "Default is ", ToString(EModelType::CatboostBinary), ".");
}

void NCB::TAnalyticalModeCommonParams::BindParserOpts(NLastGetopt::TOpts& parser) {
    DatasetReadingParams.BindParserOpts(&parser);
    BindModelFileParams(&parser, &ModelFileName, &ModelFormat);
    parser.AddLongOption('o', "output-path", "output result path")
        .DefaultValue("output.tsv")
        .Handler1T<TStringBuf>([&](const TStringBuf& pathWithScheme) {
            OutputPath = TPathWithScheme(pathWithScheme, "dsv");
        });

    parser.AddLongOption('T', "thread-count", "worker thread count (default: core count)")
        .StoreResult(&ThreadCount);
}

void NCB::BindModelFileParams(NLastGetopt::TOpts* parser, TString* modelFileName, EModelType* modelFormat) {
    parser->AddLongOption('m', "model-file", "model file name")
            .AddLongName("model-path")
            .RequiredArgument("PATH")
            .StoreResult(modelFileName)
            .Handler1T<TString>([modelFileName, modelFormat](const TString& path) {
                *modelFileName = path;
                *modelFormat = NCatboostOptions::DefineModelFormat(path);

            })
            .DefaultValue("model.bin");
    parser->AddLongOption("model-format")
            .RequiredArgument("model format")
            .StoreResult(modelFormat)
            .Help(BuildModelFormatHelpMessage());
}
