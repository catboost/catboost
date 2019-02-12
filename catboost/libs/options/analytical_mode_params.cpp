#include "analytical_mode_params.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/strbuf.h>
#include <util/generic/serialized_enum.h>
#include <util/string/iterator.h>
#include <util/string/vector.h>
#include <util/system/info.h>


using namespace NCB;

TString NCB::BuildModelFormatHelpMessage() {
    return TString::Join(
        "Alters format of output file for the model. ",
        "Supported values {", GetEnumAllNames<EModelType>(), "}",
        "Default is ", ToString(EModelType::CatboostBinary), ".");
}

void NCB::TAnalyticalModeCommonParams::BindParserOpts(NLastGetopt::TOpts& parser) {
    BindDsvPoolFormatParams(&parser, &DsvPoolFormatParams);
    BindModelFileParams(&parser, &ModelFileName, &ModelFormat);
    parser.AddLongOption("input-path", "input path")
        .DefaultValue("input.tsv")
        .Handler1T<TStringBuf>([&](const TStringBuf& pathWithScheme) {
            InputPath = TPathWithScheme(pathWithScheme, "dsv");
        });
    parser.AddLongOption('o', "output-path", "output result path")
        .DefaultValue("output.tsv")
        .Handler1T<TStringBuf>([&](const TStringBuf& pathWithScheme) {
            OutputPath = TPathWithScheme(pathWithScheme, "dsv");
        });
    parser.AddLongOption("input-pairs", "PATH")
        .Handler1T<TStringBuf>([&](const TStringBuf& pathWithScheme) {
            PairsFilePath = TPathWithScheme(pathWithScheme, "dsv");
        });

    parser.AddLongOption('T', "thread-count", "worker thread count (default: core count)")
        .StoreResult(&ThreadCount);
}

void NCB::BindDsvPoolFormatParams(
    NLastGetopt::TOpts* parser,
    NCatboostOptions::TDsvPoolFormatParams* dsvPoolFormatParams) {

    parser->AddLongOption("column-description", "[for dsv format] column description file path")
        .AddLongName("cd")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([dsvPoolFormatParams](const TStringBuf& str) {
            dsvPoolFormatParams->CdFilePath = TPathWithScheme(str, "file");
        });

    parser->AddLongOption("delimiter",
        "[for dsv format] Learning and training sets delimiter (single char, '<tab>' by default)")
        .RequiredArgument("SYMBOL")
        .Handler1T<TString>([dsvPoolFormatParams](const TString& oneChar) {
            CB_ENSURE(oneChar.size() == 1, "only single char delimiters supported");
            dsvPoolFormatParams->Format.Delimiter = oneChar[0];
        });

    parser->AddLongOption("has-header", "[for dsv format] Read first line as header")
        .NoArgument()
        .StoreValue(&dsvPoolFormatParams->Format.HasHeader,
                    true);
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
