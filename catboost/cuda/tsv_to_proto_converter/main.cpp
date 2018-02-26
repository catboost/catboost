#include "pool_converter.h"
#include <library/protobuf/protofile/protofile.h>
#include <library/getopt/opt.h>
#include <library/threading/local_executor/local_executor.h>
#include <catboost/libs/options/binarization_options.h>

using namespace NCatboostCuda;

struct TPoolConvertConfig {
    TString FeaturesPath;
    TString ColumnDescriptionPath;
    TString TempDir = "tmp";

    TString Output;

    ui32 NumThreads = 4;

    TString InputGridFile = "";
    TString OutputGridFile = "";

    bool DontBinarizeIt = false;
    NCatboostOptions::TBinarizationOptions FloatBinarization;
};

void ParseCommandLine(int argc, const char** argv, TPoolConvertConfig& settings) {
    auto parser = NLastGetopt::TOpts();
    parser.AddVersionOption('v');
    parser.AddHelpOption();

    parser.AddLongOption('f', "features", "features file")
        .Required()
        .RequiredArgument()
        .StoreResult(&settings.FeaturesPath);

    parser.AddLongOption("cd", "column description file")
        .Required()
        .RequiredArgument()
        .StoreResult(&settings.ColumnDescriptionPath);

    parser.AddLongOption("tmp", "temp dir")
        .Optional()
        .RequiredArgument()
        .StoreResult(&settings.TempDir);

    parser.AddLongOption('o', "output-file", "output")
        .Required()
        .RequiredArgument()
        .StoreResult(&settings.Output);

    parser.AddLongOption("dont-binarize", "Don't binarize pool")
        .Optional()
        .SetFlag(&settings.DontBinarizeIt);

    parser.AddLongOption("input-grid", "Dump grid (borders and catfeaure index)")
        .Optional()
        .RequiredArgument()
        .StoreResult(&settings.InputGridFile);

    parser.AddLongOption("save-grid", "Dump grid (borders and catfeaure index)")
        .Optional()
        .RequiredArgument()
        .StoreResult(&settings.OutputGridFile);

    parser.AddLongOption('T', "threads", "Num threads")
        .Optional()
        .RequiredArgument()
        .StoreResult(&settings.NumThreads);

    parser.AddLongOption('x', "border-count", "count of borders per float feature. Should be in range [1, 255]")
        .RequiredArgument("int")
        .Handler1T<int>([&](int count) {
            settings.FloatBinarization.BorderCount = count;
        });

    parser.AddLongOption("feature-border-type",
                         "Should be one of: Median, GreedyLogSum, UniformAndQuantiles, MinEntropy, MaxLogSum")
        .RequiredArgument("border-type")
        .Handler1T<TString>([&](const TString& type) {
            settings.FloatBinarization.BorderSelectionType = FromString<EBorderSelectionType>(type);
        });

    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};
}

int DoMain(int argc, const char** argv) {
    TPoolConvertConfig config;
    ParseCommandLine(argc, argv, config);
    SetLogingLevel(ELoggingLevel::Debug);
    NPar::LocalExecutor().RunAdditionalThreads(config.NumThreads);
    TOnCpuGridBuilderFactory gridBuilderFactory;

    TCatBoostProtoPoolConverter converter(config.FeaturesPath, config.ColumnDescriptionPath, config.TempDir);

    if (!config.DontBinarizeIt && config.InputGridFile.empty()) {
        converter.SetGridBuilderFactory(gridBuilderFactory)
            .SetBinarization(config.FloatBinarization);
    }
    if (config.InputGridFile.size()) {
        converter.SetInputBinarizationFile(config.InputGridFile);
    }

    Y_ENSURE(config.Output.size(), "Error: provide output file");
    converter.SetOutputFile(config.Output);

    if (config.OutputGridFile.size()) {
        converter.SetOutputBinarizationFile(config.OutputGridFile);
    }

    converter.Convert();

    return 0;
}

int main(int argc, const char** argv) {
#ifdef NDEBUG
    try {
#endif
        return DoMain(argc, argv);
#ifdef NDEBUG
    } catch (...) {
        Cerr << CurrentExceptionMessage() << Endl;
        return 1;
    }
#endif
}
