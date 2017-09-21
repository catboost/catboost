#include "pool_converter.h"
#include <catboost/cuda/data/binarization_config.h>
#include <library/protobuf/protofile/protofile.h>
#include <library/getopt/opt.h>
#include <library/threading/local_executor/local_executor.h>

struct TPoolConvertConfig {
    TString FeaturesPath;
    TString ColumnDescriptionPath;
    TString TempDir = "tmp";

    TString Output;

    ui32 NumThreads = 4;

    TString InputGridFile = "";
    TString OutputGridFile = "";

    bool DontBinarizeIt = false;
    TBinarizationConfiguration Binarization;
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

    parser.AddLongOption('x', "discretization", "Discretization for binarization")
        .Optional()
        .RequiredArgument()
        .StoreResult(&settings.Binarization.DefaultFloatBinarization.Discretization);

    parser.AddLongOption('g', "grid", "Grid")
        .Optional()
        .RequiredArgument()
        .StoreResult(&settings.Binarization.DefaultFloatBinarization.BorderSelectionType);

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

    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};
}

int DoMain(int argc, const char** argv) {
    TPoolConvertConfig config;
    ParseCommandLine(argc, argv, config);

    NPar::LocalExecutor().RunAdditionalThreads(config.NumThreads);
    TOnCpuGridBuilderFactory gridBuilderFactory;

    TCatBoostProtoPoolConverter converter(config.FeaturesPath, config.ColumnDescriptionPath, config.TempDir);

    if (!config.DontBinarizeIt && config.InputGridFile.empty()) {
        converter.SetGridBuilderFactory(gridBuilderFactory)
            .SetBinarization(config.Binarization);
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
