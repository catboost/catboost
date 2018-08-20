#include "bind_options.h"
#include "cmd_line.h"
#include "modes.h"

#include <catboost/libs/algo/roc_curve.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/model/model.h>

#include <library/getopt/small/last_getopt.h>

#include <util/system/fs.h>
#include <util/string/iterator.h>


using namespace NCB;


struct TRocParams {
    TString ModelFileName;
    TString OutputPath;
    TPathWithScheme PoolPath;
    NCatboostOptions::TDsvPoolFormatParams DsvPoolFormatParams;
    int ThreadCount = NSystemInfo::CachedNumberOfCpus();

    void BindParserOpts(NLastGetopt::TOpts& parser) {
        parser.AddLongOption('m', "model-path", "path to model")
            .StoreResult(&ModelFileName)
            .DefaultValue("model.bin");
        parser.AddLongOption('f', "pool-path", "learn set path")
            .StoreResult(&PoolPath)
            .RequiredArgument("PATH");
        BindDsvPoolFormatParams(&parser, &DsvPoolFormatParams);
        parser.AddLongOption('o', "output-path", "output result path")
            .StoreResult(&OutputPath)
            .DefaultValue("roc_data.tsv");
        parser.AddLongOption('T', "thread-count", "worker thread count (default: core count)")
            .StoreResult(&ThreadCount);
    }
};

int mode_roc(int argc, const char* argv[]) {
    TRocParams params;

    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    params.BindParserOpts(parser);
    parser.SetFreeArgsNum(0);
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};

    CB_ENSURE(NFs::Exists(params.ModelFileName), "Model file doesn't exist: " << params.ModelFileName);
    TFullModel model = ReadModel(params.ModelFileName);

    TPool pool;
    NCB::ReadPool(
        params.PoolPath,
        /*pairsFilePath=*/NCB::TPathWithScheme(),
        /*groupWeightsFilePath=*/NCB::TPathWithScheme(),
        params.DsvPoolFormatParams,
        /*ignoredFeatures*/ {},
        params.ThreadCount,
        /*verbose=*/false,
        &pool
    );

    NJson::TJsonValue paramsJson = ReadTJsonValue(model.ModelInfo["params"]);
    ELossFunction lossFunction = FromString<ELossFunction>(paramsJson["loss_function"]["type"].GetString());
    CB_ENSURE(IsBinaryClassError(lossFunction), "ROC data evaluated only for binary classification tasks.");

    TRocCurve rocCurve(model, {pool}, params.ThreadCount);
    rocCurve.Output(params.OutputPath);
    return 0;
}
