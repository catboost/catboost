#include "bind_options.h"
#include "cmd_line.h"
#include "modes.h"

#include <catboost/libs/algo/roc_curve.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/options/output_file_options.h>

#include <library/getopt/small/last_getopt.h>

#include <util/system/fs.h>
#include <util/string/iterator.h>


using namespace NCB;


struct TRocParams {
    TString ModelFileName;
    EModelType ModelFormat = EModelType::CatboostBinary;
    TString OutputPath;
    TPathWithScheme PoolPath;
    NCatboostOptions::TDsvPoolFormatParams DsvPoolFormatParams;
    int ThreadCount = NSystemInfo::CachedNumberOfCpus();

    void BindParserOpts(NLastGetopt::TOpts& parser) {
        BindModelFileParams(&parser, &ModelFileName, &ModelFormat);
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

    TFullModel model = ReadModel(params.ModelFileName, params.ModelFormat);

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
