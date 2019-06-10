#include "modes.h"

#include <catboost/libs/data_new/load_data.h>
#include <catboost/libs/documents_importance/docs_importance.h>
#include <catboost/libs/documents_importance/enums.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/options/analytical_mode_params.h>

#include <library/getopt/small/last_getopt.h>

#include <util/stream/fwd.h>
#include <util/system/info.h>


using namespace NCB;


struct TObjectImportancesParams {
    TString ModelFileName;
    EModelType ModelFormat = EModelType::CatboostBinary;
    TString OutputPath;
    TPathWithScheme LearnSetPath;
    TPathWithScheme TestSetPath;
    NCatboostOptions::TDsvPoolFormatParams DsvPoolFormatParams;
    TString UpdateMethod = ToString(EUpdateType::SinglePoint);
    int ThreadCount = NSystemInfo::CachedNumberOfCpus();
    char Delimiter = '\t';
    bool HasHeader = false;

    void BindParserOpts(NLastGetopt::TOpts& parser) {
        NCB::BindModelFileParams(&parser, &ModelFileName, &ModelFormat);
        parser.AddLongOption('f', "learn-set", "learn set path")
            .Required()
            .StoreResult(&LearnSetPath)
            .RequiredArgument("PATH");
        parser.AddLongOption('t', "test-set", "test set path")
            .Required()
            .StoreResult(&TestSetPath)
            .RequiredArgument("PATH");
        BindDsvPoolFormatParams(&parser, &DsvPoolFormatParams);
        parser.AddLongOption('o', "output-path", "output result path")
            .StoreResult(&OutputPath)
            .DefaultValue("object_importances.tsv");
        parser.AddLongOption('T', "thread-count", "worker thread count (default: core count)")
            .StoreResult(&ThreadCount);
        parser.AddLongOption("update-method", "Should be one of: SinglePoint, TopKLeaves, AllPoints or TopKLeaves:top=2 to set the top size in TopKLeaves method.")
            .StoreResult(&UpdateMethod)
            .DefaultValue("SinglePoint");
    }
};

int mode_ostr(int argc, const char* argv[]) {
    TObjectImportancesParams params;

    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    params.BindParserOpts(parser);
    parser.SetFreeArgsNum(0);
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};

    NCatboostOptions::ValidatePoolParams(params.LearnSetPath, params.DsvPoolFormatParams);
    NCatboostOptions::ValidatePoolParams(params.TestSetPath, params.DsvPoolFormatParams);

    TFullModel model = ReadModel(params.ModelFileName, params.ModelFormat);

    //TODO(eermishkina): support non symmetric trees
    CB_ENSURE(model.IsOblivious(), "Object importance is supported only for symmetric trees");

    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(params.ThreadCount - 1);

    NCB::TDataProviderPtr trainPool = NCB::ReadDataset(params.LearnSetPath,
                                                       /*pairsFilePath=*/NCB::TPathWithScheme(),
                                                       /*groupWeightsFilePath=*/NCB::TPathWithScheme(),
                                                       /*baselineFilePath=*/NCB::TPathWithScheme(),
                                                       params.DsvPoolFormatParams,
                                                       /*ignoredFeatures*/ {},
                                                       EObjectsOrder::Undefined,
                                                       TDatasetSubset::MakeColumns(),
                                                       /*classNames=*/Nothing(),
                                                       &localExecutor);

    NCB::TDataProviderPtr testPool = NCB::ReadDataset(params.TestSetPath,
                                                      /*pairsFilePath=*/NCB::TPathWithScheme(),
                                                      /*groupWeightsFilePath=*/NCB::TPathWithScheme(),
                                                      /*baselineFilePath=*/NCB::TPathWithScheme(),
                                                      params.DsvPoolFormatParams,
                                                      /*ignoredFeatures*/ {},
                                                      EObjectsOrder::Undefined,
                                                      TDatasetSubset::MakeColumns(),
                                                      /*classNames=*/Nothing(),
                                                      &localExecutor);

    CB_ENSURE(model.ModelInfo.contains("params"), "Need model with params to calculate object importances");
    CB_ENSURE(model.GetLossFunctionName(), "Optimized objective must be known to calculate object importances. It is not present in the model.");
    TDStrResult results = GetDocumentImportances(
        model,
        *trainPool,
        *testPool,
        /*dstrTypeStr=*/ToString(EDocumentStrengthType::Raw),
        /*topSize=*/-1,
        params.UpdateMethod,
        /*importanceValuesSignStr=*/ToString(EImportanceValuesSign::All),
        params.ThreadCount
    );

    TFileOutput output(params.OutputPath);
    for (const auto& row : results.Scores) {
        for (double value : row) {
            output.Write(ToString(value) + '\t');
        }
        output.Write('\n');
    }
    return 0;
}
