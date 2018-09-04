#include "modes.h"
#include "bind_options.h"
#include "cmd_line.h"

#include <catboost/libs/documents_importance/docs_importance.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/options/output_file_options.h>

#include <catboost/libs/model/model.h>

#include <library/getopt/small/last_getopt.h>

#include <util/system/fs.h>
#include <util/string/iterator.h>


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
        BindModelFileParams(&parser, &ModelFileName, &ModelFormat);
        parser.AddLongOption('f', "learn-set", "learn set path")
            .StoreResult(&LearnSetPath)
            .RequiredArgument("PATH");
        parser.AddLongOption('t', "test-set", "test set path")
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

    TFullModel model = ReadModel(params.ModelFileName, params.ModelFormat);

    TPool trainPool;
    NCB::ReadPool(params.LearnSetPath,
                  /*pairsFilePath=*/NCB::TPathWithScheme(),
                  /*groupWeightsFilePath=*/NCB::TPathWithScheme(),
                  params.DsvPoolFormatParams,
                  /*ignoredFeatures*/ {},
                  params.ThreadCount,
                  /*verbose=*/false,
                  &trainPool);

    TPool testPool;
    NCB::ReadPool(params.TestSetPath,
                  /*pairsFilePath=*/NCB::TPathWithScheme(),
                  /*groupWeightsFilePath=*/NCB::TPathWithScheme(),
                  params.DsvPoolFormatParams,
                  /*ignoredFeatures=*/{},
                  params.ThreadCount,
                  /*verbose=*/false,
                  &testPool);

    TDStrResult results = GetDocumentImportances(
        model,
        trainPool,
        testPool,
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
