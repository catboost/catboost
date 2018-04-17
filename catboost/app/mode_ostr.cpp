#include "modes.h"
#include "cmd_line.h"

#include <catboost/libs/documents_importance/docs_importance.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/model/model.h>

#include <library/getopt/small/last_getopt.h>

#include <util/system/fs.h>
#include <util/string/iterator.h>

struct TObjectImportancesParams {
    TString ModelFileName;
    TString OutputPath;
    TString LearnSetPath;
    TString TestSetPath;
    TString CdFile;
    TString LearnPairsPath = "";
    TString UpdateMethod = ToString(EUpdateType::SinglePoint);
    int ThreadCount = NSystemInfo::CachedNumberOfCpus();
    char Delimiter = '\t';
    bool HasHeader = false;

    void BindParserOpts(NLastGetopt::TOpts& parser) {
        parser.AddLongOption('m', "model-path", "path to model")
            .StoreResult(&ModelFileName)
            .DefaultValue("model.bin");
        parser.AddLongOption('f', "learn-set", "learn set path")
            .StoreResult(&LearnSetPath)
            .RequiredArgument("PATH");
        parser.AddLongOption('t', "test-set", "test set path")
            .StoreResult(&TestSetPath)
            .RequiredArgument("PATH");
        parser.AddLongOption("column-description", "column descriptions path")
            .AddLongName("cd")
            .StoreResult(&CdFile)
            .DefaultValue("");
        parser.AddLongOption('o', "output-path", "output result path")
            .StoreResult(&OutputPath)
            .DefaultValue("object_importances.tsv");
        parser.AddLongOption('T', "thread-count", "worker thread count (default: core count)")
            .StoreResult(&ThreadCount);
        parser.AddLongOption("delimiter", "delimiter")
            .DefaultValue("\t")
            .Handler1T<TString>([&](const TString& oneChar) {
                CB_ENSURE(oneChar.size() == 1, "only single char delimiters supported");
                Delimiter = oneChar[0];
            });
        parser.AddLongOption("has-header", "has header flag")
            .NoArgument()
            .StoreValue(&HasHeader, true);
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

    CB_ENSURE(NFs::Exists(params.ModelFileName), "Model file doesn't exist: " << params.ModelFileName);
    TFullModel model = ReadModel(params.ModelFileName);
    TPool trainPool;
    ReadPool(params.CdFile, params.LearnSetPath, params.LearnPairsPath, /*ignoredFeatures=*/{}, params.ThreadCount, /*verbose=*/false, params.Delimiter, params.HasHeader, /*classNames=*/{}, &trainPool);
    TPool testPool;
    ReadPool(params.CdFile, params.TestSetPath, /*pairsFilePath=*/"", /*ignoredFeatures=*/{}, params.ThreadCount, /*verbose=*/false, params.Delimiter, params.HasHeader, /*classNames=*/{}, &testPool);

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
