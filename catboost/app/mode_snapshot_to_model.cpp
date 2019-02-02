#include "modes.h"
#include "bind_options.h"

#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/algo/split.h>
#include <catboost/libs/data_new/quantized_features_info.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/model/model_build_helper.h>

#include <library/getopt/small/last_getopt.h>


struct TLoadSnapshotParams {
    TString SnapshotPath;
    TString ModelPath;

    void BindParserOpts(NLastGetopt::TOpts& parser) {
        parser.AddLongOption('s', "snapshot-file", "path to load snapshot from")
                .StoreResult(&SnapshotPath)
                .RequiredArgument("PATH");
        parser.AddLongOption('m', "model-file", "path to save model at")
                .StoreResult(&ModelPath)
                .RequiredArgument("PATH");
    }
};


TLearnProgress LoadSnapshot(const TString& snapshotPath) {
    CB_ENSURE(NFs::Exists(snapshotPath), "Snapshot file doesn't exist: " << snapshotPath);
    try {
        TLearnProgress learnProgress;
        TProgressHelper(ToString(ETaskType::CPU)).CheckedLoad(snapshotPath, [&](TIFStream* in) {
            TRestorableFastRng64 rand(0);
            ::Load(in, rand);
            learnProgress.Load(in, true);
        });
        CATBOOST_INFO_LOG << "Snapshot is loaded from: " << snapshotPath << Endl;
        return learnProgress;
    } catch (...) {
        CATBOOST_ERROR_LOG << "Can't load progress from snapshot file: " << snapshotPath << Endl;
        throw;
    }
}


TObliviousTrees BuildObliviousTrees(const TLearnProgress& learnProgress) {
    try {
        TObliviousTreeBuilder builder(learnProgress.FloatFeatures, learnProgress.CatFeatures,
                                      learnProgress.ApproxDimension);
        for (size_t treeId = 0; treeId < learnProgress.TreeStruct.size(); ++treeId) {
            TVector <TModelSplit> modelSplits;
            for (const auto &split : learnProgress.TreeStruct[treeId].Splits) {
                auto modelSplit = split.GetModelSplit(learnProgress, NCB::TPerfectHashedToHashedCatValuesMap(), TCtrHelper());
                modelSplits.push_back(modelSplit);
            }
            builder.AddTree(modelSplits, learnProgress.LeafValues[treeId],
                            learnProgress.TreeStats[treeId].LeafWeightsSum);
        }
        TObliviousTrees obliviousTrees = builder.Build();
        CATBOOST_DEBUG_LOG << "TObliviousTrees is built" << Endl;
        return obliviousTrees;
    } catch (...) {
        CATBOOST_ERROR_LOG << "Can't build TObliviousTrees" << Endl;
        throw;
    }
}


int mode_snapshot_to_model(int argc, const char* argv[]) {
    TLoadSnapshotParams params;
    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    params.BindParserOpts(parser);
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};

    TLearnProgress learnProgress = LoadSnapshot(params.SnapshotPath);
    TObliviousTrees obliviousTrees = BuildObliviousTrees(learnProgress);
    TFullModel Model;
    Model.ObliviousTrees = std::move(obliviousTrees);
    Model.ModelInfo["params"] = learnProgress.SerializedTrainParams;

    ExportModel(Model, params.ModelPath, EModelType::CatboostBinary);
    CATBOOST_INFO_LOG << "Model is saved at: " << params.ModelPath << Endl;

    return 0;
}
