#include "modes.h"

#include <catboost/libs/model/model.h>

#include <library/getopt/small/last_getopt.h>

int mode_model_sum(int argc, const char* argv[]) {
    TVector<std::pair<TString, double>> modelPathsWithWeights;
    TString outputModelPath;
    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    parser.AddLongOption('m', "model", "Model path with default weight 1.0")
        .Handler1T<TString>([&modelPathsWithWeights](const TString& modelPath) {
            modelPathsWithWeights.emplace_back(std::make_pair(modelPath, 1.0));
        });
    parser.AddLongOption("model-with-weight", "Model path with custom weight")
        .RequiredArgument("PATH=WEIGHT")
        .KVHandler([&modelPathsWithWeights](TString modelPath, TString weight) {
            modelPathsWithWeights.emplace_back(std::make_pair(modelPath, FromString<double>(weight)));
        });
    parser.AddLongOption('o', "output-path")
        .Required()
        .RequiredArgument("PATH")
        .StoreResult(&outputModelPath);
    parser.SetFreeArgsNum(0);
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};
    TVector<TFullModel> models;
    TVector<const TFullModel*> modelPtrs;
    TVector<double> weights;
    for (const auto& [path, weight] : modelPathsWithWeights) {
        models.emplace_back(ReadModel(path));
        modelPtrs.emplace_back(&models.back());
        weights.emplace_back(weight);
    }
    TFullModel result = SumModels(modelPtrs, weights);
    OutputModel(result, outputModelPath);
    return 0;
}
