#include "modes.h"

#include <catboost/libs/model/model.h>
#include <catboost/libs/model/model_export/model_exporter.h>

#include <library/cpp/getopt/small/last_getopt.h>
#include <library/cpp/getopt/small/last_getopt_support.h>

#include <util/generic/serialized_enum.h>

int mode_model_sum(int argc, const char* argv[]) {
    TVector<std::pair<TString, double>> modelPathsWithWeights;
    TString outputModelPath;
    EModelType outputModelFormat = EModelType::CatboostBinary;
    ECtrTableMergePolicy ctrMergePolicy = ECtrTableMergePolicy::IntersectingCountersAverage;

    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    parser.AddLongOption('m', "model", "Model path with default weight 1.0")
        .Handler1T<TString>([&modelPathsWithWeights](const TString& modelPath) {
            modelPathsWithWeights.emplace_back(std::make_pair(modelPath, 1.0));
        });
    parser.AddLongOption("model-with-weight", "Model path with custom weight")
        .RequiredArgument("PATH=WEIGHT")
        .Handler1T<TStringBuf>([&modelPathsWithWeights](auto path_weight) {
            TStringBuf path, weight;
            if (!path_weight.TryRSplit('=', path, weight)) {
                throw NLastGetopt::TUsageException() << "bad option value `" << path_weight << "`, expected PATH=WEIGHT";
            }
            modelPathsWithWeights.emplace_back(std::make_pair(path, FromString<double>(weight)));
        });
    parser.AddLongOption('o', "output-path")
        .Required()
        .RequiredArgument("PATH")
        .StoreResult(&outputModelPath);
    parser.AddLongOption("output-model-format")
        .OptionalArgument("output model format")
        .Handler1T<TString>([&outputModelFormat](const TString& format) {
            outputModelFormat = FromString<EModelType>(format);
        });
    parser.AddLongOption("ctr-merge-policy",
         TString::Join(
            "One of ",
            GetEnumAllNames<ECtrTableMergePolicy>()))
        .Optional()
        .StoreResult(&ctrMergePolicy);
    parser.SetFreeArgsNum(0);
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};
    TVector<THolder<TFullModel>> models;
    TVector<const TFullModel*> modelPtrs;
    TVector<double> weights;
    for (const auto& [path, weight] : modelPathsWithWeights) {
        models.emplace_back(MakeHolder<TFullModel>(ReadModel(path)));
        modelPtrs.emplace_back(models.back().Get());
        weights.emplace_back(weight);
    }
    TFullModel result = SumModels(modelPtrs, weights, ctrMergePolicy);
    NCB::ExportModel(result, outputModelPath, outputModelFormat);
    return 0;
}
