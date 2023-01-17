#include "modes.h"

#include <catboost/libs/model/model.h>
#include <catboost/libs/model/model_export/model_exporter.h>

#include <library/cpp/getopt/small/last_getopt.h>
#include <library/cpp/getopt/small/last_getopt_support.h>

#include <util/generic/serialized_enum.h>
#include <util/string/split.h>


int mode_model_sum(int argc, const char* argv[]) {
    TVector<TString> modelPaths;
    TVector<double> modelWeights;
    TVector<TString> modelParamsPrefixes;
    TString outputModelPath;
    EModelType outputModelFormat = EModelType::CatboostBinary;
    ECtrTableMergePolicy ctrMergePolicy = ECtrTableMergePolicy::IntersectingCountersAverage;

    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    parser.AddLongOption('m', "model", "Model path with default weight 1.0")
        .Handler1T<TString>([&modelPaths, &modelWeights](const TString& modelPath) {
            modelPaths.emplace_back(modelPath);
            modelWeights.emplace_back(1.0);
        });
    parser.AddLongOption("model-with-weight", "Model path with custom weight")
        .RequiredArgument("PATH=WEIGHT")
        .Handler1T<TStringBuf>([&modelPaths, &modelWeights](auto path_weight) {
            TStringBuf path, weight;
            if (!path_weight.TryRSplit('=', path, weight)) {
                throw NLastGetopt::TUsageException() << "bad option value `" << path_weight << "`, expected PATH=WEIGHT";
            }
            modelPaths.emplace_back(path);
            modelWeights.emplace_back(FromString<double>(weight));
        });
    parser.AddLongOption("model-with-weight-and-prefix", "Model path with custom weight and params prefix")
        .RequiredArgument("PATH,WEIGHT,PREFIX")
        .Handler1T<TStringBuf>([&modelPaths,&modelWeights,&modelParamsPrefixes](auto path_weight_prefix) {
            TStringBuf path, weight, paramsPrefix;
            Split(path_weight_prefix, ',', path, weight, paramsPrefix);
            modelPaths.emplace_back(path);
            modelWeights.emplace_back(FromString<double>(weight));
            modelParamsPrefixes.emplace_back(paramsPrefix);
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
    for (const auto& path : modelPaths) {
        models.emplace_back(MakeHolder<TFullModel>(ReadModel(path)));
        modelPtrs.emplace_back(models.back().Get());
    }
    TFullModel result = SumModels(modelPtrs, modelWeights, modelParamsPrefixes, ctrMergePolicy);
    NCB::ExportModel(result, outputModelPath, outputModelFormat);
    return 0;
}
