#include "decl.h"
#include "pmml.h"

#include <contrib/libs/onnx/proto/onnx_ml.pb.h>

#include <catboost/libs/model/model.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/private/libs/options/json_helper.h>
#include <library/getopt/small/last_getopt.h>

#include <util/generic/set.h>
#include <util/string/split.h>

#include <cmath>


using namespace NCB;



struct TSubmodelComparison {
    bool StructureIsDifferent = false;
    double MaxElementwiseDiff = 0.0;

    bool Update(const TSubmodelComparison& other) {
        bool updated = false;
        if (other.StructureIsDifferent && !StructureIsDifferent) {
            updated = true;
            StructureIsDifferent = true;
        }
        if (!(other.MaxElementwiseDiff <= MaxElementwiseDiff)) {
            updated = true;
            MaxElementwiseDiff = other.MaxElementwiseDiff;
        }
        return updated;
    }

    bool Update(double diff) {
        bool updated = false;
        if (!(diff <= MaxElementwiseDiff)) {
            MaxElementwiseDiff = diff;
            updated = true;
        }
        return updated;
    }
};

double Diff(double x, double y) {
    double maxAbs = std::max(std::abs(x), std::abs(y));
    return maxAbs != 0.0 ? std::abs((x - y) / maxAbs) : 0.0;
}

template <typename TBorders>
static TSubmodelComparison FeatureBordersDiff(const TStringBuf feature, size_t featureId, const TBorders& borders1, const TBorders& borders2) {
    TSubmodelComparison result;
    if (borders1.size() != borders2.size()) {
        Clog << feature << " " << featureId << " borders sizes differ: "
            << borders1.size() << " != " << borders2.size() << Endl;
        result.StructureIsDifferent = true;
        return result;
    }

    for (size_t i = 0; i < borders1.size(); ++i) {
        if (result.Update(Diff(borders1[i], borders2[i]))) {
            Clog << feature << " " << featureId << " border " << i << " differ: "
                << borders1[i] << " vs " << borders2[i]
                << ", diff = " << result.MaxElementwiseDiff << Endl;
        }
    }
    return result;
}

TFullModel ReadModelAny(const TString& fileName) {
    TFullModel model;
    bool loaded = false;
    for (EModelType modelType : {EModelType::CatboostBinary, EModelType::AppleCoreML}) {
        try {
            model = ReadModel(fileName, modelType);
        } catch (TCatBoostException& e) {
            continue;
        }
        loaded = true;
        break;
    }
    CB_ENSURE(loaded, "Cannot load model " << fileName);
    return model;
}

static bool CompareModelInfo(const THashMap<TString, TString>& modelInfo1, const THashMap<TString, TString>& modelInfo2,  bool verbose, const TSet<TString>& ignoreKeys) {
    if (modelInfo1.size() != modelInfo2.size()) {
        if (verbose) {
            Clog << " Different modelInfo size: " << modelInfo1.size() << " vs " << modelInfo2.size() << Endl;
        }
        return false;
    }
    for (const auto& key1: modelInfo1) {
        const auto& key2 = modelInfo2.find(key1.first);
        if (key2 == modelInfo2.end()) {
            if (verbose) {
                Clog << " Key1 not found in modelInfo2: " << key1.first << Endl;
            }
            return false;
        }
        if (key1.first != "params") {
            if (key1 != *key2) {
                if (verbose) {
                    Clog << " Values differ for key " << key1.first << ": " << key1.second << " vs " << key2->second << Endl;
                }
                if (ignoreKeys.contains(key1.first)) {
                    continue;
                }
                return false;
            }
        } else {
            if (ReadTJsonValue(key1.second) != ReadTJsonValue(key2->second)) {
                if (verbose) {
                    Clog << " Value of `params` differ: " << ReadTJsonValue(key1.second) << " vs " << ReadTJsonValue(key2->second) << Endl;
                }
                return false;
            }
        }
    }
    return true;
}


// returns Nothing() if both lhs and rhs are not of TModel type
template <class TModel>
TMaybe<int> ProcessSubType(const TStringBuf modelTypeName, const TStringBuf modelPath1, const TStringBuf modelPath2) {
    TMaybe<TModel> model1 = TryLoadModel<TModel>(modelPath1);
    TMaybe<TModel> model2 = TryLoadModel<TModel>(modelPath2);

    if (model1 && model2) {
        TString diffString;
        bool modelsAreEqual = CompareModels<TModel>(*model1, *model2, &diffString);
        if (modelsAreEqual) {
            Clog << "Models are equal" << Endl;
            return 0;
        }
        Clog << modelTypeName << " models differ:\n" << diffString << Endl
             << "MODEL1 = " << modelPath1 << Endl
             << "MODEL2 = " << modelPath2 << Endl;
        return 1;
    }
    if (model1 && !model2) {
        Clog << "Cannot compare (not implemented)\n"
            << modelTypeName << " MODEL1 = " << modelPath1 << Endl
            << "non-" << modelTypeName << " MODEL2 = " << modelPath2 << Endl;
        return 2;
    }
    if (!model1 && model2) {
        Clog << "Cannot compare (not implemented)\n"
            << "non-" << modelTypeName << " MODEL1 = " << modelPath1 << Endl
            << modelTypeName << " MODEL2 = " << modelPath2 << Endl;
        return 2;
    }

    return Nothing();
}


int main(int argc, char** argv) {
    using namespace NLastGetopt;
    double diffLimit = 0.0;
    bool verbose = false;
    TSet<TString> ignoreKeys;
    TOpts opts = NLastGetopt::TOpts::Default();
    opts.AddLongOption("diff-limit").RequiredArgument("THR")
        .Help("Tolerate elementwise relative difference less than THR")
        .DefaultValue(0.0)
        .StoreResult(&diffLimit);
    opts.AddLongOption("verbose")
        .StoreTrue(&verbose);
    opts.AddLongOption("ignore-keys")
        .RequiredArgument("KEY[,...]")
        .Help("Ignore differences for these keys")
        .DefaultValue("model_guid")
        .Handler1T<TStringBuf>([&ignoreKeys](const TStringBuf& arg) {
            for (const auto& key : StringSplitter(arg).Split(',').SkipEmpty()) {
                ignoreKeys.insert(TString(key));
            }
        });
    opts.SetFreeArgsMin(2);
    opts.SetFreeArgsMax(2);
    opts.SetFreeArgTitle(0, "MODEL1");
    opts.SetFreeArgTitle(1, "MODEL2");
    TOptsParseResult args(&opts, argc, argv);
    TVector<TString> freeArgs = args.GetFreeArgs();

    TMaybe<int> subTypeResult = ProcessSubType<onnx::ModelProto>("ONNX", freeArgs[0], freeArgs[1]);
    if (subTypeResult) {
        return *subTypeResult;
    }

    subTypeResult = ProcessSubType<TPmmlModel>("PMML", freeArgs[0], freeArgs[1]);
    if (subTypeResult) {
        return *subTypeResult;
    }

    // both models are non-ONNX and non-PMML - compare loaded as TFullModel

    TFullModel model1 = ReadModelAny(freeArgs[0]);
    TFullModel model2 = ReadModelAny(freeArgs[1]);
    if (model1 == model2) {
        Clog << "Models are equal" << Endl;
        return 0;
    }
    TSubmodelComparison result;
    const TModelTrees& trees1 = *model1.ModelTrees;
    const TModelTrees& trees2 = *model2.ModelTrees;
    if (true) {
        if (trees1.GetFloatFeatures().size() != trees2.GetFloatFeatures().size()) {
            Clog << "FloatFeatures size differ: "
                << trees1.GetFloatFeatures().size() << " vs " << trees2.GetFloatFeatures().size() << Endl;
            result.StructureIsDifferent = true;
        } else {
            for (size_t i = 0; !result.StructureIsDifferent && i < trees1.GetFloatFeatures().size(); ++i) {
                auto& floatFeature1 = trees1.GetFloatFeatures()[i];
                auto& floatFeature2 = trees2.GetFloatFeatures()[i];
                result.Update(FeatureBordersDiff("FloatFeature", i, floatFeature1.Borders, floatFeature2.Borders));
                if (floatFeature1.FeatureId != floatFeature2.FeatureId) {
                    Clog << "FloatFeature " << i << " FeatureId differ: "
                        << floatFeature1.FeatureId << " vs " << floatFeature2.FeatureId << Endl;
                    result.StructureIsDifferent = true;
                }
            }
        }
    }
    if (trees1.GetCatFeatures().size() != trees2.GetCatFeatures().size()) {
        Clog << "CatFeatures differ" << Endl;
        result.StructureIsDifferent = true;
    } else {
        for (size_t i = 0; !result.StructureIsDifferent && i < trees1.GetCatFeatures().size(); ++i) {
            const auto& catFeature1 = trees1.GetCatFeatures()[i];
            const auto& catFeature2 = trees2.GetCatFeatures()[i];
            if (catFeature1 != catFeature2) {
                Clog << "CatFeatures differ" << Endl;
                result.StructureIsDifferent = true;
            }
        }
    }
    if (true) {
        if (trees1.GetOneHotFeatures().size() != trees2.GetOneHotFeatures().size()) {
            Clog << "OneHotFeatures size differ: "
                << trees1.GetOneHotFeatures().size() << " vs " << trees2.GetOneHotFeatures().size() << Endl;
            result.StructureIsDifferent = true;
        } else {
            for (size_t i = 0; i < trees1.GetOneHotFeatures().size(); ++i) {
                auto& feature1 = trees1.GetOneHotFeatures()[i];
                auto& feature2 = trees2.GetOneHotFeatures()[i];
                result.Update(FeatureBordersDiff("OneHotFeatures.Values", i, feature1.Values, feature2.Values));
            }
        }
    }
    if (true) {
        if (trees1.GetCtrFeatures().size() != trees2.GetCtrFeatures().size()) {
            Clog << "CTRFeatures size differ: "
                << trees1.GetCtrFeatures().size() << " vs " << trees2.GetCtrFeatures().size() << Endl;
            result.StructureIsDifferent = true;
        } else {
            for (size_t i = 0; i < trees1.GetCtrFeatures().size(); ++i) {
                auto& feature1 = trees1.GetCtrFeatures()[i];
                auto& feature2 = trees2.GetCtrFeatures()[i];
                result.Update(FeatureBordersDiff("CTRFeatures", i, feature1.Borders, feature2.Borders));
            }
        }
    }
    if (true) {
        if (trees1.GetDimensionsCount() != trees2.GetDimensionsCount()) {
            Clog << "ModelTrees.ApproxDimension differs" << Endl;
            result.StructureIsDifferent = true;
        }
        if (trees1.GetTreeSplits().size() != trees2.GetTreeSplits().size()) {
            Clog << "ModelTrees.TreeSplits differs" << Endl;
            result.StructureIsDifferent = true;
        } else {
            for (size_t i = 0; i < !result.StructureIsDifferent && i < trees1.GetTreeSplits().size(); ++i) {
                const auto treeSplit1 = trees1.GetTreeSplits()[i];
                const auto treeSplit2 = trees2.GetTreeSplits()[i];
                if (treeSplit1 != treeSplit2) {
                    Clog << "ModelTrees.TreeSplits differs" << Endl;
                    result.StructureIsDifferent = true;
                }
            }
        }
        if (trees1.GetTreeSizes().size() != trees2.GetTreeSizes().size()) {
            Clog << "ModelTrees.TreeSizes differs" << Endl;
            result.StructureIsDifferent = true;
        } else {
            for (size_t i = 0; i < !result.StructureIsDifferent && i < trees1.GetTreeSizes().size(); ++i) {
                const auto treeSize1 = trees1.GetTreeSizes()[i];
                const auto treeSize2 = trees2.GetTreeSizes()[i];
                if (treeSize1 != treeSize2) {
                    Clog << "ModelTrees.TreeSizes differs" << Endl;
                    result.StructureIsDifferent = true;
                }
            }
        }
        if (trees1.GetTreeStartOffsets().size() != trees2.GetTreeStartOffsets().size()) {
            Clog << "ModelTrees.TreeStartOffsets differs" << Endl;
            result.StructureIsDifferent = true;
        } else {
            for (size_t i = 0; i < !result.StructureIsDifferent && i < trees1.GetTreeStartOffsets().size(); ++i) {
                const auto treeStartOffset1 = trees1.GetTreeStartOffsets()[i];
                const auto treeStartOffset2 = trees2.GetTreeStartOffsets()[i];
                if (treeStartOffset1 != treeStartOffset2) {
                    Clog << "ModelTrees.TreeStartOffsets differs" << Endl;
                    result.StructureIsDifferent = true;
                }
            }
        }
        if (!result.StructureIsDifferent) {
            Y_ASSERT(trees1.GetLeafValues().size() == trees2.GetLeafValues().size());
            for (int i = 0; i < trees1.GetLeafValues().ysize(); ++i) {
                if (result.Update(Diff(trees1.GetLeafValues()[i], trees2.GetLeafValues()[i]))) {
                    Clog << "ModelTrees.LeafValues[" << i << "] differ: "
                        << trees1.GetLeafValues()[i] << " vs " << trees2.GetLeafValues()[i]
                        << ", diff = " << result.MaxElementwiseDiff << Endl;
                }
            }
        }
        if (!result.StructureIsDifferent) {
            Y_ASSERT(trees1.GetLeafWeights().size() == trees2.GetLeafWeights().size());
            for (int i = 0; i < trees1.GetLeafWeights().ysize(); ++i) {
                if (result.Update(Diff(trees1.GetLeafWeights()[i], trees2.GetLeafWeights()[i]))) {
                    Clog << "ModelTrees.LeafWeights[" << i << "] differ: "
                        << trees1.GetLeafWeights()[i] << " vs " << trees2.GetLeafWeights()[i]
                        << ", diff = " << result.MaxElementwiseDiff << Endl;
                }
            }
        }
    }
    if (!CompareModelInfo(model1.ModelInfo, model2.ModelInfo, verbose, ignoreKeys)) {
        Clog << "ModelInfo differ" << Endl;
        model1.ModelInfo = THashMap<TString, TString>();
        model2.ModelInfo = THashMap<TString, TString>();
        if (model1 != model2) {
            result.StructureIsDifferent = true;
        }
    }
    Clog << "MODEL1 = " << freeArgs[0] << Endl;
    Clog << "MODEL2 = " << freeArgs[1] << Endl;
    Clog << "Structure of models is " << (result.StructureIsDifferent ? "different" : "same") << Endl;
    Clog << "Maximum observed elementwise diff is " << result.MaxElementwiseDiff << ", limit is " << diffLimit << Endl;
    return result.StructureIsDifferent || !(result.MaxElementwiseDiff <= diffLimit) ? 1 : 0;
}
