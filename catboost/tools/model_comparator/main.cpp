
#include "onnx.h"

#include <catboost/libs/model/model.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/options/json_helper.h>
#include <library/getopt/small/last_getopt.h>

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

static bool CompareModelInfo(const THashMap<TString, TString>& modelInfo1, const THashMap<TString, TString>& modelInfo2) {
    if (modelInfo1.size() != modelInfo2.size()) {
        return false;
    }
    for (const auto& key1: modelInfo1) {
        const auto& key2 = modelInfo2.find(key1.first);
        if (key2 == modelInfo2.end()) {
            return false;
        }
        if (key1.first != "params") {
            if (key1 != *key2) {
                return false;
            }
        } else {
            if (ReadTJsonValue(key1.second) != ReadTJsonValue(key2->second)) {
                return false;
            }
        }
    }
    return true;
}


int main(int argc, char** argv) {
    using namespace NLastGetopt;
    double diffLimit = 0.0;
    TOpts opts = NLastGetopt::TOpts::Default();
    opts.AddLongOption("diff-limit").RequiredArgument("THR")
        .Help("Tolerate elementwise relative difference less than THR")
        .DefaultValue(0.0)
        .StoreResult(&diffLimit);
    opts.SetFreeArgsMin(2);
    opts.SetFreeArgsMax(2);
    opts.SetFreeArgTitle(0, "MODEL1");
    opts.SetFreeArgTitle(1, "MODEL2");
    TOptsParseResult args(&opts, argc, argv);
    TVector<TString> freeArgs = args.GetFreeArgs();


    TMaybe<onnx::ModelProto> onnxModel1 = TryLoadOnnxModel(freeArgs[0]);
    TMaybe<onnx::ModelProto> onnxModel2 = TryLoadOnnxModel(freeArgs[1]);

    if (onnxModel1 && onnxModel2) {
        TString diffString;
        bool modelsAreEqual = Compare(*onnxModel1, *onnxModel2, &diffString);
        if (modelsAreEqual) {
            Clog << "Models are equal" << Endl;
            return 0;
        }
        Clog << "ONNX models differ:\n" << diffString << Endl
             << "MODEL1 = " << freeArgs[0] << Endl
             << "MODEL2 = " << freeArgs[1] << Endl;
        return 1;
    }
    if (onnxModel1 && !onnxModel2) {
        Clog << "Cannot compare (not implemented)\n"
            << "ONNX MODEL1 = " << freeArgs[0] << Endl
            << "non-ONNX MODEL2 = " << freeArgs[1] << Endl;
        return 2;
    }
    if (!onnxModel1 && onnxModel2) {
        Clog << "Cannot compare (not implemented)\n"
            << "non-ONNX MODEL1 = " << freeArgs[0] << Endl
            << "ONNX MODEL2 = " << freeArgs[1] << Endl;
        return 2;
    }

    // both models are non-ONNX - compare loaded as TFullModel

    TFullModel model1 = ReadModelAny(freeArgs[0]);
    TFullModel model2 = ReadModelAny(freeArgs[1]);
    if (model1 == model2) {
        Clog << "Models are equal" << Endl;
        return 0;
    }
    TSubmodelComparison result;
    const TObliviousTrees& trees1 = model1.ObliviousTrees;
    const TObliviousTrees& trees2 = model2.ObliviousTrees;
    if (true) {
        if (trees1.FloatFeatures.size() != trees2.FloatFeatures.size()) {
            Clog << "FloatFeatures size differ: "
                << trees1.FloatFeatures.size() << " vs " << trees2.FloatFeatures.size() << Endl;
            result.StructureIsDifferent = true;
        } else {
            for (size_t i = 0; !result.StructureIsDifferent && i < trees1.FloatFeatures.size(); ++i) {
                auto& floatFeature1 = trees1.FloatFeatures[i];
                auto& floatFeature2 = trees2.FloatFeatures[i];
                result.Update(FeatureBordersDiff("FloatFeature", i, floatFeature1.Borders, floatFeature2.Borders));
                if (floatFeature1.FeatureId != floatFeature2.FeatureId) {
                    Clog << "FloatFeature " << i << " FeatureId differ: "
                        << floatFeature1.FeatureId << " vs " << floatFeature2.FeatureId << Endl;
                    result.StructureIsDifferent = true;
                }
            }
        }
    }
    if (trees1.CatFeatures != trees2.CatFeatures) {
        Clog << "CatFeatures differ" << Endl;
        result.StructureIsDifferent = true;
    }
    if (true) {
        if (trees1.OneHotFeatures.size() != trees2.OneHotFeatures.size()) {
            Clog << "OneHotFeatures size differ: "
                << trees1.OneHotFeatures.size() << " vs " << trees2.OneHotFeatures.size() << Endl;
            result.StructureIsDifferent = true;
        } else {
            for (size_t i = 0; i < trees1.OneHotFeatures.size(); ++i) {
                auto& feature1 = trees1.OneHotFeatures[i];
                auto& feature2 = trees2.OneHotFeatures[i];
                result.Update(FeatureBordersDiff("OneHotFeatures.Values", i, feature1.Values, feature2.Values));
            }
        }
    }
    if (true) {
        if (trees1.CtrFeatures.size() != trees2.CtrFeatures.size()) {
            Clog << "CTRFeatures size differ: "
                << trees1.CtrFeatures.size() << " vs " << trees2.CtrFeatures.size() << Endl;
            result.StructureIsDifferent = true;
        } else {
            for (size_t i = 0; i < trees1.CtrFeatures.size(); ++i) {
                auto& feature1 = trees1.CtrFeatures[i];
                auto& feature2 = trees2.CtrFeatures[i];
                result.Update(FeatureBordersDiff("CTRFeatures", i, feature1.Borders, feature2.Borders));
            }
        }
    }
    if (true) {
        if (trees1.ApproxDimension != trees2.ApproxDimension) {
            Clog << "ObliviousTrees.ApproxDimension differs" << Endl;
            result.StructureIsDifferent = true;
        }
        if (trees1.TreeSplits != trees2.TreeSplits) {
            Clog << "ObliviousTrees.TreeSplits differ" << Endl;
            result.StructureIsDifferent = true;
        }
        if (trees1.TreeSizes != trees2.TreeSizes) {
            Clog << "ObliviousTrees.TreeSizes differ" << Endl;
            result.StructureIsDifferent = true;
        }
        if (trees1.TreeStartOffsets != trees2.TreeStartOffsets) {
            Clog << "ObliviousTrees.TreeStartOffsets differ" << Endl;
            result.StructureIsDifferent = true;
        }
        if (!result.StructureIsDifferent) {
            Y_ASSERT(trees1.LeafValues.size() == trees2.LeafValues.size());
            for (int i = 0; i < trees1.LeafValues.ysize(); ++i) {
                if (result.Update(Diff(trees1.LeafValues[i], trees2.LeafValues[i]))) {
                    Clog << "ObliviousTrees.LeafValues[" << i << "] differ: "
                        << trees1.LeafValues[i] << " vs " << trees2.LeafValues[i]
                        << ", diff = " << result.MaxElementwiseDiff << Endl;
                }
            }
        }
        if (!result.StructureIsDifferent) {
            Y_ASSERT(trees1.LeafWeights.size() == trees2.LeafWeights.size());
            for (int i = 0; i < trees1.LeafWeights.ysize(); ++i) {
                const TVector<double>& weights1 = trees1.LeafWeights[i];
                const TVector<double>& weights2 = trees2.LeafWeights[i];
                Y_ASSERT(weights1.size() == weights2.size());
                for (int j = 0; j < weights1.ysize(); ++j) {
                    if (result.Update(Diff(weights1[j], weights2[j]))) {
                        Cerr << "ObliviousTrees.LeafWeights[" << i << "][" << j << "] differ: "
                            << weights1[j] << " vs " << weights2[j]
                            << ", diff = " << result.MaxElementwiseDiff << Endl;
                    }
                }
            }
        }
        if (trees1.CatFeatures != trees2.CatFeatures) {
            result.StructureIsDifferent = true;
        }
    }
    if (!CompareModelInfo(model1.ModelInfo, model2.ModelInfo)) {
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
