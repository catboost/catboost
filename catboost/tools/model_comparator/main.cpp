#include "decl.h"
#include "pmml.h"

#include <contrib/libs/onnx/onnx/onnx_pb.h>

#include <catboost/libs/model/model.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/private/libs/options/json_helper.h>
#include <library/cpp/getopt/small/last_getopt.h>
#include <library/cpp/json/writer/json.h>

#include <util/generic/algorithm.h>
#include <util/generic/hash_set.h>
#include <util/generic/xrange.h>
#include <util/string/split.h>

#include <cmath>
#include <regex>


using namespace NCB;

static double MinAbsRelErr(double x, double y) {
    double maxAbs = std::max(std::abs(x), std::abs(y));
    double absErr = std::abs(x - y);
    double relErr = absErr ? absErr / maxAbs : 0.0;
    return Min(absErr, relErr);
}

static bool IsMatchedBy(const TString& regex, const TString& name) {
    std::regex pattern(regex.c_str());
    return !!std::regex_match(name.c_str(), pattern);
}

static TString ShortQuoted(const TString& s) {
    constexpr int maxShortLen = 50;
    if (s.length() < maxShortLen) {
        return "\"" + s + "\"";
    } else {
        constexpr int partLen = maxShortLen / 2 - 5;
        static_assert(partLen > 0);
        return "\"" + s.substr(0, partLen) + "..." + s.substr(s.length() - partLen) + "\"";
    }
}

class TComparator {
public:
    TComparator(double eps, const THashSet<TString>& ignoredKeys)
    : Eps(eps)
    , IgnoredKeys(ignoredKeys)
    {}

    bool IsIgnored(const TString& name) {
        return AnyOf(IgnoredKeys, [&](const TString& pattern) {
            return IsMatchedBy(pattern, name);
        });
    }

    bool NotIgnored(const TString& name) {
        return !IsIgnored(name);
    }

    bool AlmostEqual(const TString& name, double a, double b) {
        if (IsIgnored(name))
            return true;
        double e = MinAbsRelErr(a, b);
        if (!(e <= MaxObservedDiff)) {
            Clog << name << ": " << a << " vs " << b
                << ", diff " << e
                << Endl;
            MaxObservedDiff = e;
        }
        return e <= Eps;
    }

    template <typename T>
    bool AlmostEqual(const TString& name, const T& a, const T& b) {
        if (IsIgnored(name))
            return true;
        if (!(a == b)) {
            Clog << name << ": " << a << " vs " << b << Endl;
            return false;
        }
        return true;
    }

    bool AlmostEqual(const TString& name, const NJson::TJsonValue::TMapType& a, const NJson::TJsonValue::TMapType& b) {
        if (IsIgnored(name))
            return true;
        TVector<TString> allKeys;
        for (const auto& kv : a) {
            allKeys.push_back(kv.first);
        }
        for (const auto& kv : b) {
            allKeys.push_back(kv.first);
        }
        SortUnique(allKeys);
        bool equal = true;
        for (const auto& key : allKeys) {
            equal &= AlmostEqual(name + '.' + key,
                                 a.Value(key, NJson::TJsonValue(NJson::EJsonValueType::JSON_NULL)),
                                 b.Value(key, NJson::TJsonValue(NJson::EJsonValueType::JSON_NULL)));
        }
        return equal;
    }

    bool AlmostEqual(const TString& name, const NJson::TJsonValue::TArray& a, const NJson::TJsonValue::TArray& b) {
        if (IsIgnored(name))
            return true;
        if (a.size() != b.size()) {
            Clog << name << ": array size is "
                << a.size()
                << " vs "
                << b.size()
                << Endl;
            return false;
        }
        return AllOf(xrange(a.size()), [&](size_t idx) {
            return AlmostEqual(name + '[' + ToString(idx) + ']', a[idx], b[idx]);
        });
    }

    bool AlmostEqual(const TString& name, const NJson::TJsonValue& a, const NJson::TJsonValue& b) {
        if (IsIgnored(name))
            return true;
        if (a.GetType() != b.GetType()) {
            Clog << name << ": element type is "
                << a.GetType()
                << " vs "
                << b.GetType()
                << Endl;
            return false;
        }
        switch (a.GetType()) {

            case NJson::EJsonValueType::JSON_ARRAY:
                return AlmostEqual(name, a.GetArraySafe(), b.GetArraySafe());

            case NJson::EJsonValueType::JSON_MAP:
                return AlmostEqual(name, a.GetMapSafe(), b.GetMapSafe());

            case NJson::EJsonValueType::JSON_DOUBLE:
                return AlmostEqual(name, a.GetDoubleSafe(), b.GetDoubleSafe());

            default:
                return AlmostEqual(name, a.GetString(), b.GetString());
        }
    }

    bool AlmostEqual(const TString& name, const TModelTrees& a, const TModelTrees& b) {
        if (IsIgnored(name)) {
            return true;
        }
        bool result = true;
        result &= AlmostEqual(name + ".FloatFeatures", a.GetFloatFeatures(), b.GetFloatFeatures());
        result &= AlmostEqual(name + ".CatFeatures", a.GetCatFeatures(), b.GetCatFeatures());
        result &= AlmostEqual(name + ".OneHotFeatures", a.GetOneHotFeatures(), b.GetOneHotFeatures());
        result &= AlmostEqual(name + ".CtrFeatures", a.GetCtrFeatures(), b.GetCtrFeatures());
        result &= AlmostEqual(name + ".ApproxDimension", a.GetDimensionsCount(), b.GetDimensionsCount());
        result &= AlmostEqual(name + ".TreeSplits", a.GetModelTreeData()->GetTreeSplits(), b.GetModelTreeData()->GetTreeSplits());
        result &= AlmostEqual(name + ".TreeSizes", a.GetModelTreeData()->GetTreeSizes(), b.GetModelTreeData()->GetTreeSizes());
        result &= AlmostEqual(name + ".TreeStartOffsets", a.GetModelTreeData()->GetTreeStartOffsets(), b.GetModelTreeData()->GetTreeStartOffsets());
        result &= AlmostEqualLeafValues(name + ".LeafValues", a, b);
        result &= AlmostEqual(name + ".LeafWeights", a.GetModelTreeData()->GetLeafWeights(), b.GetModelTreeData()->GetLeafWeights());
        return result;
    }

    template <typename T>
    bool AlmostEqual(const TString& name, TConstArrayRef<T> a, TConstArrayRef<T> b) {
        if (IsIgnored(name))
            return true;
        if (a.size() != b.size()) {
            Clog << name << ": array size is "
                << a.size()
                << " vs "
                << b.size()
                << Endl;
            return false;
        }
        return AllOf(xrange(a.size()), [&](size_t idx){
            return AlmostEqual(name + '[' + ToString(idx) + ']', a[idx], b[idx]);
        });
    }

    template <typename T>
    bool AlmostEqual(const TString& name, const TVector<T>& a, const TVector<T>& b) {
        return AlmostEqual(name, TConstArrayRef<T>(a.begin(), a.end()), TConstArrayRef<T>(b.begin(), b.end()));
    }

    bool AlmostEqualLeafValues(const TString& name, const TModelTrees& a, const TModelTrees& b) {
        if (a.GetScaleAndBias() == b.GetScaleAndBias()) {
            return AlmostEqual(name, a.GetModelTreeData()->GetLeafValues(), b.GetModelTreeData()->GetLeafValues());
        }

        Clog << name << ".ScaleAndBias: "
            << a.GetScaleAndBias()
            << " vs "
            << b.GetScaleAndBias()
            << ", will compare normalized LeafValues..."
            << Endl;

        auto normedLeafValues = [](const TModelTrees& trees) -> TVector<double> {
            TVector<double> result(trees.GetModelTreeData()->GetLeafValues().begin(), trees.GetModelTreeData()->GetLeafValues().end());
            int firstTreeLeafCount = trees.GetTreeCount() > 0 ? trees.GetTreeLeafCounts()[0] : 0;
            const auto norm = trees.GetScaleAndBias();

            double bias = norm.GetOneDimensionalBias(
                "Non single-dimension approxes are not supported");
            for (int i = 0; i < result.ysize(); ++i) {
                result[i] *= norm.Scale;
                if (i < firstTreeLeafCount) {
                    result[i] += bias;
                }
            }
            return result;
        };
        bool equal = AlmostEqual(name + "(normalized)", normedLeafValues(a), normedLeafValues(b));
        if (equal) {
            Clog << name + "(normalized): equal" << Endl;
        }
        return equal;
    }

    bool AlmostEqual(const TString& name, const TFloatFeature& a, const TFloatFeature& b) {
        return IsIgnored(name)
            || AlmostEqual(name + ".HasNans", a.HasNans, b.HasNans)
            && AlmostEqual(name + ".Position", a.Position, b.Position)
            && AlmostEqual(name + ".Borders", a.Borders, b.Borders)
            && AlmostEqual(name + ".FeatureId", a.FeatureId, b.FeatureId)
            && AlmostEqual(name + ".NanValueTreatment", a.NanValueTreatment, b.NanValueTreatment)
            ;
    }

    bool AlmostEqual(const TString& name, const TOneHotFeature& a, const TOneHotFeature& b) {
        return IsIgnored(name)
            || AlmostEqual(name + ".CatFeatureIndex", a.CatFeatureIndex, b.CatFeatureIndex)
            && AlmostEqual(name + ".Values", a.Values, b.Values)
            && AlmostEqual(name + ".StringValues", a.StringValues, b.StringValues)
            ;
    }

    bool AlmostEqual(const TString& name, const TCtrFeature& a, const TCtrFeature& b) {
        return IsIgnored(name)
            || AlmostEqual(name + ".Ctr", a.Ctr, b.Ctr)
            && AlmostEqual(name + ".Borders", a.Borders, b.Borders)
            ;
    }

    bool AlmostEqual(const TString& name, const TModelCtr& a, const TModelCtr& b) {
        return IsIgnored(name)
            || AlmostEqual(name + ".Base", a.Base, b.Base)
            && AlmostEqual(name + ".PriorNum", a.PriorNum, b.PriorNum)
            && AlmostEqual(name + ".PriorDenom", a.PriorDenom, b.PriorDenom)
            && AlmostEqual(name + ".Scale", a.Scale, b.Scale)
            && AlmostEqual(name + ".Shift", a.Shift, b.Shift)
            && AlmostEqual(name + ".TargetBorderIdx", a.TargetBorderIdx, b.TargetBorderIdx)
            ;
    }

    bool AlmostEqual(const TString& name, const TModelCtrBase& a, const TModelCtrBase& b) {
        return IsIgnored(name)
            || AlmostEqual(name + ".CtrType", a.CtrType, b.CtrType)
            && AlmostEqual(name + ".Projection", a.Projection, b.Projection)
            && AlmostEqual(name + ".TargetBorderClassifierIdx", a.TargetBorderClassifierIdx, b.TargetBorderClassifierIdx)
            ;
    }

    bool AlmostEqual(const TString& name, const TFeatureCombination& a, const TFeatureCombination& b) {
        return IsIgnored(name)
            || AlmostEqual(name + ".CatFeatures", a.CatFeatures, b.CatFeatures)
            && AlmostEqual(name + ".BinFeatures", a.BinFeatures, b.BinFeatures)
            && AlmostEqual(name + ".OneHotFeatures", a.OneHotFeatures, b.OneHotFeatures)
            ;
    }

    bool AlmostEqualModelInfo(const TString& name, const THashMap<TString, TString>& a, const THashMap<TString, TString>& b) {
        if (IsIgnored(name))
            return true;
        TVector<TString> allKeys;
        for (const auto& kv : a) {
            allKeys.push_back(kv.first);
        }
        for (const auto& kv : b) {
            allKeys.push_back(kv.first);
        }
        SortUnique(allKeys);
        bool equal = true;
        for (const auto& key : allKeys) {
            TString nameKey = name + '.' + key;
            if (IsIgnored(nameKey))
                continue;
            if (a.contains(key) != b.contains(key)) {
                Clog << nameKey << ": key is "
                    << (a.contains(key) ? ShortQuoted(a.at(key)) : "null")
                    << " vs "
                    << (b.contains(key) ? ShortQuoted(b.at(key)) : "null")
                    << Endl;
                equal &= false;
            } else if (key.EndsWith("params")) {
                equal &= AlmostEqual(nameKey, ReadTJsonValue(a.at(key)), ReadTJsonValue(b.at(key)));
            } else {
                equal &= AlmostEqual(nameKey, a.at(key), b.at(key));
            }
        }
        return equal;
    }

private:
    double Eps;
    THashSet<TString> IgnoredKeys;
public:
    double MaxObservedDiff = 0;
};

template <>
void Out<TScaleAndBias>(IOutputStream& out, TTypeTraits<TScaleAndBias>::TFuncParam norm) {
    out << "{" << norm.Scale << "," << "[";
    bool firstItem = true;
    auto bias = norm.GetBiasRef();
    for (auto b : bias) {
        out << (firstItem ? "" : ",") << b;
        firstItem = false;
    }
    out << "]}";
}

template <>
void Out<TFeaturePosition>(IOutputStream& out, TTypeTraits<TFeaturePosition>::TFuncParam position) {
    out << "{" << position.Index << "," << position.FlatIndex << "}";
}

template <>
void Out<TCatFeature>(IOutputStream& out, TTypeTraits<TCatFeature>::TFuncParam feature) {
    out << "{" << feature.FeatureId << ";" << feature.Position << ";" << feature.UsedInModel() << "}";
}

template <>
void Out<TFloatSplit>(IOutputStream& out, TTypeTraits<TFloatSplit>::TFuncParam split) {
    out << "{" << split.FloatFeature << "," << split.Split << "}";
}

template <>
void Out<TOneHotSplit>(IOutputStream& out, TTypeTraits<TOneHotSplit>::TFuncParam split) {
    out << "{" << split.CatFeatureIdx << "," << split.Value << "}";
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

// returns Nothing() if both lhs and rhs are not of TModel type
template <class TModel>
TMaybe<int> ProcessSubType(const TStringBuf modelTypeName, const TStringBuf modelPath1, const TStringBuf modelPath2, double diffLimit) {
    TMaybe<TModel> model1 = TryLoadModel<TModel>(modelPath1);
    TMaybe<TModel> model2 = TryLoadModel<TModel>(modelPath2);

    if (model1 && model2) {
        TString diffString;
        bool modelsAreEqual = CompareModels<TModel>(*model1, *model2, diffLimit, &diffString);
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
    THashSet<TString> ignoreKeys;
    TOpts opts = NLastGetopt::TOpts::Default();
    opts.AddLongOption("diff-limit").RequiredArgument("THR")
        .Help("Tolerate elementwise relative difference less than THR")
        .DefaultValue(0.0)
        .StoreResult(&diffLimit);
    opts.AddLongOption("verbose")
        .StoreTrue(&verbose);
    opts.AddLongOption("ignore-keys-default")
        .RequiredArgument("KEY[,...]")
        .Help("Ignore differences for these key regexps")
        .DefaultValue(".*model_guid,.*catboost_version_info,.*train_finish_time")
        .Handler1T<TStringBuf>([&ignoreKeys](const TStringBuf& arg) {
            for (const auto& key : StringSplitter(arg).Split(',').SkipEmpty()) {
                ignoreKeys.insert(TString(key));
            }
        });
    opts.AddLongOption("ignore-keys")
        .RequiredArgument("KEY[,...]")
        .Help("Ignore differences for these key regexps, in addition to ignore-keys-default")
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

    TMaybe<int> subTypeResult = ProcessSubType<onnx::ModelProto>("ONNX", freeArgs[0], freeArgs[1], diffLimit);
    if (subTypeResult) {
        return *subTypeResult;
    }

    subTypeResult = ProcessSubType<TPmmlModel>("PMML", freeArgs[0], freeArgs[1], diffLimit);
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

    TComparator withinEps(diffLimit, ignoreKeys);
    bool equal = true
        && withinEps.AlmostEqual("ModelTrees", *model1.ModelTrees, *model2.ModelTrees)
        && withinEps.AlmostEqualModelInfo("ModelInfo", model1.ModelInfo, model2.ModelInfo)
        ;

    Clog << "MODEL1 = " << freeArgs[0] << Endl;
    Clog << "MODEL2 = " << freeArgs[1] << Endl;

    Clog << "Maximum observed elementwise diff is " << withinEps.MaxObservedDiff << ", limit is " << diffLimit << Endl;
    return equal ? 0 : 1;
}
