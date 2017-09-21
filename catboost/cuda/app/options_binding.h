#pragma once

#include <catboost/cuda/data/binarization_config.h>
#include <catboost/cuda/data/load_config.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/methods/oblivious_tree.h>
#include <catboost/cuda/methods/boosting.h>
#include <catboost/cuda/targets/target_options.h>
#include <library/getopt/small/last_getopt_opts.h>
#include <sys/time.h>
#include <util/string/iterator.h>
#include <util/charset/utf8.h>
#include <cstdio>

inline double GetTime() {
    timeval tv;
    gettimeofday(&tv, nullptr);
    return ((double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0);
}

template <class TConfig>
class TOptionsBinder;

struct TFoldOption {
    int FoldId = 0;
    int FoldCount = 0;
};

template <>
inline TFoldOption FromString(const TStringBuf& folding) {
    TFoldOption result;
    int n = sscanf(folding.c_str(), "%d/%d", &result.FoldId, &result.FoldCount);
    if (n != 2 || result.FoldId <= 0 || result.FoldCount < 2) {
        ythrow TCatboostException() << "Invalid folding: " << folding;
    }
    return result;
};

template <>
class TOptionsBinder<TPoolLoadOptions> {
public:
    static void Bind(TPoolLoadOptions& poolLoadOptions,
                     NLastGetopt::TOpts& options,
                     bool needTest = true) {
        options
            .AddLongOption('f', "learn-set")
            .RequiredArgument("FILE")
            .Help("Training pool file name. Default is features.txt.")
            .StoreResult(&poolLoadOptions.FeaturesName);

        options
            .AddLongOption("cd")
            .AddLongName("column-description")
            .RequiredArgument("FILE")
            .Help("Column description path")
            .StoreResult(&poolLoadOptions.ColumnDescriptionName);

        options.AddLongOption("dev-catfeature-binarization-temp-file")
            .RequiredArgument("FILE")
            .Help("Temp file to store cat feature index")
            .DefaultValue(TStringBuilder() << "/tmp/cat_feature_index." << CreateGuidAsString() << ".tmp")
            .StoreResult(&poolLoadOptions.CatFeatureBinarizationTempName);

        options.AddLongOption("delimiter", "Learning and training sets delimiter")
            .RequiredArgument("SYMBOL")
            .Handler1T<TString>([&](const TString& oneChar) {
                CB_ENSURE(oneChar.size() == 1, "only single char delimiters supported");
                poolLoadOptions.Delimiter = oneChar[0];
            });

        options.AddLongOption("has-header", "Read first line as header")
            .NoArgument()
            .StoreValue(&poolLoadOptions.HasHeaderFlag, true);

        options
            .AddLongOption('I', "ignored-features")
            .RequiredArgument("RANGE")
            .Help("Sets range of feature indices to remove from pool.")
            .Handler1T<TString>([&](const TString& range) mutable {
                poolLoadOptions.IgnoredFeatures = ::NHelpers::ParseRangeString(range);
            });

        if (needTest) {
            options
                .AddLongOption('t', "test-set")
                .RequiredArgument("FILE")
                .Help("TestPool file name.")
                .StoreResult(&poolLoadOptions.TestName);
        }
    }
};

template <>
class TOptionsBinder<TBinarizationConfiguration> {
public:
    static void Bind(TBinarizationConfiguration& binarizationConfiguration,
                     NLastGetopt::TOpts& options) {
        options
            .AddLongOption("feature-border-type")
            .RequiredArgument("Feature border type")
            .Help("Sets grid type [ UniformAndQuantiles | GreedyLogSum | MaxLogSum | MinEntropy | Median] or file with custom grid.")
            .DefaultValue("MinEntropy")
            .StoreResult(&binarizationConfiguration.DefaultFloatBinarization.BorderSelectionType)
            .StoreResult(&binarizationConfiguration.TargetBinarization.BorderSelectionType);

        options
            .AddLongOption('x', "border-count")
            .RequiredArgument("INT")
            .Help("Sets number of conditions per float feature. Default is 32.")
            .DefaultValue("32")
            .Handler1T<ui32>([&](ui32 discretization) {
                if (discretization > 255) {
                    ythrow TCatboostException() << "Maximum supported binarization is -x 255";
                }
                binarizationConfiguration.DefaultFloatBinarization.Discretization = discretization;
            });

        options
            .AddLongOption("ctr-border-count")
            .RequiredArgument("INT")
            .Help("Sets number of conditions per float feature. Default is 16.")
            .DefaultValue("16")
            .Handler1T<ui32>([&](ui32 discretization) {
                if (discretization > 255) {
                    ythrow TCatboostException() << "Maximum supported binarization is -x 255";
                }
                binarizationConfiguration.DefaultCtrBinarization.Discretization = discretization;
                binarizationConfiguration.FreqCtrBinarization.Discretization = discretization;
            });

        options
            .AddLongOption("dev-ctr-border-type")
            .RequiredArgument("INT")
            .Help("Sets number of conditions per float feature. Default is 16.")
            .DefaultValue("UniformAndQuantiles")
            .Handler1T<EBorderSelectionType>([&](EBorderSelectionType type) {
                binarizationConfiguration.DefaultCtrBinarization.BorderSelectionType = type;
            });

        options
            .AddLongOption("dev-tree-ctr-border-count")
            .RequiredArgument("INT")
            .Help("Sets number of conditions per float feature. Default is 16.")
            .DefaultValue("16")
            .Handler1T<ui32>([&](ui32 discretization) {
                if (discretization > 255) {
                    ythrow TCatboostException() << "Maximum supported binarization is -x 255";
                }
                binarizationConfiguration.DefaultTreeCtrBinarization.Discretization = discretization;
            });

        options
                .AddLongOption("dev-freq-ctr-border-count")
                .RequiredArgument("INT")
                .Help("Sets number of conditions per float feature. Default is 16.")
                .DefaultValue("16")
                .Handler1T<ui32>([&](ui32 discretization) {
                    binarizationConfiguration.FreqCtrBinarization.Discretization = discretization;
                });


        options
                .AddLongOption("dev-freq-ctr-border-type")
                .RequiredArgument("INT")
                .Help("Sets number of conditions per float feature. Default is 16.")
                .DefaultValue("GreedyLogSum")
                .Handler1T<EBorderSelectionType>([&](EBorderSelectionType type) {
                    binarizationConfiguration.FreqCtrBinarization.BorderSelectionType = type;
                });

        options
                .AddLongOption("dev-freq-tree-ctr-border-count")
                .RequiredArgument("INT")
                .Help("Sets number of conditions per float feature. Default is 16.")
                .DefaultValue("16")
                .Handler1T<ui32>([&](ui32 discretization) {
                    binarizationConfiguration.FreqTreeCtrBinarization.Discretization = discretization;
                });

        options
                .AddLongOption("dev-tree-ctr-border-type")
                .RequiredArgument("Grid")
                .Help("Sets  tree ctrs borders selection type")
                .DefaultValue("Uniform")
                .StoreResult(&binarizationConfiguration.DefaultTreeCtrBinarization.BorderSelectionType);


        options
            .AddLongOption("dev-target-binarization")
            .RequiredArgument("INT")
            .Help("Sets number of conditions per target. Default is 16.")
            .DefaultValue("16")
            .Handler1T<ui32>([&](ui32 discretization) {
                if (discretization > 255) {
                    ythrow TCatboostException() << "Maximum supported binarization is -x 255";
                }
                binarizationConfiguration.TargetBinarization.Discretization = discretization;
            });
    }
};

template <>
class TOptionsBinder<TFeatureManagerOptions> {
public:
    static void Bind(TFeatureManagerOptions& featureManagerOptions, NLastGetopt::TOpts& options) {
        TOptionsBinder<TBinarizationConfiguration>::Bind(featureManagerOptions.BinarizationConfiguration, options);

        options
            .AddLongOption("one-hot-max-size")
            .RequiredArgument("Int")
            .Help("Limit for one hot. Max is 255")
            .DefaultValue("0")
            .StoreResult(&featureManagerOptions.OneHotLimit);

        options
            .AddLongOption("max-ctr-complexity")
            .RequiredArgument("Int")
            .Help("Limit for one hot. Max is 255")
            .DefaultValue("1")
            .Handler1T<ui32>([&](ui32 count) {
                featureManagerOptions.MaxTensorComplexity = count;
            });

        options
            .AddLongOption("dev-ctrs")
            .RequiredArgument("String")
            .Help("Enabled ctrs")
            .DefaultValue("Borders,FeatureFreq")
            .Handler1T<TString>([&](const TString& ctrs) {
                if (ctrs == "None") {
                    return;
                }
                for (const auto& ctr : StringSplitter(ctrs).Split(',')) {
                    ECtrType ctrType = FromString<ECtrType>(ctr.Token());
                    CB_ENSURE(IsSupportedCtrType(ctrType), "Error: unsupported ctr type");
                    featureManagerOptions.EnabledCtrTypes.insert(ctrType);
                }
                featureManagerOptions.CustomCtrTypes = true;
            });
    }
};

template <>
class TOptionsBinder<TBootstrapConfig> {
public:
    static void Bind(TBootstrapConfig& bootstrapOptions, NLastGetopt::TOpts& options) {
        options
            .AddLongOption("dev-sample-rate")
            .RequiredArgument("Float")
            .Help("Sample rate")
            .DefaultValue("0.66")
            .StoreResult(&bootstrapOptions.TakenFraction);

        options
                .AddLongOption("bagging-temperature")
                .RequiredArgument("Float")
                .Help("Sample rate")
                .DefaultValue("1.0")
                .StoreResult(&bootstrapOptions.BaggingTemperature);

        options
            .AddLongOption("dev-bootstrap-type")
            .RequiredArgument("STRING")
            .Help("Bootstrap type")
            .DefaultValue("Bayesian")
            .StoreResult(&bootstrapOptions.BootstrapType);



        if (options.HasLongOption("random-seed")) {
            options
                .GetLongOption("random-seed")
                .StoreResult(&bootstrapOptions.Seed);
        } else {
            options
                .AddLongOption('r', "random-seed")
                .RequiredArgument("INT")
                .Help("Sets random generators seed.")
                .DefaultValue(ToString<long>(GetTime()))
                .StoreResult(&bootstrapOptions.Seed);
        }
    }
};

template <>
class TOptionsBinder<TObliviousTreeLearnerOptions> {
public:
    static void Bind(TObliviousTreeLearnerOptions& treeOptions, NLastGetopt::TOpts& options) {
        TOptionsBinder<TBootstrapConfig>::Bind(treeOptions.BootstrapConfig, options);

        options
            .AddLongOption('n', "depth")
            .RequiredArgument("INT")
            .Help("Sets number of splits per each tree. Default is 6.")
            .DefaultValue("6")
            .StoreResult(&treeOptions.MaxDepth);

        options
            .AddLongOption("gradient-iterations")
            .RequiredArgument("INT")
            .Help("Sets leaf estimation iterations")
            .Handler1T<ui32>([&](ui32 iter) mutable {
                treeOptions.LeavesEstimationIters = iter;
            });

        options
            .AddLongOption("dev-score-function")
            .RequiredArgument("STRING")
            .Help("Score function")
            .DefaultValue("Correlation")
            .StoreResult(&treeOptions.ScoreFunction);

        options
            .AddLongOption("dev-normalize-loss-in-estimation")
            .RequiredArgument("FLAG")
            .Help("Normalize loss")
            .DefaultValue("FALSE")
            .NoArgument()
            .SetFlag(&treeOptions.NormalizeLossInEstimation);

        options
            .AddLongOption("dev-add-ridge-to-target-function")
            .RequiredArgument("FLAG")
            .Help("Gradient descent in leaves will account for full target, not only for loss-part")
            .DefaultValue("FALSE")
            .NoArgument()
            .SetFlag(&treeOptions.AddRidgeToTargetFunctionFlag);

        options
            .AddLongOption("dev-bootstrap-learn")
            .RequiredArgument("FLAG")
            .Help("Use bootstraped weights on learn and test")
            .DefaultValue("FALSE")
            .NoArgument()
            .SetFlag(&treeOptions.BootstrapLearn);

        options
            .AddLongOption("l2-leaf-reg")
            .AddLongName("l2-leaf-regularizer")
            .RequiredArgument("Float")
            .Help("L2 leaf reg")
            .DefaultValue("3")
            .StoreResult(&treeOptions.L2Reg);

        options
            .AddLongOption("dev-dump-free-memory")
            .RequiredArgument("FLAG")
            .SetFlag(&treeOptions.DumpFreeMemoryFlag)
            .NoArgument();

        options
            .AddLongOption("dev-max-ctr-complexity-for-borders-cache")
            .RequiredArgument("INT")
            .DefaultValue("1")
            .StoreResult(&treeOptions.MaxCtrComplexityForBordersCaching);

        options
            .AddLongOption("leaf-estimation-method")
            .RequiredArgument("Leaf estimation method")
            .DefaultValue("Newton")
            .Handler1T<TString>([&](const TString& method) {
                auto lowerCaseMethodName = ToLowerUTF8(method);
                if (lowerCaseMethodName == "newton") {
                    treeOptions.UseNewton = true;
                } else if (lowerCaseMethodName == "gradient") {
                    treeOptions.UseNewton = false;
                } else {
                    ythrow TCatboostException() << "Error: unknown leaf estimation method " << method;
                }
            });
    }
};

template <>
class TOptionsBinder<TBoostingOptions> {
public:
    static void Bind(TBoostingOptions& boostingOptions,
                     NLastGetopt::TOpts& options) {
        options
            .AddLongOption('w', "learning-rate")
            .RequiredArgument("FLOAT")
            .Help("Sets regularization multiplier. Default is 0.03.")
            .DefaultValue("0.03")
            .StoreResult(&boostingOptions.Regularization);

        options
            .AddLongOption('i', "iterations")
            .RequiredArgument("INT")
            .DefaultValue("500")
            .Help("Sets iteration counts")
            .StoreResult(&boostingOptions.IterationCount);

        options
                .AddLongOption("random-strength")
                .RequiredArgument("DOUBLE")
                .DefaultValue("1.0")
                .StoreResult(&boostingOptions.RandomStrength);

        options
            .AddLongOption("dev-min-fold-size")
            .RequiredArgument("INT")
            .Help("Sets min fold size")
            .DefaultValue("128")
            .StoreResult(&boostingOptions.MinFoldSize);

        options
            .AddLongOption("fold-len-multiplier")
            .RequiredArgument("FLOAT")
            .Help("Fold len growth rate")
            .DefaultValue("2")
            .StoreResult(&boostingOptions.GrowthRate);

        options
            .AddLongOption('p', "dev-permutations")
            .RequiredArgument("INT")
            .Help("Sets permutation count")
            .StoreResult(&boostingOptions.PermutationCount);

        options
                .AddLongOption("dev-use-cpu-ram-for-catfeatures")
                .RequiredArgument("INT")
                .Help("Store")
                .SetFlag(&boostingOptions.UseCpuRamForCatFeaturesFlag)
                .NoArgument();

        options.AddLongOption("fold-permutation-block", "Enables fold permutation by blocks of given length, preserving documents order inside each block. Block size should be power of two. ")
                .RequiredArgument("BLOCKSIZE")
                .DefaultValue("32")
                .StoreResult(&boostingOptions.PermutationBlockSize);

        options
                .AddLongOption("dev-disable-dontlookahead")
                .RequiredArgument("INT")
                .Help("Store")
                .SetFlag(&boostingOptions.DisableDontLookAheadFlag)
                .NoArgument();

        options
            .AddLongOption("has-time")
            .RequiredArgument("INT")
            .Help("Use time from dataSet")
            .NoArgument()
            .Handler0([&]() {
                boostingOptions.HasTimeFlag = true;
                boostingOptions.PermutationCount = 1;
            });

        options
            .AddLongOption("dev-skip-calc-scores")
            .RequiredArgument("Flag")
            .Help("Calc scores")
            .Optional()
            .StoreResult(&boostingOptions.CalcScores, false)
            .NoArgument();

        options.AddLongOption("learn-err-log", "file to log error function on train")
            .RequiredArgument("file")
            .Handler1T<TString>([&](const TString& log) {
                boostingOptions.CalcScores = true;
                boostingOptions.LearnErrorLogPath = log;
            });

        options.AddLongOption("test-err-log", "file to log error function on test")
            .RequiredArgument("file")
            .Handler1T<TString>([&](const TString& log) {
                boostingOptions.CalcScores = true;
                boostingOptions.TestErrorLogPath = log;
            });
    }
};

template <>
class TOptionsBinder<TTargetOptions> {
public:
    static void Bind(TTargetOptions& targetOptions,
                     NLastGetopt::TOpts& options) {
        options
            .AddLongOption("loss-function")
            .RequiredArgument("Target")
            .Help("Target (loss function)")
            .Handler1T<ETargetFunction>([&](ETargetFunction target) {
                targetOptions.TargetType = target;
                if (target == ETargetFunction::Logloss) {
                    targetOptions.UseBorderForClassification = true;
                }
            });

        options
            .AddLongOption("border")
            .RequiredArgument("float")
            .Help("Border for binary classification")
            .Handler1T<float>([&](float border) {
                targetOptions.BinClassBorder = border;
                targetOptions.UseBorderForClassification = true;
            });
    }
};
