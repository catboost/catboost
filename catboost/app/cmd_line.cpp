#include "cmd_line.h"
#include <library/grid_creator/binarization.h>

#include <util/generic/strbuf.h>
#include <util/string/iterator.h>
#include <util/string/vector.h>
#include <util/system/info.h>

static yvector<int> ParseIndicesLine(const TStringBuf indicesLine) {
    yvector<int> result;
    for (const auto& t : StringSplitter(indicesLine).Split(':')) {
        const auto s = t.Token();
        int from = FromString<int>(s.Before('-'));
        int to = FromString<int>(s.After('-')) + 1;
        for (int i = from; i < to; ++i) {
            result.push_back(i);
        }
    }
    return result;
}

static NJson::TJsonValue ParsePriors(int indexCount, TStringBuf& priorsLine) {
    NJson::TJsonValue priors;
    while (indexCount--)
    {
        int index;
        GetNext<int>(priorsLine, ':', index);
        priors.AppendValue(index);
    }
    TMaybe<float> prior;
    GetNext<float>(priorsLine, ':', prior);
    while (prior.Defined()) {
        priors.AppendValue(*prior.Get());
        GetNext<float>(priorsLine, ':', prior);
    }
    return priors;
}

void ParseCommandLine(int argc, const char* argv[],
                      NJson::TJsonValue* trainJson,
                      TCmdLineParams* params,
                      TString* paramsPath) {
    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();

    parser.AddLongOption("verbose", "produce verbose output")
        .NoArgument()
        .Handler0([&trainJson]() {
            trainJson->InsertValue("verbose", true);
        });

    parser.AddLongOption("loss-function", "Should be one of: Logloss, CrossEntropy, RMSE, MAE, Quantile, LogLinQuantile, MAPE, Poisson, MultiClass, MultiClassOneVsAll, PairLogit. A loss might have params, then params should be written in format Loss:paramName=value.")
        .RequiredArgument("string")
        .Handler1T<TString>([&trainJson](const TString& target) {
            trainJson->InsertValue("loss_function", target);
        });

    parser.AddLongOption("custom-loss", "A loss might have params, then params should be written in format Loss:paramName=value. Loss should be one of: Logloss, CrossEntropy, RMSE, MAE, Quantile, LogLinQuantile, MAPE, Poisson, MultiClass, MultiClassOneVsAll, PairLogit, R2, AUC, Accuracy, Precision, Recall, F1, TotalF1, MCC, PairAccuracy")
        .RequiredArgument("comma separated list of loss functions")
        .Handler1T<TString>([&trainJson](const TString& lossFunctionsLine) {
            for (const auto& t : StringSplitter(lossFunctionsLine).Split(',')) {
                (*trainJson)["custom_loss"].AppendValue(t.Token());
            }
        });

    parser.AddLongOption('f', "learn-set", "learn set path")
        .RequiredArgument("PATH")
        .StoreResult(&params->LearnFile);

    parser.AddLongOption('t', "test-set", "test set path")
        .RequiredArgument("PATH")
        .DefaultValue("")
        .StoreResult(&params->TestFile);

    parser.AddLongOption("learn-pairs", "path to learn pairs")
        .RequiredArgument("PATH")
        .DefaultValue("")
        .StoreResult(&params->LearnPairsFile);

    parser.AddLongOption("test-pairs", "path to test pairs")
        .RequiredArgument("PATH")
        .DefaultValue("")
        .StoreResult(&params->TestPairsFile);

    parser.AddLongOption("column-description", "column desctiption file name")
        .AddLongName("cd")
        .RequiredArgument("PATH")
        .StoreResult(&params->CdFile);

    parser.AddLongOption("delimiter", "Learning and training sets delimiter")
        .RequiredArgument("SYMBOL")
        .Handler1T<TString>([&params](const TString& oneChar) {
            CB_ENSURE(oneChar.size() == 1, "only single char delimiters supported");
            params->Delimiter = oneChar[0];
        });

    parser.AddLongOption("has-header", "Read first line as header")
        .NoArgument()
        .StoreValue(&params->HasHeaders, true);

    parser.AddLongOption('m', "model-file", "model file name")
        .RequiredArgument("PATH")
        .DefaultValue("model.bin")
        .StoreResult(&params->ModelFileName);

    parser.AddLongOption("eval-file", "eval output file name")
        .RequiredArgument("PATH")
        .DefaultValue("")
        .StoreResult(&params->EvalFileName);

    parser.AddLongOption('i', "iterations", "iterations count")
        .RequiredArgument("ITERATIONS")
        .DefaultValue("500")
        .Handler1T<int>([&trainJson](int iterations) {
            trainJson->InsertValue("iterations", iterations);
        });

    parser.AddLongOption("border", "target border for Logloss mode")
        .RequiredArgument("float")
        .Handler1T<float>([&trainJson](float border) {
            trainJson->InsertValue("border", border);
        });

    parser.AddLongOption("gradient-iterations", "gradient iterations count")
        .RequiredArgument("int")
        .Handler1T<int>([&trainJson](int gsteps) {
            trainJson->InsertValue("gradient_iterations", gsteps);
        });

    parser.AddLongOption('n', "depth", "tree depth")
        .RequiredArgument("int")
        .DefaultValue("6")
        .Handler1T<int>([&trainJson](int depth) {
            trainJson->InsertValue("depth", depth);
        });

    parser.AddLongOption('w', "learning-rate", "learning rate")
        .RequiredArgument("float")
        .Handler1T<float>([&trainJson](float w) {
            trainJson->InsertValue("learning_rate", w);
        });

    parser.AddLongOption("rsm", "random subspace method (feature bagging)")
        .RequiredArgument("float")
        .Handler1T<float>([&trainJson](float rsm) {
            trainJson->InsertValue("rsm", rsm);
        });

    parser.AddCharOption('T', "worker thread count, default - min(core count, 8)")
        .AddLongName("thread-count")
        .RequiredArgument("count")
        .Handler1T<int>([&trainJson](int count) {
            trainJson->InsertValue("thread_count", count);
        });

    parser.AddCharOption('r', "random seed")
        .AddLongName("random-seed")
        .RequiredArgument("count")
        .Handler1T<int>([&trainJson](int seed) {
            trainJson->InsertValue("random_seed", seed);
        });

    parser.AddLongOption("ctr-border-count", "count of ctr borders. Should be in range [1, 255]")
        .RequiredArgument("int")
        .Handler1T<int>([&trainJson](int cnt) {
            trainJson->InsertValue("ctr_border_count", cnt);
        });

    parser.AddLongOption("max-ctr-complexity", "max count of cat features for tree ctr")
        .RequiredArgument("int")
        .Handler1T<int>([&trainJson](int cnt) {
            trainJson->InsertValue("max_ctr_complexity", cnt);
        });

    // TODO(annaveronika): Save properly in json.
    parser.AddLongOption("ctr-description", "Ctr description should be written in format CtrType[:TargetBorderCount][:TargetBorderType]. CtrType should be one of: Borders, Buckets, BinarizedTargetMeanValue, Counter. TargetBorderType should be one of: Median, GreedyLogSum, UniformAndQuantiles, MinEntropy, MaxLogSum, Uniform")
        .AddLongName("ctr")
        .RequiredArgument("comma separated list of ctr descriptions")
        .Handler1T<TString>([&trainJson](const TString& ctrDescriptionLine) {
            for (const auto& t : StringSplitter(ctrDescriptionLine).Split(',')) {
                (*trainJson)["ctr_description"].AppendValue(t.Token());
        } });

    parser.AddLongOption('x', "border-count", "count of borders per float feature. Should be in range [1, 255]")
        .RequiredArgument("int")
        .Handler1T<int>([&trainJson](int cnt) {
            trainJson->InsertValue("border_count", cnt);
        });

    parser.AddLongOption("auto-stop-pval", "set threshold for overfitting detector and stop matrixnet automaticaly. For good results use threshold in [1e-10, 1e-2]. Requires any test part.")
        .AddLongName("od-pval")
        .RequiredArgument("float")
        .Handler1T<float>([&trainJson](float pval) {
            trainJson->InsertValue("od_pval", pval);
        });

    parser.AddLongOption("overfitting-detector-iterations-wait", "number of iterations which overfitting detector will wait after new best error")
        .AddLongName("od-wait")
        .RequiredArgument("int")
        .Handler1T<int>([&trainJson](int iters) {
            trainJson->InsertValue("od_wait", iters);
        });

    parser.AddLongOption("overfitting-detector-type", "Should be one of {IncToDec, Iter}")
        .AddLongName("od-type")
        .RequiredArgument("detector-type")
        .Handler1T<TString>([&trainJson](const TString& type) {
            trainJson->InsertValue("od_type", type);
        });

    parser.AddLongOption("use-best-model", "save all trees until best iteration on test")
        .NoArgument()
        .Handler0([&trainJson]() {
            trainJson->InsertValue("use_best_model", true);
        });

    parser.AddLongOption("fstr-file", "Save fstr to this file")
        .RequiredArgument("filename")
        .StoreResult(&params->FstrRegularFileName);

    parser.AddLongOption("fstr-internal-file", "Save internal fstr values to this file")
        .RequiredArgument("filename")
        .StoreResult(&params->FstrInternalFileName);

    parser.AddLongOption("fold-permutation-block", "Enables fold permutation by blocks of given length, preserving documents order inside each block.")
        .RequiredArgument("BLOCKSIZE")
        .Handler1T<int>([&trainJson](int cnt) {
            trainJson->InsertValue("fold_permutation_block_size", cnt);
        });

    parser.AddLongOption("detailed-profile", "use detailed profile")
        .NoArgument()
        .Handler0([&trainJson]() {
            trainJson->InsertValue("detailed_profile", true);
        });

    parser.AddLongOption("leaf-estimation-method", "One of {Newton, Gradient}")
        .RequiredArgument("method-name")
        .Handler1T<TString>([&trainJson](const TString& method) {
            trainJson->InsertValue("leaf_estimation_method", method);
        });

    parser.AddLongOption("counter-calc-method", "Should be one of {Full, SkipTest}")
        .RequiredArgument("method-name")
        .Handler1T<TString>([&trainJson](const TString& method) {
            trainJson->InsertValue("counter_calc_method", method);
        });

    parser.AddLongOption('I', "ignore-features", "don't use the specified features in the learn set (the features are separated by colon and can be specified as an inclusive interval, for example: -I 4:78-89:312)")
        .RequiredArgument("INDEXES")
        .Handler1T<TString>([&trainJson](const TString& indicesLine) {
            auto ignoredFeatures = ParseIndicesLine(indicesLine);
            for (int f : ignoredFeatures) {
                (*trainJson)["ignored_features"].AppendValue(f);
            }
        });

    parser.AddCharOption('X', "xross validation, test on fold n of k, n is 0-based")
        .RequiredArgument("n/k")
        .Handler1T<TStringBuf>([&params](const TStringBuf& str) {
            Split(str, '/', params->CvParams.FoldIdx, params->CvParams.FoldCount);
            params->CvParams.Inverted = false;
        });

    parser.AddCharOption('Y', "inverted xross validation, train on fold n of k, n is 0-based")
        .RequiredArgument("n/k")
        .Handler1T<TStringBuf>([&params](const TStringBuf& str) {
            Split(str, '/', params->CvParams.FoldIdx, params->CvParams.FoldCount);
            params->CvParams.Inverted = true;
        });

    parser.AddLongOption("cv-rand", "cross-validation random seed")
        .RequiredArgument("seed")
        .StoreResult(&params->CvParams.RandSeed);

    parser.AddLongOption("learn-err-log", "file to log error function on train")
        .RequiredArgument("file")
        .Handler1T<TString>([&trainJson](const TString& log) {
            trainJson->InsertValue("learn_error_log", log);
        });

    parser.AddLongOption("test-err-log", "file to log error function on test")
        .RequiredArgument("file")
        .Handler1T<TString>([&trainJson](const TString& log) {
            trainJson->InsertValue("test_error_log", log);
        });

    parser.AddLongOption("feature-border-type", "Should be one of: Median, GreedyLogSum, UniformAndQuantiles, MinEntropy, MaxLogSum")
        .RequiredArgument("border-type")
        .Handler1T<TString>([&trainJson](const TString& type) {
            trainJson->InsertValue("feature_border_type", type);
        });

    parser.AddLongOption("l2-leaf-reg", "Regularization value. Should be >= 0")
        .RequiredArgument("float")
        .Handler1T<float>([&trainJson](float reg) {
            trainJson->InsertValue("l2_leaf_reg", reg);
        });

    parser.AddLongOption("has-time", "Use dataset order as time")
        .NoArgument()
        .Handler0([&trainJson]() {
            trainJson->InsertValue("has_time", true);
        });

    parser.AddLongOption("priors", "Priors in format p1:p2:p3")
        .RequiredArgument("priorsLine")
        .Handler1T<TString>([&trainJson](const TString& priorsLine) {
            TStringBuf priorsLineBuf(priorsLine);
            (*trainJson)["priors"] = ParsePriors(0, priorsLineBuf);
        });

    parser.AddLongOption("ctr-priors")
        .RequiredArgument("priors-line")
        .Handler1T<TString>([&trainJson](const TString& priorsDesctiption) {
        for (const auto& t : StringSplitter(priorsDesctiption).Split(',')) {
            TStringBuf priorsLineBuf(t.Token());
            (*trainJson)["ctr_priors"].AppendValue(ParsePriors(1, priorsLineBuf));
        } })
        .Help("You might provide custom priors for some ctr. They will be used instead of default ones. Format is: c1Idx:prior1:prior2:prior3,c2Idx:prior1");

    parser.AddLongOption("feature-priors")
        .RequiredArgument("priors-line")
        .Handler1T<TString>([&trainJson](const TString& priorsDesctiption) {
            for (const auto& t : StringSplitter(priorsDesctiption).Split(',')) {
                TStringBuf priorsLineBuf(t.Token());
                (*trainJson)["feature_priors"].AppendValue(ParsePriors(1, priorsLineBuf));
        } })
        .Help("You might provide custom priors for some features. They will be used instead of default ones or ones given in ctr-priors. Format is: f1Idx:prior1:prior2:prior3,f2Idx:prior1");

    parser.AddLongOption("feature-ctr-priors")
        .RequiredArgument("priors-line")
        .Handler1T<TString>([&trainJson](const TString& priorsDesctiption) {
            for (const auto& t : StringSplitter(priorsDesctiption).Split(',')) {
                TStringBuf priorsLineBuf(t.Token());
                (*trainJson)["feature_ctr_priors"].AppendValue(ParsePriors(2, priorsLineBuf));
            } })
        .Help("You might provide custom priors for some pairs of feature and ctr. They will be used instead of default ones or given in ctr-priors or feature-priors. Format is: f1Idx:c1Idx:prior1:prior2:prior3,f2Idx:c2Idx:prior1");

    parser.AddLongOption("name", "name to be displayed in visualizator")
        .RequiredArgument("name")
        .Handler1T<TString>([&trainJson](const TString& name) {
            trainJson->InsertValue("name", name);
        });

    parser.AddLongOption("train-dir", "directory to store train logs")
        .RequiredArgument("PATH")
        .Handler1T<TString>([&trainJson](const TString& path) {
            trainJson->InsertValue("train_dir", path);
        });

    parser.AddLongOption("snapshot-file", "use progress file for restoring progress after crashes")
        .RequiredArgument("PATH")
        .Handler1T<TString>([&trainJson](const TString& path) {
            trainJson->InsertValue("save_snapshot", true);
            trainJson->InsertValue("snapshot_file", path);
        });

    parser.AddLongOption("fold-len-multiplier", "Fold length multiplier. Should be greater than 1")
        .RequiredArgument("float")
        .Handler1T<float>([&trainJson](float multiplier) {
            trainJson->InsertValue("fold_len_multiplier", multiplier);
        });

    parser.AddLongOption("ctr-leaf-count-limit",
                         "Limit maximum ctr leaf count. If there are more leafs than limit, it'll select top values by frequency and put the rest into trashbucket. This option reduces resulting model size and amount of memory used during training. But it might affect the resulting quality.")
        .RequiredArgument("maxLeafCount")
        .Handler1T<ui64>([&trainJson](ui64 maxLeafCount) {
            trainJson->InsertValue("ctr_leaf_count_limit", maxLeafCount);
        });

    parser.AddLongOption("store-all-simple-ctr", "Do not limit simple ctr leafs count to topN, store all values from learn set")
        .NoArgument()
        .Handler0([&trainJson]() {
            trainJson->InsertValue("store_all_simple_ctr", true);
        });

    parser.AddLongOption("eval-metric")
        .RequiredArgument("string")
        .Handler1T<TString>([&trainJson](const TString& metric) {
            trainJson->InsertValue("eval_metric", metric);
        })
        .Help("evaluation metric for overfitting detector (if enabled) and best model "
              "selection in format MetricName:param=value. If not specified default metric for objective is used.");

    parser.AddLongOption("prediction-type", "Should be one of: Probability, Class, RawFormulaVal")
        .RequiredArgument("prediction-type")
        .Handler1T<TString>([&trainJson](const TString& predictionType) {
            trainJson->InsertValue("prediction_type", predictionType);
        });

    parser.AddLongOption("nan-mode", "Should be one of: {Min, Max, Forbidden}")
        .RequiredArgument("nan-mode")
        .Handler1T<TString>([&trainJson](const TString& nanMode) {
            trainJson->InsertValue("nan_mode", nanMode);
    });

    parser.AddLongOption("params-file", "Path to JSON file with params.")
        .RequiredArgument("PATH")
        .StoreResult(paramsPath)
        .Help("If param is given in json file and in command line then one from command line will be used.");

    parser.AddLongOption("classes-count", "number of classes")
        .RequiredArgument("int")
        .Handler1T<int>([&trainJson](const int classesCount) {
            trainJson->InsertValue("classes_count", classesCount);
        })
        .Help("Takes effect only with MutliClass loss function. If classes-count is given (and class-names is not given), then each class label should be less than that number.");

    parser.AddLongOption("class-names", "names for classes.")
        .RequiredArgument("comma separated list of names")
        .Handler1T<TString>([&trainJson](const TString& namesLine) {
            for (const auto& t : StringSplitter(namesLine).Split(',')) {
                (*trainJson)["class_names"].AppendValue(t.Token());
            }
        })
        .Help("Takes effect only with MutliClass/LogLoss loss functions. Wihout this parameter classes are 0, 1, ..., classes-count - 1");

    parser.AddLongOption("class-weights", "Weights for classes.")
        .RequiredArgument("comma separated list of weights")
        .Handler1T<TString>([&trainJson](const TString& weightsLine) {
            for (const auto& t : StringSplitter(weightsLine).Split(',')) {
                (*trainJson)["class_weights"].AppendValue(FromString<float>(t.Token()));
            }
        })
        .Help("Takes effect only with MutliClass/LogLoss loss functions. Number of classes indicated by classes-count, class-names and class-weights should be the same");

    parser.AddLongOption("one-hot-max-size")
        .RequiredArgument("size_t")
        .Handler1T<size_t>([&trainJson](const size_t oneHotMaxSize) {
            trainJson->InsertValue("one_hot_max_size", oneHotMaxSize);
        })
        .Help("If parameter is specified than features with no more than specified value different values will be converted to float features using one-hot encoding. No ctrs will be calculated on this features.");

    parser.AddLongOption("random-strength")
        .RequiredArgument("float")
        .Handler1T<float>([&trainJson](float randomStrength) {
            trainJson->InsertValue("random_strength", randomStrength);
        })
        .Help("score stdandart deviation multiplier");

    parser.AddLongOption("used-ram-limit", "Try to limit used memory. WARNING: This option affects CTR memory usage only.\nAllowed suffixes: GB, MB, KB in different cases")
        .RequiredArgument("TARGET_RSS")
        .Handler1T<TString>([&trainJson](const TString& param) {
            TString sizeLine = param;
            ui64 sizeMultiplier = 1;
            if (sizeLine.back() == 'b' || sizeLine.back() == 'B') {
                sizeLine.pop_back();
                switch (sizeLine.back()) {
                    case 'k':
                    case 'K':
                        sizeMultiplier = 1024;
                        break;
                    case 'm':
                    case 'M':
                        sizeMultiplier = 1024 * 1024;
                        break;
                    case 'g':
                    case 'G':
                        sizeMultiplier = 1024 * 1024 * 1024;
                        break;
                    default:
                        CB_ENSURE(false, "unknown size suffix: " << param);
                }
                sizeLine.pop_back();
            }
            ui64 ramLimit = sizeMultiplier * FromString<ui64>(sizeLine);
            trainJson->InsertValue("used_ram_limit", ramLimit);
        });

    parser.AddLongOption("print-trees")
        .NoArgument()
        .Handler0([&trainJson]() {
            trainJson->InsertValue("print_trees", true);
        })
        .Help("Print tree structure and split scores to stdout.");

    parser.AddLongOption("developer-mode")
        .AddLongName("dev")
        .NoArgument()
        .Handler0([&trainJson]() {
            trainJson->InsertValue("verbose", true);
            trainJson->InsertValue("developer_mode", true);
        })
        .Help("Profile mode for developers.");

    parser.AddLongOption("bagging-temperature")
        .AddLongName("tmp")
        .RequiredArgument("float")
        .Handler1T<float>([&trainJson](float baggingTemperature) {
            trainJson->InsertValue("bagging_temperature", baggingTemperature);
        })
        .Help("Controls intensity of Bayesian bagging. The higher the temperature the more aggressive bagging is. Typical values are in range [0, 1] (0 - no bagging, 1 - default).");

    parser.AddLongOption("approx-on-full-history")
        .NoArgument()
        .Handler0([&trainJson]() {
            trainJson->InsertValue("approx_on_full_history", true);
        })
        .Help("Use full history to calculate approxes.");

    parser.SetFreeArgsNum(0);

    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};
}

void TAnalyticalModeCommonParams::BindParserOpts(NLastGetopt::TOpts& parser) {
    parser.AddLongOption('m', "model-path", "path to model")
        .StoreResult(&ModelFileName)
        .DefaultValue("model.bin");
    parser.AddLongOption("input-path", "input path, use \"-\" to read from stdin")
        .StoreResult(&InputPath)
        .DefaultValue("-");
    parser.AddLongOption("column-description", "path to columns descriptions")
        .AddLongName("cd")
        .StoreResult(&CdFile)
        .DefaultValue("");
    parser.AddLongOption('o', "output-path", "output result path, use \"-\" to output to stdout")
        .StoreResult(&OutputPath)
        .DefaultValue("-");
    parser.AddLongOption('T', "thread-count")
        .StoreResult(&ThreadCount)
        .DefaultValue("1");
    parser.AddLongOption("delimiter", "delimiter")
            .DefaultValue("\t")
            .Handler1T<TString>([&](const TString& oneChar) {
                CB_ENSURE(oneChar.size() == 1, "only single char delimiters supported");
                Delimiter = oneChar[0];
            });
    parser.AddLongOption("has-header", "has header flag")
            .NoArgument()
            .StoreValue(&HasHeader, true);
    parser.AddLongOption("class-names", "names for classes.")
            .RequiredArgument("comma separated list of names")
            .Handler1T<TString>([&](const TString& namesLine) {
                for (const auto& t : StringSplitter(namesLine).Split(',')) {
                    ClassNames.push_back(FromString<TString>(t.Token()));
                }
            });
}
