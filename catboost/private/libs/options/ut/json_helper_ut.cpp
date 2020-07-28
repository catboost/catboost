#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/json/json_reader.h>
#include <catboost/private/libs/options/option.h>
#include <catboost/private/libs/options/json_helper.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/plain_options_helper.h>

#include <util/generic/xrange.h>
#include <util/generic/ymath.h>


Y_UNIT_TEST_SUITE(TJsonHelperTest) {
    using namespace NCatboostOptions;

    Y_UNIT_TEST(TestSimpleSerializtion) {
        TStringBuf json = ""
                          "{\n"
                          "  \"double_val\": 10.01,\n"
                          "  \"int_val\": 42,\n"
                          "  \"option_val\": 10,\n"
                          "  \"enum_val\": \"GPU\",\n"
                          "  \"string_val\": \"text\",\n"
                          "  \"enum_arr\": [\"GPU\",\"CPU\",\"GPU\"],\n"
                          "  \"bool_val\": true\n"
                          "}"
                          "";

        NJson::TJsonValue tree;
        NJson::ReadJsonTree(json, &tree);

        double refDouble = 10.01;
        TOption<int> option("option_val", 10);
        int refInt = 42;
        ETaskType refType = ETaskType::GPU;
        TString refString = "text";
        bool refBool = true;
        TVector<ETaskType> enumArrRef = {ETaskType::GPU, ETaskType::CPU, ETaskType::GPU};

        double parsedDouble = 1;
        int parsedInt = 4;
        ETaskType parsedType = ETaskType::CPU;
        TString parsedString = "sss";
        bool parsedBool = false;
        TVector<ETaskType> parsedEnumArr;
        TOption<int> parsedOption("option_val", 22);

        TJsonFieldHelper<double>::Read(tree["double_val"], &parsedDouble);
        TJsonFieldHelper<int>::Read(tree["int_val"], &parsedInt);
        TJsonFieldHelper<ETaskType>::Read(tree["enum_val"], &parsedType);
        TJsonFieldHelper<TString>::Read(tree["string_val"], &parsedString);
        TJsonFieldHelper<decltype(enumArrRef)>::Read(tree["enum_arr"], &parsedEnumArr);
        TJsonFieldHelper<decltype(option)>::Read(tree, &parsedOption);
        TJsonFieldHelper<bool>::Read(tree["bool_val"], &parsedBool);

        UNIT_ASSERT_VALUES_EQUAL(parsedDouble, refDouble);
        UNIT_ASSERT_VALUES_EQUAL(parsedInt, refInt);
        UNIT_ASSERT_VALUES_EQUAL(parsedType, refType);
        UNIT_ASSERT_VALUES_EQUAL(parsedEnumArr, enumArrRef);
        UNIT_ASSERT_VALUES_EQUAL(parsedString, refString);
        UNIT_ASSERT_VALUES_EQUAL(parsedBool, refBool);
        UNIT_ASSERT_VALUES_EQUAL(parsedOption.Get(), option.Get());

        NJson::TJsonValue serializedTree;
        TJsonFieldHelper<double>::Write(parsedDouble, &serializedTree["double_val"]);
        TJsonFieldHelper<int>::Write(parsedInt, &serializedTree["int_val"]);
        TJsonFieldHelper<ETaskType>::Write(parsedType, &serializedTree["enum_val"]);
        TJsonFieldHelper<TString>::Write(parsedString, &serializedTree["string_val"]);
        TJsonFieldHelper<decltype(enumArrRef)>::Write(parsedEnumArr, &serializedTree["enum_arr"]);
        TJsonFieldHelper<bool>::Write(parsedBool, &serializedTree["bool_val"]);
        TJsonFieldHelper<decltype(parsedOption)>::Write(parsedOption, &serializedTree);

        UNIT_ASSERT_VALUES_EQUAL(serializedTree["double_val"], tree["double_val"]);
        UNIT_ASSERT_VALUES_EQUAL(serializedTree["int_val"], tree["int_val"]);
        UNIT_ASSERT_VALUES_EQUAL(serializedTree["enum_val"], tree["enum_val"]);
        UNIT_ASSERT_VALUES_EQUAL(serializedTree["enum_arr"], tree["enum_arr"]);
        UNIT_ASSERT_VALUES_EQUAL(serializedTree["string_val"], tree["string_val"]);
        UNIT_ASSERT_VALUES_EQUAL(serializedTree["bool_val"], tree["bool_val"]);
        UNIT_ASSERT_VALUES_EQUAL(serializedTree["option_val"], tree["option_val"]);
    }

    Y_UNIT_TEST(TestJsonSerializationWithFloatingPointValues) {
        TVector<double> values = {1.0f, 0.4f, 12.33f, 1.e-6f, 0.0f};

        NJson::TJsonValue jsonArray(NJson::JSON_ARRAY);
        for (auto value : values) {
            jsonArray.AppendValue(value);
        }

        const TString serialized = WriteTJsonValue(jsonArray);
        const NJson::TJsonValue restoredJson = ReadTJsonValue(serialized);

        const NJson::TJsonValue::TArray restoredJsonArray = restoredJson.GetArraySafe();

        UNIT_ASSERT_VALUES_EQUAL(restoredJsonArray.size(), values.size());

        for (auto i : xrange(values.size())) {
            UNIT_ASSERT(FuzzyEquals(restoredJsonArray[i].GetDoubleSafe(), values[i]));
        }
    }

    Y_UNIT_TEST(TestUnimplementedAwareOptions) {
        TStringBuf jsonStr = ""
                             "{\n"
                             " \"cpu_unimplemented\": 10.01,\n"
                             "  \"gpu_unimplemented\": 42,\n"
                             "}"
                             "";

        NJson::TJsonValue tree;
        NJson::ReadJsonTree(jsonStr, &tree);

        TUnimplementedAwareOption<double, TSupportedTasks<ETaskType::GPU>> cpuUnimplemented("cpu_unimplemented", 1.0, ETaskType::GPU);
        cpuUnimplemented.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::SkipWithWarning);
        TUnimplementedAwareOption<int, TSupportedTasks<ETaskType::CPU>> gpuUnimplemented("gpu_unimplemented", 10, ETaskType::GPU);
        gpuUnimplemented.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::SkipWithWarning);

        TUnimplementedAwareOptionsLoader gpuLoader(tree);
        gpuLoader.LoadMany(&cpuUnimplemented,
                           &gpuUnimplemented);

        UNIT_ASSERT_VALUES_EQUAL(cpuUnimplemented.Get(), 10.01);
        UNIT_ASSERT_VALUES_EQUAL(gpuUnimplemented.GetUnchecked(), 10);

        cpuUnimplemented = 100;
        gpuUnimplemented = 105;
        cpuUnimplemented.SetTaskType(ETaskType::CPU);
        gpuUnimplemented.SetTaskType(ETaskType::CPU);

        TUnimplementedAwareOptionsLoader cpuLoader(tree);
        cpuLoader.LoadMany(&cpuUnimplemented,
                           &gpuUnimplemented);

        UNIT_ASSERT_VALUES_EQUAL(cpuUnimplemented.GetUnchecked(), 100);
        UNIT_ASSERT_VALUES_EQUAL(gpuUnimplemented.Get(), 42);
    }

    Y_UNIT_TEST(TestDisableOption) {
        TStringBuf jsonStr = ""
                             "{\n"
                             " \"option_1\": 102,\n"
                             "  \"option2\": 42,\n"
                             "}"
                             "";

        NJson::TJsonValue tree;
        NJson::ReadJsonTree(jsonStr, &tree);
        TOption<i32> option1("option_1", 100);
        TOption<i32> option2("option_2", 40);
        option2.SetDisabledFlag(true);

        TUnimplementedAwareOptionsLoader loader(tree);
        loader.LoadMany(&option1, &option2);

        UNIT_ASSERT_VALUES_EQUAL(option1.Get(), 102);
        UNIT_ASSERT_EXCEPTION(option2.Get(), TCatBoostException);

        NJson::TJsonValue tree2;
        SaveFields(&tree2, option1, option2);
        UNIT_ASSERT_VALUES_EQUAL(ToString<NJson::TJsonValue>(tree2), "{\"option_1\":102}");
    }
        Y_UNIT_TEST(TestPlainAndReversePlainSerialization) {
            TString jsonPlain = ""
                "{\n"
                "  \"loss_function\": \"Logloss\",\n"
                "  \"custom_metric\": \"CrossEntropy\",\n"
                "  \"eval_metric\": \"RMSE\",\n"

                "  \"use_best_model\": true,\n"
                "  \"verbose\": 0,\n"

                "  \"iterations\": 100,\n"
                "  \"learning_rate\": 0.1,\n"
                "  \"depth\": 3,\n"
                "  \"fold_len_multiplier\": 2,\n"
                "  \"approx_on_full_history\": false,\n"
                "  \"fold_permutation_block\": 16,\n"

                "  \"od_pval\": 0.001,\n"
                "  \"od_wait\": 10,\n"
                "  \"od_type\": \"IncToDec\",\n"

                "  \"leaf_estimation_iterations\": 15,\n"
                "  \"leaf_estimation_backtracking\": \"AnyImprovement\",\n"

                "  \"bootstrap_type\": \"Bernoulli\",\n"
                "  \"bagging_temperature\": 36.6,\n"

                "  \"per_float_feature_quantization\": [\"1:border_count=4\", \"2:nan_mode=Max,border_type=MinEntropy\"]\n"

                "}"
                "";

            NJson::TJsonValue plainOptions;
            NJson::ReadJsonTree(jsonPlain, &plainOptions);

            // reference variables
            int refIterations = 100;
            double refLearningRate = 0.1;
            int refFoldLenMultiplier = 2;
            bool refApproxOnFullHistory = false;
            int refFoldPermutationBlock = 16;

            double refOdPval = 0.001;
            int refOdWait = 10;
            TString refOdType = "IncToDec";

            int refDepth = 3;
            int refLeafEstimationIterations = 15;
            TString refLeafEstimationBacktracking = "AnyImprovement";

            TString refBootstrapType = "Bernoulli";
            double refBaggingTemperature = 36.6;

            TVector<TString> refPerFloatFeatureQuantization = {"1:border_count=4", "2:nan_mode=Max,border_type=MinEntropy"};

            // parsed variables
            int parsedIterations = 7;
            double parsedLearningRate = 10.9;
            int parsedFoldLenMultiplier = 8;
            bool parsedApproxOnFullHistory = true;
            int parsedFoldPermutationBlock = 33;

            double parsedOdPval = 1.5;
            int parsedOdWait = 1000;
            TString parsedOdType = "Moscow";

            int parsedDepth = 10000;
            int parsedLeafEstimationIterations = 1000;
            TString parsedLeafEstimationBacktracking = "Minsk";

            TString parsedBootstrapType = "Laplace";
            double parsedBaggingTemperature = 39.0;

            TVector<TString> parsedTextProcessing = {"foo", "bar"};
            TVector<TString> parsedPerFloatFeatureQuantization = {"foo", "bar"};

            NJson::TJsonValue trainOptionsJson;
            NJson::TJsonValue outputFilesOptionsJson;
            NCatboostOptions::PlainJsonToOptions(plainOptions, &trainOptionsJson, &outputFilesOptionsJson);

            // boosting options
            auto& boostingOptions = trainOptionsJson["boosting_options"];
            TJsonFieldHelper<int>::Read(boostingOptions["iterations"], &parsedIterations);
            TJsonFieldHelper<double>::Read(boostingOptions["learning_rate"], &parsedLearningRate);
            TJsonFieldHelper<int>::Read(boostingOptions["fold_len_multiplier"], &parsedFoldLenMultiplier);
            TJsonFieldHelper<bool>::Read(boostingOptions["approx_on_full_history"], &parsedApproxOnFullHistory);
            TJsonFieldHelper<int>::Read(boostingOptions["fold_permutation_block"], &parsedFoldPermutationBlock);

            // od_config options
            auto& odConfig = boostingOptions["od_config"];
            TJsonFieldHelper<double>::Read(odConfig["stop_pvalue"], &parsedOdPval);
            TJsonFieldHelper<int>::Read(odConfig["wait_iterations"], &parsedOdWait);
            TJsonFieldHelper<TString>::Read(odConfig["type"], &parsedOdType);

            // tree_learner options
            auto& treeOptions = trainOptionsJson["tree_learner_options"];
            TJsonFieldHelper<int>::Read(treeOptions["depth"], &parsedDepth);
            TJsonFieldHelper<int>::Read(treeOptions["leaf_estimation_iterations"], &parsedLeafEstimationIterations);
            TJsonFieldHelper<TString>::Read(treeOptions["leaf_estimation_backtracking"], &parsedLeafEstimationBacktracking);

            // bootstrap
            auto& bootstrapOptions = treeOptions["bootstrap"];
            TJsonFieldHelper<TString>::Read(bootstrapOptions["type"], &parsedBootstrapType);
            TJsonFieldHelper<double>::Read(bootstrapOptions["bagging_temperature"], &parsedBaggingTemperature);

            // plainOptions to trainOptionsJson and outputFilesOptionsJson using PlainJsonToOptions
            UNIT_ASSERT_VALUES_EQUAL(parsedIterations, refIterations);
            UNIT_ASSERT_VALUES_EQUAL(parsedLearningRate, refLearningRate);
            UNIT_ASSERT_VALUES_EQUAL(parsedFoldLenMultiplier, refFoldLenMultiplier);
            UNIT_ASSERT_VALUES_EQUAL(parsedApproxOnFullHistory, refApproxOnFullHistory);
            UNIT_ASSERT_VALUES_EQUAL(parsedFoldPermutationBlock, refFoldPermutationBlock);

            UNIT_ASSERT_VALUES_EQUAL(parsedOdPval, refOdPval);
            UNIT_ASSERT_VALUES_EQUAL(parsedOdWait, refOdWait);
            UNIT_ASSERT_VALUES_EQUAL(parsedOdType, refOdType);

            UNIT_ASSERT_VALUES_EQUAL(parsedDepth, refDepth);
            UNIT_ASSERT_VALUES_EQUAL(parsedLeafEstimationIterations, refLeafEstimationIterations);
            UNIT_ASSERT_VALUES_EQUAL(parsedLeafEstimationBacktracking, refLeafEstimationBacktracking);

            UNIT_ASSERT_VALUES_EQUAL(parsedBootstrapType, refBootstrapType);
            UNIT_ASSERT_VALUES_EQUAL(parsedBaggingTemperature, refBaggingTemperature);

            // now test reverse transformation
            NJson::TJsonValue reversePlainOptions;
            NCatboostOptions::ConvertOptionsToPlainJson(trainOptionsJson, outputFilesOptionsJson, &reversePlainOptions);

            int reverseParsedIterations = 7;
            double reverseParsedLearningRate = 10.9;
            int reverseParsedFoldLenMultiplier = 8;
            bool reverseParsedApproxOnFullHistory = true;
            int reverseParsedFoldPermutationBlock = 33;

            double reverseParsedOdPval = 1.5;
            int reverseParsedOdWait = 1000;
            TString reverseParsedOdType = "Moscow";

            int reverseParsedDepth = 10000;
            int reverseParsedLeafEstimationIterations = 1000;
            TString reverseParsedLeafEstimationBacktracking = "Minsk";

            TString reverseParsedBootstrapType = "Laplace";
            double reverseParsedBaggingTemperature = 39.0;

            TVector<TString> reversePerFloatFeatureQuantization = {"foo", "bar"};

            TJsonFieldHelper<int>::Read(reversePlainOptions["iterations"], &reverseParsedIterations);
            TJsonFieldHelper<double>::Read(reversePlainOptions["learning_rate"], &reverseParsedLearningRate);
            TJsonFieldHelper<int>::Read(reversePlainOptions["fold_len_multiplier"], &reverseParsedFoldLenMultiplier);
            TJsonFieldHelper<bool>::Read(reversePlainOptions["approx_on_full_history"], &reverseParsedApproxOnFullHistory);
            TJsonFieldHelper<int>::Read(reversePlainOptions["fold_permutation_block"], &reverseParsedFoldPermutationBlock);

            TJsonFieldHelper<double>::Read(reversePlainOptions["od_pval"], &reverseParsedOdPval);
            TJsonFieldHelper<int>::Read(reversePlainOptions["od_wait"], &reverseParsedOdWait);
            TJsonFieldHelper<TString>::Read(reversePlainOptions["od_type"], &reverseParsedOdType);

            TJsonFieldHelper<int>::Read(reversePlainOptions["depth"], &reverseParsedDepth);
            TJsonFieldHelper<int>::Read(reversePlainOptions["leaf_estimation_iterations"], &reverseParsedLeafEstimationIterations);
            TJsonFieldHelper<TString>::Read(reversePlainOptions["leaf_estimation_backtracking"], &reverseParsedLeafEstimationBacktracking);

            TJsonFieldHelper<TString>::Read(reversePlainOptions["bootstrap_type"], &reverseParsedBootstrapType);
            TJsonFieldHelper<double>::Read(reversePlainOptions["bagging_temperature"], &reverseParsedBaggingTemperature);

            TJsonFieldHelper<TVector<TString>>::Read(reversePlainOptions["per_float_feature_quantization"], &reversePerFloatFeatureQuantization);

            // plainOptions == reversePlainOptions
            UNIT_ASSERT_VALUES_EQUAL(reverseParsedIterations, refIterations);
            UNIT_ASSERT_VALUES_EQUAL(reverseParsedLearningRate, refLearningRate);
            UNIT_ASSERT_VALUES_EQUAL(reverseParsedFoldLenMultiplier, refFoldLenMultiplier);
            UNIT_ASSERT_VALUES_EQUAL(reverseParsedApproxOnFullHistory, refApproxOnFullHistory);
            UNIT_ASSERT_VALUES_EQUAL(reverseParsedFoldPermutationBlock, refFoldPermutationBlock);

            UNIT_ASSERT_VALUES_EQUAL(reverseParsedOdPval, refOdPval);
            UNIT_ASSERT_VALUES_EQUAL(reverseParsedOdWait, refOdWait);
            UNIT_ASSERT_VALUES_EQUAL(reverseParsedOdType, refOdType);

            UNIT_ASSERT_VALUES_EQUAL(reverseParsedDepth, refDepth);
            UNIT_ASSERT_VALUES_EQUAL(reverseParsedLeafEstimationIterations, refLeafEstimationIterations);
            UNIT_ASSERT_VALUES_EQUAL(reverseParsedLeafEstimationBacktracking, refLeafEstimationBacktracking);

            UNIT_ASSERT_VALUES_EQUAL(reverseParsedBootstrapType, refBootstrapType);
            UNIT_ASSERT_VALUES_EQUAL(reverseParsedBaggingTemperature, refBaggingTemperature);

            UNIT_ASSERT_VALUES_EQUAL(reversePerFloatFeatureQuantization, refPerFloatFeatureQuantization);
        }
}
