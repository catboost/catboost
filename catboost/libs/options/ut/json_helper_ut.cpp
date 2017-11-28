#include <library/unittest/registar.h>
#include <library/json/json_reader.h>
#include <catboost/libs/options/option.h>
#include <catboost/libs/options/json_helper.h>
#include <catboost/libs/options/enums.h>

SIMPLE_UNIT_TEST_SUITE(TJsonHelperTest) {
    using namespace NCatboostOptions;

    SIMPLE_UNIT_TEST(TestSimpleSerializtion) {
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

    SIMPLE_UNIT_TEST(TestUnimplementedAwareOptions) {
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

    SIMPLE_UNIT_TEST(TestDisableOption) {
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
        UNIT_ASSERT_EXCEPTION(option2.Get(), TCatboostException);

        NJson::TJsonValue tree2;
        SaveFields(&tree2, option1, option2);
        UNIT_ASSERT_VALUES_EQUAL(ToString<NJson::TJsonValue>(tree2), "{\"option_1\":102}");
    }
}
