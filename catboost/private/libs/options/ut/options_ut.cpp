#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/json/json_reader.h>
#include <catboost/private/libs/options/option.h>
#include <catboost/private/libs/options/json_helper.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/system_options.h>
#include <catboost/private/libs/options/catboost_options.h>

Y_UNIT_TEST_SUITE(TOptionsTest) {
    using namespace NCatboostOptions;

    template <class TOptions>
    inline void TestSaveLoad(const TOptions& options, ETaskType taskType) {
        NJson::TJsonValue tree;
        options.Save(&tree);
        TOptions loaded(taskType);
        loaded.Load(tree);
        UNIT_ASSERT_VALUES_EQUAL(loaded == options, true);
    }

    Y_UNIT_TEST(TestApplicationOptions) {
        TSystemOptions options(ETaskType::GPU);
        options.NumThreads = 100;
        TestSaveLoad(options, ETaskType::GPU);
    }

    Y_UNIT_TEST(TestEnumSerialization) {
        TOption<ECtrType> type("type", ECtrType::Borders);
        TString jsonStr = "{ \"type\":\"Counter\"}";
        NJson::TJsonValue json = ReadTJsonValue(jsonStr);
        CheckedLoad(json, &type);
        UNIT_ASSERT_VALUES_EQUAL(type.Get(), ECtrType::Counter);
    }

    Y_UNIT_TEST(TestBoostingOptions) {
        {
            TBoostingOptions options(ETaskType::GPU);
            TestSaveLoad(options, ETaskType::GPU);
        }
        {
            TBoostingOptions options(ETaskType::CPU);
            TestSaveLoad(options, ETaskType::CPU);
        }
    }

    Y_UNIT_TEST(TestParseCtrParams) {
        TString ctr1 = "Buckets:TargetBorderType=GreedyLogSum:TargetBorderCount=2:Prior=1/2:Prior=2/4";
        TString ctr2 = "Borders:CtrBorderCount=33:CtrBorderType=GreedyLogSum";
        TString perFeatureCtrs = "1:Buckets:CtrBorderCount=33:CtrBorderType=GreedyLogSum;2:Counter:CtrBorderCount=13:CtrBorderType=GreedyLogSum";
        NJson::TJsonValue ctr1Json = ParseCtrDescription(ctr1);
        NJson::TJsonValue ctr2Json = ParseCtrDescription(ctr2);
        NJson::TJsonValue perFeatureCtrJson = ParsePerFeatureCtrs(perFeatureCtrs);
        TCtrDescription description1;
        description1.Load(ctr1Json);

        TCtrDescription description2;
        description2.Load(ctr2Json);

        TMap<ui32, TVector<TCtrDescription>> perFeatureCtr;
        TJsonFieldHelper<decltype(perFeatureCtr)>::Read(perFeatureCtrJson, &perFeatureCtr);

        UNIT_ASSERT_VALUES_EQUAL(description1.GetPriors()[0], TVector<float>({1, 2}));
        UNIT_ASSERT_VALUES_EQUAL(description1.GetPriors()[1], TVector<float>({2, 4}));
        UNIT_ASSERT_VALUES_EQUAL(description1.TargetBinarization->BorderCount.Get(), 2);
        UNIT_ASSERT_VALUES_EQUAL(description1.TargetBinarization->BorderSelectionType.Get(), EBorderSelectionType::GreedyLogSum);

        UNIT_ASSERT_VALUES_EQUAL(description2.TargetBinarization->BorderCount.Get(), 1);
        UNIT_ASSERT_VALUES_EQUAL(description2.CtrBinarization->BorderSelectionType.Get(), EBorderSelectionType::GreedyLogSum);
        UNIT_ASSERT_VALUES_EQUAL(description2.CtrBinarization->BorderCount.Get(), 33);

        UNIT_ASSERT_VALUES_EQUAL(perFeatureCtr.size(), 2);
        UNIT_ASSERT_VALUES_EQUAL(perFeatureCtr.at(1)[0].Type.Get(), ECtrType::Buckets);
        UNIT_ASSERT_VALUES_EQUAL(perFeatureCtr.at(2)[0].Type.Get(), ECtrType::Counter);
        UNIT_ASSERT_VALUES_EQUAL(perFeatureCtr.at(1)[0].CtrBinarization->BorderCount.Get(), 33);
        UNIT_ASSERT_VALUES_EQUAL(perFeatureCtr.at(2)[0].CtrBinarization->BorderCount.Get(), 13);
    }

    template <class TOptions>
    inline void TestLoadForOtherTaskType() {
        {
            TOptions options(ETaskType::GPU);
            NJson::TJsonValue tree;
            options.Save(&tree);
            TOptions options2(ETaskType::CPU);
            options2.Load(tree);
        }

        {
            TOptions options(ETaskType::CPU);
            NJson::TJsonValue tree;
            options.Save(&tree);
            TOptions options2(ETaskType::GPU);
            options2.Load(tree);
        }
    }

    Y_UNIT_TEST(TestLoadOtherTaskType) {
        TestLoadForOtherTaskType<TBoostingOptions>();
        TestLoadForOtherTaskType<TObliviousTreeLearnerOptions>();
        TestLoadForOtherTaskType<TSystemOptions>();

        {
            TCatBoostOptions options(ETaskType::GPU);
            NJson::TJsonValue tree;
            options.Save(&tree);
            tree["task_type"] = ToString<ETaskType>(ETaskType::CPU);
            TCatBoostOptions options2(ETaskType::CPU);
            options2.Load(tree);
        }

        {
            TCatBoostOptions options(ETaskType::CPU);
            NJson::TJsonValue tree;
            options.Save(&tree);
            tree["task_type"] = ToString<ETaskType>(ETaskType::GPU);
            TCatBoostOptions options2(ETaskType::GPU);
            options2.Load(tree);
        }
    }

    Y_UNIT_TEST(TestCpuOptions) {
        TCatBoostOptions options(ETaskType::CPU);
        options.SetNotSpecifiedOptionsToDefaults();
        TestSaveLoad(options, ETaskType::CPU);
    }

    Y_UNIT_TEST(TestGpuOptions) {
        TCatBoostOptions options(ETaskType::GPU);
        options.SetNotSpecifiedOptionsToDefaults();
        TestSaveLoad(options, ETaskType::GPU);
    }
}
