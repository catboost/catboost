#include <library/unittest/registar.h>

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/data_new/objects.h>
#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/ut_helpers/data_provider.h>

#include <util/generic/array_ref.h>
#include <util/stream/labeled.h>
#include <util/string/builder.h>

using NCB::TFloatValuesHolder;
using NCB::THashedCatValuesHolder;
using NCB::TMaybeData;
using NCB::TRawObjectsDataProvider;

static void CompareNumericArrays(
    const TMaybeData<const TFloatValuesHolder*> maybeSample,
    const TConstArrayRef<float> expected,
    const TString& context)
{
    UNIT_ASSERT_VALUES_EQUAL_C(
        static_cast<bool>(expected),
        static_cast<bool>(maybeSample),
        context.c_str());

    const auto* const samplePtr = maybeSample.GetRef();

    UNIT_ASSERT_VALUES_EQUAL_C(
        static_cast<bool>(expected),
        static_cast<bool>(samplePtr),
        context.c_str());

    const auto sample = samplePtr->GetArrayData();

    UNIT_ASSERT_VALUES_EQUAL_C(
        expected.size(),
        sample.Size(),
        context.c_str());

    sample.ForEach([&](const auto idx, const auto value) {
        const TString extendedContext = TStringBuilder() << context << "; " << LabeledOutput(idx);
        UNIT_ASSERT_C(idx < expected.size(), extendedContext.c_str())
        UNIT_ASSERT_VALUES_EQUAL_C(expected[idx], value, extendedContext.c_str());
    });
}

static void CompareCategoricalArrays(
    const TMaybeData<const THashedCatValuesHolder*> maybeSample,
    const TConstArrayRef<TStringBuf> expected,
    const TString& context)
{
    UNIT_ASSERT_VALUES_EQUAL_C(
        static_cast<bool>(expected),
        static_cast<bool>(maybeSample),
        context.c_str());

    const auto* const samplePtr = maybeSample.GetRef();

    UNIT_ASSERT_VALUES_EQUAL_C(
        static_cast<bool>(expected),
        static_cast<bool>(samplePtr),
        context.c_str());

    const auto sample = samplePtr->GetArrayData();

    UNIT_ASSERT_VALUES_EQUAL_C(
        expected.size(),
        sample.Size(),
        context.c_str());

    sample.ForEach([&](const auto idx, const auto value) {
        const TString extendedContext = TStringBuilder() << context << "; " << LabeledOutput(idx);
        const auto expectedHash = CalcCatFeatureHash(expected[idx]);
        UNIT_ASSERT_C(idx < expected.size(), extendedContext.c_str())
        UNIT_ASSERT_VALUES_EQUAL_C(expectedHash, value, extendedContext.c_str());
    });
}

static void CompareStringArrays(
    const TMaybeData<TConstArrayRef<TString>> maybeSample,
    const TConstArrayRef<TStringBuf> expected,
    const TString& context)
{
    UNIT_ASSERT_VALUES_EQUAL_C(
        static_cast<bool>(expected),
        static_cast<bool>(maybeSample),
        context.c_str());

    const auto sample = maybeSample.GetRef();

    UNIT_ASSERT_VALUES_EQUAL_C(
        expected.size(),
        sample.size(),
        context.c_str());

    for (size_t i = 0; i < expected.size(); ++i) {
        const TString extendedContext = TStringBuilder() << context << "; " << LabeledOutput(i);
        UNIT_ASSERT_VALUES_EQUAL_C(expected[i], sample[i], extendedContext.c_str());
    }
}

static void CompareNumericArrays(
    const TMaybeData<TConstArrayRef<TConstArrayRef<float>>> maybeSample,
    const TConstArrayRef<float> expected,
    const TString& context)
{
    UNIT_ASSERT_VALUES_EQUAL_C(
        static_cast<bool>(expected),
        static_cast<bool>(maybeSample),
        context.c_str());

    const auto sample = maybeSample.GetRef();

    UNIT_ASSERT_VALUES_EQUAL_C(
            1,
            sample.size(),
            context.c_str());

    UNIT_ASSERT_VALUES_EQUAL_C(
        expected.size(),
        sample.front().size(),
        context.c_str());

    for (size_t i = 0; i < expected.size(); ++i) {
        const TString extendedContext = TStringBuilder() << context << "; " << LabeledOutput(i);
        UNIT_ASSERT_VALUES_EQUAL_C(expected[i], sample.front()[i], extendedContext.c_str());
    }
}

static void CompareNumericArrays(
    const NCB::TWeights<float>& sample,
    const TConstArrayRef<float> expected,
    const TString& context)
{
    UNIT_ASSERT_VALUES_EQUAL_C(
        expected.size(),
        sample.GetSize(),
        context.c_str());

    for (size_t i = 0; i < expected.size(); ++i) {
        const TString extendedContext = TStringBuilder() << context << "; " << LabeledOutput(i);
        UNIT_ASSERT_VALUES_EQUAL_C(expected[i], sample[i], extendedContext.c_str());
    }
}

Y_UNIT_TEST_SUITE(MakeDataProviderFromTextTests) {
    Y_UNIT_TEST(Test1) {
        const auto provider = NCB::MakeDataProviderFromText(
            "0\tTarget\n"
            "1\tNum\n",
            R"(
                1 2
                3 4
            )");

        UNIT_ASSERT_VALUES_EQUAL(
            2,
            provider->ObjectsData->GetObjectCount());

        const auto* const objects = dynamic_cast<const TRawObjectsDataProvider*>(provider->ObjectsData.Get());
        const auto* const target = &provider->RawTargetData;

        CompareStringArrays(
            target->GetTarget(),
            {"1", "3"},
            "target");

        CompareNumericArrays(
            objects->GetFloatFeature(0),
            {2, 4},
            TStringBuilder() << "numeric feature " << 0);
    }

    Y_UNIT_TEST(Test2) {
        const auto provider = NCB::MakeDataProviderFromText(
            "0\tTarget\n"
            "1\tNum\n"
            "2\tCateg\n"
            "3\tAuxiliary\n"
            "4\tBaseline\n"
            "5\tWeight\n",
            R"(
                1 2 3 4 5 6
                7 8 9 10 11 12
            )");

        UNIT_ASSERT_VALUES_EQUAL(
            2,
            provider->ObjectsData->GetObjectCount());

        const auto* const metaInfo = &provider->MetaInfo;
        const auto* const layout = provider->MetaInfo.FeaturesLayout.Get();
        const auto* const objects = dynamic_cast<const TRawObjectsDataProvider*>(provider->ObjectsData.Get());
        const auto* const target = &provider->RawTargetData;

        UNIT_ASSERT_VALUES_EQUAL(metaInfo->HasTarget, true);
        UNIT_ASSERT_VALUES_EQUAL(metaInfo->BaselineCount, 1);
        UNIT_ASSERT_VALUES_EQUAL(metaInfo->HasGroupId, false);
        UNIT_ASSERT_VALUES_EQUAL(metaInfo->HasGroupWeight, false);
        UNIT_ASSERT_VALUES_EQUAL(metaInfo->HasWeights, true);
        UNIT_ASSERT_VALUES_EQUAL(metaInfo->HasTimestamp, false);
        UNIT_ASSERT_VALUES_EQUAL(metaInfo->HasPairs, false);

        UNIT_ASSERT_VALUES_EQUAL(layout->GetFloatFeatureCount(), 1);
        UNIT_ASSERT_VALUES_EQUAL(layout->GetCatFeatureCount(), 1);

        CompareStringArrays(
            target->GetTarget(),
            {"1", "7"},
            "target");

        CompareNumericArrays(
            target->GetBaseline(),
            {5.f, 11.f},
            "baseline");

        CompareNumericArrays(
            target->GetWeights(),
            {6.f, 12.f},
            "weights");

        CompareNumericArrays(
            objects->GetFloatFeature(0),
            {2.f, 8.f},
            TStringBuilder() << "numeric feature " << 0);

        CompareCategoricalArrays(
            objects->GetCatFeature(0),
            {"3", "9"},
            TStringBuilder() << "categorical feature " << 0);
    }
}
