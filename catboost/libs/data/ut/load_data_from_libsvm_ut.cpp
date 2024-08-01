
#include <catboost/libs/data/ut/lib/for_loader.h>

#include <catboost/libs/data/libsvm_loader.h>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;
using namespace NCB::NDataNewUT;

Y_UNIT_TEST_SUITE(LoadDataFromLibSvm) {
    template <class T>
    TConstPolymorphicValuesSparseArray<T, ui32> MakeConstPolymorphicValuesSparseArray(
        ui32 size,
        TVector<ui32>&& indices,
        TVector<T>&& values,
        T defaultValue = T()
    ) {
        return MakeConstPolymorphicValuesSparseArrayWithArrayIndex(
            size,
            TMaybeOwningConstArrayHolder<ui32>::CreateOwning(std::move(indices)),
            TMaybeOwningConstArrayHolder<T>::CreateOwning(std::move(values)),
            /*ordered*/ true,
            T(defaultValue)
        );
    }


    Y_UNIT_TEST(ReadDataset) {
        TVector<TReadDatasetTestCase> testCases;

        {
            TReadDatasetTestCase testCase;
            TSrcData srcData;
            srcData.Scheme = "libsvm";
            srcData.DatasetFileData = TStringBuf(
                "0 1:0.1 3:0.2\n"
                "1 2:0.97 5:0.82 6:0.11 8:1.2\n"
                "0 3:0.13 7:0.22 8:0.17\n"
            );
            testCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            expectedData.MetaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                ui32(8),
                /*catFeatureIndices*/ TVector<ui32>(),
                /*textFeatureIndices*/ TVector<ui32>(),
                /*embeddingFeatureIndices*/ TVector<ui32>(),
                /*featureId*/ TVector<TString>(),
                /*hasGraph*/ false,
                /*featureTags*/ THashMap<TString, NCB::TTagDescription>(),
                /*allFeaturesAreSparse*/ true
            );
            expectedData.MetaInfo.TargetType = ERawTargetType::Float;
            expectedData.MetaInfo.TargetCount = 1;

            expectedData.Objects.FloatFeatures = {
                MakeConstPolymorphicValuesSparseArray<float>(3, {0}, {0.1f}), // 0
                MakeConstPolymorphicValuesSparseArray<float>(3, {1}, {0.97f}), // 1
                MakeConstPolymorphicValuesSparseArray<float>(3, {0, 2}, {0.2f, 0.13f}), // 2
                MakeConstPolymorphicValuesSparseArray<float>(3, {}, {}), // 3
                MakeConstPolymorphicValuesSparseArray<float>(3, {1}, {0.82f}), // 4
                MakeConstPolymorphicValuesSparseArray<float>(3, {1}, {0.11f}), // 5
                MakeConstPolymorphicValuesSparseArray<float>(3, {2}, {0.22f}), // 6
                MakeConstPolymorphicValuesSparseArray<float>(3, {1, 2}, {1.2f, 0.17f}), // 7
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(3);
            expectedData.Target.TargetType = ERawTargetType::Float;
            TVector<TVector<TString>> rawTarget{{"0", "1", "0"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(3);
            expectedData.Target.GroupWeights = TWeights<float>(3);

            testCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(testCase));
        }

        for (const auto& testCase : testCases) {
            TestReadDataset(testCase);
        }
    }

    Y_UNIT_TEST(ReadDatasetWithCatFeatures) {
        TVector<TReadDatasetTestCase> testCases;

        {
            TReadDatasetTestCase testCase;
            TSrcData srcData;
            srcData.Scheme = "libsvm";
            srcData.CdFileData = TStringBuf(
                "0\tTarget\n"
                "3\tCateg\tCat0\n"
                "9\tCateg\tCat1\n"
                "11\tCateg\tCat2\n"
            );
            srcData.DatasetFileData = TStringBuf(
                "0 1:0.1 3:0 9:12 12:0.66\n"
                "1 2:0.97 5:0.82 6:0.11 8:1.2 11:7\n"
                "0 3:1 7:0.22 8:0.17 9:0 11:2\n"
            );
            testCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            expectedData.MetaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                ui32(12),
                /*catFeatureIndices*/ TVector<ui32>{2, 8, 10},
                /*textFeatureIndices*/ TVector<ui32>(),
                /*embeddingFeatureIndices*/ TVector<ui32>(),
                /*featureId*/ TVector<TString>{
                    "", // 0
                    "", // 1
                    "Cat0", // 2
                    "", // 3
                    "", // 4
                    "", // 5
                    "", // 6
                    "", // 7
                    "Cat1", // 8
                    "", // 9
                    "Cat2", // 10
                    "" // 11
                },
                /*hasGraph*/ false,
                /*featureTags*/ THashMap<TString, NCB::TTagDescription>(),
                /*allFeaturesAreSparse*/ true
            );
            expectedData.MetaInfo.TargetType = ERawTargetType::Float;
            expectedData.MetaInfo.TargetCount = 1;

            expectedData.Objects.FloatFeatures = {
                MakeConstPolymorphicValuesSparseArray<float>(3, {0}, {0.1f}), // 0
                MakeConstPolymorphicValuesSparseArray<float>(3, {1}, {0.97f}), // 1
                MakeConstPolymorphicValuesSparseArray<float>(3, {}, {}), // 3
                MakeConstPolymorphicValuesSparseArray<float>(3, {1}, {0.82f}), // 4
                MakeConstPolymorphicValuesSparseArray<float>(3, {1}, {0.11f}), // 5
                MakeConstPolymorphicValuesSparseArray<float>(3, {2}, {0.22f}), // 6
                MakeConstPolymorphicValuesSparseArray<float>(3, {1, 2}, {1.2f, 0.17f}), // 7
                MakeConstPolymorphicValuesSparseArray<float>(3, {}, {}), // 9
                MakeConstPolymorphicValuesSparseArray<float>(3, {0}, {0.66f}), // 11
            };

            expectedData.Objects.CatFeatures = {
                MakeConstPolymorphicValuesSparseArray<TStringBuf>(3, {0, 2}, {"0", "1"}, "0"), // 2
                MakeConstPolymorphicValuesSparseArray<TStringBuf>(3, {0, 2}, {"12", "0"}, "0"), // 8
                MakeConstPolymorphicValuesSparseArray<TStringBuf>(3, {1, 2}, {"7", "2"}, "0"), // 10
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(3);
            expectedData.Target.TargetType = ERawTargetType::Float;
            TVector<TVector<TString>> rawTarget{{"0", "1", "0"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(3);
            expectedData.Target.GroupWeights = TWeights<float>(3);

            testCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(testCase));
        }

        for (const auto& testCase : testCases) {
            TestReadDataset(testCase);
        }
    }

    Y_UNIT_TEST(ReadDatasetWithQid) {
        TVector<TReadDatasetTestCase> testCases;

        {
            TReadDatasetTestCase testCase;
            TSrcData srcData;
            srcData.Scheme = "libsvm";
            srcData.DatasetFileData = TStringBuf(
                "0 qid:0 1:0.1 3:0.2\n"
                "1 qid:1 2:0.97 5:0.82 6:0.11 8:1.2\n"
                "0 qid:1 3:0.13 7:0.22 8:0.17\n"
            );
            testCase.SrcData = std::move(srcData);


            TExpectedRawData expectedData;

            expectedData.MetaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
                ui32(8),
                /*catFeatureIndices*/ TVector<ui32>(),
                /*textFeatureIndices*/ TVector<ui32>(),
                /*embeddingFeatureIndices*/ TVector<ui32>(),
                /*featureId*/ TVector<TString>(),
                /*hasGraph*/ false,
                /*featureTags*/ THashMap<TString, NCB::TTagDescription>(),
                /*allFeaturesAreSparse*/ true
            );
            expectedData.MetaInfo.TargetType = ERawTargetType::Float;
            expectedData.MetaInfo.TargetCount = 1;
            expectedData.MetaInfo.HasGroupId = true;

            expectedData.Objects.GroupIds = TVector<TStringBuf>{
                "0",
                "1",
                "1"
            };
            expectedData.Objects.TreatGroupIdsAsIntegers = true;
            expectedData.Objects.FloatFeatures = {
                MakeConstPolymorphicValuesSparseArray<float>(3, {0}, {0.1f}), // 0
                MakeConstPolymorphicValuesSparseArray<float>(3, {1}, {0.97f}), // 1
                MakeConstPolymorphicValuesSparseArray<float>(3, {0, 2}, {0.2f, 0.13f}), // 2
                MakeConstPolymorphicValuesSparseArray<float>(3, {}, {}), // 3
                MakeConstPolymorphicValuesSparseArray<float>(3, {1}, {0.82f}), // 4
                MakeConstPolymorphicValuesSparseArray<float>(3, {1}, {0.11f}), // 5
                MakeConstPolymorphicValuesSparseArray<float>(3, {2}, {0.22f}), // 6
                MakeConstPolymorphicValuesSparseArray<float>(3, {1, 2}, {1.2f, 0.17f}), // 7
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(
                TVector<TGroupBounds>{{0, 1}, {1, 3}}
            );
            expectedData.Target.TargetType = ERawTargetType::Float;
            TVector<TVector<TString>> rawTarget{{"0", "1", "0"}};
            expectedData.Target.Target.assign(rawTarget.begin(), rawTarget.end());
            expectedData.Target.Weights = TWeights<float>(3);
            expectedData.Target.GroupWeights = TWeights<float>(3);

            testCase.ExpectedData = std::move(expectedData);

            testCases.push_back(std::move(testCase));
        }

        for (const auto& testCase : testCases) {
            TestReadDataset(testCase);
        }
    }
}
