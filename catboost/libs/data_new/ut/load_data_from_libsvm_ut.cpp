
#include <catboost/libs/data_new/ut/lib/for_loader.h>

#include <catboost/libs/data_new/libsvm_loader.h>

#include <library/unittest/registar.h>


using namespace NCB;
using namespace NCB::NDataNewUT;

Y_UNIT_TEST_SUITE(LoadDataFromLibSvm) {
    template <class T>
    TConstSparseArray<T, ui32> MakeConstSparseArray(
        ui32 size,
        TVector<ui32>&& indices,
        TVector<T>&& values,
        T defaultValue = T()
    ) {
        return MakeConstSparseArrayWithArrayIndex(
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
            srcData.DatasetFileData = AsStringBuf(
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
                /*featureId*/ TVector<TString>(),
                /*allFeaturesAreSparse*/ true
            );
            expectedData.MetaInfo.HasTarget = true;

            expectedData.Objects.FloatFeatures = {
                MakeConstSparseArray<float>(3, {0}, {0.1f}), // 0
                MakeConstSparseArray<float>(3, {1}, {0.97f}), // 1
                MakeConstSparseArray<float>(3, {0, 2}, {0.2f, 0.13f}), // 2
                MakeConstSparseArray<float>(3, {}, {}), // 3
                MakeConstSparseArray<float>(3, {1}, {0.82f}), // 4
                MakeConstSparseArray<float>(3, {1}, {0.11f}), // 5
                MakeConstSparseArray<float>(3, {2}, {0.22f}), // 6
                MakeConstSparseArray<float>(3, {1, 2}, {1.2f, 0.17f}), // 7
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(3);
            expectedData.Target.Target = TVector<TString>{"0", "1", "0"};
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
            srcData.CdFileData = AsStringBuf(
                "0\tTarget\n"
                "3\tCateg\tCat0\n"
                "9\tCateg\tCat1\n"
                "11\tCateg\tCat2\n"
            );
            srcData.DatasetFileData = AsStringBuf(
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
                /*allFeaturesAreSparse*/ true
            );
            expectedData.MetaInfo.HasTarget = true;

            expectedData.Objects.FloatFeatures = {
                MakeConstSparseArray<float>(3, {0}, {0.1f}), // 0
                MakeConstSparseArray<float>(3, {1}, {0.97f}), // 1
                MakeConstSparseArray<float>(3, {}, {}), // 3
                MakeConstSparseArray<float>(3, {1}, {0.82f}), // 4
                MakeConstSparseArray<float>(3, {1}, {0.11f}), // 5
                MakeConstSparseArray<float>(3, {2}, {0.22f}), // 6
                MakeConstSparseArray<float>(3, {1, 2}, {1.2f, 0.17f}), // 7
                MakeConstSparseArray<float>(3, {}, {}), // 9
                MakeConstSparseArray<float>(3, {0}, {0.66f}), // 11
            };

            expectedData.Objects.CatFeatures = {
                MakeConstSparseArray<TStringBuf>(3, {0, 2}, {"0", "1"}, "0"), // 2
                MakeConstSparseArray<TStringBuf>(3, {0, 2}, {"12", "0"}, "0"), // 8
                MakeConstSparseArray<TStringBuf>(3, {1, 2}, {"7", "2"}, "0"), // 10
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(3);
            expectedData.Target.Target = TVector<TString>{"0", "1", "0"};
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
            srcData.DatasetFileData = AsStringBuf(
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
                /*featureId*/ TVector<TString>(),
                /*allFeaturesAreSparse*/ true
            );
            expectedData.MetaInfo.HasTarget = true;
            expectedData.MetaInfo.HasGroupId = true;

            expectedData.Objects.GroupIds = TVector<TStringBuf>{
                "0",
                "1",
                "1"
            };
            expectedData.Objects.TreatGroupIdsAsIntegers = true;
            expectedData.Objects.FloatFeatures = {
                MakeConstSparseArray<float>(3, {0}, {0.1f}), // 0
                MakeConstSparseArray<float>(3, {1}, {0.97f}), // 1
                MakeConstSparseArray<float>(3, {0, 2}, {0.2f, 0.13f}), // 2
                MakeConstSparseArray<float>(3, {}, {}), // 3
                MakeConstSparseArray<float>(3, {1}, {0.82f}), // 4
                MakeConstSparseArray<float>(3, {1}, {0.11f}), // 5
                MakeConstSparseArray<float>(3, {2}, {0.22f}), // 6
                MakeConstSparseArray<float>(3, {1, 2}, {1.2f, 0.17f}), // 7
            };

            expectedData.ObjectsGrouping = TObjectsGrouping(
                TVector<TGroupBounds>{{0, 1}, {1, 3}}
            );
            expectedData.Target.Target = TVector<TString>{"0", "1", "0"};
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
