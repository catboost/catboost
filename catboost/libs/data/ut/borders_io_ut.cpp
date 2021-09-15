#include <catboost/libs/data/borders_io.h>

#include <catboost/libs/data/features_layout.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <util/stream/file.h>
#include <util/system/tempfile.h>

#include <limits>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(BordersIo) {
    Y_UNIT_TEST(Save) {
        TFeaturesLayout featuresLayout(ui32(3), {}, {}, {}, {});

        TQuantizedFeaturesInfo quantizedFeaturesInfo(
            featuresLayout,
            TConstArrayRef<ui32>(),
            NCatboostOptions::TBinarizationOptions(),
            TMap<ui32, NCatboostOptions::TBinarizationOptions>());

        quantizedFeaturesInfo.SetBorders(TFloatFeatureIdx(0), {0.11f, 0.22f, 0.34f});
        quantizedFeaturesInfo.SetBorders(TFloatFeatureIdx(1), {0.9f, 1.2f});

        quantizedFeaturesInfo.SetBorders(
            TFloatFeatureIdx(2),
            {std::numeric_limits<float>::lowest(), 1.2f, 4.5f, 11.0f});
        quantizedFeaturesInfo.SetNanMode(TFloatFeatureIdx(2), ENanMode::Min);

        TTempFile bordersFile(MakeTempName());
        SaveBordersAndNanModesToFileInMatrixnetFormat(bordersFile.Name(), quantizedFeaturesInfo);

        {
            UNIT_ASSERT_VALUES_EQUAL(
                TIFStream(bordersFile.Name()).ReadAll(),
                "0\t0.1099999994\n"
                "0\t0.2199999988\n"
                "0\t0.3400000036\n"
                "1\t0.8999999762\n"
                "1\t1.200000048\n"
                "2\t-3.402823466e+38\tMin\n"
                "2\t1.200000048\tMin\n"
                "2\t4.5\tMin\n"
                "2\t11\tMin\n");
        }
    }

    Y_UNIT_TEST(Load) {
        TTempFile bordersFile(MakeTempName());

        {
            TOFStream(bordersFile.Name()).Write(
                "0\t0.1099999994\n"
                "0\t0.2199999988\n"
                "0\t0.3400000036\n"
                "1\t0.8999999762\n"
                "1\t1.200000048\n"
                "2\t-3.402823466e+38\tMin\n"
                "2\t1.200000048\tMin\n"
                "2\t4.5\tMin\n"
                "2\t11\tMin\n");
        }

        TFeaturesLayout featuresLayout(ui32(3), {}, {}, {}, {});

        TQuantizedFeaturesInfo quantizedFeaturesInfo(
            featuresLayout,
            TConstArrayRef<ui32>(),
            NCatboostOptions::TBinarizationOptions(),
            TMap<ui32, NCatboostOptions::TBinarizationOptions>());

        LoadBordersAndNanModesFromFromFileInMatrixnetFormat(bordersFile.Name(), &quantizedFeaturesInfo);

        UNIT_ASSERT(
            ApproximatelyEqual<float>(
                quantizedFeaturesInfo.GetBorders(TFloatFeatureIdx(0)),
                {0.11f, 0.22f, 0.34f},
                1.e-13f)
        );

    }
}
