#include "print.h"
#include "pool.h"

#include <library/cpp/testing/unittest/registar.h>

#include <contrib/libs/flatbuffers/include/flatbuffers/flatbuffers.h>

#include <catboost/idl/pool/flat/quantized_chunk_t.fbs.h>
#include <catboost/idl/pool/proto/quantization_schema.pb.h>
#include <catboost/libs/column_description/column.h>

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/memory/blob.h>
#include <util/stream/mem.h>
#include <util/stream/output.h>
#include <util/stream/str.h>
#include <util/system/types.h>

using NCB::NIdl::TFeatureQuantizationSchema;

static NCB::TQuantizedPool MakeQuantizedPool(bool hasGroupId = false, bool hasIgnoredFeatures = false) {
    TVector<TBlob> blobs;
    {
        static const ui8 borders[] = {2, 0, 0};
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(NCB::NIdl::CreateTQuantizedFeatureChunk(
            builder,
            NCB::NIdl::EBitsPerDocumentFeature_BPDF_8,
            builder.CreateVector(borders, Y_ARRAY_SIZE(borders))));
        blobs.push_back(TBlob::Copy(
            builder.GetBufferPointer(),
            builder.GetSize()));
    }
    {
        static const float labels[] = {0.5, 1.5, 0};
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(NCB::NIdl::CreateTQuantizedFeatureChunk(
            builder,
            NCB::NIdl::EBitsPerDocumentFeature_BPDF_32,
            builder.CreateVector(
                reinterpret_cast<const ui8*>(labels),
                sizeof(float) * Y_ARRAY_SIZE(labels))));
        blobs.push_back(TBlob::Copy(
            builder.GetBufferPointer(),
            builder.GetSize()));
    }
    if (hasGroupId) {
        {
            static const ui64 groupIds[] = {1, 2, 3};
            flatbuffers::FlatBufferBuilder builder;
            builder.Finish(NCB::NIdl::CreateTQuantizedFeatureChunk(
                        builder,
                        NCB::NIdl::EBitsPerDocumentFeature_BPDF_64,
                        builder.CreateVector(
                            reinterpret_cast<const ui8*>(groupIds),
                            sizeof(ui64) * Y_ARRAY_SIZE(groupIds))));
            blobs.push_back(TBlob::Copy(
                        builder.GetBufferPointer(),
                        builder.GetSize()));
        }
        {
            static const ui8 stringGroupIds[18] = {
                0x02, 0x00, 0x00, 0x00, '0', '1',
                0x02, 0x00, 0x00, 0x00, '0', '2',
                0x02, 0x00, 0x00, 0x00, '0', '3',
            };
            flatbuffers::FlatBufferBuilder builder;
            builder.Finish(NCB::NIdl::CreateTQuantizedFeatureChunk(
                        builder,
                        NCB::NIdl::EBitsPerDocumentFeature_BPDF_8,
                        builder.CreateVector(
                            reinterpret_cast<const ui8*>(stringGroupIds),
                            Y_ARRAY_SIZE(stringGroupIds))));
            blobs.push_back(TBlob::Copy(
                        builder.GetBufferPointer(),
                        builder.GetSize()));
        }
    }

    NCB::TQuantizedPool pool;
    pool.Blobs = std::move(blobs);
    pool.ColumnIndexToLocalIndex.emplace(1, 0);
    pool.ColumnIndexToLocalIndex.emplace(5, 1);
    pool.ColumnTypes = {EColumn::Num, EColumn::Label};
    if (hasGroupId) {
        pool.HasStringColumns = true;
        pool.StringGroupIdLocalIndex = 3;
        pool.ColumnIndexToLocalIndex.emplace(7, 2);
        pool.ColumnTypes.push_back(EColumn::GroupId);
    }
    pool.DocumentCount = 2;
    if (!hasIgnoredFeatures) {
        TFeatureQuantizationSchema featureSchema;
        featureSchema.AddBorders(0.25);
        featureSchema.AddBorders(0.5);
        featureSchema.AddBorders(0.75);
        pool.QuantizationSchema.MutableFeatureIndexToSchema()->insert({
            0,
            std::move(featureSchema)});
    }
    {
        TVector<NCB::TQuantizedPool::TChunkDescription> chunks;
        chunks.emplace_back(
            0,
            3,
            flatbuffers::GetRoot<NCB::NIdl::TQuantizedFeatureChunk>(pool.Blobs[0].AsCharPtr()));
        pool.Chunks.push_back(std::move(chunks));
    }
    {
        TVector<NCB::TQuantizedPool::TChunkDescription> chunks;
        chunks.emplace_back(
            0,
            3,
            flatbuffers::GetRoot<NCB::NIdl::TQuantizedFeatureChunk>(pool.Blobs[1].AsCharPtr()));
        pool.Chunks.push_back(std::move(chunks));
    }
    if (hasGroupId) {
        {
            TVector<NCB::TQuantizedPool::TChunkDescription> chunks;
            chunks.emplace_back(
                    0,
                    3,
                    flatbuffers::GetRoot<NCB::NIdl::TQuantizedFeatureChunk>(pool.Blobs[2].AsCharPtr()));
            pool.Chunks.push_back(std::move(chunks));
        }
        {
            TVector<NCB::TQuantizedPool::TChunkDescription> chunks;
            chunks.emplace_back(
                    0,
                    3,
                    flatbuffers::GetRoot<NCB::NIdl::TQuantizedFeatureChunk>(pool.Blobs[3].AsCharPtr()));
            pool.Chunks.push_back(std::move(chunks));
        }
    }

    return pool;
}

Y_UNIT_TEST_SUITE(PrintTests) {
    Y_UNIT_TEST(TestFloatAndLabelColumns) {
        const auto pool = MakeQuantizedPool();

        TString humanReadablePool;
        TStringOutput humanReadablePoolInput{humanReadablePool};
        NCB::PrintQuantizedPool(
            pool,
            {NCB::EQuantizedPoolPrintFormat::HumanReadableChunkWise},
            &humanReadablePoolInput);

        static const TStringBuf expected = R"(2 2
1 Num 1 0
0 3
2 0 0
5 Label 1
0 3
0.5 1.5 0
)";
        UNIT_ASSERT_VALUES_EQUAL(humanReadablePool, expected);
    }

    Y_UNIT_TEST(TestFloatAndLabelColumnsResolveBorders) {
        const auto pool = MakeQuantizedPool();

        TString humanReadablePool;
        TStringOutput humanReadablePoolInput{humanReadablePool};
        NCB::PrintQuantizedPool(
            pool,
            {NCB::EQuantizedPoolPrintFormat::HumanReadableChunkWise, true},
            &humanReadablePoolInput);

        static const TStringBuf expected = R"(2 2
1 Num 1 0
0 3
<0.75 <0.25 <0.25
5 Label 1
0 3
0.5 1.5 0
)";
        UNIT_ASSERT_VALUES_EQUAL(humanReadablePool, expected);
    }

    Y_UNIT_TEST(TestIgnoredFeature) {
        const auto pool = MakeQuantizedPool(/*hasGroupId=*/false, /*hasIgnoredFeatures=*/true);

        TString humanReadablePool;
        TStringOutput humanReadablePoolInput{humanReadablePool};
        NCB::PrintQuantizedPool(
            pool,
            {NCB::EQuantizedPoolPrintFormat::HumanReadableChunkWise},
            &humanReadablePoolInput);

        static const TStringBuf expected = R"(2 2
5 Label 1
0 3
0.5 1.5 0
)";
        UNIT_ASSERT_VALUES_EQUAL(humanReadablePool, expected);
    }

    Y_UNIT_TEST(TestFloatLabelAndGroupIdColumns) {
        const auto pool = MakeQuantizedPool(/*hasGroupId=*/true);

        TString humanReadablePool;
        TStringOutput humanReadablePoolInput{humanReadablePool};
        NCB::PrintQuantizedPool(
            pool,
            {NCB::EQuantizedPoolPrintFormat::HumanReadableChunkWise},
            &humanReadablePoolInput);

        static const TStringBuf expected = R"(3 2
1 Num 1 0
0 3
2 0 0
5 Label 1
0 3
0.5 1.5 0
7 GroupId 1
0 3
1 2 3
0 3
01 02 03
)";
        UNIT_ASSERT_VALUES_EQUAL(humanReadablePool, expected);
    }

    Y_UNIT_TEST(TestColumnWiseFormat) {
        const auto pool = MakeQuantizedPool();

        TString humanReadablePool;
        TStringOutput humanReadablePoolInput{humanReadablePool};
        NCB::PrintQuantizedPool(
            pool,
            {NCB::EQuantizedPoolPrintFormat::HumanReadableColumnWise},
            &humanReadablePoolInput);

        static const TStringBuf expected = R"(2 2
1 Num 1 0
2 0 0
5 Label 1
0.5 1.5 0
)";
        UNIT_ASSERT_VALUES_EQUAL(humanReadablePool, expected);
    }

}
