#include "print.h"
#include "pool.h"

#include <library/unittest/registar.h>

#include <contrib/libs/flatbuffers/include/flatbuffers/flatbuffers.h>

#include <catboost/idl/pool/flat/quantized_chunk_t.fbs.h>
#include <catboost/idl/pool/proto/quantization_schema.pb.h>

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/memory/blob.h>
#include <util/stream/mem.h>
#include <util/stream/output.h>
#include <util/stream/str.h>
#include <util/system/types.h>

static NCB::TQuantizedPool MakeQuantizedPool() {
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
        static const ui8 borders[] = {1, 2, 0};
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(NCB::NIdl::CreateTQuantizedFeatureChunk(
            builder,
            NCB::NIdl::EBitsPerDocumentFeature_BPDF_8,
            builder.CreateVector(borders, Y_ARRAY_SIZE(borders))));
        blobs.push_back(TBlob::Copy(
            builder.GetBufferPointer(),
            builder.GetSize()));
    }

    NCB::TQuantizedPool pool;
    pool.Blobs = std::move(blobs);
    pool.TrueFeatureIndexToLocalIndex.emplace(1, 0);
    pool.TrueFeatureIndexToLocalIndex.emplace(5, 1);
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

    return pool;
}

static NCB::NIdl::TPoolQuantizationSchema MakeQuantizationSchema() {
    NCB::NIdl::TPoolQuantizationSchema schema;
    {
        NCB::NIdl::TFeatureQuantizationSchema featureSchema;
        featureSchema.AddBorders(-1);
        featureSchema.AddBorders(0);
        featureSchema.AddBorders(1.5);
        schema.MutableFeatureIndexToSchema()->insert({
            1,
            std::move(featureSchema)});
    }
    {
        NCB::NIdl::TFeatureQuantizationSchema featureSchema;
        featureSchema.AddBorders(0);
        featureSchema.AddBorders(0.5);
        featureSchema.AddBorders(3);
        schema.MutableFeatureIndexToSchema()->insert({
            5,
            std::move(featureSchema)});
    }
    return schema;
}

Y_UNIT_TEST_SUITE(PrintTests) {
    Y_UNIT_TEST(TestWithoutSchema) {
        const auto pool = MakeQuantizedPool();

        TString humanReadablePool;
        TStringOutput humanReadablePoolInput{humanReadablePool};
        NCB::PrintQuantizedPool(
            pool,
            {NCB::EQuantizedPoolPrintFormat::HumanReadable},
            &humanReadablePoolInput);

        static const TStringBuf expected = R"(2
1 1
0 3
2 0 0
5 1
0 3
1 2 0
)";
        UNIT_ASSERT_VALUES_EQUAL(humanReadablePool, expected);
    }

    Y_UNIT_TEST(TestWithSchema) {
        const auto pool = MakeQuantizedPool();
        const auto schema = MakeQuantizationSchema();

        TString humanReadablePool;
        TStringOutput humanReadablePoolInput{humanReadablePool};
        NCB::PrintQuantizedPool(
            pool,
            {NCB::EQuantizedPoolPrintFormat::HumanReadable},
            &humanReadablePoolInput,
            &schema);

        static const TStringBuf expected = R"(2
1 1
0 3
1.5 -1 -1
5 1
0 3
0.5 3 0
)";
        UNIT_ASSERT_VALUES_EQUAL(humanReadablePool, expected);
    }
}
