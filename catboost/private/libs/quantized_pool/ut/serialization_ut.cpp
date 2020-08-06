#include "pool.h"
#include "print.h"
#include "serialization.h"

#include <library/cpp/testing/unittest/registar.h>

#include <contrib/libs/flatbuffers/include/flatbuffers/flatbuffers.h>
#include <google/protobuf/util/message_differencer.h>

#include <catboost/idl/pool/flat/quantized_chunk_t.fbs.h>
#include <catboost/idl/pool/proto/quantization_schema.pb.h>

#include <util/folder/dirut.h>
#include <util/folder/path.h>
#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/memory/blob.h>
#include <util/stream/file.h>
#include <util/stream/input.h>
#include <util/stream/length.h>
#include <util/stream/output.h>
#include <util/system/fstat.h>

using NCB::NIdl::TFeatureQuantizationSchema;
using NCB::NIdl::TPoolQuantizationSchema;

static TPoolQuantizationSchema MakeQuantizationSchema() {
    TPoolQuantizationSchema quantizationSchema;
    {
        TFeatureQuantizationSchema featureSchema;
        featureSchema.AddBorders(0.25);
        featureSchema.AddBorders(0.5);
        featureSchema.AddBorders(0.75);
        quantizationSchema.MutableFeatureIndexToSchema()->insert({
            0,
            std::move(featureSchema)});
    }
    return quantizationSchema;
}

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

    NCB::TQuantizedPool pool;
    pool.Blobs = std::move(blobs);
    pool.ColumnIndexToLocalIndex.emplace(1, 0);
    pool.ColumnIndexToLocalIndex.emplace(5, 1);
    pool.ColumnTypes = {EColumn::Num, EColumn::Label};
    pool.QuantizationSchema = MakeQuantizationSchema();
    pool.DocumentCount = 2;
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

static TString QuantizedPoolToString(const NCB::TQuantizedPool& pool) {
    TString str;
    TStringOutput output{str};
    NCB::PrintQuantizedPool(
        pool,
        {NCB::EQuantizedPoolPrintFormat::HumanReadableChunkWise},
        &output);
    return str;
}

static void CompareQuantizedPoolDigests(
    const NCB::TQuantizedPoolDigest& lhs,
    const NCB::TQuantizedPoolDigest& rhs) {

    UNIT_ASSERT_VALUES_EQUAL(lhs.CategoricFeatureCount, rhs.CategoricFeatureCount);

    UNIT_ASSERT_VALUES_EQUAL(lhs.NumericFeatureCount, rhs.NumericFeatureCount);
    UNIT_ASSERT_VALUES_EQUAL(lhs.NumericFeature1BitCount, rhs.NumericFeature1BitCount);
    UNIT_ASSERT_VALUES_EQUAL(lhs.NumericFeature4BitCount, rhs.NumericFeature4BitCount);
    UNIT_ASSERT_VALUES_EQUAL(lhs.NumericFeature8BitCount, rhs.NumericFeature8BitCount);
    UNIT_ASSERT_VALUES_EQUAL(lhs.NumericFeature16BitCount, rhs.NumericFeature16BitCount);

    UNIT_ASSERT_VALUES_EQUAL(lhs.NonFeatureColumnCount, rhs.NonFeatureColumnCount);

    UNIT_ASSERT_VALUES_EQUAL(lhs.ClassesCount, rhs.ClassesCount);
}

static bool IsEqual(
    const google::protobuf::Message& lhs,
    const google::protobuf::Message& rhs,
    TString* const diff) {

    google::protobuf::util::MessageDifferencer differencer;
    differencer.set_float_comparison(google::protobuf::util::MessageDifferencer::APPROXIMATE);
    if (diff) {
        differencer.ReportDifferencesToString(diff);
    }

    return differencer.Compare(lhs, rhs);
}

// TODO(yazevnul): compare schemas as C++ objects too

Y_UNIT_TEST_SUITE(SerializationTests) {
    Y_UNIT_TEST(TestSerializeDeserialize) {
        const auto pool = MakeQuantizedPool();
        const auto path = TFsPath(GetSystemTempDir()) / "quantized_pool.bin";

        {
            TFileOutput output(path.GetPath());
            NCB::SaveQuantizedPool(pool, &output);
        }

        const auto loadedPool = NCB::LoadQuantizedPool(NCB::TPathWithScheme(path.GetPath(), "quantized"), {false, false, NCB::TDatasetSubset::MakeColumns()});

        const auto poolAsText = QuantizedPoolToString(pool);
        const auto loadedPoolAsText = QuantizedPoolToString(loadedPool);

        UNIT_ASSERT_VALUES_EQUAL(loadedPoolAsText, poolAsText);
    }

    Y_UNIT_TEST(TestLoadQuantizationSchema) {
        const auto pool = MakeQuantizedPool();
        const auto path = TFsPath(GetSystemTempDir()) / "quantized_pool.bin";

        {
            TFileOutput output(path.GetPath());
            NCB::SaveQuantizedPool(pool, &output);
        }

        const auto expectedQuantizationSchema = MakeQuantizationSchema();
        const auto quantizationSchema = NCB::LoadQuantizationSchemaFromPool(path.GetPath());

        TString diff;
        UNIT_ASSERT_C(IsEqual(expectedQuantizationSchema, quantizationSchema, &diff), diff.data());
    }
}

Y_UNIT_TEST_SUITE(DigestTests) {
    Y_UNIT_TEST(Test) {
        const auto pool = MakeQuantizedPool();

        const auto path = TFsPath(GetSystemTempDir()) / "quantized_pool.bin";
        {
            TFileOutput output(path.GetPath());
            NCB::SaveQuantizedPool(pool, &output);
        }

        const auto digest = NCB::CalculateQuantizedPoolDigest(path.GetPath());
        NCB::TQuantizedPoolDigest expectedDigest;
        expectedDigest.NumericFeatureCount = 1;
        expectedDigest.NumericFeature4BitCount = 1;
        expectedDigest.NonFeatureColumnCount = 1;

        CompareQuantizedPoolDigests(digest, expectedDigest);
    }
}
