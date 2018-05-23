#include <library/unittest/registar.h>

#include "validate.h"

#include <catboost/idl/pool/flat/quantization_schema.fbs.h>
#include <catboost/libs/helpers/exception.h>

#include <contrib/libs/flatbuffers/include/flatbuffers/flatbuffers.h>

#include <util/generic/vector.h>
#include <util/system/types.h>

using flatbuffers::FlatBufferBuilder;
using flatbuffers::GetRoot;
using NCB::NIdl::TFeatureQuantizationSchemaBuilder;
using NCB::NIdl::TFeatureQuantizationSchema;

namespace {
    enum class EResult {
        Valid,
        Invalid
    };
}

template <typename T>
static void DoChecks(const FlatBufferBuilder& b, const EResult case_) {
    switch (case_) {
        case EResult::Valid:
            UNIT_ASSERT_NO_EXCEPTION(NCB::ValidateFlatBuffer<T>(
                b.GetBufferPointer(),
                b.GetSize()));
            UNIT_ASSERT(NCB::IsValidFlatBuffer<T>(
                b.GetBufferPointer(),
                b.GetSize()));
            break;
        case EResult::Invalid:
            UNIT_ASSERT_EXCEPTION(
                NCB::ValidateFlatBuffer<T>(
                    b.GetBufferPointer(),
                    b.GetSize()),
                TCatboostException);
            UNIT_ASSERT(!NCB::IsValidFlatBuffer<T>(
                b.GetBufferPointer(),
                b.GetSize()));
            break;
    }
}

Y_UNIT_TEST_SUITE(QuantizationSchemaValidationTests) {
    Y_UNIT_TEST(TestValidBorders) {
        FlatBufferBuilder b;
        {
            const auto borders = b.CreateVector<float>({1.f});
            TFeatureQuantizationSchemaBuilder sb(b);
            sb.add_Borders(borders);

            b.Finish(sb.Finish());
        }

        DoChecks<TFeatureQuantizationSchema>(b, EResult::Valid);
    }

    Y_UNIT_TEST(TestValidUniqueHashes) {
        FlatBufferBuilder b;
        {
            const auto hashes = b.CreateVector<ui32>({1});
            TFeatureQuantizationSchemaBuilder sb(b);
            sb.add_UniqueHashes(hashes);
            b.Finish(sb.Finish());
        }

        DoChecks<TFeatureQuantizationSchema>(b, EResult::Valid);
    }

    Y_UNIT_TEST(TestValidTwoBorders) {
        FlatBufferBuilder b;
        {
            const auto borders = b.CreateVector<float>({1.f, 2.f});
            TFeatureQuantizationSchemaBuilder sb(b);
            sb.add_Borders(borders);

            b.Finish(sb.Finish());
        }

        DoChecks<TFeatureQuantizationSchema>(b, EResult::Valid);
    }

    Y_UNIT_TEST(TestValidTwoUniqueHashes) {
        FlatBufferBuilder b;
        {
            const auto hashes = b.CreateVector<ui32>({1, 2});
            TFeatureQuantizationSchemaBuilder sb(b);
            sb.add_UniqueHashes(hashes);
            b.Finish(sb.Finish());
        }

        DoChecks<TFeatureQuantizationSchema>(b, EResult::Valid);
    }

    Y_UNIT_TEST(TestBordersAndUniqueHashesPresent) {
        FlatBufferBuilder b;
        {
            const auto borders = b.CreateVector<float>({1.f});
            const auto hashes = b.CreateVector<ui32>({1});
            TFeatureQuantizationSchemaBuilder sb(b);
            sb.add_Borders(borders);
            sb.add_UniqueHashes(hashes);

            b.Finish(sb.Finish());
        }

        DoChecks<TFeatureQuantizationSchema>(b, EResult::Invalid);
    }

    Y_UNIT_TEST(TestTestUnsortedBorders) {
        FlatBufferBuilder b;
        {
            const auto borders = b.CreateVector<float>({2.f, 1.f});
            TFeatureQuantizationSchemaBuilder sb(b);
            sb.add_Borders(borders);

            b.Finish(sb.Finish());
        }

        DoChecks<TFeatureQuantizationSchema>(b, EResult::Invalid);
    }

    Y_UNIT_TEST(TestUnsortedUniqueHashes) {
        FlatBufferBuilder b;
        {
            const auto hashes = b.CreateVector<ui32>({2, 1});
            TFeatureQuantizationSchemaBuilder sb(b);
            sb.add_UniqueHashes(hashes);
            b.Finish(sb.Finish());
        }

        DoChecks<TFeatureQuantizationSchema>(b, EResult::Invalid);
    }

    Y_UNIT_TEST(TestTestNonUniqueBorders) {
        FlatBufferBuilder b;
        {
            const auto borders = b.CreateVector<float>({1.f, 1.f});
            TFeatureQuantizationSchemaBuilder sb(b);
            sb.add_Borders(borders);

            b.Finish(sb.Finish());
        }

        DoChecks<TFeatureQuantizationSchema>(b, EResult::Invalid);
    }

    Y_UNIT_TEST(TestUnsortedNonUniqueUniqueHashes) {
        FlatBufferBuilder b;
        {
            const auto hashes = b.CreateVector<ui32>({1, 1});
            TFeatureQuantizationSchemaBuilder sb(b);
            sb.add_UniqueHashes(hashes);
            b.Finish(sb.Finish());
        }

        DoChecks<TFeatureQuantizationSchema>(b, EResult::Invalid);
    }
}
