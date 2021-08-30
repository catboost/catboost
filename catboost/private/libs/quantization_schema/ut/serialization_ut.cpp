#include <library/cpp/testing/unittest/registar.h>

#include <catboost/idl/pool/proto/quantization_schema.pb.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/quantization_schema/schema.h>
#include <catboost/private/libs/quantization_schema/serialization.h>

#include <google/protobuf/util/message_differencer.h>
#include <google/protobuf/text_format.h>

#include <util/generic/strbuf.h>
#include <util/stream/mem.h>
#include <util/stream/str.h>

static NCB::NIdl::TPoolQuantizationSchema MakeProtoQuantizationSchema(const TStringBuf text) {
    NCB::NIdl::TPoolQuantizationSchema schema;
    google::protobuf::TextFormat::Parser parser;
    parser.ParseFromString({text.data(), text.size()}, &schema);
    return schema;
}

static NCB::TPoolQuantizationSchema MakeQuantizationSchema(const TStringBuf text) {
    const auto proto = MakeProtoQuantizationSchema(text);
    return NCB::QuantizationSchemaFromProto(proto);
}

static bool IsEqual(
    const google::protobuf::Message& lhs,
    const google::protobuf::Message& rhs,
    TString* const report) {

    google::protobuf::util::MessageDifferencer differencer;
    differencer.set_float_comparison(google::protobuf::util::MessageDifferencer::APPROXIMATE);
    if (report) {
        differencer.ReportDifferencesToString(report);
    }

    return differencer.Compare(lhs, rhs);
}

static void DoCheck(const google::protobuf::Message& lhs, const google::protobuf::Message& rhs) {
    TString report;
    UNIT_ASSERT_C(IsEqual(lhs, rhs, &report), report.data());
}

Y_UNIT_TEST_SUITE(SerializationTests) {
    Y_UNIT_TEST(TestConversionToProto) {
        NCB::TPoolQuantizationSchema schema;
        schema.FloatFeatureIndices = {1, 3, 5};
        schema.Borders = {
            {.25f, .5f, .75f},
            {-0.5f, .0f},
            {.0f, 1.f, 10.f}};
        schema.NanModes = {ENanMode::Min, ENanMode::Min, ENanMode::Forbidden};

        const auto proto = NCB::QuantizationSchemaToProto(schema);
        const auto expected = MakeProtoQuantizationSchema(R"(
            FeatureIndexToSchema: {
                key: 1
                value: {
                    Borders: [0.25, 0.5, 0.75]
                    NanMode: NM_MIN
                }
            }
            FeatureIndexToSchema: {
                key: 3
                value: {
                    Borders: [-0.5, 0]
                    NanMode: NM_MIN
                }
            }
            FeatureIndexToSchema: {
                key: 5
                value: {
                    Borders: [0, 1, 10]
                    NanMode: NM_FORBIDDEN
                }
            }
        )");

        DoCheck(proto, expected);
    }

    Y_UNIT_TEST(TestConversionFromProto) {
        const auto proto = MakeProtoQuantizationSchema(R"(
            FeatureIndexToSchema: {
                key: 1
                value: {
                    Borders: [0.25, 0.5, 0.75]
                    NanMode: NM_MIN
                }
            }
            FeatureIndexToSchema: {
                key: 3
                value: {
                    Borders: [-0.5, 0]
                    NanMode: NM_MIN
                }
            }
            FeatureIndexToSchema: {
                key: 5
                value: {
                    Borders: [0, 1, 10]
                    NanMode: NM_FORBIDDEN
                }
            }
        )");
        const auto schema = NCB::QuantizationSchemaFromProto(proto);

        NCB::TPoolQuantizationSchema expected;
        expected.FloatFeatureIndices = {1, 3, 5};
        expected.Borders = {
            {.25f, .5f, .75f},
            {-0.5f, .0f},
            {.0f, 1.f, 10.f}};
        expected.NanModes = {ENanMode::Min, ENanMode::Min, ENanMode::Forbidden};

        UNIT_ASSERT_VALUES_EQUAL(schema.FloatFeatureIndices, expected.FloatFeatureIndices);
        UNIT_ASSERT_VALUES_EQUAL(schema.Borders.size(), expected.Borders.size());
        for (size_t i = 0; i < schema.Borders.size(); ++i) {
            UNIT_ASSERT_VALUES_EQUAL(schema.Borders[i], expected.Borders[i]);
        }
    }

    Y_UNIT_TEST(TestLoadInMatrixNetFormat) {
        const TStringBuf mxFormatBordersStr =
            "1\t0.25\n"
            "1\t0.5\n"
            "1\t0.75\n"
            "5\t0\n"
            "5\t1\n"
            "5\t10\n"
            "3\t-0.5\n"
            "3\t0\n";
        TMemoryInput mxFormatBorders(mxFormatBordersStr.data(), mxFormatBordersStr.size());
        const auto expected = MakeProtoQuantizationSchema(R"(
            FeatureIndexToSchema: {
                key: 1
                value: {
                    Borders: [0.25, 0.5, 0.75]
                    NanMode: NM_FORBIDDEN
                }
            }
            FeatureIndexToSchema: {
                key: 3
                value: {
                    Borders: [-0.5, 0]
                    NanMode: NM_FORBIDDEN
                }
            }
            FeatureIndexToSchema: {
                key: 5
                value: {
                    Borders: [0, 1, 10]
                    NanMode: NM_FORBIDDEN
                }
            }
        )");

        const auto proto = NCB::QuantizationSchemaToProto(NCB::LoadQuantizationSchema(
            NCB::EQuantizationSchemaSerializationFormat::Matrixnet,
            &mxFormatBorders));

        DoCheck(proto, expected);
    }

    Y_UNIT_TEST(TestLoadInMatrixNetFormatInconsistentNanModes1) {
        const TStringBuf mxFormatBordersStr =
            "1\t0.25\n"
            "1\t0.5\tMin\n";
        TMemoryInput mxFormatBorders(mxFormatBordersStr.data(), mxFormatBordersStr.size());

        UNIT_ASSERT_EXCEPTION(
            NCB::LoadQuantizationSchema(
                NCB::EQuantizationSchemaSerializationFormat::Matrixnet,
                &mxFormatBorders),
            TCatBoostException);
    }

    Y_UNIT_TEST(TestLoadInMatrixNetFormatInconsistentNanModes2) {
        const TStringBuf mxFormatBordersStr =
            "1\t0.25\tMax\n"
            "1\t0.5\tMin\n";
        TMemoryInput mxFormatBorders(mxFormatBordersStr.data(), mxFormatBordersStr.size());

        UNIT_ASSERT_EXCEPTION(
            NCB::LoadQuantizationSchema(
                NCB::EQuantizationSchemaSerializationFormat::Matrixnet,
                &mxFormatBorders),
            TCatBoostException);
    }

    Y_UNIT_TEST(TestSaveInMatrixNetFormat1) {
        const auto schema = MakeQuantizationSchema(R"(
            FeatureIndexToSchema: {
                key: 1
                value: {
                    Borders: [0.25, 0.5, 0.75]
                    NanMode: NM_MIN
                }
            }
            FeatureIndexToSchema: {
                key: 3
                value: {
                    Borders: [-0.5, 0]
                    NanMode: NM_MIN
                }
            }
            FeatureIndexToSchema: {
                key: 5
                value: {
                    Borders: [0, 1, 10]
                    NanMode: NM_FORBIDDEN
                }
            }
        )");
        const TStringBuf expected =
            "1\t0.25\tMin\n"
            "1\t0.5\tMin\n"
            "1\t0.75\tMin\n"
            "3\t-0.5\tMin\n"
            "3\t0\tMin\n"
            "5\t0\n"
            "5\t1\n"
            "5\t10\n";

        TStringStream ss;
        NCB::SaveQuantizationSchema(
            schema,
            NCB::EQuantizationSchemaSerializationFormat::Matrixnet,
            &ss);

        UNIT_ASSERT_VALUES_EQUAL(ss.Str(), expected);
    }

    Y_UNIT_TEST(TestSaveInMatrixNetFormat2) {
        const auto schema = MakeQuantizationSchema(R"(
            FeatureIndexToSchema: {
                key: 1
                value: {
                    Borders: [0.25, 0.5, 0.75]
                    NanMode: NM_FORBIDDEN
                }
            }
            FeatureIndexToSchema: {
                key: 3
                value: {
                    Borders: [-0.5, 0]
                    NanMode: NM_FORBIDDEN
                }
            }
            FeatureIndexToSchema: {
                key: 5
                value: {
                    Borders: [0, 1, 10]
                    NanMode: NM_FORBIDDEN
                }
            }
        )");
        const TStringBuf expected =
            "1\t0.25\n"
            "1\t0.5\n"
            "1\t0.75\n"
            "3\t-0.5\n"
            "3\t0\n"
            "5\t0\n"
            "5\t1\n"
            "5\t10\n";

        TStringStream ss;
        NCB::SaveQuantizationSchema(
            schema,
            NCB::EQuantizationSchemaSerializationFormat::Matrixnet,
            &ss);

        UNIT_ASSERT_VALUES_EQUAL(ss.Str(), expected);
    }
}
