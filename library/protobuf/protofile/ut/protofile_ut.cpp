#include <library/protobuf/protofile/protofile.h>
#include <library/protobuf/protofile/ut/protofile_ut.pb.h>

#include <library/unittest/registar.h>

#include <util/system/tempfile.h>

static float seed = 1.f;

static float FloatGen() {
    seed *= 3.5;
    seed += 3.14;
    if (seed > 1e8)
        seed /= 1000;
    return seed;
}

SIMPLE_UNIT_TEST_SUITE(ProtoFileTest) {
    static const char TempFileName[] = "./ProtoFile-test";
    static const size_t MCount = 1000;

    void Write() {
        TProtoTest message;
        TFixedBufferFileOutput output(TempFileName);
        NFastTier::TBinaryProtoWriter<TProtoTest> writer;
        writer.Open(&output);
        TString randomCrap = "Lorem ipsum dot sir amet, и съешь ещё этих мягких французских булок! ";
        for (size_t i = 0; i < MCount; ++i) {
            TString tmp = randomCrap;
            message.Clear();

            message.SetHash(i * i * 5000000);
            for (size_t j = 0; j < (i % 800) + 10; ++j)
                message.AddFactors(FloatGen());
            for (size_t j = 0; j < (i % 8) + 3; ++j) {
                tmp += randomCrap + ToString(i);
                message.AddBlob(tmp);
            }
            if (i % 3)
                message.SetVersion(ui32(i));
            writer.Write(message);
        }
        writer.Finish();
    }

    void Read() {
        TProtoTest message;
        TFileInput input(TempFileName);
        NFastTier::TBinaryProtoReader<TProtoTest> reader;
        reader.Open(&input);
        TString randomCrap = "Lorem ipsum dot sir amet, и съешь ещё этих мягких французских булок! ";
        for (size_t i = 0; i < MCount; ++i) {
            UNIT_ASSERT_EQUAL(reader.GetNext(message), true);
            TString tmp = randomCrap;
            UNIT_ASSERT_EQUAL(message.GetHash(), i * i * 5000000);
            for (size_t j = 0; j < (i % 800) + 10; ++j)
                UNIT_ASSERT_EQUAL(message.GetFactors(j), FloatGen());
            for (size_t j = 0; j < (i % 8) + 3; ++j) {
                tmp += randomCrap + ToString(i);
                UNIT_ASSERT_EQUAL(message.GetBlob(j), tmp);
            }
            if (i % 3)
                UNIT_ASSERT_EQUAL(message.GetVersion(), ui32(i));
        }
        UNIT_ASSERT_EQUAL(reader.GetNext(message), false);
    }

    SIMPLE_UNIT_TEST(TestWriteBinary) {
        seed = 1.0f;
        Write();
    }

    SIMPLE_UNIT_TEST(TestReadBinary) {
        seed = 1.0f;
        Read();
    }

    SIMPLE_UNIT_TEST(TestCleanup) {
        TTempFile tmpFile(TempFileName);
    }
};
