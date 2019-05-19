#include "factory.h"

#include <library/streams/lz/lz.h>

#include <library/unittest/registar.h>

#include <util/generic/buffer.h>
#include <util/generic/string.h>
#include <util/stream/buffer.h>
#include <util/stream/file.h>
#include <util/stream/mem.h>
#include <util/stream/zlib.h>
#include <util/system/env.h>
#include <util/generic/vector.h>

static const TString plain = "aaaaaaaaaaabbbbbbbbbbbdddddd22222222000000aldkfa9s3jsfkjlkja909090909090q3lkjalkjf3aldjl";

static const ui8 gz[] = {31, 139, 8, 8, 126, 193, 203, 80, 0, 3, 97, 46, 116, 120, 116, 0, 75, 76, 132, 131, 36, 4, 72, 1, 3, 35, 40, 48, 0, 131, 196, 156, 148, 236, 180, 68, 203, 98, 227, 172, 226, 180, 236, 172, 156, 236, 172, 68, 75, 3, 4, 44, 52, 6, 137, 0, 113, 154, 49, 80, 97, 86, 14, 0, 5, 203, 67, 131, 88, 0, 0, 0};
static const auto gzLength = Y_ARRAY_SIZE(gz);

static const ui8 gzLvl6[] = {31, 139, 8, 0, 0, 0, 0, 0, 0, 3, 75, 76, 132, 131, 36, 4, 72, 1, 3, 35, 40, 48, 0, 131, 196, 156, 148, 236, 180, 68, 203, 98, 227, 172, 226, 180, 236, 172, 156, 236, 172, 68, 75, 3, 4, 44, 52, 6, 137, 0, 113, 154, 49, 80, 97, 86, 14, 0, 5, 203, 67, 131, 88, 0, 0, 0};
static const auto gzLvl6Length = Y_ARRAY_SIZE(gzLvl6);

static const ui8 bz2[] = {66, 90, 104, 57, 49, 65, 89, 38, 83, 89, 140, 92, 215, 106, 0, 0, 17, 73, 128, 20, 128, 88, 32, 53, 28, 40, 0, 32, 0, 84, 66, 52, 211, 0, 6, 72, 122, 140, 131, 36, 97, 60, 92, 230, 1, 71, 91, 170, 135, 33, 135, 149, 133, 75, 174, 153, 146, 217, 24, 174, 177, 76, 246, 69, 254, 225, 195, 236, 95, 180, 93, 201, 20, 225, 66, 66, 49, 115, 93, 168};
static const auto bz2Length = Y_ARRAY_SIZE(bz2);

static const ui8 bz2Lvl6[] = {66, 90, 104, 54, 49, 65, 89, 38, 83, 89, 140, 92, 215, 106, 0, 0, 17, 73, 128, 20, 128, 88, 32, 53, 28, 40, 0, 32, 0, 84, 66, 52, 211, 0, 6, 72, 122, 140, 131, 36, 97, 60, 92, 230, 1, 71, 91, 170, 135, 33, 135, 149, 133, 75, 174, 153, 146, 217, 24, 174, 177, 76, 246, 69, 254, 225, 195, 236, 95, 180, 93, 201, 20, 225, 66, 66, 49, 115, 93, 168};
static const auto bz2Lvl6Length = Y_ARRAY_SIZE(bz2Lvl6);

Y_UNIT_TEST_SUITE(TRecognizeCompressorTest) {
    static void TestRawData(const void* data, size_t len, const TString& orig) {
        TMemoryInput mem(data, len);

        THolder<IInputStream> input = OpenMaybeCompressedInput(&mem);
        UNIT_ASSERT_VALUES_UNEQUAL(input.Get(), nullptr);
        UNIT_ASSERT_VALUES_EQUAL(input->ReadAll(), orig);
    }

    static void TestRawDataOwned(const void* data, size_t len, const TString& orig) {
        THolder<IInputStream> input = OpenOwnedMaybeCompressedInput(new TMemoryInput(data, len));
        UNIT_ASSERT_VALUES_UNEQUAL(input.Get(), nullptr);
        UNIT_ASSERT_VALUES_EQUAL(input->ReadAll(), orig);
    }

    static inline void TestSame(const TString& text) {
        TestRawData(text.data(), text.size(), text);
        TestRawDataOwned(text.data(), text.size(), text);
    }

    Y_UNIT_TEST(TestPlain) {
        TestSame(plain);
        TestSame("");
        TestSame("a");
        TestSame("ab");
        TestSame("abc");
        TestSame("abcd");
    }

    Y_UNIT_TEST(TestGzip) {
        TestRawData(gz, gzLength, plain);
        TestRawDataOwned(gz, gzLength, plain);
    }

    Y_UNIT_TEST(TestBzip2) {
        TestRawData(bz2, bz2Length, plain);
        TestRawDataOwned(bz2, bz2Length, plain);
    }

    template <typename TCompress>
    static void TestCompress() {
        TBufferStream buf;
        {
            TCompress z(&buf);
            z.Write(plain.data(), plain.size());
        }
        TestRawData(buf.Buffer().Data(), buf.Buffer().Size(), plain);
    }

    Y_UNIT_TEST(TestLz) {
        TestCompress<TLz4Compress>();
        TestCompress<TSnappyCompress>();
        TestCompress<TLzoCompress>();
        TestCompress<TLzqCompress>();
        TestCompress<TLzfCompress>();
    }

    Y_UNIT_TEST(TestZlib) {
        TestCompress<TZLibCompress>();
    }

    Y_UNIT_TEST(TestOpenInput) {
        const auto fileName = TString("test_open_input.file");
        TFileOutput{fileName}.Write(plain);
        UNIT_ASSERT_VALUES_EQUAL(OpenInput(fileName)->ReadAll(), plain);
    }

    Y_UNIT_TEST(TestOpenInputZlib) {
        const auto fileName = TString("test_open_input_zlib.file.gz");
        TFileOutput{fileName}.Write(gz, gzLength);
        UNIT_ASSERT_VALUES_EQUAL(OpenInput(fileName)->ReadAll(), plain);
    }

    Y_UNIT_TEST(TestOpenInputBZ2) {
        const auto fileName = TString("test_open_input_bz2.file.bz2");
        TFileOutput{fileName}.Write(bz2, bz2Length);
        UNIT_ASSERT_VALUES_EQUAL(OpenInput(fileName)->ReadAll(), plain);
    }

    Y_UNIT_TEST(TestOpenOutput) {
        const auto fileName = TString("test_open_output.file");
        OpenOutput(fileName)->Write(plain);
        UNIT_ASSERT_VALUES_EQUAL(TFileInput{fileName}.ReadAll(), plain);
    }

    Y_UNIT_TEST(TestOpenOutputZlib) {
        const auto fileName = TString("test_open_output_zlib.file.gz");
        OpenOutput(fileName)->Write(plain);
        const auto expected = TStringBuf{(const char*)gzLvl6, gzLvl6Length};
        UNIT_ASSERT_VALUES_EQUAL(TFileInput{fileName}.ReadAll(), expected);
    }

    Y_UNIT_TEST(TestOpenOutputBZ2) {
        const auto fileName = TString("test_open_output_bz2.file.bz2");
        OpenOutput(fileName)->Write(plain);
        const auto expected = TStringBuf{(const char*)bz2Lvl6, bz2Lvl6Length};
        UNIT_ASSERT_VALUES_EQUAL(TFileInput{fileName}.ReadAll(), expected);
    }

    static void TestReadWrite(const TString& fileName, const TString& data) {
        OpenOutput(fileName)->Write(data.data(), data.size());
        UNIT_ASSERT_VALUES_EQUAL(OpenInput(fileName)->ReadAll(), data);
    }

    Y_UNIT_TEST(TestOpenInputOpenOutputSimple) {
        TestReadWrite(TString("test_open_input_open_output.file"), plain);
    }

    Y_UNIT_TEST(TestOpenInputOpenOutputZLib) {
        TestReadWrite(TString("test_open_input_open_output_zlib.file.gz"), plain);
    }

    Y_UNIT_TEST(TestOpenInputOpenOutputBZ2) {
        TestReadWrite(TString("test_open_input_open_output_bz2.file.bz2"), plain);
    }
}
