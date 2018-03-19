#include "brotli.h"

#include <library/unittest/registar.h>

#include <util/random/fast.h>

SIMPLE_UNIT_TEST_SUITE(TBrotliTestSuite) {
    TString Compress(TString data) {
        TString compressed;
        TStringOutput output(compressed);
        TBrotliCompress compressStream(&output, 11);
        compressStream.Write(~data, +data);
        compressStream.Finish();
        output.Finish();
        return compressed;
    }

    TString Decompress(TString data) {
        TStringInput input(data);
        TBrotliDecompress decompressStream(&input);
        return decompressStream.ReadAll();
    }

    void TestCase(const TString& s) {
        UNIT_ASSERT_VALUES_EQUAL(s, Decompress(Compress(s)));
    }

    TString GenerateRandomString(size_t size) {
        TReallyFastRng32 rng(42);
        TString result;
        result.reserve(size + sizeof(ui64));
        while (result.size() < size) {
            ui64 value = rng.GenRand64();
            result += TStringBuf(reinterpret_cast<const char*>(&value), sizeof(value));
        }
        result.resize(size);
        return result;
    }

    SIMPLE_UNIT_TEST(TestHelloWorld) {
        TestCase("hello world");
    }

    SIMPLE_UNIT_TEST(TestSeveralStreams) {
        auto s1 = GenerateRandomString(1 << 15);
        auto s2 = GenerateRandomString(1 << 15);
        auto c1 = Compress(s1);
        auto c2 = Compress(s2);
        UNIT_ASSERT_VALUES_EQUAL(s1 + s2, Decompress(c1 + c2));
    }

    SIMPLE_UNIT_TEST(TestIncompleteStream) {
        TString manyAs(64 * 1024, 'a');
        auto compressed = Compress(manyAs);
        TString truncated(compressed.Data(), compressed.Size() - 1);
        UNIT_CHECK_GENERATED_EXCEPTION(Decompress(truncated), std::exception);
    }

    SIMPLE_UNIT_TEST(Test64KB) {
        auto manyAs = TString(64 * 1024, 'a');
        TString str("Hello from the Matrix!@#% How are you?}{\n\t\a");
        TestCase(manyAs + str + manyAs);
    }

    SIMPLE_UNIT_TEST(Test1MB) {
        TestCase(GenerateRandomString(1 * 1024 * 1024));
    }

    SIMPLE_UNIT_TEST(TestEmpty) {
        TestCase("");
    }
}
