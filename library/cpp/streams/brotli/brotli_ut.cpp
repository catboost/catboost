#include "brotli.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/random/fast.h>

Y_UNIT_TEST_SUITE(TBrotliTestSuite) {
    TString Compress(const TString& data, TStringBuf dict = "") {
        TString compressed;
        TStringOutput output(compressed);
        size_t quality = NBrotli::BEST_BROTLI_QUALITY;
        TBrotliDictionary dictionary(dict, quality);
        TBrotliCompress compressStream(&output, quality, &dictionary);
        compressStream.Write(data.data(), data.size());
        compressStream.Finish();
        output.Finish();
        return compressed;
    }

    TString Decompress(const TString& data, TStringBuf dict = "") {
        TStringInput input(data);
        TBrotliDictionary dictionary(dict);
        TBrotliDecompress decompressStream(&input, NBrotli::DEFAULT_BROTLI_BUFFER_SIZE, &dictionary);
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

    Y_UNIT_TEST(TestHelloWorld) {
        TestCase("hello world");
    }

    Y_UNIT_TEST(TestFlush) {
        TStringStream ss;
        TBrotliCompress compressStream(&ss);
        TBrotliDecompress decompressStream(&ss);

        for (size_t i = 0; i < 3; ++i) {
            TString s = GenerateRandomString(1 << 15);
            compressStream.Write(s.data(), s.size());
            compressStream.Flush();

            TString r(s.size(), '*');
            decompressStream.Load((char*)r.data(), r.size());

            UNIT_ASSERT_VALUES_EQUAL(s, r);
        }
    }

    Y_UNIT_TEST(TestSeveralStreams) {
        auto s1 = GenerateRandomString(1 << 15);
        auto s2 = GenerateRandomString(1 << 15);
        auto c1 = Compress(s1);
        auto c2 = Compress(s2);
        UNIT_ASSERT_VALUES_EQUAL(s1 + s2, Decompress(c1 + c2));
    }

    Y_UNIT_TEST(TestIncompleteStream) {
        TString manyAs(64 * 1024, 'a');
        auto compressed = Compress(manyAs);
        TString truncated(compressed.data(), compressed.size() - 1);
        UNIT_CHECK_GENERATED_EXCEPTION(Decompress(truncated), std::exception);
    }

    Y_UNIT_TEST(Test64KB) {
        auto manyAs = TString(64 * 1024, 'a');
        TString str("Hello from the Matrix!@#% How are you?}{\n\t\a");
        TestCase(manyAs + str + manyAs);
    }

    Y_UNIT_TEST(Test1MB) {
        TestCase(GenerateRandomString(1 * 1024 * 1024));
    }

    Y_UNIT_TEST(TestEmpty) {
        TestCase("");
    }

    Y_UNIT_TEST(TestDictionary) {
        TString str = "Bond, James Bond";
        TStringBuf dict = "Bond";
        UNIT_ASSERT_VALUES_EQUAL(str, Decompress(Compress(str, dict), dict));
    }

    Y_UNIT_TEST(TestStreamOffset) {
        TString first = "apple pen";
        TString second = " pineapple pen";

        TString compressed;
        TStringOutput out1(compressed);
        TBrotliCompress stream1(&out1, NBrotli::BEST_BROTLI_QUALITY);
        stream1.Write(first);
        stream1.Flush();

        TStringOutput out2(compressed);
        TBrotliCompress stream2(&out2, NBrotli::BEST_BROTLI_QUALITY, nullptr, first.size());
        stream2.Write(second);
        stream2.Finish();

        UNIT_ASSERT_VALUES_EQUAL(first + second, Decompress(compressed));
    }
} // Y_UNIT_TEST_SUITE(TBrotliTestSuite)
