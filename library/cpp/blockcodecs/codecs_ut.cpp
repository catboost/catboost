#include "codecs.h"
#include "stream.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/str.h>
#include <util/string/join.h>
#include <util/digest/multi.h>

Y_UNIT_TEST_SUITE(TBlockCodecsTest) {
    using namespace NBlockCodecs;

    TBuffer Buffer(TStringBuf b) {
        TBuffer bb;
        bb.Assign(b.data(), b.size());
        return bb;
    }

    void TestAllAtOnce(size_t n, size_t m) {
        TVector<TBuffer> datas;

        datas.emplace_back();
        datas.push_back(Buffer("na gorshke sidel korol"));
        datas.push_back(Buffer(TStringBuf("", 1)));
        datas.push_back(Buffer(" "));
        datas.push_back(Buffer("  "));
        datas.push_back(Buffer("   "));
        datas.push_back(Buffer("    "));

        {
            TStringStream data;

            for (size_t i = 0; i < 1024; ++i) {
                data << " " << i;
            }

            datas.push_back(Buffer(data.Str()));
        }

        TCodecList lst = ListAllCodecs();

        for (size_t i = 0; i < lst.size(); ++i) {
            const ICodec* c = Codec(lst[i]);
            const auto h = MultiHash(c->Name(), i, 1);

            if (h % n == m) {
            } else {
                continue;
            }

            for (size_t j = 0; j < datas.size(); ++j) {
                const TBuffer& data = datas[j];
                TString res;

                try {
                    TBuffer e, d;
                    c->Encode(data, e);
                    c->Decode(e, d);
                    d.AsString(res);
                    UNIT_ASSERT_EQUAL(NBlockCodecs::TData(res), NBlockCodecs::TData(data));
                } catch (...) {
                    Cerr << c->Name() << "(" << res.Quote() << ")(" << TString{NBlockCodecs::TData(data)}.Quote() << ")" << Endl;

                    throw;
                }
            }
        }
    }

    Y_UNIT_TEST(TestAllAtOnce0) {
        TestAllAtOnce(20, 0);
    }

    Y_UNIT_TEST(TestAllAtOnce1) {
        TestAllAtOnce(20, 1);
    }

    Y_UNIT_TEST(TestAllAtOnce2) {
        TestAllAtOnce(20, 2);
    }

    Y_UNIT_TEST(TestAllAtOnce3) {
        TestAllAtOnce(20, 3);
    }

    Y_UNIT_TEST(TestAllAtOnce4) {
        TestAllAtOnce(20, 4);
    }

    Y_UNIT_TEST(TestAllAtOnce5) {
        TestAllAtOnce(20, 5);
    }

    Y_UNIT_TEST(TestAllAtOnce6) {
        TestAllAtOnce(20, 6);
    }

    Y_UNIT_TEST(TestAllAtOnce7) {
        TestAllAtOnce(20, 7);
    }

    Y_UNIT_TEST(TestAllAtOnce8) {
        TestAllAtOnce(20, 8);
    }

    Y_UNIT_TEST(TestAllAtOnce9) {
        TestAllAtOnce(20, 9);
    }

    Y_UNIT_TEST(TestAllAtOnce10) {
        TestAllAtOnce(20, 10);
    }

    Y_UNIT_TEST(TestAllAtOnce12) {
        TestAllAtOnce(20, 12);
    }

    Y_UNIT_TEST(TestAllAtOnce13) {
        TestAllAtOnce(20, 13);
    }

    Y_UNIT_TEST(TestAllAtOnce14) {
        TestAllAtOnce(20, 14);
    }

    Y_UNIT_TEST(TestAllAtOnce15) {
        TestAllAtOnce(20, 15);
    }

    Y_UNIT_TEST(TestAllAtOnce16) {
        TestAllAtOnce(20, 16);
    }

    Y_UNIT_TEST(TestAllAtOnce17) {
        TestAllAtOnce(20, 17);
    }

    Y_UNIT_TEST(TestAllAtOnce18) {
        TestAllAtOnce(20, 18);
    }

    Y_UNIT_TEST(TestAllAtOnce19) {
        TestAllAtOnce(20, 19);
    }

    void TestStreams(size_t n, size_t m) {
        TVector<TString> datas;
        TString res;

        for (size_t i = 0; i < 256; ++i) {
            datas.push_back(TString(i, (char)(i % 128)));
        }

        for (size_t i = 0; i < datas.size(); ++i) {
            res += datas[i];
        }

        TCodecList lst = ListAllCodecs();

        for (size_t i = 0; i < lst.size(); ++i) {
            TStringStream ss;

            const ICodec* c = Codec(lst[i]);
            const auto h = MultiHash(c->Name(), i, 2);

            if (h % n == m) {
            } else {
                continue;
            }

            {
                TCodedOutput out(&ss, c, 1234);

                for (size_t j = 0; j < datas.size(); ++j) {
                    out << datas[j];
                }

                out.Finish();
            }

            const TString resNew = TDecodedInput(&ss).ReadAll();

            try {
                UNIT_ASSERT_EQUAL(resNew, res);
            } catch (...) {
                Cerr << c->Name() << Endl;

                throw;
            }
        }
    }

    Y_UNIT_TEST(TestStreams0) {
        TestStreams(20, 0);
    }

    Y_UNIT_TEST(TestStreams1) {
        TestStreams(20, 1);
    }

    Y_UNIT_TEST(TestStreams2) {
        TestStreams(20, 2);
    }

    Y_UNIT_TEST(TestStreams3) {
        TestStreams(20, 3);
    }

    Y_UNIT_TEST(TestStreams4) {
        TestStreams(20, 4);
    }

    Y_UNIT_TEST(TestStreams5) {
        TestStreams(20, 5);
    }

    Y_UNIT_TEST(TestStreams6) {
        TestStreams(20, 6);
    }

    Y_UNIT_TEST(TestStreams7) {
        TestStreams(20, 7);
    }

    Y_UNIT_TEST(TestStreams8) {
        TestStreams(20, 8);
    }

    Y_UNIT_TEST(TestStreams9) {
        TestStreams(20, 9);
    }

    Y_UNIT_TEST(TestStreams10) {
        TestStreams(20, 10);
    }

    Y_UNIT_TEST(TestStreams11) {
        TestStreams(20, 11);
    }

    Y_UNIT_TEST(TestStreams12) {
        TestStreams(20, 12);
    }

    Y_UNIT_TEST(TestStreams13) {
        TestStreams(20, 13);
    }

    Y_UNIT_TEST(TestStreams14) {
        TestStreams(20, 14);
    }

    Y_UNIT_TEST(TestStreams15) {
        TestStreams(20, 15);
    }

    Y_UNIT_TEST(TestStreams16) {
        TestStreams(20, 16);
    }

    Y_UNIT_TEST(TestStreams17) {
        TestStreams(20, 17);
    }

    Y_UNIT_TEST(TestStreams18) {
        TestStreams(20, 18);
    }

    Y_UNIT_TEST(TestStreams19) {
        TestStreams(20, 19);
    }

    Y_UNIT_TEST(TestMaxPossibleDecompressedSize) {

        UNIT_ASSERT_VALUES_EQUAL(GetMaxPossibleDecompressedLength(), Max<size_t>());

        TVector<char> input(10001, ' ');
        TCodecList codecs = ListAllCodecs();
        SetMaxPossibleDecompressedLength(10000);

        for (const auto& codec : codecs) {
            const ICodec* c = Codec(codec);
            TBuffer inputBuffer(input.data(), input.size());
            TBuffer output;
            TBuffer decompressed;
            c->Encode(inputBuffer, output);
            UNIT_ASSERT_EXCEPTION(c->Decode(output, decompressed), yexception);
        }

        // restore status quo
        SetMaxPossibleDecompressedLength(Max<size_t>());
    }

    Y_UNIT_TEST(TestListAllCodecs) {
        static const TString ALL_CODECS =
            "brotli_1,brotli_10,brotli_11,brotli_2,brotli_3,brotli_4,brotli_5,brotli_6,brotli_7,brotli_8,brotli_9,"

            "bzip2,bzip2-1,bzip2-2,bzip2-3,bzip2-4,bzip2-5,bzip2-6,bzip2-7,bzip2-8,bzip2-9,"

            "fastlz,fastlz-0,fastlz-1,fastlz-2,"

            "lz4,lz4-fast-fast,lz4-fast-safe,lz4-fast10-fast,lz4-fast10-safe,lz4-fast11-fast,lz4-fast11-safe,"
            "lz4-fast12-fast,lz4-fast12-safe,lz4-fast13-fast,lz4-fast13-safe,lz4-fast14-fast,lz4-fast14-safe,"
            "lz4-fast15-fast,lz4-fast15-safe,lz4-fast16-fast,lz4-fast16-safe,lz4-fast17-fast,lz4-fast17-safe,"
            "lz4-fast18-fast,lz4-fast18-safe,lz4-fast19-fast,lz4-fast19-safe,lz4-fast20-fast,lz4-fast20-safe,"
            "lz4-hc-fast,lz4-hc-safe,lz4fast,lz4hc,"

            "lzma,lzma-0,lzma-1,lzma-2,lzma-3,lzma-4,lzma-5,lzma-6,lzma-7,lzma-8,lzma-9,"

            "null,"

            "snappy,"

            "zlib,zlib-0,zlib-1,zlib-2,zlib-3,zlib-4,zlib-5,zlib-6,zlib-7,zlib-8,zlib-9,"

            "zstd06_1,zstd06_10,zstd06_11,zstd06_12,zstd06_13,zstd06_14,zstd06_15,zstd06_16,zstd06_17,zstd06_18,"
            "zstd06_19,zstd06_2,zstd06_20,zstd06_21,zstd06_22,zstd06_3,zstd06_4,zstd06_5,zstd06_6,zstd06_7,zstd06_8,"
            "zstd06_9,"

            "zstd08_1,zstd08_10,zstd08_11,zstd08_12,zstd08_13,zstd08_14,zstd08_15,zstd08_16,zstd08_17,zstd08_18,"
            "zstd08_19,zstd08_2,zstd08_20,zstd08_21,zstd08_22,zstd08_3,zstd08_4,zstd08_5,zstd08_6,zstd08_7,zstd08_8,"
            "zstd08_9,zstd_1,zstd_10,zstd_11,zstd_12,zstd_13,zstd_14,zstd_15,zstd_16,zstd_17,zstd_18,zstd_19,zstd_2,"
            "zstd_20,zstd_21,zstd_22,zstd_3,zstd_4,zstd_5,zstd_6,zstd_7,zstd_8,zstd_9";

        UNIT_ASSERT_VALUES_EQUAL(ALL_CODECS, JoinSeq(",", ListAllCodecs()));
    }

    Y_UNIT_TEST(TestEncodeDecodeIntoString) {
        TStringBuf data = "na gorshke sidel korol";

        TCodecList codecs = ListAllCodecs();
        for (const auto& codec : codecs) {
            const ICodec* c = Codec(codec);
            TString encoded = c->Encode(data);
            TString decoded = c->Decode(encoded);

            UNIT_ASSERT_VALUES_EQUAL(decoded, data);
        }
    }
}
