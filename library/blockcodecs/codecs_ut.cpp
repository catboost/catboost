#include "codecs.h"
#include "stream.h"

#include <library/unittest/registar.h>

#include <util/stream/str.h>
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
                    Cerr << c->Name() << "(" << res.Quote() << ")(" << NBlockCodecs::TData(data).ToString().Quote() << ")" << Endl;

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
}
