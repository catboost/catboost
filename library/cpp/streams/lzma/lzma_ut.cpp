#include "lzma.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/mem.h>
#include <util/random/fast.h>
#include <util/random/random.h>

class TStrokaByOneByte: public IZeroCopyInput {
public:
    TStrokaByOneByte(const TString& s)
        : Data(s)
        , Pos(s.data())
    {
    }

private:
    size_t DoNext(const void** ptr, size_t len) override {
        if (Pos < Data.end()) {
            len = Min(len, static_cast<size_t>(1));
            *ptr = Pos;
            Pos += len;
            return len;
        } else {
            return 0;
        }
    }

    TString Data;
    const char* Pos;
};

class TLzmaTest: public TTestBase {
    UNIT_TEST_SUITE(TLzmaTest);
    UNIT_TEST(Test1)
    UNIT_TEST(Test2)
    UNIT_TEST_SUITE_END();

private:
    inline TString GenData() {
        TString data;
        TReallyFastRng32 rnd(RandomNumber<ui64>());

        for (size_t i = 0; i < 50000; ++i) {
            const char ch = rnd.Uniform(256);
            const size_t len = 1 + rnd.Uniform(10);

            data += TString(len, ch);
        }

        return data;
    }

    inline void Test2() {
        class TExcOutput: public IOutputStream {
        public:
            ~TExcOutput() override {
            }

            void DoWrite(const void*, size_t) override {
                throw 12345;
            }
        };

        TString data(GenData());
        TMemoryInput mi(data.data(), data.size());
        TExcOutput out;

        try {
            TLzmaCompress c(&out);

            TransferData(&mi, &c);
        } catch (int i) {
            UNIT_ASSERT_EQUAL(i, 12345);
        }
    }

    inline void Test1() {
        TString data(GenData());
        TString data1;
        TString res;

        {
            TMemoryInput mi(data.data(), data.size());
            TStringOutput so(res);
            TLzmaCompress c(&so);

            TransferData(&mi, &c);

            c.Finish();
        }

        {
            TMemoryInput mi(res.data(), res.size());
            TStringOutput so(data1);
            TLzmaDecompress d((IInputStream*)&mi);

            TransferData(&d, &so);
        }

        UNIT_ASSERT_EQUAL(data, data1);

        data1.clear();
        {
            TMemoryInput mi(res.data(), res.size());
            TStringOutput so(data1);
            TLzmaDecompress d(&mi);

            TransferData(&d, &so);
        }

        UNIT_ASSERT_EQUAL(data, data1);

        data1.clear();
        {
            TStrokaByOneByte mi(res);
            TStringOutput so(data1);
            TLzmaDecompress d(&mi);

            TransferData(&d, &so);
        }

        UNIT_ASSERT_EQUAL(data, data1);
    }
};

UNIT_TEST_SUITE_REGISTRATION(TLzmaTest);
