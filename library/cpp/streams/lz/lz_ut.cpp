#include "lz.h"

#include <library/cpp/unittest/registar.h>
#include <library/cpp/resource/resource.h>

#include <util/stream/file.h>
#include <util/generic/vector.h>
#include <util/system/tempfile.h>
#include <util/generic/singleton.h>

#define LDATA "./ldata"
#define LDATA_RANDOM "./ldata.random"

static const TString data = "aa aaa aa aaa aa aaa bb bbb bb bbb bb bbb";

namespace {
    /**
     * Produces well-formed random crap
     **/
    TString RandomString(size_t size) {
        TString entropy(NResource::Find("/random.data"));
        TString result;
        size_t seed = 1;
        size_t j = 0;
        for (size_t i = 0; i < size; ++i) {
            seed *= 3;
            char sym;
            do {
                sym = char((seed ^ i) % 256);
                if (!sym) {
                    seed += 1;
                }
            } while (!sym);
            Y_ASSERT(sym);
            j = (j + 1) % entropy.size();
            result += char(sym + entropy[j]);
        }
        return result;
    }

    TVector<TString> InitRandomData() {
        static const TVector<size_t> sizes = {
            0,
            1,
            127,
            2017,
            32767,
        };

        TVector<TString> result;
        for (auto size : sizes) {
            result.push_back(RandomString(size));
        }
        result.push_back(NResource::Find("/request.data"));
        return result;
    }

    TString TestFileName(const TString& d, size_t bufferSize) {
        return LDATA_RANDOM + TString(".") + ToString(d.size()) + TString(".") + ToString(bufferSize);
    }

    struct TRandomData: public TVector<TString> {
        inline TRandomData() {
            InitRandomData().swap(*this);
        }
    };
}

static const TVector<size_t> bufferSizes = {
    127,
    1024,
    32768,
};

namespace {
    template <TLzqCompress::EVersion Ver, int Level, TLzqCompress::EMode Mode>
    struct TLzqCompressX: public TLzqCompress {
        inline TLzqCompressX(IOutputStream* out, size_t bufLen)
            : TLzqCompress(out, bufLen, Ver, Level, Mode)
        {
        }
    };
}

template <class C>
static inline void TestGoodDataCompress() {
    TFixedBufferFileOutput o(LDATA);
    C c(&o, 1024);

    TString d = data;

    for (size_t i = 0; i < 10; ++i) {
        c.Write(d.data(), d.size());
        c << Endl;
        d = d + d;
    }

    c.Finish();
    o.Finish();
}

template <class C>
static inline void TestIncompressibleDataCompress(const TString& d, size_t bufferSize) {
    TString testFileName = TestFileName(d, bufferSize);
    TFixedBufferFileOutput o(testFileName);
    C c(&o, bufferSize);
    c.Write(d.data(), d.size());
    c.Finish();
    o.Finish();
}

template <class C>
static inline void TestCompress() {
    TestGoodDataCompress<C>();
    for (auto bufferSize : bufferSizes) {
        for (auto rd : *Singleton<TRandomData>()) {
            TestIncompressibleDataCompress<C>(rd, bufferSize);
        }
    }
}

template <class D>
static inline void TestGoodDataDecompress() {
    TTempFile tmpFile(LDATA);

    {
        TFileInput i1(LDATA);
        D ld(&i1);

        TString d = data;

        for (size_t i2 = 0; i2 < 10; ++i2) {
            UNIT_ASSERT_EQUAL(ld.ReadLine(), d);

            d = d + d;
        }
    }
}

template <class D>
static inline void TestIncompressibleDataDecompress(const TString& d, size_t bufferSize) {
    TString testFileName = TestFileName(d, bufferSize);
    TTempFile tmpFile(testFileName);

    {
        TFileInput i(testFileName);
        D ld(&i);

        UNIT_ASSERT_EQUAL(ld.ReadAll(), d);
    }
}

template <class D>
static inline void TestDecompress() {
    TestGoodDataDecompress<D>();
    for (auto bufferSize : bufferSizes) {
        for (auto rd : *Singleton<TRandomData>()) {
            TestIncompressibleDataDecompress<D>(rd, bufferSize);
        }
    }
}

class TMixedDecompress: public IInputStream {
public:
    TMixedDecompress(IInputStream* input)
        : Slave_(OpenLzDecompressor(input).Release())
    {
    }

private:
    size_t DoRead(void* buf, size_t len) override {
        return Slave_->Read(buf, len);
    }

private:
    THolder<IInputStream> Slave_;
};

template <class C>
static inline void TestMixedDecompress() {
    TestCompress<C>();
    TestDecompress<TMixedDecompress>();
}

template <class D, class C>
static inline void TestDecompressError() {
    TestCompress<C>();
    UNIT_ASSERT_EXCEPTION(TestDecompress<D>(), TDecompressorError);
}

Y_UNIT_TEST_SUITE(TLzTest) {
    Y_UNIT_TEST(TestLzo) {
        TestCompress<TLzoCompress>();
        TestDecompress<TLzoDecompress>();
    }

    Y_UNIT_TEST(TestLzf) {
        TestCompress<TLzfCompress>();
        TestDecompress<TLzfDecompress>();
    }

    Y_UNIT_TEST(TestLzq) {
        TestCompress<TLzqCompress>();
        TestDecompress<TLzqDecompress>();
    }

    Y_UNIT_TEST(TestLzq151_1) {
        TestCompress<TLzqCompressX<TLzqCompress::V_1_51, 1, TLzqCompress::M_0>>();
        TestDecompress<TLzqDecompress>();
    }

    Y_UNIT_TEST(TestLzq151_2) {
        TestCompress<TLzqCompressX<TLzqCompress::V_1_51, 2, TLzqCompress::M_100000>>();
        TestDecompress<TLzqDecompress>();
    }

    Y_UNIT_TEST(TestLzq151_3) {
        TestCompress<TLzqCompressX<TLzqCompress::V_1_51, 3, TLzqCompress::M_1000000>>();
        TestDecompress<TLzqDecompress>();
    }

    Y_UNIT_TEST(TestLzq140_1) {
        TestCompress<TLzqCompressX<TLzqCompress::V_1_40, 1, TLzqCompress::M_0>>();
        TestDecompress<TLzqDecompress>();
    }

    Y_UNIT_TEST(TestLzq140_2) {
        TestCompress<TLzqCompressX<TLzqCompress::V_1_40, 2, TLzqCompress::M_100000>>();
        TestDecompress<TLzqDecompress>();
    }

    Y_UNIT_TEST(TestLzq140_3) {
        TestCompress<TLzqCompressX<TLzqCompress::V_1_40, 3, TLzqCompress::M_1000000>>();
        TestDecompress<TLzqDecompress>();
    }

    Y_UNIT_TEST(TestLz4) {
        TestCompress<TLz4Compress>();
        TestDecompress<TLz4Decompress>();
    }

    Y_UNIT_TEST(TestSnappy) {
        TestCompress<TSnappyCompress>();
        TestDecompress<TSnappyDecompress>();
    }

    Y_UNIT_TEST(TestGeneric) {
        TestMixedDecompress<TLzoCompress>();
        TestMixedDecompress<TLzfCompress>();
        TestMixedDecompress<TLzqCompress>();
        TestMixedDecompress<TLz4Compress>();
        TestMixedDecompress<TSnappyCompress>();
    }

    Y_UNIT_TEST(TestDecompressorError) {
        TestDecompressError<TLzoDecompress, TLzfCompress>();
        TestDecompressError<TLzfDecompress, TLzqCompress>();
        TestDecompressError<TLzqDecompress, TLz4Compress>();
        TestDecompressError<TLz4Decompress, TSnappyCompress>();
        TestDecompressError<TSnappyDecompress, TBufferedOutput>();
        TestDecompressError<TMixedDecompress, TBufferedOutput>();
    }

    Y_UNIT_TEST(TestFactory) {
        TStringStream ss;

        {
            TLz4Compress c(&ss);

            c.Write("123456789", 9);
            c.Finish();
        }

        TAutoPtr<IInputStream> is(OpenOwnedLzDecompressor(new TStringInput(ss.Str())));

        UNIT_ASSERT_EQUAL(is->ReadAll(), "123456789");
    }
}
