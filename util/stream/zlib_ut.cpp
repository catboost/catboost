#include "zlib.h"

#include <library/unittest/registar.h>

#include "file.h"
#include <util/system/tempfile.h>
#include <util/random/entropy.h>
#include <util/random/random.h>

#define ZDATA "./zdata"

class TThrowingStream: public IOutputStream {
public:
    TThrowingStream(int limit)
        : Limit_(limit)
    {
    }

    void DoWrite(const void*, size_t size) override {
        if (Ignore) {
            return;
        }

        Limit_ -= size;
        if (Limit_ < 0) {
            throw yexception() << "catch this";
        }
    }

    void DoFinish() override {
        if (Ignore) {
            return;
        }
        if (Limit_ < 0) {
            throw yexception() << "catch this";
        }
    }

    void DoFlush() override {
        if (Ignore) {
            return;
        }
        if (Limit_ < 0) {
            throw yexception() << "catch this";
        }
    }

    bool Ignore = false;

private:
    int Limit_;
};

Y_UNIT_TEST_SUITE(TZLibTest) {
    static const TString DATA = "8s7d5vc6s5vc67sa4c65ascx6asd4xcv76adsfxv76s";
    static const TString DATA2 = "cn8wk2bd9vb3vdfif83g1ks94bfiovtwv";

    Y_UNIT_TEST(Compress) {
        TUnbufferedFileOutput o(ZDATA);
        TZLibCompress c(&o, ZLib::ZLib);

        c.Write(DATA.data(), DATA.size());
        c.Finish();
        o.Finish();
    }

    Y_UNIT_TEST(Decompress) {
        TTempFile tmpFile(ZDATA);

        {
            TUnbufferedFileInput i(ZDATA);
            TZLibDecompress d(&i);

            UNIT_ASSERT_EQUAL(d.ReadLine(), DATA);
        }
    }

    Y_UNIT_TEST(Dictionary) {
        static constexpr auto data = AsStringBuf("<html><body></body></html>");
        static constexpr auto dict = AsStringBuf("</<html><body>");
        for (auto type : {ZLib::Raw, ZLib::ZLib}) {
            TStringStream compressed;
            {
                TZLibCompress compressor(TZLibCompress::TParams(&compressed).SetDict(dict).SetType(type));
                compressor.Write(data);
            }

            TZLibDecompress decompressor(&compressed, type, ZLib::ZLIB_BUF_LEN, dict);
            UNIT_ASSERT_STRINGS_EQUAL(decompressor.ReadAll(), data);
        }
    }

    Y_UNIT_TEST(DecompressTwoStreams) {
        // Check that Decompress(Compress(X) + Compress(Y)) == X + Y
        TTempFile tmpFile(ZDATA);
        {
            TUnbufferedFileOutput o(ZDATA);
            TZLibCompress c1(&o, ZLib::ZLib);
            c1.Write(DATA.data(), DATA.size());
            c1.Finish();
            TZLibCompress c2(&o, ZLib::ZLib);
            c2.Write(DATA2.data(), DATA2.size());
            c2.Finish();
            o.Finish();
        }
        {
            TUnbufferedFileInput i(ZDATA);
            TZLibDecompress d(&i);

            UNIT_ASSERT_EQUAL(d.ReadLine(), DATA + DATA2);
        }
    }

    Y_UNIT_TEST(CompressionExceptionSegfault) {
        TVector<char> buf(512 * 1024);
        EntropyPool().Load(buf.data(), buf.size());

        TThrowingStream o(128 * 1024);
        TZLibCompress c(&o, ZLib::GZip, 4, 1 << 15);
        try {
            c.Write(buf.data(), buf.size());
        } catch (...) {
        }

        o.Ignore = true;
        TVector<char>().swap(buf);
    }

    Y_UNIT_TEST(DecompressFirstOfTwoStreams) {
        // Check that Decompress(Compress(X) + Compress(Y)) == X when single stream is allowed
        TTempFile tmpFile(ZDATA);
        {
            TUnbufferedFileOutput o(ZDATA);
            TZLibCompress c1(&o, ZLib::ZLib);
            c1.Write(DATA.data(), DATA.size());
            c1.Finish();
            TZLibCompress c2(&o, ZLib::ZLib);
            c2.Write(DATA2.data(), DATA2.size());
            c2.Finish();
            o.Finish();
        }
        {
            TUnbufferedFileInput i(ZDATA);
            TZLibDecompress d(&i);
            d.SetAllowMultipleStreams(false);

            UNIT_ASSERT_EQUAL(d.ReadLine(), DATA);
        }
    }

    Y_UNIT_TEST(CompressFlush) {
        TString data = "";

        for (size_t i = 0; i < 32; ++i) {
            TTempFile tmpFile(ZDATA);

            TUnbufferedFileOutput output(ZDATA);
            TZLibCompress compressor(&output, ZLib::ZLib);

            compressor.Write(data.data(), data.size());
            compressor.Flush();

            {
                TUnbufferedFileInput input(ZDATA);
                TZLibDecompress decompressor(&input);

                TString s;
                UNIT_ASSERT_EQUAL(decompressor.ReadLine(s), i);
                UNIT_ASSERT_EQUAL(s, data);
            }

            data += 'A' + i;
        }
    }

    Y_UNIT_TEST(CompressEmptyFlush) {
        TTempFile tmpFile(ZDATA);

        TUnbufferedFileOutput output(ZDATA);
        TZLibCompress compressor(&output, ZLib::ZLib);

        compressor.Write(DATA.data(), DATA.size());

        for (size_t i = 0; i < 32; ++i) {
            compressor.Flush();
        }

        {
            TUnbufferedFileInput input(ZDATA);
            TZLibDecompress decompressor(&input);

            UNIT_ASSERT_EQUAL(decompressor.ReadLine(), DATA);
        }
    }

    Y_UNIT_TEST(CompressFlushSmallBuffer) {
        for (size_t bufferSize = 16; bufferSize < 32; ++bufferSize) {
            TString data1 = "";

            for (size_t data1Size = 0; data1Size < 16; ++data1Size) {
                TString data2 = "";

                for (size_t data2Size = 0; data2Size < 16; ++data2Size) {
                    TTempFile tmpFile(ZDATA);

                    TUnbufferedFileOutput output(ZDATA);
                    TZLibCompress compressor(TZLibCompress::TParams(&output).SetType(ZLib::ZLib).SetBufLen(bufferSize));

                    TUnbufferedFileInput input(ZDATA);
                    TZLibDecompress decompressor(&input);

                    compressor.Write(data1.data(), data1.size());
                    compressor.Flush();

                    {
                        TString s;
                        UNIT_ASSERT_EQUAL(decompressor.ReadLine(s), data1Size);
                        UNIT_ASSERT_EQUAL(s, data1);
                    }

                    compressor.Write(data2.data(), data2.size());
                    compressor.Flush();

                    {
                        TString s;
                        UNIT_ASSERT_EQUAL(decompressor.ReadLine(s), data2Size);
                        UNIT_ASSERT_EQUAL(s, data2);
                    }

                    data2 += 'A' + data2Size;
                }

                data1 += 'A' + data1Size;
            }
        }
    }
}
