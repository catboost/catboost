#include "chunk.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/file.h>
#include <util/system/tempfile.h>
#include <util/stream/null.h>

#define CDATA "./chunkedio"

Y_UNIT_TEST_SUITE(TestChunkedIO) {
    static const char test_data[] = "87s6cfbsudg cuisg s igasidftasiy tfrcua6s";

    TString CombString(const TString& s, size_t chunkSize) {
        TString result;
        for (size_t pos = 0; pos < s.size(); pos += 2 * chunkSize)
            result += s.substr(pos, chunkSize);
        return result;
    }

    void WriteTestData(IOutputStream * stream, TString * contents) {
        contents->clear();
        for (size_t i = 0; i < sizeof(test_data); ++i) {
            stream->Write(test_data, i);
            contents->append(test_data, i);
        }
    }

    void ReadInSmallChunks(IInputStream * stream, TString * contents) {
        char buf[11];
        size_t read = 0;

        contents->clear();
        do {
            read = stream->Read(buf, sizeof(buf));
            contents->append(buf, read);
        } while (read > 0);
    }

    void ReadCombed(IInputStream * stream, TString * contents, size_t chunkSize) {
        Y_ASSERT(chunkSize < 128);
        char buf[128];

        contents->clear();
        while (true) {
            size_t read = stream->Load(buf, chunkSize);
            contents->append(buf, read);
            if (read == 0)
                break;

            size_t toSkip = chunkSize;
            size_t skipped = 0;
            do {
                skipped = stream->Skip(toSkip);
                toSkip -= skipped;
            } while (skipped != 0 && toSkip != 0);
        }
    }

    Y_UNIT_TEST(TestChunkedIo) {
        TTempFile tmpFile(CDATA);
        TString tmp;

        {
            TUnbufferedFileOutput fo(CDATA);
            TChunkedOutput co(&fo);
            WriteTestData(&co, &tmp);
        }

        {
            TUnbufferedFileInput fi(CDATA);
            TChunkedInput ci(&fi);
            TString r;

            ReadInSmallChunks(&ci, &r);

            UNIT_ASSERT_EQUAL(r, tmp);
        }

        {
            TUnbufferedFileInput fi(CDATA);
            TChunkedInput ci(&fi);
            TString r;

            ReadCombed(&ci, &r, 11);

            UNIT_ASSERT_EQUAL(r, CombString(tmp, 11));
        }
    }

    Y_UNIT_TEST(TestBadChunk) {
        bool hasError = false;

        try {
            TString badChunk = "10\r\nqwerty";
            TMemoryInput mi(badChunk.data(), badChunk.size());
            TChunkedInput ci(&mi);
            TransferData(&ci, &Cnull);
        } catch (...) {
            hasError = true;
        }

        UNIT_ASSERT(hasError);
    }
}
