#include "stream.h"
#include "compression.h"

#include <library/unittest/registar.h>
#include <library/unittest/tests_data.h>

#include <util/stream/zlib.h>

Y_UNIT_TEST_SUITE(THttpCompressionTest) {
    static const TString DATA = "I'm a teapot";

    Y_UNIT_TEST(TestGetBestCodecs) {
        UNIT_ASSERT(TCompressionCodecFactory::Instance().GetBestCodecs().size() > 0);
    }

    Y_UNIT_TEST(TestEncoder) {
        TStringStream buffer;

        {
            auto encoder = TCompressionCodecFactory::Instance().FindEncoder("gzip");
            UNIT_ASSERT(encoder);

            auto encodedStream = (*encoder)(&buffer);
            encodedStream->Write(DATA);
        }

        TZLibDecompress decompressor(&buffer);
        UNIT_ASSERT_EQUAL(decompressor.ReadAll(), DATA);
    }

    Y_UNIT_TEST(TestDecoder) {
        TStringStream buffer;

        {
            TZLibCompress compressor(TZLibCompress::TParams(&buffer).SetType(ZLib::GZip));
            compressor.Write(DATA);
        }

        auto decoder = TCompressionCodecFactory::Instance().FindDecoder("gzip");
        UNIT_ASSERT(decoder);

        auto decodedStream = (*decoder)(&buffer);
        UNIT_ASSERT_EQUAL(decodedStream->ReadAll(), DATA);
    }
} // THttpCompressionTest suite
