#include "stream.h"
#include "compression.h"

#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/testing/unittest/tests_data.h>

#include <util/stream/zlib.h>
#include <util/generic/hash_set.h>

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

    Y_UNIT_TEST(TestChooseBestCompressionScheme) {
        THashSet<TString> accepted;

        auto checkAccepted = [&accepted](const TString& v) {
            return accepted.contains(v);
        };

        UNIT_ASSERT_VALUES_EQUAL("identity", NHttp::ChooseBestCompressionScheme(checkAccepted, {"gzip", "deflate"}));
        accepted.insert("deflate");
        UNIT_ASSERT_VALUES_EQUAL("deflate", NHttp::ChooseBestCompressionScheme(checkAccepted, {"gzip", "deflate"}));
        accepted.insert("*");
        UNIT_ASSERT_VALUES_EQUAL("gzip", NHttp::ChooseBestCompressionScheme(checkAccepted, {"gzip", "deflate"}));
    }
} // THttpCompressionTest suite
