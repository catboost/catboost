#include <library/cpp/testing/unittest/registar.h>

#include <library/cpp/string_utils/base64/base64.h>

Y_UNIT_TEST_SUITE(TBase64DecodeUneven) {
    Y_UNIT_TEST(Base64DecodeUneven) {
        const TString wikipedia_slogan =
            "Man is distinguished, not only by his reason, "
            "but by this singular passion from other animals, which is a lust of the "
            "mind, that by a perseverance of delight in the continued and "
            "indefatigable generation of knowledge, exceeds the short "
            "vehemence of any carnal pleasure.";
        const TString encoded =
            "TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0"
            "aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1"
            "c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0"
            "aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdl"
            "LCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4=";

        UNIT_ASSERT_VALUES_EQUAL(encoded, Base64Encode(wikipedia_slogan));
        UNIT_ASSERT_VALUES_EQUAL(wikipedia_slogan, Base64DecodeUneven(encoded));

        const TString encoded_url1 =
            "TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0"
            "aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1"
            "c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0"
            "aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdl"
            "LCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4,";
        const TString encoded_url2 =
            "TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0"
            "aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1"
            "c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0"
            "aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdl"
            "LCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4";
        UNIT_ASSERT_VALUES_EQUAL(wikipedia_slogan, Base64DecodeUneven(encoded_url1));
        UNIT_ASSERT_VALUES_EQUAL(wikipedia_slogan, Base64DecodeUneven(encoded_url2));

        const TString lp = "Linkin Park";
        UNIT_ASSERT_VALUES_EQUAL(lp, Base64DecodeUneven(Base64Encode(lp)));
        UNIT_ASSERT_VALUES_EQUAL(lp, Base64DecodeUneven(Base64EncodeUrl(lp)));

        const TString dp = "ADP GmbH\nAnalyse Design & Programmierung\nGesellschaft mit beschränkter Haftung";
        UNIT_ASSERT_VALUES_EQUAL(dp, Base64DecodeUneven(Base64Encode(dp)));
        UNIT_ASSERT_VALUES_EQUAL(dp, Base64DecodeUneven(Base64EncodeUrl(dp)));
    }
}
