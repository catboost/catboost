#include <library/cpp/testing/unittest/registar.h>

#include <library/cpp/string_utils/base64/base64.h>

#include <util/generic/yexception.h>

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

Y_UNIT_TEST_SUITE(TBase64StrictDecodeUneven) {
    Y_UNIT_TEST(PaddedAndUnpadded) {
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven(""), "");
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("QQ=="), "A");
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("QQ="), "A");
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("QQ"), "A");
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("MTI="), "12");
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("MTI"), "12");
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("YWFh"), "aaa");
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("YWJjZA=="), "abcd");
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("YWJjZA"), "abcd");
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("YWJjZGU="), "abcde");
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("YWJjZGU"), "abcde");
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("YWFhYWFh"), "aaaaaa");
    }

    Y_UNIT_TEST(Base64Url) {
        const TString binary("\xfb\xfb", 2);
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("+/s="), binary);
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("+/s"), binary);
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("-_s="), binary);
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("-_s,"), binary);
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("-_s"), binary);
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("QQ,,"), "A");
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("QQ,"), "A");
    }

    Y_UNIT_TEST(PaddingInside) {
        // Preserve Base64StrictDecode's support for concatenated padded strings.
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("QQ==Qg=="), "AB");
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("QQ==Qg"), "AB");
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecodeUneven("QQ==Qg="), "AB");
    }

    Y_UNIT_TEST(InvalidLength) {
        for (const TStringBuf encoded : {"A", "YWFhA", "YWFhYWFhA", "QQ==="}) {
            UNIT_ASSERT_EXCEPTION(Base64StrictDecodeUneven(encoded), yexception);
        }
    }

    Y_UNIT_TEST(InvalidSymbols) {
        for (const TStringBuf encoded :
             {"!A", "AA!", "!AAAQQ", "!AAAQQQ", "YWFh!A",
              " Q", "QQ\n", "QQ\r", "Q\t", "\xffQ"}) {
            UNIT_ASSERT_EXCEPTION(Base64StrictDecodeUneven(encoded), yexception);
        }
        UNIT_ASSERT_EXCEPTION(Base64StrictDecodeUneven("Q\0"_sb), yexception);
    }

    Y_UNIT_TEST(InvalidPadding) {
        for (const TStringBuf encoded : {"=Q", "Q=", "=QQ", "Q=Q", "====", "=AAA", "A=AA", "AA=A",
                                         "AA,A", "YWFh=Q", "YWFhQ=Q"}) {
            UNIT_ASSERT_EXCEPTION(Base64StrictDecodeUneven(encoded), yexception);
        }
    }
}
