#include "md5.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/system/fs.h>
#include <util/stream/file.h>

Y_UNIT_TEST_SUITE(TMD5Test) {
    Y_UNIT_TEST(TestMD5) {
        // echo -n 'qwertyuiopqwertyuiopasdfghjklasdfghjkl' | md5sum
        constexpr const char* b = "qwertyuiopqwertyuiopasdfghjklasdfghjkl";

        MD5 r;
        r.Update((const unsigned char*)b, 15);
        r.Update((const unsigned char*)b + 15, strlen(b) - 15);

        char rs[33];
        TString s(r.End(rs));
        s.to_lower();

        UNIT_ASSERT_NO_DIFF(s, TStringBuf("3ac00dd696b966fd74deee3c35a59d8f"));

        TString result = r.Calc(TStringBuf(b));
        result.to_lower();
        UNIT_ASSERT_NO_DIFF(result, TStringBuf("3ac00dd696b966fd74deee3c35a59d8f"));
    }

    Y_UNIT_TEST(TestFile) {
        TString s = NUnitTest::RandomString(1000000, 1);
        const TString tmpFile = "tmp";

        {
            TFixedBufferFileOutput fo(tmpFile);
            fo.Write(s.data(), s.size());
        }

        char fileBuf[100];
        char memBuf[100];
        TString fileHash = MD5::File(tmpFile.data(), fileBuf);
        TString memoryHash = MD5::Data((const unsigned char*)s.data(), s.size(), memBuf);

        UNIT_ASSERT_NO_DIFF(fileHash, memoryHash);

        fileHash = MD5::File(tmpFile);
        UNIT_ASSERT_NO_DIFF(fileHash, memoryHash);

        NFs::Remove(tmpFile);
        fileHash = MD5::File(tmpFile);
        UNIT_ASSERT_EQUAL(fileHash.size(), 0);
    }

    Y_UNIT_TEST(TestIsMD5) {
        UNIT_ASSERT_EQUAL(false, MD5::IsMD5(TStringBuf()));
        UNIT_ASSERT_EQUAL(false, MD5::IsMD5(TStringBuf("4136ebb0e4c45d21e2b09294c75cfa0")));   // length 31
        UNIT_ASSERT_EQUAL(false, MD5::IsMD5(TStringBuf("4136ebb0e4c45d21e2b09294c75cfa000"))); // length 33
        UNIT_ASSERT_EQUAL(false, MD5::IsMD5(TStringBuf("4136ebb0e4c45d21e2b09294c75cfa0g")));  // wrong character 'g'
        UNIT_ASSERT_EQUAL(true, MD5::IsMD5(TStringBuf("4136EBB0E4C45D21E2B09294C75CFA08")));
        UNIT_ASSERT_EQUAL(true, MD5::IsMD5(TStringBuf("4136ebb0E4C45D21e2b09294C75CfA08")));
        UNIT_ASSERT_EQUAL(true, MD5::IsMD5(TStringBuf("4136ebb0e4c45d21e2b09294c75cfa08")));
    }

    Y_UNIT_TEST(TestMd5HalfMix) {
        UNIT_ASSERT_EQUAL(MD5::CalcHalfMix(""), 7203772011789518145ul);
        UNIT_ASSERT_EQUAL(MD5::CalcHalfMix("qwertyuiopqwertyuiopasdfghjklasdfghjkl"), 11753545595885642730ul);
    }
}
