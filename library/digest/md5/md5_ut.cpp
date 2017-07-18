#include "md5.h"

#include <library/unittest/registar.h>

#include <util/system/fs.h>
#include <util/stream/file.h>

SIMPLE_UNIT_TEST_SUITE(TMD5Test) {
    SIMPLE_UNIT_TEST(TestOverflow) {
        if (sizeof(size_t) > sizeof(unsigned int)) {
            const size_t len = (size_t)(5) + (size_t)Max<unsigned int>();
            TArrayHolder<char> buf = new char[len];

            memset(buf.Get(), 0, len);

            UNIT_ASSERT_VALUES_EQUAL(MD5::Calc(TStringBuf(buf.Get(), len)), "6e92cd744269e7ce7f65df408e2ed4d3");
        }
    }

    SIMPLE_UNIT_TEST(TestMD5) {
        // echo -n 'qwertyuiopqwertyuiopasdfghjklasdfghjkl' | md5sum
        char b[] = "qwertyuiopqwertyuiopasdfghjklasdfghjkl";

        MD5 r;
        r.Update((const unsigned char*)b, 15);
        r.Update((const unsigned char*)b + 15, strlen(b) - 15);

        char rs[33];
        TString s(r.End(rs));
        s.to_lower();

        UNIT_ASSERT_EQUAL(s, STRINGBUF("3ac00dd696b966fd74deee3c35a59d8f"));

        TString result = r.Calc(STRINGBUF(b));
        result.to_lower();
        UNIT_ASSERT_EQUAL(result, STRINGBUF("3ac00dd696b966fd74deee3c35a59d8f"));
    }

    SIMPLE_UNIT_TEST(TestFile) {
        TString s = NUnitTest::RandomString(1000000, 1);
        const TString tmpFile = "tmp";

        {
            TBufferedFileOutput fo(tmpFile);
            fo.Write(~s, +s);
        }

        char fileBuf[100];
        char memBuf[100];
        TString fileHash = MD5::File(~tmpFile, fileBuf);
        TString memoryHash = MD5::Data((const unsigned char*)~s, +s, memBuf);

        UNIT_ASSERT_EQUAL(fileHash, memoryHash);

        fileHash = MD5::File(tmpFile);
        UNIT_ASSERT_EQUAL(fileHash, memoryHash);

        NFs::Remove(tmpFile);
        fileHash = MD5::File(tmpFile);
        UNIT_ASSERT_EQUAL(fileHash.size(), 0);
    }

    SIMPLE_UNIT_TEST(TestIsMD5) {
        UNIT_ASSERT_EQUAL(false, MD5::IsMD5(TStringBuf()));
        UNIT_ASSERT_EQUAL(false, MD5::IsMD5(STRINGBUF("4136ebb0e4c45d21e2b09294c75cfa0")));   // length 31
        UNIT_ASSERT_EQUAL(false, MD5::IsMD5(STRINGBUF("4136ebb0e4c45d21e2b09294c75cfa000"))); // length 33
        UNIT_ASSERT_EQUAL(false, MD5::IsMD5(STRINGBUF("4136ebb0e4c45d21e2b09294c75cfa0g")));  // wrong character 'g'
        UNIT_ASSERT_EQUAL(true, MD5::IsMD5(STRINGBUF("4136EBB0E4C45D21E2B09294C75CFA08")));
        UNIT_ASSERT_EQUAL(true, MD5::IsMD5(STRINGBUF("4136ebb0E4C45D21e2b09294C75CfA08")));
        UNIT_ASSERT_EQUAL(true, MD5::IsMD5(STRINGBUF("4136ebb0e4c45d21e2b09294c75cfa08")));
    }
}
