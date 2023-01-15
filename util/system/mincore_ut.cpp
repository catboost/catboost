#include <library/cpp/unittest/registar.h>

#ifdef _unix_
#include <sys/resource.h>
#endif

#include "filemap.h"
#include "info.h"
#include "mincore.h"
#include "mlock.h"
#include "tempfile.h"

#include <util/generic/size_literals.h>
#include <util/system/fs.h>

#include <cstring>
#include <cstdio>

Y_UNIT_TEST_SUITE(MincoreSuite) {
    static const char* FileName_("./mappped_file");

    Y_UNIT_TEST(TestLockAndInCore) {
        TVector<char> content(2_MB);

        TTempFile cleanup(FileName_);
        TFile file(FileName_, CreateAlways | WrOnly);
        file.Write(content.data(), content.size());
        file.Close();

        TFileMap mappedFile(FileName_, TMemoryMapCommon::oRdWr);
        mappedFile.Map(0, mappedFile.Length());
        UNIT_ASSERT_EQUAL(mappedFile.MappedSize(), content.size());
        UNIT_ASSERT_EQUAL(mappedFile.Length(), static_cast<i64>(content.size()));

        LockMemory(mappedFile.Ptr(), mappedFile.Length());

        TVector<unsigned char> incore(mappedFile.Length() / NSystemInfo::GetPageSize());
        InCoreMemory(mappedFile.Ptr(), mappedFile.Length(), incore.data(), incore.size());

        // compile and run on all platforms, but assume non-zero results only on Linux
#if defined(_linux_)
        for (const auto& flag : incore) {
            UNIT_ASSERT(IsPageInCore(flag));
        }
#endif
    }
}
