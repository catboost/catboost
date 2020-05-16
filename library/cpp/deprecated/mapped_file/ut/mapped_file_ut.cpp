#include <library/cpp/deprecated/mapped_file/mapped_file.h>
#include <library/cpp/unittest/registar.h>

#include <util/system/fs.h>

Y_UNIT_TEST_SUITE(TMappedFileTest) {
    static const char* FileName_("./mappped_file");
    Y_UNIT_TEST(TestFileMapEmpty) {
        TFile file(FileName_, CreateAlways | WrOnly);
        file.Close();

        TMappedFile map;
        map.init(FileName_);
        map.getData(0);

        NFs::Remove(FileName_);
    }
};
