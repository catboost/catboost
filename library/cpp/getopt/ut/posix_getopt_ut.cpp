#include <library/cpp/getopt/posix_getopt.h>

#include <library/cpp/testing/unittest/registar.h>

using namespace NLastGetopt;

Y_UNIT_TEST_SUITE(TPosixGetoptTest) {
    Y_UNIT_TEST(TestSimple) {
        int argc = 6;
        const char* argv0[] = {"program", "-b", "-f1", "-f", "2", "zzzz"};
        char** const argv = (char**)argv0;

        NLastGetopt::optreset = 1;
        UNIT_ASSERT_VALUES_EQUAL('b', NLastGetopt::getopt(argc, argv, "bf:"));
        UNIT_ASSERT_VALUES_EQUAL('f', NLastGetopt::getopt(argc, argv, "bf:"));
        UNIT_ASSERT_VALUES_EQUAL(NLastGetopt::optarg, TString("1"));
        UNIT_ASSERT_VALUES_EQUAL('f', NLastGetopt::getopt(argc, argv, "bf:"));
        UNIT_ASSERT_VALUES_EQUAL(NLastGetopt::optarg, TString("2"));
        UNIT_ASSERT_VALUES_EQUAL(-1, NLastGetopt::getopt(argc, argv, "bf:"));

        UNIT_ASSERT_VALUES_EQUAL(5, NLastGetopt::optind);
    }

    Y_UNIT_TEST(TestLong) {
        int daggerset = 0;
        /* options descriptor */
        const NLastGetopt::option longopts[] = {
            {"buffy", no_argument, nullptr, 'b'},
            {"fluoride", required_argument, nullptr, 'f'},
            {"daggerset", no_argument, &daggerset, 1},
            {nullptr, 0, nullptr, 0}};

        int argc = 7;
        const char* argv0[] = {"program", "-b", "--buffy", "-f1", "--fluoride=2", "--daggerset", "zzzz"};
        char** const argv = (char**)argv0;

        int longIndex;

        NLastGetopt::optreset = 1;
        UNIT_ASSERT_VALUES_EQUAL('b', NLastGetopt::getopt_long(argc, argv, "bf:", longopts, &longIndex));
        UNIT_ASSERT_VALUES_EQUAL(0, longIndex);
        UNIT_ASSERT_VALUES_EQUAL('b', NLastGetopt::getopt_long(argc, argv, "bf:", longopts, nullptr));
        UNIT_ASSERT_VALUES_EQUAL('f', NLastGetopt::getopt_long(argc, argv, "bf:", longopts, &longIndex));
        UNIT_ASSERT_VALUES_EQUAL(1, longIndex);
        UNIT_ASSERT_VALUES_EQUAL('f', NLastGetopt::getopt_long(argc, argv, "bf:", longopts, nullptr));
        UNIT_ASSERT_VALUES_EQUAL(0, NLastGetopt::getopt_long(argc, argv, "bf:", longopts, nullptr));
        UNIT_ASSERT_VALUES_EQUAL(-1, NLastGetopt::getopt_long(argc, argv, "bf:", longopts, nullptr));

        UNIT_ASSERT_VALUES_EQUAL(6, NLastGetopt::optind);
    }

    Y_UNIT_TEST(TestLongPermutation) {
        int daggerset = 0;
        /* options descriptor */
        const NLastGetopt::option longopts[] = {
            {"buffy", no_argument, nullptr, 'b'},
            {"fluoride", required_argument, nullptr, 'f'},
            {"daggerset", no_argument, &daggerset, 1},
            {nullptr, 0, nullptr, 0}};

        int argc = 7;
        const char* argv0[] = {"program", "aa", "-b", "bb", "cc", "--buffy", "dd"};
        char** const argv = (char**)argv0;

        NLastGetopt::optreset = 1;
        UNIT_ASSERT_VALUES_EQUAL('b', NLastGetopt::getopt_long(argc, argv, "bf:", longopts, nullptr));
        UNIT_ASSERT_VALUES_EQUAL('b', NLastGetopt::getopt_long(argc, argv, "bf:", longopts, nullptr));
        UNIT_ASSERT_VALUES_EQUAL(-1, NLastGetopt::getopt_long(argc, argv, "bf:", longopts, nullptr));

        UNIT_ASSERT_VALUES_EQUAL(3, NLastGetopt::optind);
    }

    Y_UNIT_TEST(TestNoOptionsOptionsWithDoubleDash) {
        const NLastGetopt::option longopts[] = {
            {"buffy", no_argument, nullptr, 'b'},
            {"fluoride", no_argument, nullptr, 'f'},
            {nullptr, 0, nullptr, 0}};

        int argc = 2;
        const char* argv0[] = {"program", "--bf"};
        char** const argv = (char**)argv0;

        NLastGetopt::optreset = 1;
        UNIT_ASSERT_VALUES_EQUAL('?', NLastGetopt::getopt_long(argc, argv, "bf", longopts, nullptr));
    }

    Y_UNIT_TEST(TestLongOnly) {
        const NLastGetopt::option longopts[] = {
            {"foo", no_argument, nullptr, 'F'},
            {"fluoride", no_argument, nullptr, 'f'},
            {"ogogo", no_argument, nullptr, 'o'},
            {nullptr, 0, nullptr, 0}};

        int argc = 4;
        const char* argv0[] = {"program", "--foo", "-foo", "-fo"};
        char** const argv = (char**)argv0;

        NLastGetopt::optreset = 1;
        UNIT_ASSERT_VALUES_EQUAL('F', NLastGetopt::getopt_long_only(argc, argv, "fo", longopts, nullptr));
        UNIT_ASSERT_VALUES_EQUAL('F', NLastGetopt::getopt_long_only(argc, argv, "fo", longopts, nullptr));
        UNIT_ASSERT_VALUES_EQUAL('f', NLastGetopt::getopt_long_only(argc, argv, "fo", longopts, nullptr));
        UNIT_ASSERT_VALUES_EQUAL('o', NLastGetopt::getopt_long_only(argc, argv, "fo", longopts, nullptr));
        UNIT_ASSERT_VALUES_EQUAL(-1, NLastGetopt::getopt_long_only(argc, argv, "fo", longopts, nullptr));
    }

    Y_UNIT_TEST(TestLongWithoutOnlySingleDashNowAllowed) {
        const NLastGetopt::option longopts[] = {
            {"foo", no_argument, nullptr, 'F'},
            {"zoo", no_argument, nullptr, 'z'},
            {nullptr, 0, nullptr, 0}};

        int argc = 2;
        const char* argv0[] = {"program", "-foo"};
        char** const argv = (char**)argv0;

        NLastGetopt::optreset = 1;
        UNIT_ASSERT_VALUES_EQUAL('?', NLastGetopt::getopt_long(argc, argv, "z", longopts, nullptr));
    }
}
