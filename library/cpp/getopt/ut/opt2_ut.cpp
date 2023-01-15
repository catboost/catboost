#include <library/cpp/getopt/opt2.h>

#include <library/cpp/testing/unittest/registar.h>

//using namespace NLastGetopt;

Y_UNIT_TEST_SUITE(Opt2Test) {
    Y_UNIT_TEST(TestSimple) {
        int argc = 8;
        char* argv[] = {
            (char*)"cmd",
            (char*)"--aaaa=aaaa",
            (char*)"zz",
            (char*)"-x1",
            (char*)"-x2",
            (char*)"-c",
            (char*)"-d8",
            (char*)"ww",
        };

        Opt2 opt(argc, argv, "A:b:cd:e:x:", 2, "aaaa=A");

        const char* edef = "edef";
        const char* a = opt.Arg('A', "<var_name> - usage of -A");
        int b = opt.Int('b', "<var_name> - usage of -b", 2);
        bool c = opt.Has('c', "usage of -c");
        int d = opt.Int('d', "<var_name> - usage of -d", 13);
        const char* e = opt.Arg('e', "<unused> - only default is really used", edef);
        const TVector<const char*>& x = opt.MArg('x', "<var_name> - usage of -x");

        UNIT_ASSERT(!opt.AutoUsage("<L> <M>"));
        UNIT_ASSERT_VALUES_EQUAL("aaaa", a);
        UNIT_ASSERT_VALUES_EQUAL(2, b);
        UNIT_ASSERT(c);
        UNIT_ASSERT_VALUES_EQUAL(8, d);
        UNIT_ASSERT_VALUES_EQUAL((void*)edef, e);

        UNIT_ASSERT_VALUES_EQUAL(2u, opt.Pos.size());
        UNIT_ASSERT_STRINGS_EQUAL("zz", opt.Pos.at(0));
        UNIT_ASSERT_VALUES_EQUAL((void*)argv[2], opt.Pos.at(0));
        UNIT_ASSERT_STRINGS_EQUAL("ww", opt.Pos.at(1));
        UNIT_ASSERT_STRINGS_EQUAL("1", x.at(0));
        UNIT_ASSERT_STRINGS_EQUAL("2", x.at(1));
    }

    Y_UNIT_TEST(TestErrors1) {
        int argc = 4;
        char* argv[] = {
            (char*)"cmd",
            (char*)"zz",
            (char*)"-c",
            (char*)"-e",
        };

        Opt2 opt(argc, argv, "ce:", 2);

        const char* edef = "edef";
        bool c = opt.Has('c', "usage of -c");
        const char* e = opt.Arg('e', "<unused> - only default is really used", edef);
        UNIT_ASSERT(c);
        UNIT_ASSERT_VALUES_EQUAL((void*)edef, e);
    }
}
