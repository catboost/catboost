#include <library/cpp/getopt/opt.h>

#include <library/cpp/testing/unittest/registar.h>
#include <util/string/vector.h>

Y_UNIT_TEST_SUITE(OptTest) {
    Y_UNIT_TEST(TestSimple) {
        int argc = 3;
        char* argv[] = {
            (char*)"cmd", (char*)"-x"};
        Opt opt(argc, argv, "");
        opt.Err = false; // be quiet
        UNIT_ASSERT_VALUES_EQUAL('?', opt.Get());
        UNIT_ASSERT_VALUES_EQUAL(EOF, opt.Get());
        UNIT_ASSERT_VALUES_EQUAL(EOF, opt.Get());
        UNIT_ASSERT_VALUES_EQUAL(EOF, opt.Get());
    }

    Y_UNIT_TEST(TestFreeArguments) {
        Opt::Ion options[] = {
            {"some-option", Opt::WithArg, nullptr, 123},
            {nullptr, Opt::WithoutArg, nullptr, 0}};
        const char* argv[] = {"cmd", "ARG1", "-some-option", "ARG2", "ARG3", nullptr};
        int argc = 5;
        Opt opts(argc, argv, "", options);

        UNIT_ASSERT_VALUES_EQUAL(JoinStrings(opts.GetFreeArgs(), ", "), "ARG1, ARG3");
    }

    Y_UNIT_TEST(TestLongOption) {
        const int SOME_OPTION_ID = 12345678;
        Opt::Ion options[] = {
            {"some-option", Opt::WithArg, nullptr, SOME_OPTION_ID},
            {nullptr, Opt::WithoutArg, nullptr, 0}};
        for (int doubleDash = 0; doubleDash <= 1; ++doubleDash) {
            const char* argv[] = {"cmd", "ARG1", (doubleDash ? "--some-option" : "-some-option"), "ARG2", "ARG3", nullptr};
            int argc = 5;
            Opt opts(argc, argv, "", options);

            TString optionValue = "";
            int optlet = 0;
            while ((optlet = opts.Get()) != EOF) {
                if (optlet == SOME_OPTION_ID) {
                    optionValue = opts.GetArg();
                } else {
                    UNIT_FAIL("don't expected any options, except -some-option");
                }
            }
            UNIT_ASSERT_VALUES_EQUAL(optionValue, "ARG2");
        }
    }
}
