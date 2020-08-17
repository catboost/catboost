#include <library/cpp/getopt/modchooser.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/str.h>

void ValidateArgcArgv(int argc, const char** argv) {
    UNIT_ASSERT_EQUAL(argc, 1);
    UNIT_ASSERT_EQUAL(argv[argc], nullptr);
}

int One(int argc, const char** argv) {
    ValidateArgcArgv(argc, argv);
    return 1;
}

int Two(int argc, const char** argv) {
    ValidateArgcArgv(argc, argv);
    return 2;
}

int Three(int argc, const char** argv) {
    ValidateArgcArgv(argc, argv);
    return 3;
}

int Four(int argc, const char** argv) {
    ValidateArgcArgv(argc, argv);
    return 4;
}

int Five(int argc, const char** argv) {
    ValidateArgcArgv(argc, argv);
    return 5;
}

typedef int (*F_PTR)(int, const char**);
static const F_PTR FUNCTIONS[] = {One, Two, Three, Four, Five};
static const char* NAMES[] = {"one", "two", "three", "four", "five"};
static_assert(Y_ARRAY_SIZE(FUNCTIONS) == Y_ARRAY_SIZE(NAMES), "Incorrect input tests data");

Y_UNIT_TEST_SUITE(TModChooserTest) {
    Y_UNIT_TEST(TestModesSimpleRunner) {
        TModChooser chooser;
        for (size_t idx = 0; idx < Y_ARRAY_SIZE(NAMES); ++idx) {
            chooser.AddMode(NAMES[idx], FUNCTIONS[idx], NAMES[idx]);
        }

        // test argc, argv
        for (size_t idx = 0; idx < Y_ARRAY_SIZE(NAMES); ++idx) {
            int argc = 2;
            const char* argv[] = {"UNITTEST", NAMES[idx], nullptr};
            UNIT_ASSERT_EQUAL(static_cast<int>(idx) + 1, chooser.Run(argc, argv));
        }

        // test TVector<TString> argv
        for (size_t idx = 0; idx < Y_ARRAY_SIZE(NAMES); ++idx) {
            const TVector<TString> argv = {"UNITTEST", NAMES[idx]};
            UNIT_ASSERT_EQUAL(static_cast<int>(idx) + 1, chooser.Run(argv));
        }
    }

    Y_UNIT_TEST(TestHelpMessage) {
        TModChooser chooser;

        int argc = 2;
        const char* argv[] = {"UNITTEST", "-?", nullptr};

        chooser.Run(argc, argv);
    }
}
