#include <library/cpp/getopt/modchooser.h>

#include <library/cpp/unittest/registar.h>

#include <util/stream/str.h>

int One(int, const char**) {
    return 1;
}

int Two(int, const char**) {
    return 2;
}

int Three(int, const char**) {
    return 3;
}

int Four(int, const char**) {
    return 4;
}

int Five(int, const char**) {
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

        for (size_t idx = 0; idx < Y_ARRAY_SIZE(NAMES); ++idx) {
            int argc = 2;
            const char* argv[] = {"UNITTEST", NAMES[idx]};
            UNIT_ASSERT_EQUAL(static_cast<int>(idx) + 1, chooser.Run(argc, argv));
        }
    }

    Y_UNIT_TEST(TestHelpMessage) {
        TModChooser chooser;

        int argc = 2;
        const char* argv[] = {"UNITTEST", "-?"};

        chooser.Run(argc, argv);
    }
}
