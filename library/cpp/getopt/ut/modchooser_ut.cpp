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

class TRecordingAction: public TMainClassArgs {
public:
    int DoRun(NLastGetopt::TOptsParseResult&& /*res*/) override {
        CapturedSubcommandPath = GetSubcommandPath();
        return 0;
    }

    void RegisterOptions(NLastGetopt::TOpts& opts) override {
        opts.SetFreeArgsMax(2);
        opts.AddLongOption("options-flag")
            .Optional()
            .NoArgument()
            .StoreTrue(&OptionsFlag);
    }

public:
    bool OptionsFlag = false;
    TVector<TString> CapturedSubcommandPath;
};

class TOuterModes: public TMainClassModes {
public:
    explicit TOuterModes(TMainClass* inner)
        : Inner_(inner)
    {
    }

protected:
    void RegisterModes(TModChooser& modes) override {
        modes.AddMode("inner", Inner_, "inner");
    }

private:
    TMainClass* Inner_;
};

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

    Y_UNIT_TEST(TestSubcommandPathPropagation) {
        TRecordingAction innerAction;
        TOuterModes outer(&innerAction);
        TModChooser chooser;
        chooser.AddMode("outer", &outer, "outer");

        const char* argv[] = {"UNITTEST", "outer", "inner", "--options-flag", "free-arg1", "free-arg2", nullptr};
        UNIT_ASSERT_NO_EXCEPTION(chooser.Run(6, argv));

        const TVector<TString> expected = {"outer", "inner"};
        UNIT_ASSERT_EQUAL(innerAction.OptionsFlag, true);
        UNIT_ASSERT_VALUES_EQUAL(expected, innerAction.CapturedSubcommandPath);
    }
}
