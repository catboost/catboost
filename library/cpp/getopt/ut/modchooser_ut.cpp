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

class TTestAction: public TMainClass {
public:
    int operator()(int argc, const char** argv) override {
        Args.assign(argv, argv + argc);
        CapturedSubcommandPath = GetSubcommandPath();
        return ReturnCode;
    }

    int ReturnCode = 42;
    TVector<TString> Args;
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

    Y_UNIT_TEST(TestDefaultActionWithNoOtherMode) {
        // An invocation without a mode runs the default action and preserves the current subcommand path.
        TTestAction defaultAction;
        TModChooser chooser;
        chooser.SetDefaultAction(&defaultAction);
        chooser.SetSubcommandPath({"outer"});

        const char* argv[] = {"UNITTEST", nullptr};
        UNIT_ASSERT_EQUAL(chooser.Run(1, argv), defaultAction.ReturnCode);
        UNIT_ASSERT_VALUES_EQUAL(defaultAction.Args, TVector<TString>{"UNITTEST"});
        UNIT_ASSERT_VALUES_EQUAL(defaultAction.CapturedSubcommandPath, TVector<TString>{"outer"});
    }

    Y_UNIT_TEST(TestDefaultActionWithUnknownMode) {
        // Unknown mode-like arguments are passed unchanged to the default action.
        TTestAction defaultAction;
        TModChooser chooser;
        chooser.SetDefaultAction(&defaultAction);

        const char* argv[] = {"UNITTEST", "mount-point", "--flag", nullptr};
        UNIT_ASSERT_EQUAL(chooser.Run(3, argv), defaultAction.ReturnCode);
        UNIT_ASSERT_VALUES_EQUAL(
            defaultAction.Args,
            (TVector<TString>{"UNITTEST", "mount-point", "--flag"}));
        UNIT_ASSERT(defaultAction.CapturedSubcommandPath.empty());
    }

    Y_UNIT_TEST(TestDefaultActionWithNamedMode) {
        // A named mode takes precedence over the default action and propagates its distinct return code.
        TTestAction defaultAction;
        TTestAction namedAction;
        // Use a random distinct return code to verify that the named action
        // is selected and its result is propagated.
        namedAction.ReturnCode = 17;
        TModChooser chooser;
        chooser.SetDefaultAction(&defaultAction);
        chooser.AddMode("list", &namedAction, "list entries");

        const char* argv[] = {"UNITTEST", "list", "argument", nullptr};
        UNIT_ASSERT_EQUAL(chooser.Run(3, argv), namedAction.ReturnCode);
        UNIT_ASSERT(defaultAction.Args.empty());
        UNIT_ASSERT_VALUES_EQUAL(namedAction.Args, (TVector<TString>{"UNITTEST list", "argument"}));
        UNIT_ASSERT_VALUES_EQUAL(namedAction.CapturedSubcommandPath, TVector<TString>{"list"});
    }

    // The default action must be non-null and cannot coexist with a default mode.
    Y_UNIT_TEST(TestDefaultConfiguration) {
        TTestAction defaultAction;
        TModChooser chooserWithNullAction;
        UNIT_ASSERT_EXCEPTION(chooserWithNullAction.SetDefaultAction(nullptr), yexception);

        TModChooser chooserWithAction;
        chooserWithAction.SetDefaultAction(&defaultAction);
        UNIT_ASSERT_EXCEPTION(chooserWithAction.SetDefaultMode("mode"), yexception);

        TModChooser chooserWithMode;
        chooserWithMode.SetDefaultMode("mode");
        UNIT_ASSERT_EXCEPTION(chooserWithMode.SetDefaultAction(&defaultAction), yexception);
    }
}
