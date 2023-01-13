#include <library/cpp/getopt/last_getopt.h>

#include <library/cpp/colorizer/colors.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/array_size.h>
#include <util/string/subst.h>
#include <util/string/vector.h>
#include <util/string/split.h>

using namespace NLastGetopt;

namespace {
    struct TOptsNoDefault: public TOpts {
        TOptsNoDefault(const TStringBuf& optstring = TStringBuf())
            : TOpts(optstring)
        {
        }
    };

    class TOptsParseResultTestWrapper: public TOptsParseResultException {
        TVector<const char*> Argv_;

    public:
        TOptsParseResultTestWrapper(const TOpts* opts, TVector<const char*> argv)
            : Argv_(argv)
        {
            Init(opts, (int)Argv_.size(), Argv_.data());
        }
    };

    using V = TVector<const char*>;
}

struct TOptsParserTester {
    TOptsNoDefault Opts_;
    TVector<const char*> Argv_;

    THolder<TOptsParser> Parser_;

    void Initialize() {
        if (!Parser_)
            Parser_.Reset(new TOptsParser(&Opts_, (int)Argv_.size(), Argv_.data()));
    }

    void Accept() {
        Initialize();
        UNIT_ASSERT(Parser_->Next());
    }

    void AcceptOption() {
        Accept();
        UNIT_ASSERT(!!Parser_->CurOpt());
    }

    void AcceptOption(char c) {
        AcceptOption();
        UNIT_ASSERT(Parser_->CurOpt()->CharIs(c));
    }

    void AcceptOption(const TString& optName) {
        AcceptOption();
        UNIT_ASSERT(Parser_->CurOpt()->NameIs(optName));
    }

    template <typename TOpt>
    void AcceptOptionWithValue(TOpt optName, const TString& value) {
        AcceptOption(optName);
        UNIT_ASSERT_VALUES_EQUAL_C(value, Parser_->CurValStr(), "; option " << optName);
    }

    template <typename TOpt>
    void AcceptOptionWithoutValue(TOpt optName) {
        AcceptOption(optName);
        UNIT_ASSERT_C(!Parser_->CurVal(), ": opt " << optName << " must have no param");
    }

    void AcceptFreeArgInOrder(const TString& expected) {
        Accept();
        UNIT_ASSERT(!Parser_->CurOpt());
        UNIT_ASSERT_VALUES_EQUAL(expected, Parser_->CurValStr());
    }

    size_t Pos_;

    void AcceptEndOfOptions() {
        Initialize();
        UNIT_ASSERT(!Parser_->Next());
        Pos_ = Parser_->Pos_;

        // pos must not be changed after last meaningful invocation of Next()
        UNIT_ASSERT(!Parser_->Next());
        UNIT_ASSERT_VALUES_EQUAL(Pos_, Parser_->Pos_);
        UNIT_ASSERT(!Parser_->Next());
        UNIT_ASSERT_VALUES_EQUAL(Pos_, Parser_->Pos_);
    }

    void AcceptError() {
        Initialize();
        try {
            Parser_->Next();
            UNIT_FAIL("expecting exception");
        } catch (const TUsageException&) {
            // expecting
        }
    }

    void AcceptUnexpectedOption() {
        Initialize();
        size_t pos = Parser_->Pos_;
        size_t sop = Parser_->Sop_;
        AcceptError();
        UNIT_ASSERT_VALUES_EQUAL(pos, Parser_->Pos_);
        UNIT_ASSERT_VALUES_EQUAL(sop, Parser_->Sop_);
    }

    void AcceptFreeArg(const TString& expected) {
        UNIT_ASSERT(Pos_ < Parser_->Argc_);
        UNIT_ASSERT_VALUES_EQUAL(expected, Parser_->Argv_[Pos_]);
        ++Pos_;
    }

    void AcceptEndOfFreeArgs() {
        UNIT_ASSERT_VALUES_EQUAL(Argv_.size(), Pos_);
    }
};

namespace {
    bool gSimpleFlag = false;
    void SimpleHander(void) {
        gSimpleFlag = true;
    }
}

Y_UNIT_TEST_SUITE(TLastGetoptTests) {
    Y_UNIT_TEST(TestEqual) {
        TOptsNoDefault opts;
        opts.AddLongOption("from");
        opts.AddLongOption("to");
        TOptsParseResultTestWrapper r(&opts, V({"copy", "--from=/", "--to=/etc"}));

        UNIT_ASSERT_VALUES_EQUAL("copy", r.GetProgramName());
        UNIT_ASSERT_VALUES_EQUAL("/", r.Get("from"));
        UNIT_ASSERT_VALUES_EQUAL("/etc", r.Get("to"));
        UNIT_ASSERT_VALUES_EQUAL("/etc", r.GetOrElse("to", "trash"));
        UNIT_ASSERT(r.Has("from"));
        UNIT_ASSERT(r.Has("to"));

        UNIT_ASSERT_EXCEPTION(r.Get("left"), TException);
    }

    Y_UNIT_TEST(TestCharOptions) {
        TOptsNoDefault opts;
        opts.AddCharOption('R', NO_ARGUMENT);
        opts.AddCharOption('l', NO_ARGUMENT);
        opts.AddCharOption('h', NO_ARGUMENT);
        TOptsParseResultTestWrapper r(&opts, V({"cp", "/etc", "-Rl", "/tmp/etc"}));
        UNIT_ASSERT(r.Has('R'));
        UNIT_ASSERT(r.Has('l'));
        UNIT_ASSERT(!r.Has('h'));

        UNIT_ASSERT_VALUES_EQUAL(2u, r.GetFreeArgs().size());
        UNIT_ASSERT_VALUES_EQUAL(2u, r.GetFreeArgCount());
        UNIT_ASSERT_VALUES_EQUAL("/etc", r.GetFreeArgs()[0]);
        UNIT_ASSERT_VALUES_EQUAL("/tmp/etc", r.GetFreeArgs()[1]);
    }

    Y_UNIT_TEST(TestFreeArgs) {
        TOptsNoDefault opts;
        opts.SetFreeArgsNum(1, 3);
        TOptsParseResultTestWrapper r11(&opts, V({"cp", "/etc"}));
        TOptsParseResultTestWrapper r12(&opts, V({"cp", "/etc", "/tmp/etc"}));
        TOptsParseResultTestWrapper r13(&opts, V({"cp", "/etc", "/tmp/etc", "verbose"}));

        UNIT_ASSERT_EXCEPTION(
            TOptsParseResultTestWrapper(&opts, V({"cp", "/etc", "/tmp/etc", "verbose", "nosymlink"})),
            yexception);

        UNIT_ASSERT_EXCEPTION(
            TOptsParseResultTestWrapper(&opts, V({"cp"})),
            yexception);

        opts.SetFreeArgsNum(2);
        TOptsParseResultTestWrapper r22(&opts, V({"cp", "/etc", "/var/tmp"}));
    }

    Y_UNIT_TEST(TestCharOptionsRequiredOptional) {
        TOptsNoDefault opts;
        opts.AddCharOption('d', REQUIRED_ARGUMENT);
        opts.AddCharOption('e', REQUIRED_ARGUMENT);
        opts.AddCharOption('x', REQUIRED_ARGUMENT);
        opts.AddCharOption('y', REQUIRED_ARGUMENT);
        opts.AddCharOption('l', NO_ARGUMENT);
        TOptsParseResultTestWrapper r(&opts, V({"cmd", "-ld11", "-e", "22", "-lllx33", "-y", "44"}));
        UNIT_ASSERT_VALUES_EQUAL("11", r.Get('d'));
        UNIT_ASSERT_VALUES_EQUAL("22", r.Get('e'));
        UNIT_ASSERT_VALUES_EQUAL("33", r.Get('x'));
        UNIT_ASSERT_VALUES_EQUAL("44", r.Get('y'));
    }

    Y_UNIT_TEST(TestReturnInOrder) {
        TOptsParserTester tester;
        tester.Opts_.AddLongOption('v', "value");
        tester.Opts_.ArgPermutation_ = RETURN_IN_ORDER;

        tester.Argv_.push_back("cmd");
        tester.Argv_.push_back("--value=11");
        tester.Argv_.push_back("xx");
        tester.Argv_.push_back("-v12");
        tester.Argv_.push_back("yy");
        tester.Argv_.push_back("--");
        tester.Argv_.push_back("-v13");
        tester.Argv_.push_back("--");

        tester.AcceptOptionWithValue("value", "11");
        tester.AcceptFreeArgInOrder("xx");
        tester.AcceptOptionWithValue('v', "12");
        tester.AcceptFreeArgInOrder("yy");
        tester.AcceptFreeArgInOrder("-v13");
        tester.AcceptFreeArgInOrder("--");
        tester.AcceptEndOfOptions();
        tester.AcceptEndOfFreeArgs();
    }

    Y_UNIT_TEST(TestRequireOrder) {
        TOptsParserTester tester;
        tester.Opts_.ArgPermutation_ = REQUIRE_ORDER;
        tester.Opts_.AddLongOption('v', "value");

        tester.Argv_.push_back("cmd");
        tester.Argv_.push_back("--value=11");
        tester.Argv_.push_back("xx");
        tester.Argv_.push_back("-v12");
        tester.Argv_.push_back("yy");

        tester.AcceptOptionWithValue("value", "11");
        tester.AcceptEndOfOptions();

        tester.AcceptFreeArg("xx");
        tester.AcceptFreeArg("-v12");
        tester.AcceptFreeArg("yy");
        tester.AcceptEndOfFreeArgs();
    }

    Y_UNIT_TEST(TestPlusForLongOption) {
        TOptsParserTester tester;
        tester.Opts_.AddLongOption('v', "value");
        tester.Opts_.AllowPlusForLong_ = true;
        tester.Argv_.push_back("cmd");
        tester.Argv_.push_back("+value=11");
        tester.Argv_.push_back("xx");
        tester.Argv_.push_back("-v12");
        tester.Argv_.push_back("yy");

        tester.AcceptOptionWithValue("value", "11");
        tester.AcceptOptionWithValue("value", "12");
        tester.AcceptEndOfOptions();

        tester.AcceptFreeArg("xx");
        tester.AcceptFreeArg("yy");
        tester.AcceptEndOfFreeArgs();
    }

    Y_UNIT_TEST(TestBug1) {
        TOptsParserTester tester;
        tester.Opts_.AddCharOptions("A:b:cd:");

        tester.Argv_.push_back("cmd");
        tester.Argv_.push_back("-A");
        tester.Argv_.push_back("aaaa");
        tester.Argv_.push_back("zz");
        tester.Argv_.push_back("-c");
        tester.Argv_.push_back("-d8");
        tester.Argv_.push_back("ww");

        tester.AcceptOptionWithValue('A', "aaaa");
        tester.AcceptOptionWithoutValue('c');
        tester.AcceptOptionWithValue('d', "8");
        tester.AcceptEndOfOptions();

        tester.AcceptFreeArg("zz");
        tester.AcceptFreeArg("ww");
        tester.AcceptEndOfFreeArgs();
    }

    Y_UNIT_TEST(TestPermuteComplex) {
        TOptsParserTester tester;

        tester.Opts_.AddCharOption('x').NoArgument();
        tester.Opts_.AddCharOption('y').RequiredArgument();
        tester.Opts_.AddCharOption('z').NoArgument();
        tester.Opts_.AddCharOption('w').RequiredArgument();
        tester.Opts_.ArgPermutation_ = PERMUTE;

        tester.Argv_.push_back("cmd");
        tester.Argv_.push_back("-x");
        tester.Argv_.push_back("-y");
        tester.Argv_.push_back("val");
        tester.Argv_.push_back("freearg1");
        tester.Argv_.push_back("-zw");
        tester.Argv_.push_back("val2");
        tester.Argv_.push_back("freearg2");

        tester.AcceptOptionWithoutValue('x');
        tester.AcceptOptionWithValue('y', "val");
        tester.AcceptOptionWithoutValue('z');
        tester.AcceptOptionWithValue('w', "val2");
        tester.AcceptEndOfOptions();
        tester.AcceptFreeArg("freearg1");
        tester.AcceptFreeArg("freearg2");
        tester.AcceptEndOfFreeArgs();
    }

    Y_UNIT_TEST(TestFinalDashDash) {
        TOptsParserTester tester;
        tester.Opts_.AddLongOption("size");

        tester.Argv_.push_back("cmd");
        tester.Argv_.push_back("--");

        tester.AcceptEndOfOptions();
        tester.AcceptEndOfFreeArgs();
    }

    Y_UNIT_TEST(TestDashDashAfterDashDash) {
        TOptsParserTester tester;
        tester.Opts_.AddLongOption("size");

        tester.Argv_.push_back("cmd");
        tester.Argv_.push_back("--");
        tester.Argv_.push_back("--");
        tester.Argv_.push_back("--");

        tester.AcceptEndOfOptions();
        tester.AcceptFreeArg("--");
        tester.AcceptFreeArg("--");
        tester.AcceptEndOfFreeArgs();
    }

    Y_UNIT_TEST(TestUnexpectedUnknownOption) {
        TOptsParserTester tester;

        tester.Argv_.push_back("cmd");
        tester.Argv_.push_back("-x");

        tester.AcceptUnexpectedOption();
    }

    Y_UNIT_TEST(TestDuplicatedOptionCrash) {
        // this test is broken, cause UNIT_ASSERT(false) always throws
        return;

        bool exception = false;
        try {
            TOpts opts;
            opts.AddLongOption('x', "one");
            opts.AddLongOption('x', "two");
            UNIT_ASSERT(false);
        } catch (...) {
            // we should go here, duplicating options are forbidden
            exception = true;
        }
        UNIT_ASSERT(exception);
    }

    Y_UNIT_TEST(TestPositionWhenNoArgs) {
        TOptsParserTester tester;

        tester.Argv_.push_back("cmd");

        tester.Opts_.AddCharOption('c');

        tester.AcceptEndOfOptions();

        UNIT_ASSERT_VALUES_EQUAL(1u, tester.Parser_->Pos_);
    }

    Y_UNIT_TEST(TestExpectedUnknownCharOption) {
        TOptsParserTester tester;

        tester.Argv_.push_back("cmd");
        tester.Argv_.push_back("-x");
        tester.Argv_.push_back("-y");
        tester.Argv_.push_back("val");
        tester.Argv_.push_back("freearg1");
        tester.Argv_.push_back("-zw");
        tester.Argv_.push_back("val2");
        tester.Argv_.push_back("freearg2");

        tester.Opts_.AllowUnknownCharOptions_ = true;

        tester.AcceptOptionWithoutValue('x');
        tester.AcceptOptionWithValue('y', "val");
        tester.AcceptOptionWithoutValue('z');
        tester.AcceptOptionWithValue('w', "val2");
        tester.AcceptEndOfOptions();
        tester.AcceptFreeArg("freearg1");
        tester.AcceptFreeArg("freearg2");
        tester.AcceptEndOfFreeArgs();
    }

#if 0
    Y_UNIT_TEST(TestRequiredParams) {
        TOptsParserTester tester;

        tester.Argv_.push_back("cmd");
        tester.Argv_.push_back("--port=1231");
        tester.Argv_.push_back("asas");

        tester.Opts_.AddLongOption("port");
        tester.Opts_.AddLongOption("home").Required();

        tester.AcceptOptionWithValue("port", "1231");
        tester.AcceptError();
    }
#endif

    Y_UNIT_TEST(TestEqParseOnly) {
        TOptsParserTester tester;

        tester.Argv_.push_back("cmd");
        tester.Argv_.push_back("--data=jjhh");
        tester.Argv_.push_back("-n");
        tester.Argv_.push_back("11");
        tester.Argv_.push_back("--optional-number-1=8");
        tester.Argv_.push_back("--optional-string-1=os1");
        tester.Argv_.push_back("--optional-number-2");
        tester.Argv_.push_back("10");
        tester.Argv_.push_back("--optional-string-2");
        tester.Argv_.push_back("freearg");

        tester.Opts_.AddLongOption('d', "data");
        tester.Opts_.AddLongOption('n', "number");
        tester.Opts_.AddLongOption("optional-string-0");
        tester.Opts_.AddLongOption("optional-number-0");
        tester.Opts_.AddLongOption("optional-string-1");
        tester.Opts_.AddLongOption("optional-number-1");
        tester.Opts_.AddLongOption("optional-string-2").OptionalArgument().DisableSpaceParse();
        tester.Opts_.AddLongOption("optional-number-2").OptionalArgument();

        tester.AcceptOptionWithValue("data", "jjhh");
        tester.AcceptOptionWithValue('n', "11");
        tester.AcceptOptionWithValue("optional-number-1", "8");
        tester.AcceptOptionWithValue("optional-string-1", "os1");
        tester.AcceptOptionWithValue("optional-number-2", "10");
        tester.AcceptOptionWithoutValue("optional-string-2");
        tester.AcceptEndOfOptions();
        tester.AcceptFreeArg("freearg");
        tester.AcceptEndOfFreeArgs();
    }

    Y_UNIT_TEST(TestStoreResult) {
        TOptsNoDefault opts;
        TString data;
        int number;
        TMaybe<TString> optionalString0, optionalString1;
        TMaybe<int> optionalNumber0, optionalNumber1;
        opts.AddLongOption('d', "data").StoreResult(&data);
        opts.AddLongOption('n', "number").StoreResult(&number);
        opts.AddLongOption("optional-string-0").StoreResult(&optionalString0);
        opts.AddLongOption("optional-number-0").StoreResult(&optionalNumber0);
        opts.AddLongOption("optional-string-1").StoreResult(&optionalString1);
        opts.AddLongOption("optional-number-1").StoreResult(&optionalNumber1);
        TOptsParseResultTestWrapper r(&opts, V({"cmd", "--data=jjhh", "-n", "11", "--optional-number-1=8", "--optional-string-1=os1"}));
        UNIT_ASSERT_VALUES_EQUAL("jjhh", data);
        UNIT_ASSERT_VALUES_EQUAL(11, number);
        UNIT_ASSERT(!optionalString0.Defined());
        UNIT_ASSERT(!optionalNumber0.Defined());
        UNIT_ASSERT_VALUES_EQUAL(*optionalString1, "os1");
        UNIT_ASSERT_VALUES_EQUAL(*optionalNumber1, 8);
    }

    Y_UNIT_TEST(TestStoreValue) {
        int a = 0, b = 0;
        size_t c = 0;
        EHasArg e = NO_ARGUMENT;

        TOptsNoDefault opts;
        opts.AddLongOption('a', "alpha").NoArgument().StoreValue(&a, 42);
        opts.AddLongOption('b', "beta").NoArgument().StoreValue(&b, 24);
        opts.AddLongOption('e', "enum").NoArgument().StoreValue(&e, REQUIRED_ARGUMENT).StoreValue(&c, 12345);

        TOptsParseResultTestWrapper r(&opts, V({"cmd", "-a", "-e"}));

        UNIT_ASSERT_VALUES_EQUAL(42, a);
        UNIT_ASSERT_VALUES_EQUAL(0, b);
        UNIT_ASSERT(e == REQUIRED_ARGUMENT);
        UNIT_ASSERT_VALUES_EQUAL(12345u, c);
    }

    Y_UNIT_TEST(TestSetFlag) {
        bool a = false, b = true, c = false, d = true;

        TOptsNoDefault opts;
        opts.AddLongOption('a', "alpha").NoArgument().SetFlag(&a);
        opts.AddLongOption('b', "beta").NoArgument().SetFlag(&b);
        opts.AddCharOption('c').StoreTrue(&c);
        opts.AddCharOption('d').StoreTrue(&d);

        TOptsParseResultTestWrapper r(&opts, V({"cmd", "-a", "-c"}));

        UNIT_ASSERT(a);
        UNIT_ASSERT(!b);
        UNIT_ASSERT(c);
        UNIT_ASSERT(!d);
    }

    Y_UNIT_TEST(TestDefaultValue) {
        TOptsNoDefault opts;
        opts.AddLongOption("path").DefaultValue("/etc");
        int value = 42;
        opts.AddLongOption("value").StoreResult(&value).DefaultValue(32);
        TOptsParseResultTestWrapper r(&opts, V({"cmd", "dfdf"}));
        UNIT_ASSERT_VALUES_EQUAL("/etc", r.Get("path"));
        UNIT_ASSERT_VALUES_EQUAL(32, value);
    }

    Y_UNIT_TEST(TestSplitValue) {
        TOptsNoDefault opts;
        TVector<TString> vals;
        opts.AddLongOption('s', "split").SplitHandler(&vals, ',');
        TOptsParseResultTestWrapper r(&opts, V({"prog", "--split=a,b,c"}));
        UNIT_ASSERT_EQUAL(vals.size(), 3);
        UNIT_ASSERT_EQUAL(vals[0], "a");
        UNIT_ASSERT_EQUAL(vals[1], "b");
        UNIT_ASSERT_EQUAL(vals[2], "c");
    }

    Y_UNIT_TEST(TestRangeSplitValue) {
        TOptsNoDefault opts;
        TVector<ui32> vals;
        opts.AddLongOption('s', "split").RangeSplitHandler(&vals, ',', '-');
        TOptsParseResultTestWrapper r(&opts, V({"prog", "--split=1,8-10", "--split=12-14"}));
        UNIT_ASSERT_EQUAL(vals.size(), 7);
        UNIT_ASSERT_EQUAL(vals[0], 1);
        UNIT_ASSERT_EQUAL(vals[1], 8);
        UNIT_ASSERT_EQUAL(vals[2], 9);
        UNIT_ASSERT_EQUAL(vals[3], 10);
        UNIT_ASSERT_EQUAL(vals[4], 12);
        UNIT_ASSERT_EQUAL(vals[5], 13);
        UNIT_ASSERT_EQUAL(vals[6], 14);
    }

    Y_UNIT_TEST(TestParseArgs) {
        TOptsNoDefault o("AbCx:y:z::");
        UNIT_ASSERT_EQUAL(o.GetCharOption('A').HasArg_, NO_ARGUMENT);
        UNIT_ASSERT_EQUAL(o.GetCharOption('b').HasArg_, NO_ARGUMENT);
        UNIT_ASSERT_EQUAL(o.GetCharOption('C').HasArg_, NO_ARGUMENT);
        UNIT_ASSERT_EQUAL(o.GetCharOption('x').HasArg_, REQUIRED_ARGUMENT);
        UNIT_ASSERT_EQUAL(o.GetCharOption('y').HasArg_, REQUIRED_ARGUMENT);
        UNIT_ASSERT_EQUAL(o.GetCharOption('z').HasArg_, OPTIONAL_ARGUMENT);
    }

    Y_UNIT_TEST(TestRequiredOpts) {
        TOptsNoDefault opts;
        TOpt& opt_d = opts.AddCharOption('d');

        // test 'not required'
        // makes sure that the problem will only be in 'required'
        TOptsParseResultTestWrapper r1(&opts, V({"cmd"}));

        // test 'required'
        opt_d.Required();
        UNIT_ASSERT_EXCEPTION(
            TOptsParseResultTestWrapper(&opts, V({"cmd"})),
            TUsageException);

        TOptsParseResultTestWrapper r3(&opts, V({"cmd", "-d11"}));
        UNIT_ASSERT_VALUES_EQUAL("11", r3.Get('d'));
    }

    class HandlerStoreTrue {
        bool* Flag;

    public:
        HandlerStoreTrue(bool* flag)
            : Flag(flag)
        {
        }
        void operator()() {
            *Flag = true;
        }
    };
    Y_UNIT_TEST(TestHandlers) {
        {
            TOptsNoDefault opts;
            bool flag = false;
            opts.AddLongOption("flag").Handler0(HandlerStoreTrue(&flag)).NoArgument();
            TOptsParseResultTestWrapper r(&opts, V({"cmd", "--flag"}));
            UNIT_ASSERT(flag);
        }
        {
            TOptsNoDefault opts;
            unsigned uval = 5;
            double fval = 0.0;
            opts.AddLongOption("flag1").RequiredArgument().StoreResult(&uval);
            opts.AddLongOption("flag2").RequiredArgument().StoreResultT<int>(&uval);
            opts.AddLongOption("flag3").RequiredArgument().StoreMappedResult(&fval, (double (*)(double))fabs);
            opts.AddLongOption("flag4").RequiredArgument().StoreMappedResult(&fval, (double (*)(double))sqrt);
            UNIT_ASSERT_EXCEPTION(
                TOptsParseResultTestWrapper(&opts, V({"cmd", "--flag3", "-2.0", "--flag1", "-1"})),
                yexception);
            UNIT_ASSERT_VALUES_EQUAL(uval, 5u);
            UNIT_ASSERT_VALUES_EQUAL(fval, 2.0);
            TOptsParseResultTestWrapper r1(&opts, V({"cmd", "--flag4", "9.0", "--flag2", "-1"}));
            UNIT_ASSERT_VALUES_EQUAL(uval, Max<unsigned>());
            UNIT_ASSERT_VALUES_EQUAL(fval, 3.0);
        }
    }

    Y_UNIT_TEST(TestTitleAndPrintUsage) {
        TOpts opts;
        const char* prog = "my_program";
        TString title = TString("Sample ") + TString(prog).Quote() + " application";
        opts.SetTitle(title);
        int argc = 2;
        const char* cmd[] = {prog};
        TOptsParser parser(&opts, argc, cmd);
        TStringStream out;
        parser.PrintUsage(out);
        // find title
        UNIT_ASSERT(out.Str().find(title) != TString::npos);
        // find usage
        UNIT_ASSERT(out.Str().find(" " + TString(prog) + " ") != TString::npos);
    }

    Y_UNIT_TEST(TestCustomCmdLineDescr) {
        TOpts opts;
        const char* prog = "my_program";
        TString customDescr = "<FILE|TABLE> USER [OPTIONS]";
        int argc = 2;
        const char* cmd[] = {prog};
        opts.SetCmdLineDescr(customDescr);
        TOptsParser parser(&opts, argc, cmd);
        TStringStream out;
        parser.PrintUsage(out);
        // find custom usage
        UNIT_ASSERT(out.Str().find(customDescr) != TString::npos);
    }

    Y_UNIT_TEST(TestColorPrint) {
        TOpts opts;
        const char* prog = "my_program";
        opts.AddLongOption("long_option").Required();
        opts.AddLongOption('o', "other");
        opts.AddCharOption('d').DefaultValue("42");
        opts.AddCharOption('s').DefaultValue("str_default");
        opts.SetFreeArgsNum(123, 456);
        opts.SetFreeArgTitle(0, "first_free_arg", "help");
        opts.SetFreeArgTitle(2, "second_free_arg");
        opts.AddSection("Section", "Section\n  text");
        const char* cmd[] = {prog};
        TOptsParser parser(&opts, Y_ARRAY_SIZE(cmd), cmd);
        TStringStream out;
        NColorizer::TColors colors(true);
        parser.PrintUsage(out, colors);

        // find options and green color
        UNIT_ASSERT(out.Str().find(TStringBuilder() << colors.GreenColor() << "--long_option" << colors.OldColor()) != TString::npos);
        UNIT_ASSERT(out.Str().find(TStringBuilder() << colors.GreenColor() << "--other" << colors.OldColor()) != TString::npos);
        UNIT_ASSERT(out.Str().find(TStringBuilder() << colors.GreenColor() << "-o" << colors.OldColor()) != TString::npos);
        UNIT_ASSERT(out.Str().find(TStringBuilder() << colors.GreenColor() << "-d" << colors.OldColor()) != TString::npos);
        UNIT_ASSERT(out.Str().find(TStringBuilder() << colors.GreenColor() << "-s" << colors.OldColor()) != TString::npos);

        // find default values
        UNIT_ASSERT(out.Str().find(TStringBuilder() << colors.CyanColor() << "42" << colors.OldColor()) != TString::npos);
        UNIT_ASSERT(out.Str().find(TStringBuilder() << colors.CyanColor() << "\"str_default\"" << colors.OldColor()) != TString::npos);

        // find free args
        UNIT_ASSERT(out.Str().find(TStringBuilder() << colors.GreenColor() << "123" << colors.OldColor()) != TString::npos);
        UNIT_ASSERT(out.Str().find(TStringBuilder() << colors.GreenColor() << "456" << colors.OldColor()) != TString::npos);
        UNIT_ASSERT(out.Str().find(TStringBuilder() << colors.GreenColor() << "first_free_arg" << colors.OldColor()) != TString::npos);
        // free args without help not rendered even if they have custom title
        UNIT_ASSERT(out.Str().find(TStringBuilder() << colors.GreenColor() << "second_free_arg" << colors.OldColor()) == TString::npos);

        // find signatures
        UNIT_ASSERT(out.Str().find(TStringBuilder() << colors.BoldColor() << "Usage" << colors.OldColor()) != TString::npos);
        UNIT_ASSERT(out.Str().find(TStringBuilder() << colors.BoldColor() << "Required parameters" << colors.OldColor()) != TString::npos);
        UNIT_ASSERT(out.Str().find(TStringBuilder() << colors.BoldColor() << "Optional parameters" << colors.OldColor()) != TString::npos);
        UNIT_ASSERT(out.Str().find(TStringBuilder() << colors.BoldColor() << "Free args" << colors.OldColor()) != TString::npos);

        // find sections
        UNIT_ASSERT(out.Str().find(TStringBuilder() << colors.BoldColor() << "Section" << colors.OldColor() << ":") != TString::npos);
        UNIT_ASSERT(out.Str().find(TStringBuilder() << "  Section\n    text") != TString::npos);

        // print without colors
        TStringStream out2;
        opts.PrintUsage(prog, out2);
        UNIT_ASSERT(out2.Str().find(colors.GreenColor()) == TString::npos);
        UNIT_ASSERT(out2.Str().find(colors.CyanColor()) == TString::npos);
        UNIT_ASSERT(out2.Str().find(colors.BoldColor()) == TString::npos);
        UNIT_ASSERT(out2.Str().find(colors.OldColor()) == TString::npos);
    }

    Y_UNIT_TEST(TestPadding) {
        const bool withColorsOpt[] = {false, true};
        for (bool withColors : withColorsOpt) {
            TOpts opts;
            const char* prog = "my_program";
            opts.AddLongOption("option", "description 1").Required();               // long option
            opts.AddLongOption('o', "other", "description 2");                      // char and long option
            opts.AddCharOption('d', "description 3").RequiredArgument("DD");        // char option
            opts.AddCharOption('s', "description 4\ndescription 5\ndescription 6"); // multiline desc
            opts.AddLongOption('l', "very_very_very_loooong_ooooption", "description 7").RequiredArgument("LONG_ARGUMENT");
            const char* cmd[] = {prog};
            TOptsParser parser(&opts, Y_ARRAY_SIZE(cmd), cmd);

            TStringStream out;
            NColorizer::TColors colors(withColors);
            parser.PrintUsage(out, colors);

            TString printed = out.Str();
            if (withColors) {
                // remove not printable characters
                SubstGlobal(printed, TString(colors.BoldColor()), "");
                SubstGlobal(printed, TString(colors.GreenColor()), "");
                SubstGlobal(printed, TString(colors.CyanColor()), "");
                SubstGlobal(printed, TString(colors.OldColor()), "");
            }
            TVector<TString> lines;
            StringSplitter(printed).Split('\n').SkipEmpty().Collect(&lines);
            UNIT_ASSERT(!lines.empty());
            TVector<size_t> indents;
            for (const TString& line : lines) {
                const size_t indent = line.find("description ");
                if (indent != TString::npos)
                    indents.push_back(indent);
            }
            UNIT_ASSERT_VALUES_EQUAL(indents.size(), 7);
            const size_t theOnlyIndent = indents[0];
            for (size_t indent : indents) {
                UNIT_ASSERT_VALUES_EQUAL_C(indent, theOnlyIndent, printed);
            }
        }
    }

    Y_UNIT_TEST(TestAppendTo) {
        TVector<int> ints;
        std::vector<std::string> strings;

        TOptsNoDefault opts;
        opts.AddLongOption("size").AppendTo(&ints);
        opts.AddLongOption("value").AppendTo(&strings);

        TOptsParseResultTestWrapper r(&opts, V({"cmd", "--size=17", "--size=19", "--value=v1", "--value=v2"}));

        UNIT_ASSERT_VALUES_EQUAL(size_t(2), ints.size());
        UNIT_ASSERT_VALUES_EQUAL(17, ints.at(0));
        UNIT_ASSERT_VALUES_EQUAL(19, ints.at(1));

        UNIT_ASSERT_VALUES_EQUAL(size_t(2), strings.size());
        UNIT_ASSERT_VALUES_EQUAL("v1", strings.at(0));
        UNIT_ASSERT_VALUES_EQUAL("v2", strings.at(1));
    }

    Y_UNIT_TEST(TestEmplaceTo) {
        TVector<std::tuple<TString>> richPaths;

        TOptsNoDefault opts;
        opts.AddLongOption("path").EmplaceTo(&richPaths);

        TOptsParseResultTestWrapper r(&opts, V({"cmd", "--path=<a=b>//cool", "--path=//nice"}));

        UNIT_ASSERT_VALUES_EQUAL(size_t(2), richPaths.size());
        UNIT_ASSERT_VALUES_EQUAL("<a=b>//cool", std::get<0>(richPaths.at(0)));
        UNIT_ASSERT_VALUES_EQUAL("//nice", std::get<0>(richPaths.at(1)));
    }

    Y_UNIT_TEST(TestKVHandler) {
        TStringBuilder keyvals;

        TOptsNoDefault opts;
        opts.AddLongOption("set").KVHandler([&keyvals](TString k, TString v) { keyvals << k << ":" << v << ","; });

        TOptsParseResultTestWrapper r(&opts, V({"cmd", "--set", "x=1", "--set", "y=2", "--set=z=3"}));

        UNIT_ASSERT_VALUES_EQUAL(keyvals, "x:1,y:2,z:3,");
    }

    Y_UNIT_TEST(TestEasySetup) {
        TEasySetup opts;
        bool flag = false;
        opts('v', "version", "print version information")('a', "abstract", "some abstract param", true)('b', "buffer", "SIZE", "some param with argument")('c', "count", "SIZE", "some param with required argument")('t', "true", HandlerStoreTrue(&flag), "Some arg with handler")("global", SimpleHander, "Another arg with handler");

        {
            gSimpleFlag = false;
            TOptsParseResultTestWrapper r(&opts, V({"cmd", "--abstract"}));
            UNIT_ASSERT(!flag);
            UNIT_ASSERT(!gSimpleFlag);
        }

        {
            TOptsParseResultTestWrapper r(&opts, V({"cmd", "--abstract", "--global", "-t"}));
            UNIT_ASSERT(flag);
            UNIT_ASSERT(gSimpleFlag);
        }

        {
            UNIT_ASSERT_EXCEPTION(
                TOptsParseResultTestWrapper(&opts, V({"cmd", "--true"})),
                TUsageException);
        }

        {
            TOptsParseResultTestWrapper r(&opts, V({"cmd", "--abstract", "--buffer=512"}));
            UNIT_ASSERT(r.Has('b'));
            UNIT_ASSERT_VALUES_EQUAL(r.Get('b', 0), "512");
        }
    }

    Y_UNIT_TEST(TestTOptsParseResultException) {
        // verify that TOptsParseResultException actually throws a TUsageException instead of exit()
        // not using wrapper here because it can hide bugs (see review #243810 and r2737774)
        TOptsNoDefault opts;
        opts.AddLongOption("required-opt").Required();
        const char* argv[] = {"cmd"};
        // Should throw TUsageException. Other exception types, no exceptions at all and exit(1) are failures
        UNIT_ASSERT_EXCEPTION(
            TOptsParseResultException(&opts, Y_ARRAY_SIZE(argv), argv),
            TUsageException);
    }

    Y_UNIT_TEST(TestFreeArgsStoreResult) {
        TOptsNoDefault opts;
        TString data;
        int number = 0;
        opts.AddFreeArgBinding("data", data);
        opts.AddFreeArgBinding("number", number);
        TOptsParseResultTestWrapper r(&opts, V({"cmd", "hello", "25"}));
        UNIT_ASSERT_VALUES_EQUAL("hello", data);
        UNIT_ASSERT_VALUES_EQUAL(25, number);
        UNIT_ASSERT_VALUES_EQUAL(2, r.GetFreeArgCount());
    }

    Y_UNIT_TEST(TestCheckUserTypos) {
        {
            TOptsNoDefault opts;
            opts.SetCheckUserTypos();
            opts.AddLongOption("from");
            opts.AddLongOption("to");

            UNIT_ASSERT_EXCEPTION(
                    TOptsParseResultTestWrapper(&opts, V({"copy", "-from", "/home", "--to=/etc"})),
                    TUsageException);
            UNIT_ASSERT_NO_EXCEPTION(
                    TOptsParseResultTestWrapper(&opts, V({"copy", "--from", "from", "--to=/etc"})));
        }

        {
            TOptsNoDefault opts;
            opts.SetCheckUserTypos();
            opts.AddLongOption('f', "file", "");
            opts.AddLongOption('r', "read", "");
            opts.AddLongOption("fr");
            UNIT_ASSERT_NO_EXCEPTION(
                    TOptsParseResultTestWrapper(&opts, V({"copy", "-fr"})));
        }
    }
}
