#pragma once

#include <library/dbg_output/dump.h>

#include <util/generic/bt_exception.h>
#include <util/generic/hash.h>
#include <util/generic/intrlist.h>
#include <util/generic/map.h>
#include <util/generic/ptr.h>
#include <util/generic/set.h>
#include <util/generic/type_name.h>
#include <util/generic/typetraits.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>

#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/string/printf.h>

#include <util/system/defaults.h>
#include <util/system/src_location.h>

#include <util/system/rusage.h>

#include <cmath>
#include <cstdio>
#include <functional>

extern bool CheckExceptionMessage(const char*, TString&);

namespace NUnitTest {
    class TTestBase;

    namespace NPrivate {
        void RaiseError(const char* what, const TString& msg, bool fatalFailure);
        void SetUnittestThread(bool);
        void SetCurrentTest(TTestBase*);
        TTestBase* GetCurrentTest();
    }

    extern bool ShouldColorizeDiff;
    extern bool ContinueOnFail;
    TString ColoredDiff(TStringBuf s1, TStringBuf s2, const TString& delims = TString(), bool reverse = false);
    TString GetFormatTag(const char* name);
    TString GetResetTag();

    // Raise error handler
    // Used for testing library/unittest macroses
    // and unittest helpers.
    // For all other unittests standard handler is used
    using TRaiseErrorHandler = std::function<void(const char*, const TString&, bool)>;

    void SetRaiseErrorHandler(TRaiseErrorHandler handler);

    inline void ClearRaiseErrorHandler() {
        SetRaiseErrorHandler(TRaiseErrorHandler());
    }

    class TAssertException: public yexception {
    };

    struct TTestContext {
        using TMetrics = yhash<TString, double>;
        TMetrics Metrics;
    };

    class ITestSuiteProcessor {
    public:
        struct TUnit {
            const TString name;
        };

        struct TTest {
            const TUnit* unit;
            const char* name;
        };

        struct TError {
            const TTest* test;
            const char* msg;
            TString BackTrace;
            TTestContext* Context;
        };

        struct TFinish {
            const TTest* test;
            TTestContext* Context;
            bool Success;
        };

        ITestSuiteProcessor();

        virtual ~ITestSuiteProcessor();

        void Start();

        void End();

        void UnitStart(const TUnit& unit);

        void UnitStop(const TUnit& unit);

        void Error(const TError& descr);

        void BeforeTest(const TTest& test);

        void Finish(const TFinish& descr);

        unsigned GoodTests() const noexcept;

        unsigned FailTests() const noexcept;

        unsigned GoodTestsInCurrentUnit() const noexcept;

        unsigned FailTestsInCurrentUnit() const noexcept;

        // Should execute test suite?
        virtual bool CheckAccess(TString /*name*/, size_t /*num*/);

        // Should execute a test whitin suite?
        virtual bool CheckAccessTest(TString /*suite*/, const char* /*name*/);

        virtual void Run(std::function<void()> f, const TString /*suite*/, const char* /*name*/, const bool /*forceFork*/);

        // This process is forked for current test
        virtual bool GetIsForked() const;

        // --fork-tests is set (warning: this may be false, but never the less test will be forked if called inside UNIT_FORKED_TEST)
        virtual bool GetForkTests() const;

    private:
        virtual void OnStart();

        virtual void OnEnd();

        virtual void OnUnitStart(const TUnit* /*unit*/);

        virtual void OnUnitStop(const TUnit* /*unit*/);

        virtual void OnError(const TError* /*error*/);

        virtual void OnFinish(const TFinish* /*finish*/);

        virtual void OnBeforeTest(const TTest* /*test*/);

        void AddTestError(const TTest& test);

        void AddTestFinish(const TTest& test);

    private:
        ymap<TString, size_t> TestErrors_;
        ymap<TString, size_t> CurTestErrors_;
    };

    class TTestBase;
    class TTestFactory;

    class ITestBaseFactory: public TIntrusiveListItem<ITestBaseFactory> {
    public:
        ITestBaseFactory();

        virtual ~ITestBaseFactory();

        // name of test suite
        virtual TString Name() const noexcept = 0;
        virtual TTestBase* ConstructTest() = 0;

    private:
        void Register() noexcept;
    };

    class TTestBase {
        friend class TTestFactory;
        TRusage rusage;

    public:
        TTestBase() noexcept;

        virtual ~TTestBase();

        virtual TString TypeId() const;

        virtual TString Name() const noexcept = 0;
        virtual void Execute() = 0;

        virtual void SetUp();

        virtual void TearDown();

        void AddError(const char* msg, const TString& backtrace = TString(), TTestContext* context = nullptr);

        void AddError(const char* msg, TTestContext* context);

    protected:
        bool CheckAccessTest(const char* test);

        void BeforeTest(const char* func);

        void Finish(const char* func, TTestContext* context);

        void AtStart();

        void AtEnd();

        void Run(std::function<void()> f, const TString suite, const char* name, const bool forceFork);

        class TCleanUp {
        public:
            TCleanUp(TTestBase* base);

            ~TCleanUp();

        private:
            TTestBase* Base_;
        };

        void BeforeTest();

        void AfterTest();

        bool GetIsForked() const;

        bool GetForkTests() const;

    private:
        ITestSuiteProcessor* Processor() const noexcept;

    private:
        TTestFactory* Parent_;
        size_t TestErrors_;
        const char* CurrentSubtest_;
    };

#define UNIT_TEST_SUITE(N)                           \
    typedef N TThisUnitTestSuite;                    \
                                                     \
public:                                              \
    static TString StaticName() noexcept {           \
        return TString(#N);                          \
    }                                                \
                                                     \
private:                                             \
    virtual TString Name() const noexcept override { \
        return this->StaticName();                   \
    }                                                \
                                                     \
    virtual void Execute() override {                \
        this->AtStart();

#define UNIT_TEST_SUITE_DEMANGLE(N)                        \
    typedef N TThisUnitTestSuite;                          \
                                                           \
public:                                                    \
    static TString StaticName() noexcept {                 \
        return TCppDemangler().Demangle(typeid(N).name()); \
    }                                                      \
                                                           \
private:                                                   \
    virtual TString Name() const noexcept override {       \
        return this->StaticName();                         \
    }                                                      \
                                                           \
    virtual void Execute() override {                      \
        this->AtStart();

#ifndef UT_SKIP_EXCEPTIONS
#define CATCH_REACTION(FN, e, context) this->AddError(~("(" + TypeName(&e) + ") " + e.what()), context)
#define CATCH_REACTION_BT(FN, e, context) this->AddError(~("(" + TypeName(&e) + ") " + e.what()), (e.BackTrace() ? e.BackTrace()->PrintToString() : TString()), context)
#else
#define CATCH_REACTION(FN, e, context) throw
#define CATCH_REACTION_BT(FN, e, context) throw
#endif

#define UNIT_TEST_CHECK_TEST_IS_DECLARED_ONLY_ONCE(F)                                       \
    /* If you see this message - delete multiple UNIT_TEST(TestName) with same TestName. */ \
    /* It's forbidden to declare same test twice because it breaks --fork-tests logic.  */  \
    int You_have_declared_test_##F##_multiple_times_This_is_forbidden;                      \
    Y_UNUSED(You_have_declared_test_##F##_multiple_times_This_is_forbidden);

#define UNIT_TEST_RUN(F, FF, context)                                                             \
    this->BeforeTest((#F));                                                                       \
    {                                                                                             \
        struct T##F##Caller {                                                                     \
            static void X(TThisUnitTestSuite* thiz, NUnitTest::TTestContext&) {                   \
                TCleanUp cleaner(thiz);                                                           \
                thiz->F();                                                                        \
            }                                                                                     \
        };                                                                                        \
        this->TTestBase::Run(std::bind(&T##F##Caller::X, this, context), StaticName(), (#F), FF); \
    }

#define UNIT_TEST_IMPL(F, FF)                                   \
    UNIT_TEST_CHECK_TEST_IS_DECLARED_ONLY_ONCE(F) {             \
        NUnitTest::TTestContext context;                        \
        if (this->CheckAccessTest((#F))) {                      \
            try {                                               \
                UNIT_TEST_RUN(F, FF, context)                   \
            } catch (const ::NUnitTest::TAssertException&) {    \
            } catch (const yexception& e) {                     \
                CATCH_REACTION_BT((#F), e, &context);           \
            } catch (const std::exception& e) {                 \
                CATCH_REACTION((#F), e, &context);              \
            } catch (...) {                                     \
                this->AddError("non-std exception!", &context); \
            }                                                   \
            this->Finish((#F), &context);                       \
        }                                                       \
    }

#define UNIT_TEST(F) UNIT_TEST_IMPL(F, false)

#define UNIT_FORKED_TEST(F) UNIT_TEST_IMPL(F, true)

#define UNIT_TEST_EXCEPTION(F, E)                                                                                      \
    /* main process with "--fork-tests" flag treats exceptions as errors - it's result of forked test run */           \
    if (this->GetForkTests() && !this->GetIsForked()) {                                                                \
        UNIT_TEST_IMPL(F, false);                                                                                      \
        /* forked process (or main without "--fork-tests") treats some exceptions as success - it's exception test! */ \
    } else {                                                                                                           \
        NUnitTest::TTestContext context;                                                                               \
        if (this->CheckAccessTest((#F))) {                                                                             \
            try {                                                                                                      \
                UNIT_TEST_RUN(F, false, context)                                                                       \
                this->AddError("exception expected", &context);                                                        \
            } catch (const ::NUnitTest::TAssertException&) {                                                           \
            } catch (const E& e) {                                                                                     \
                TString err;                                                                                           \
                if (!CheckExceptionMessage(e.what(), err))                                                             \
                    this->AddError(err.c_str(), &context);                                                             \
            } catch (const std::exception& e) {                                                                        \
                this->AddError(e.what(), &context);                                                                    \
            } catch (...) {                                                                                            \
                this->AddError("non-std exception!", &context);                                                        \
            }                                                                                                          \
            this->Finish((#F), &context);                                                                              \
        }                                                                                                              \
    }

#define UNIT_TEST_SUITE_END() \
    this->AtEnd();            \
    }                         \
                              \
public:                       \
    /*for ; after macros*/ void sub##F()

#define UNIT_FAIL_IMPL(R, M)                                                                                                                     \
    do {                                                                                                                                         \
        ::NUnitTest::NPrivate::RaiseError(R, TStringBuilder() << R << " at " << __LOCATION__ << ", " << __PRETTY_FUNCTION__ << ": " << M, true); \
    } while (false)

#define UNIT_FAIL_NONFATAL_IMPL(R, M)                                                                                                             \
    do {                                                                                                                                          \
        ::NUnitTest::NPrivate::RaiseError(R, TStringBuilder() << R << " at " << __LOCATION__ << ", " << __PRETTY_FUNCTION__ << ": " << M, false); \
    } while (false)

#define UNIT_FAIL(M) UNIT_FAIL_IMPL("forced failure", M)
#define UNIT_FAIL_NONFATAL(M) UNIT_FAIL_NONFATAL_IMPL("forced failure", M)

//types
#define UNIT_ASSERT_TYPES_EQUAL(A, B) \
    if (!std::is_same<A, B>::value) {                                   \
        UNIT_FAIL_IMPL("types equal assertion failed", ~(TStringBuilder() << #A << " (" << TypeName<A>() << ") != " << #B << " (" << TypeName<B>() << ")")); \
    }

//doubles
#define UNIT_ASSERT_DOUBLES_EQUAL_C(E, A, D, C)                                                            \
    if (std::abs((E) - (A)) > (D)) {                                                                       \
        const auto _es = ToString((long double)(E));                                                       \
        const auto _as = ToString((long double)(A));                                                       \
        const auto _ds = ToString((long double)(D));                                                       \
        auto&& failMsg = Sprintf("std::abs(%s - %s) > %s %s", ~_es, ~_as, ~_ds, ~(TStringBuilder() << C)); \
        UNIT_FAIL_IMPL("assertion failure", failMsg);                                                      \
    }

#define UNIT_ASSERT_DOUBLES_EQUAL(E, A, D) UNIT_ASSERT_DOUBLES_EQUAL_C(E, A, D, "")

//strings
#define UNIT_ASSERT_STRINGS_EQUAL_C(A, B, C)                                                                 \
    do {                                                                                                     \
        const TString _a(A);                                                                                 \
        const TString _b(B);                                                                                 \
        if (_a != _b) {                                                                                      \
            auto&& failMsg = Sprintf("%s != %s %s", ~ToString(_a), ~ToString(_b), ~(TStringBuilder() << C)); \
            UNIT_FAIL_IMPL("strings equal assertion failed", failMsg);                                       \
        }                                                                                                    \
    } while (false)

#define UNIT_ASSERT_STRINGS_EQUAL(A, B) UNIT_ASSERT_STRINGS_EQUAL_C(A, B, "")

#define UNIT_ASSERT_STRING_CONTAINS_C(A, B, C)                                                                                  \
    do {                                                                                                                        \
        const TString _a(A);                                                                                                    \
        const TString _b(B);                                                                                                    \
        if (!_a.Contains(_b)) {                                                                                                 \
            auto&& msg = Sprintf("\"%s\" does not contain \"%s\", %s", ~ToString(_a), ~ToString(_b), ~(TStringBuilder() << C)); \
            UNIT_FAIL_IMPL("strings contains assertion failed", msg);                                                           \
        }                                                                                                                       \
    } while (false)

#define UNIT_ASSERT_STRING_CONTAINS(A, B) UNIT_ASSERT_STRING_CONTAINS_C(A, B, "")

#define UNIT_ASSERT_NO_DIFF(A, B)                                                                                                              \
    do {                                                                                                                                       \
        const TString _a(A);                                                                                                                   \
        const TString _b(B);                                                                                                                   \
        if (_a != _b) {                                                                                                                        \
            UNIT_FAIL_IMPL("strings (" #A ") and (" #B ") are different", Sprintf("\n%s", ~::NUnitTest::ColoredDiff(_a, _b, " \t\n.,:;'\""))); \
        }                                                                                                                                      \
    } while (false)

//strings
#define UNIT_ASSERT_STRINGS_UNEQUAL_C(A, B, C)                                                           \
    do {                                                                                                 \
        const TString _a(A);                                                                             \
        const TString _b(B);                                                                             \
        if (_a == _b) {                                                                                  \
            auto&& msg = Sprintf("%s == %s %s", ~ToString(_a), ~ToString(_b), ~(TStringBuilder() << C)); \
            UNIT_FAIL_IMPL("strings unequal assertion failed", msg);                                     \
        }                                                                                                \
    } while (false)

#define UNIT_ASSERT_STRINGS_UNEQUAL(A, B) UNIT_ASSERT_STRINGS_UNEQUAL_C(A, B, "")

//bool
#define UNIT_ASSERT_C(A, C)                                                                   \
    if (!(A)) {                                                                               \
        UNIT_FAIL_IMPL("assertion failed", Sprintf("(%s) %s", #A, ~(TStringBuilder() << C))); \
    }

#define UNIT_ASSERT(A) UNIT_ASSERT_C(A, "")

//general
#define UNIT_ASSERT_EQUAL_C(A, B, C)                                                                        \
    if (!((A) == (B))) {                                                                                    \
        UNIT_FAIL_IMPL("equal assertion failed", Sprintf("%s == %s %s", #A, #B, ~(TStringBuilder() << C))); \
    }

#define UNIT_ASSERT_EQUAL(A, B) UNIT_ASSERT_EQUAL_C(A, B, "")

#define UNIT_ASSERT_UNEQUAL_C(A, B, C)                                                                        \
    if ((A) == (B)) {                                                                                         \
        UNIT_FAIL_IMPL("unequal assertion failed", Sprintf("%s != %s %s", #A, #B, ~(TStringBuilder() << C))); \
    }

#define UNIT_ASSERT_UNEQUAL(A, B) UNIT_ASSERT_UNEQUAL_C(A, B, "")

#define UNIT_CHECK_GENERATED_EXCEPTION_C(A, E, C)                                            \
    while (true) {                                                                           \
        try {                                                                                \
            (void)(A);                                                                       \
        } catch (const ::NUnitTest::TAssertException&) {                                     \
            throw;                                                                           \
        } catch (const E&) {                                                                 \
            break;                                                                           \
        }                                                                                    \
        UNIT_ASSERT_C(0, "Exception hasn't been thrown, but it should have happened " << C); \
    }

#define UNIT_CHECK_GENERATED_EXCEPTION(A, E) UNIT_CHECK_GENERATED_EXCEPTION_C(A, E, "")

#define UNIT_CHECK_GENERATED_NO_EXCEPTION_C(A, E, C)                                         \
    try {                                                                                    \
        (void)(A);                                                                           \
    } catch (const ::NUnitTest::TAssertException&) {                                         \
        throw;                                                                               \
    } catch (const E&) {                                                                     \
        UNIT_ASSERT_C(0, "Exception has been thrown, but it shouldn't have happened " << C); \
    }

#define UNIT_CHECK_GENERATED_NO_EXCEPTION(A, E) UNIT_CHECK_GENERATED_NO_EXCEPTION_C(A, E, "")

// Assert that exception is thrown and contains some part of text
#define UNIT_ASSERT_EXCEPTION_CONTAINS_C(A, E, substr, C)      \
    do {                                                       \
        bool _thrown = false;                                  \
        try {                                                  \
            (void)(A);                                         \
        } catch (const ::NUnitTest::TAssertException&) {       \
            throw;                                             \
        } catch (const E&) {                                   \
            _thrown = true;                                    \
            const TString _substr((substr));                   \
            if (!_substr.empty()) {                            \
                UNIT_ASSERT_C(CurrentExceptionMessage()        \
                              .Contains(_substr),              \
                              "Exception doesn't contain \""   \
                              << _substr << "\"");             \
            }                                                  \
        } catch (...) {                                        \
            _thrown = true;                                    \
            UNIT_FAIL_IMPL("exception assertion failed",       \
                           #A << " doesn't throw " << #E       \
                           << ", but throws other exception "  \
                           << "with message:\n"                \
                           << CurrentExceptionMessage());      \
        }                                                      \
        if (!_thrown) {                                        \
            UNIT_FAIL_IMPL("exception assertion failed",       \
                           #A << " doesn't throw " << #E       \
                           << " " << C);                       \
        }                                                      \
    } while (false)

#define UNIT_ASSERT_EXCEPTION_CONTAINS(A, E, substr) \
    UNIT_ASSERT_EXCEPTION_CONTAINS_C(A, E, substr, "")

#define UNIT_ASSERT_EXCEPTION_C(A, E, C) UNIT_ASSERT_EXCEPTION_CONTAINS_C(A, E, "", C)

#define UNIT_ASSERT_EXCEPTION(A, E) UNIT_ASSERT_EXCEPTION_C(A, E, "")

#define UNIT_ASSERT_NO_EXCEPTION_C(A, C)                                                                                                                                 \
    do {                                                                                                                                                                 \
        try {                                                                                                                                                            \
            (void)(A);                                                                                                                                                   \
        } catch (const ::NUnitTest::TAssertException&) {                                                                                                                 \
            throw;                                                                                                                                                       \
        } catch (...) {                                                                                                                                                  \
            UNIT_FAIL_IMPL("exception-free assertion failed", Sprintf("%s throws %s\nException message: %s", #A, ~(TStringBuilder() << C), ~CurrentExceptionMessage())); \
        }                                                                                                                                                                \
    } while (false)

#define UNIT_ASSERT_NO_EXCEPTION(A) UNIT_ASSERT_NO_EXCEPTION_C(A, "")

    namespace NPrivate {
        template <class T, class U, bool Integers>
        struct TCompareValuesImpl {
            static inline bool Compare(const T& a, const U& b) {
                return a == b;
            }
        };

        template <class T, class U>
        struct TCompareValuesImpl<T, U, true> {
            static inline bool Compare(const T& a, const U& b) {
                return ::ToString(a) == ::ToString(b);
            }
        };

        template <class T, class U>
        using TCompareValues = TCompareValuesImpl<T, U, std::is_integral<T>::value && std::is_integral<U>::value>;

        template <typename T, typename U>
        static inline bool CompareEqual(const T& a, const U& b) {
            return TCompareValues<T, U>::Compare(a, b);
        }

        static inline bool CompareEqual(const char* a, const char* b) {
            return 0 == strcmp(a, b);
        }

        // helper method to avoid double evaluation of A and B expressions in UNIT_ASSERT_VALUES_EQUAL_C
        template <typename T, typename U>
        static inline bool CompareAndMakeStrings(const T& a, const U& b, TString& as, TString& asInd, TString& bs, TString& bsInd, bool& usePlainDiff, bool want) {
            const bool have = CompareEqual(a, b);
            usePlainDiff = std::is_integral<T>::value && std::is_integral<U>::value;

            if (want == have) {
                return true;
            }

            as = ::TStringBuilder() << ::DbgDump(a);
            bs = ::TStringBuilder() << ::DbgDump(b);
            asInd = ::TStringBuilder() << ::DbgDump(a).SetIndent(true);
            bsInd = ::TStringBuilder() << ::DbgDump(b).SetIndent(true);

            return false;
        }
    }

//values
#define UNIT_ASSERT_VALUES_EQUAL_IMPL(A, B, C, EQflag, EQstr, NEQstr)                                                                  \
    do {                                                                                                                               \
        TString _as;                                                                                                                   \
        TString _bs;                                                                                                                   \
        TString _asInd;                                                                                                                \
        TString _bsInd;                                                                                                                \
        bool _usePlainDiff;                                                                                                            \
        if (!::NUnitTest::NPrivate::CompareAndMakeStrings(A, B, _as, _asInd, _bs, _bsInd, _usePlainDiff, EQflag)) {                    \
            auto&& failMsg = Sprintf("(%s %s %s) failed: (%s %s %s) %s", #A, EQstr, #B, ~_as, NEQstr, ~_bs, ~(TStringBuilder() << C)); \
            if (EQflag && !_usePlainDiff) {                                                                                            \
                failMsg += ", with diff:\n";                                                                                           \
                failMsg += ::NUnitTest::ColoredDiff(_asInd, _bsInd);                                                                   \
            }                                                                                                                          \
            UNIT_FAIL_IMPL("assertion failed", failMsg);                                                                               \
        }                                                                                                                              \
    } while (false)

#define UNIT_ASSERT_VALUES_EQUAL_C(A, B, C) \
    UNIT_ASSERT_VALUES_EQUAL_IMPL(A, B, C, true, "==", "!=")

#define UNIT_ASSERT_VALUES_UNEQUAL_C(A, B, C) \
    UNIT_ASSERT_VALUES_EQUAL_IMPL(A, B, C, false, "!=", "==")

#define UNIT_ASSERT_VALUES_EQUAL(A, B) UNIT_ASSERT_VALUES_EQUAL_C(A, B, "")
#define UNIT_ASSERT_VALUES_UNEQUAL(A, B) UNIT_ASSERT_VALUES_UNEQUAL_C(A, B, "")

// Checks that test will fail while executing given expression
// Macro for using in unitests for ut helpers
#define UNIT_ASSERT_TEST_FAILS_C(A, C)                                   \
    do {                                                                 \
        ::NUnitTest::TUnitTestFailChecker checker;                       \
        try {                                                            \
            auto guard = checker.InvokeGuard();                          \
            (void)(A);                                                   \
        } catch (...) {                                                  \
            UNIT_FAIL_IMPL("fail test assertion failure",                \
                           "code is expected to generate test failure, " \
                           "but it throws exception with message: "      \
                           << CurrentExceptionMessage());                \
        }                                                                \
        if (!checker.Failed()) {                                         \
            UNIT_FAIL_IMPL("fail test assertion failure",                \
                           "code is expected to generate test failure"); \
        }                                                                \
    } while (false)

#define UNIT_ASSERT_TEST_FAILS(A) UNIT_ASSERT_TEST_FAILS_C(A, "")

#define UNIT_ADD_METRIC(name, value) context.Metrics[name] = value

    class TTestFactory {
        friend class TTestBase;
        friend class ITestBaseFactory;

    public:
        static TTestFactory& Instance();

        unsigned Execute();

        void SetProcessor(ITestSuiteProcessor* processor);

    private:
        void Register(ITestBaseFactory* b) noexcept;

        ITestSuiteProcessor* Processor() const noexcept;

    private:
        TTestFactory(ITestSuiteProcessor* processor);

        ~TTestFactory();

    private:
        TIntrusiveList<ITestBaseFactory> Items_;
        ITestSuiteProcessor* Processor_;
    };

    template <class T>
    class TTestBaseFactory: public ITestBaseFactory {
    public:
        ~TTestBaseFactory() override = default;

        inline TTestBase* ConstructTest() override {
            return new T;
        }

        inline TString Name() const noexcept override {
            return T::StaticName();
        }
    };

    struct TTest {
        inline TTest()
            : TTest(nullptr, nullptr, false)
        {
        }

        inline TTest(const char* name, std::function<void(TTestContext&)> body, bool forceFork)
            : Name(name)
            , Body(body)
            , ForceFork(forceFork)
        {
        }

        const char* Name;
        std::function<void(TTestContext&)> Body;
        bool ForceFork;
    };

    // Class for checking that code raises unittest failure
    class TUnitTestFailChecker {
    public:
        struct TInvokeGuard {
            explicit TInvokeGuard(TUnitTestFailChecker& parent)
                : Parent(&parent)
            {
                Parent->SetHandler();
            }

            TInvokeGuard(TInvokeGuard&& guard)
                : Parent(guard.Parent)
            {
                guard.Parent = nullptr;
            }

            ~TInvokeGuard() {
                if (Parent) {
                    ClearRaiseErrorHandler();
                }
            }

            TUnitTestFailChecker* Parent;
        };

        TUnitTestFailChecker() = default;
        TUnitTestFailChecker(const TUnitTestFailChecker&) = delete;
        TUnitTestFailChecker(TUnitTestFailChecker&&) = delete;

        TInvokeGuard InvokeGuard() {
            return TInvokeGuard(*this);
        }

        const TString& What() const {
            return What_;
        }

        const TString& Msg() const {
            return Msg_;
        }

        bool FatalFailure() const {
            return FatalFailure_;
        }

        bool Failed() const {
            return Failed_;
        }

    private:
        void Handler(const char* what, const TString& msg, bool fatalFailure) {
            What_ = what;
            Msg_ = msg;
            FatalFailure_ = fatalFailure;
            Failed_ = true;
        }

        void SetHandler() {
            TRaiseErrorHandler handler = [this](const char* what, const TString& msg, bool fatalFailure) {
                Handler(what, msg, fatalFailure);
            };
            SetRaiseErrorHandler(std::move(handler));
        }

    private:
        TString What_;
        TString Msg_;
        bool FatalFailure_ = false;
        bool Failed_ = false;
    };

#define UNIT_TEST_SUITE_REGISTRATION(T) \
    static ::NUnitTest::TTestBaseFactory<T> Y_GENERATE_UNIQUE_ID(UTREG_);

#define SIMPLE_UNIT_TEST_SUITE_IMPL(N, T)                                                                               \
    namespace NTestSuite##N {                                                                                           \
        class TCurrentTest: public T {                                                                                  \
        private:                                                                                                        \
            typedef yvector<NUnitTest::TTest> TTests;                                                                   \
                                                                                                                        \
            static TTests& Tests() {                                                                                    \
                static TTests tests;                                                                                    \
                return tests;                                                                                           \
            }                                                                                                           \
                                                                                                                        \
        public:                                                                                                         \
            static TString StaticName() {                                                                               \
                return #N;                                                                                              \
            }                                                                                                           \
            virtual TString Name() const noexcept {                                                                     \
                return StaticName();                                                                                    \
            }                                                                                                           \
                                                                                                                        \
            static void AddTest(const char* name, std::function<void(NUnitTest::TTestContext&)> body, bool forceFork) { \
                Tests().push_back(NUnitTest::TTest(name, body, forceFork));                                             \
            }                                                                                                           \
                                                                                                                        \
            virtual void Execute() {                                                                                    \
                this->AtStart();                                                                                        \
                for (TTests::iterator i = Tests().begin(), ie = Tests().end(); i != ie; ++i) {                          \
                    if (!this->CheckAccessTest(i->Name)) {                                                              \
                        continue;                                                                                       \
                    }                                                                                                   \
                    NUnitTest::TTestContext context;                                                                    \
                    try {                                                                                               \
                        this->BeforeTest(i->Name);                                                                      \
                        {                                                                                               \
                            TCleanUp cleaner(this);                                                                     \
                            this->T::Run([i, &context]() { i->Body(context); }, StaticName(), i->Name, i->ForceFork);   \
                        }                                                                                               \
                    } catch (const ::NUnitTest::TAssertException&) {                                                    \
                    } catch (const yexception& e) {                                                                     \
                        CATCH_REACTION_BT(i->Name, e, &context);                                                        \
                    } catch (const std::exception& e) {                                                                 \
                        CATCH_REACTION(i->Name, e, &context);                                                           \
                    } catch (...) {                                                                                     \
                        this->AddError("non-std exception!", &context);                                                 \
                    }                                                                                                   \
                    this->Finish(i->Name, &context);                                                                    \
                }                                                                                                       \
                this->AtEnd();                                                                                          \
            }                                                                                                           \
        };                                                                                                              \
        UNIT_TEST_SUITE_REGISTRATION(TCurrentTest)                                                                      \
    }                                                                                                                   \
    namespace NTestSuite##N

#define SIMPLE_UNIT_TEST_SUITE(N) SIMPLE_UNIT_TEST_SUITE_IMPL(N, TTestBase)
#define RUSAGE_UNIT_TEST_SUITE(N) SIMPLE_UNIT_TEST_SUITE_IMPL(N, NUnitTest::TRusageTest)

#define SIMPLE_UNIT_TEST_IMPL_REGISTER(N, FF)                                                  \
    void N(NUnitTest::TTestContext&);                                                          \
    struct TTestRegistration##N {                                                              \
        TTestRegistration##N() {                                                               \
            TCurrentTest::AddTest(#N, static_cast<void (*)(NUnitTest::TTestContext&)>(N), FF); \
        }                                                                                      \
    };                                                                                         \
    static TTestRegistration##N testRegistration##N;

#define SIMPLE_UNIT_TEST_IMPL(N, FF)      \
    SIMPLE_UNIT_TEST_IMPL_REGISTER(N, FF) \
    void N(NUnitTest::TTestContext&)

#define SIMPLE_UNIT_TEST(N) SIMPLE_UNIT_TEST_IMPL(N, false)
#define SIMPLE_UNIT_FORKED_TEST(N) SIMPLE_UNIT_TEST_IMPL(N, true)
#define SIMPLE_UNIT_TEST_WITH_CONTEXT(N)     \
    SIMPLE_UNIT_TEST_IMPL_REGISTER(N, false) \
    void N(NUnitTest::TTestContext& context)

#define SIMPLE_UNIT_TEST_SUITE_IMPLEMENTATION(N) \
    namespace NTestSuite##N

#define SIMPLE_UNIT_TEST_DECLARE(N) \
    void N(NUnitTest::TTestContext& context)

#define SIMPLE_UNIT_TEST_FRIEND(N, T) \
    friend void NTestSuite##N::T(NUnitTest::TTestContext&)

    TString RandomString(size_t len, ui32 seed = 0);
}

using ::NUnitTest::TTestBase;
