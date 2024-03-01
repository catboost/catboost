#pragma once

#include <library/cpp/dbg_output/dump.h>

#include <util/generic/bt_exception.h>
#include <util/generic/hash.h>
#include <util/generic/intrlist.h>
#include <util/generic/map.h>
#include <util/generic/ptr.h>
#include <util/generic/scope.h>
#include <util/generic/set.h>
#include <util/generic/typetraits.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>

#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/string/printf.h>

#include <util/system/defaults.h>
#include <util/system/type_name.h>
#include <util/system/spinlock.h>
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
    TString ColoredDiff(TStringBuf s1, TStringBuf s2, const TString& delims = TString(), bool reverse = false);
    TString GetFormatTag(const char* name);
    TString GetResetTag();

    // Raise error handler
    // Used for testing library/cpp/testing/unittest macroses
    // and unittest helpers.
    // For all other unittests standard handler is used
    using TRaiseErrorHandler = std::function<void(const char*, const TString&, bool)>;

    void SetRaiseErrorHandler(TRaiseErrorHandler handler);

    inline void ClearRaiseErrorHandler() {
        SetRaiseErrorHandler(TRaiseErrorHandler());
    }

    class TAssertException: public yexception {
    };

    class ITestSuiteProcessor;

    struct TTestContext {
        TTestContext()
            : Processor(nullptr)
        {
        }

        explicit TTestContext(ITestSuiteProcessor* processor)
            : Processor(processor)
        {
        }

        using TMetrics = THashMap<TString, double>;
        TMetrics Metrics;

        ITestSuiteProcessor* Processor;
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

        virtual void Run(std::function<void()> f, const TString& /*suite*/, const char* /*name*/, bool /*forceFork*/);

        // This process is forked for current test
        virtual bool GetIsForked() const;

        // --fork-tests is set (warning: this may be false, but never the less test will be forked if called inside UNIT_FORKED_TEST)
        virtual bool GetForkTests() const;

        virtual void SetForkTestsParams(bool forkTests, bool isForked);

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
        TMap<TString, size_t> TestErrors_;
        TMap<TString, size_t> CurTestErrors_;
        bool IsForked_ = false;
        bool ForkTests_ = false;
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

        virtual void GlobalSuiteSetUp() {}

        virtual void GlobalSuiteTearDown() {}

        void AddError(const char* msg, const TString& backtrace = TString(), TTestContext* context = nullptr);

        void AddError(const char* msg, TTestContext* context);

        void RunAfterTest(std::function<void()> f); // function like atexit to run after current unit test

    protected:
        bool CheckAccessTest(const char* test);

        void BeforeTest(const char* func);

        void Finish(const char* func, TTestContext* context);

        void AtStart();

        void AtEnd();

        void Run(std::function<void()> f, const TString& suite, const char* name, bool forceFork);

        class TCleanUp {
        public:
            explicit TCleanUp(TTestBase* base);

            ~TCleanUp();

        private:
            TTestBase* Base_;
        };

        void BeforeTest();

        void AfterTest();

        bool GetIsForked() const;

        bool GetForkTests() const;

        ITestSuiteProcessor* Processor() const noexcept;

    private:
        TTestFactory* Parent_;
        size_t TestErrors_;
        const char* CurrentSubtest_;
        TAdaptiveLock AfterTestFunctionsLock_;
        TVector<std::function<void()>> AfterTestFunctions_;
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
        return TypeName<N>(); \
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
#define CATCH_REACTION(FN, e, context) this->AddError(("(" + TypeName(e) + ") " + e.what()).data(), context)
#define CATCH_REACTION_BT(FN, e, context) this->AddError(("(" + TypeName(e) + ") " + e.what()).data(), (e.BackTrace() ? e.BackTrace()->PrintToString() : TString()), context)
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

#define UNIT_TEST_IMPL(F, FF)                                          \
    UNIT_TEST_CHECK_TEST_IS_DECLARED_ONLY_ONCE(F) {                    \
        NUnitTest::TTestContext context(this->TTestBase::Processor()); \
        if (this->CheckAccessTest((#F))) {                             \
            try {                                                      \
                UNIT_TEST_RUN(F, FF, context)                          \
            } catch (const ::NUnitTest::TAssertException&) {           \
            } catch (const yexception& e) {                            \
                CATCH_REACTION_BT((#F), e, &context);                  \
            } catch (const std::exception& e) {                        \
                CATCH_REACTION((#F), e, &context);                     \
            } catch (...) {                                            \
                this->AddError("non-std exception!", &context);        \
            }                                                          \
            this->Finish((#F), &context);                              \
        }                                                              \
    }

#define UNIT_TEST(F) UNIT_TEST_IMPL(F, false)

#define UNIT_FORKED_TEST(F) UNIT_TEST_IMPL(F, true)

#define UNIT_TEST_EXCEPTION(F, E)                                                                                      \
    /* main process with "--fork-tests" flag treats exceptions as errors - it's result of forked test run */           \
    if (this->GetForkTests() && !this->GetIsForked()) {                                                                \
        UNIT_TEST_IMPL(F, false);                                                                                      \
        /* forked process (or main without "--fork-tests") treats some exceptions as success - it's exception test! */ \
    } else {                                                                                                           \
        NUnitTest::TTestContext context(this->TTestBase::Processor());                                                 \
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
        ::NUnitTest::NPrivate::RaiseError(R, ::TStringBuilder() << R << " at " << __LOCATION__ << ", " << __PRETTY_FUNCTION__ << ": " << M, true); \
    } while (false)

#define UNIT_FAIL_NONFATAL_IMPL(R, M)                                                                                                             \
    do {                                                                                                                                          \
        ::NUnitTest::NPrivate::RaiseError(R, ::TStringBuilder() << R << " at " << __LOCATION__ << ", " << __PRETTY_FUNCTION__ << ": " << M, false); \
    } while (false)

#define UNIT_FAIL(M) UNIT_FAIL_IMPL("forced failure", M)
#define UNIT_FAIL_NONFATAL(M) UNIT_FAIL_NONFATAL_IMPL("forced failure", M)

//types
#define UNIT_ASSERT_TYPES_EQUAL(A, B)                                                                                                                                  \
    do {                                                                                                                                                               \
        if (!std::is_same<A, B>::value) {                                                                                                                              \
            UNIT_FAIL_IMPL("types equal assertion failed", (::TStringBuilder() << #A << " (" << TypeName<A>() << ") != " << #B << " (" << TypeName<B>() << ")").data()); \
        }                                                                                                                                                              \
    } while (false)

//doubles
// UNIT_ASSERT_DOUBLES_EQUAL_DEPRECATED* macros do not handle NaNs correctly (see IGNIETFERRO-1419) and are for backward compatibility
// only. Consider switching to regular UNIT_ASSERT_DOUBLES_EQUAL* macros if you're still using the deprecated version.
#define UNIT_ASSERT_DOUBLES_EQUAL_DEPRECATED_C(E, A, D, C)                                                     \
    do {                                                                                                       \
        if (std::abs((E) - (A)) > (D)) {                                                                       \
            const auto _es = ToString((long double)(E));                                                       \
            const auto _as = ToString((long double)(A));                                                       \
            const auto _ds = ToString((long double)(D));                                                       \
            auto&& failMsg = Sprintf("std::abs(%s - %s) > %s %s", _es.data(), _as.data(), _ds.data(), (::TStringBuilder() << C).data()); \
            UNIT_FAIL_IMPL("assertion failure", failMsg);                                                      \
        }                                                                                                      \
    } while (false)

#define UNIT_ASSERT_DOUBLES_EQUAL_DEPRECATED(E, A, D) UNIT_ASSERT_DOUBLES_EQUAL_DEPRECATED_C(E, A, D, "")

#define UNIT_ASSERT_DOUBLES_EQUAL_C(E, A, D, C)                                                                                        \
    do {                                                                                                                               \
        const auto _ed = (E);                                                                                                          \
        const auto _ad = (A);                                                                                                          \
        const auto _dd = (D);                                                                                                          \
        if (std::isnan((long double)_ed) && !std::isnan((long double)_ad)) {                                                           \
            const auto _as = ToString((long double)_ad);                                                                               \
            auto&& failMsg = Sprintf("expected NaN, got %s %s", _as.data(), (::TStringBuilder() << C).data());                           \
            UNIT_FAIL_IMPL("assertion failure", failMsg);                                                                              \
        }                                                                                                                              \
        if (!std::isnan((long double)_ed) && std::isnan((long double)_ad)) {                                                           \
            const auto _es = ToString((long double)_ed);                                                                               \
            auto&& failMsg = Sprintf("expected %s, got NaN %s", _es.data(), (::TStringBuilder() << C).data());                           \
            UNIT_FAIL_IMPL("assertion failure", failMsg);                                                                              \
        }                                                                                                                              \
        if (std::abs((_ed) - (_ad)) > (_dd)) {                                                                                         \
            const auto _es = ToString((long double)_ed);                                                                               \
            const auto _as = ToString((long double)_ad);                                                                               \
            const auto _ds = ToString((long double)_dd);                                                                               \
            auto&& failMsg = Sprintf("std::abs(%s - %s) > %s %s", _es.data(), _as.data(), _ds.data(), (::TStringBuilder() << C).data()); \
            UNIT_FAIL_IMPL("assertion failure", failMsg);                                                                              \
        }                                                                                                                              \
    } while (false)

#define UNIT_ASSERT_DOUBLES_EQUAL(E, A, D) UNIT_ASSERT_DOUBLES_EQUAL_C(E, A, D, "")

//strings
#define UNIT_ASSERT_STRINGS_EQUAL_C(A, B, C)                                                                 \
    do {                                                                                                     \
        const TString _a(A); /* NOLINT(performance-unnecessary-copy-initialization) */                       \
        const TString _b(B); /* NOLINT(performance-unnecessary-copy-initialization) */                       \
        if (_a != _b) {                                                                                      \
            auto&& failMsg = Sprintf("%s != %s %s", ToString(_a).data(), ToString(_b).data(), (::TStringBuilder() << C).data()); \
            UNIT_FAIL_IMPL("strings equal assertion failed", failMsg);                                       \
        }                                                                                                    \
    } while (false)

#define UNIT_ASSERT_STRINGS_EQUAL(A, B) UNIT_ASSERT_STRINGS_EQUAL_C(A, B, "")

#define UNIT_ASSERT_STRING_CONTAINS_C(A, B, C)                                                                                  \
    do {                                                                                                                        \
        const TString _a(A); /* NOLINT(performance-unnecessary-copy-initialization) */                                          \
        const TString _b(B); /* NOLINT(performance-unnecessary-copy-initialization) */                                          \
        if (!_a.Contains(_b)) {                                                                                                 \
            auto&& msg = Sprintf("\"%s\" does not contain \"%s\", %s", ToString(_a).data(), ToString(_b).data(), (::TStringBuilder() << C).data()); \
            UNIT_FAIL_IMPL("strings contains assertion failed", msg);                                                           \
        }                                                                                                                       \
    } while (false)

#define UNIT_ASSERT_STRING_CONTAINS(A, B) UNIT_ASSERT_STRING_CONTAINS_C(A, B, "")

#define UNIT_ASSERT_NO_DIFF(A, B)                                                                                                              \
    do {                                                                                                                                       \
        const TString _a(A); /* NOLINT(performance-unnecessary-copy-initialization) */                                                         \
        const TString _b(B); /* NOLINT(performance-unnecessary-copy-initialization) */                                                         \
        if (_a != _b) {                                                                                                                        \
            UNIT_FAIL_IMPL("strings (" #A ") and (" #B ") are different", Sprintf("\n%s", ::NUnitTest::ColoredDiff(_a, _b, " \t\n.,:;'\"").data())); \
        }                                                                                                                                      \
    } while (false)

//strings
#define UNIT_ASSERT_STRINGS_UNEQUAL_C(A, B, C)                                                           \
    do {                                                                                                 \
        const TString _a(A); /* NOLINT(performance-unnecessary-copy-initialization) */                   \
        const TString _b(B); /* NOLINT(performance-unnecessary-copy-initialization) */                   \
        if (_a == _b) {                                                                                  \
            auto&& msg = Sprintf("%s == %s %s", ToString(_a).data(), ToString(_b).data(), (::TStringBuilder() << C).data()); \
            UNIT_FAIL_IMPL("strings unequal assertion failed", msg);                                     \
        }                                                                                                \
    } while (false)

#define UNIT_ASSERT_STRINGS_UNEQUAL(A, B) UNIT_ASSERT_STRINGS_UNEQUAL_C(A, B, "")

//bool
#define UNIT_ASSERT_C(A, C)                                                                             \
    do {                                                                                                \
        if (!(A)) {                                                                                     \
            UNIT_FAIL_IMPL("assertion failed", Sprintf("(%s) %s", #A, (::TStringBuilder() << C).data())); \
        }                                                                                               \
    } while (false)

#define UNIT_ASSERT(A) UNIT_ASSERT_C(A, "")

//general
#define UNIT_ASSERT_EQUAL_C(A, B, C)                                                                                  \
    do {                                                                                                              \
        if (!((A) == (B))) { /* NOLINT(readability-container-size-empty) */                                           \
            UNIT_FAIL_IMPL("equal assertion failed", Sprintf("%s == %s %s", #A, #B, (::TStringBuilder() << C).data())); \
        }                                                                                                             \
    } while (false)

#define UNIT_ASSERT_EQUAL(A, B) UNIT_ASSERT_EQUAL_C(A, B, "")

#define UNIT_ASSERT_UNEQUAL_C(A, B, C)                                                                                 \
    do {                                                                                                               \
        if ((A) == (B)) {  /* NOLINT(readability-container-size-empty) */                                              \
            UNIT_FAIL_IMPL("unequal assertion failed", Sprintf("%s != %s %s", #A, #B, (::TStringBuilder() << C).data()));\
        }                                                                                                              \
    } while (false)

#define UNIT_ASSERT_UNEQUAL(A, B) UNIT_ASSERT_UNEQUAL_C(A, B, "")

#define UNIT_ASSERT_LT_C(A, B, C)                                                                                        \
    do {                                                                                                                 \
        if (!((A) < (B))) {                                                                                              \
            UNIT_FAIL_IMPL("less-than assertion failed", Sprintf("%s < %s %s", #A, #B, (::TStringBuilder() << C).data())); \
        }                                                                                                                \
    } while (false)

#define UNIT_ASSERT_LT(A, B) UNIT_ASSERT_LT_C(A, B, "")

#define UNIT_ASSERT_LE_C(A, B, C)                                                                                             \
    do {                                                                                                                      \
        if (!((A) <= (B))) {                                                                                                  \
            UNIT_FAIL_IMPL("less-or-equal assertion failed", Sprintf("%s <= %s %s", #A, #B, (::TStringBuilder() << C).data())); \
        }                                                                                                                     \
    } while (false)

#define UNIT_ASSERT_LE(A, B) UNIT_ASSERT_LE_C(A, B, "")

#define UNIT_ASSERT_GT_C(A, B, C)                                                                                           \
    do {                                                                                                                    \
        if (!((A) > (B))) {                                                                                                 \
            UNIT_FAIL_IMPL("greater-than assertion failed", Sprintf("%s > %s %s", #A, #B, (::TStringBuilder() << C).data())); \
        }                                                                                                                   \
    } while (false)

#define UNIT_ASSERT_GT(A, B) UNIT_ASSERT_GT_C(A, B, "")

#define UNIT_ASSERT_GE_C(A, B, C)                                                                        \
    do { \
        if (!((A) >= (B))) {                                                                                    \
            UNIT_FAIL_IMPL("greater-or-equal assertion failed", Sprintf("%s >= %s %s", #A, #B, (::TStringBuilder() << C).data())); \
        } \
    } while (false)

#define UNIT_ASSERT_GE(A, B) UNIT_ASSERT_GE_C(A, B, "")

#define UNIT_CHECK_GENERATED_EXCEPTION_C(A, E, C)                                            \
    do {                                                                                     \
        try {                                                                                \
            (void)(A);                                                                       \
        } catch (const ::NUnitTest::TAssertException&) {                                     \
            throw;                                                                           \
        } catch (const E&) {                                                                 \
            break;                                                                           \
        }                                                                                    \
        UNIT_ASSERT_C(0, "Exception hasn't been thrown, but it should have happened " << C); \
    } while (false)

#define UNIT_CHECK_GENERATED_EXCEPTION(A, E) UNIT_CHECK_GENERATED_EXCEPTION_C(A, E, "")

#define UNIT_CHECK_GENERATED_NO_EXCEPTION_C(A, E, C)                                             \
    do {                                                                                         \
        try {                                                                                    \
            (void)(A);                                                                           \
        } catch (const ::NUnitTest::TAssertException&) {                                         \
            throw;                                                                               \
        } catch (const E&) {                                                                     \
            UNIT_ASSERT_C(0, "Exception has been thrown, but it shouldn't have happened " << C); \
        }                                                                                        \
    } while (false)

#define UNIT_CHECK_GENERATED_NO_EXCEPTION(A, E) UNIT_CHECK_GENERATED_NO_EXCEPTION_C(A, E, "and exception message is:\n" << CurrentExceptionMessage())

// Same as UNIT_ASSERT_EXCEPTION_SATISFIES but prints additional string C when nothing was thrown
#define UNIT_ASSERT_EXCEPTION_SATISFIES_C(A, E, pred, C)   \
    do {                                                                        \
        bool _thrown = false;                                                   \
        try {                                                                   \
            (void)(A);                                                          \
        } catch (const ::NUnitTest::TAssertException&) {                        \
            throw;                                                              \
        } catch (const E& e) {                                                  \
            _thrown = true;                                                     \
            UNIT_ASSERT_C(pred(e), "Exception does not satisfy predicate '"     \
                                << #pred << "'");                               \
        } catch (...) {                                                         \
            _thrown = true;                                                     \
            UNIT_FAIL_IMPL("exception assertion failed",                        \
                           #A << " did not throw " << #E                        \
                              << ", but threw other exception "                 \
                              << "with message:\n"                              \
                              << CurrentExceptionMessage());                    \
        }                                                                       \
        if (!_thrown) {                                                         \
            UNIT_FAIL_IMPL("exception assertion failed",                        \
                           #A << " did not throw any exception"                 \
                              << " (expected " << #E << ") " << C);             \
        }                                                                       \
    } while (false)

// Assert that a specific exception is thrown and satisfies predicate pred(e), where e is the exception instance.
// Example:
//      UNIT_ASSERT_EXCEPTION_SATISFIES(MakeRequest(invalidData), TError,
//          [](const TError& e){ return e.Status == HTTP_BAD_REQUEST; })
// This code validates that MakeRequest with invalidData throws TError with code 400.
#define UNIT_ASSERT_EXCEPTION_SATISFIES(A, E, pred) \
    UNIT_ASSERT_EXCEPTION_SATISFIES_C(A, E, pred, "")

// Same as UNIT_ASSERT_EXCEPTION_CONTAINS but prints additional string C when nothing was thrown
#define UNIT_ASSERT_EXCEPTION_CONTAINS_C(A, E, substr, C)                   \
    do {                                                                    \
        const TString _substr{substr};                                      \
        UNIT_ASSERT_EXCEPTION_SATISFIES_C(A, E,                             \
            [&_substr](const E&){                                           \
                if (!_substr.empty()) {                                     \
                    auto cure = CurrentExceptionMessage() ; \
                    UNIT_ASSERT_C(cure.Contains(_substr),                   \
                                  "Exception message does not contain \""   \
                                      << _substr << "\".\n"                 \
                                      << "Exception message: "              \
                                      << cure);        \
                }                                                           \
                return true;                                                \
            },                                                              \
            C);                                                             \
    } while (false)

// Assert that a specific exception is thrown and CurrentExceptionMessage() contains substr
#define UNIT_ASSERT_EXCEPTION_CONTAINS(A, E, substr) \
    UNIT_ASSERT_EXCEPTION_CONTAINS_C(A, E, substr, "")

// Same as UNIT_ASSERT_EXCEPTION but prints additional string C when nothing was thrown
#define UNIT_ASSERT_EXCEPTION_C(A, E, C) UNIT_ASSERT_EXCEPTION_SATISFIES_C(A, E, [](const E&){ return true; }, C)

// Assert that a specific exception is thrown
#define UNIT_ASSERT_EXCEPTION(A, E) UNIT_ASSERT_EXCEPTION_C(A, E, "")

#define UNIT_ASSERT_NO_EXCEPTION_RESULT_C(A, C)                 \
    [&] () mutable -> decltype(A) {                             \
        static_assert(!std::is_void_v<decltype(A)>);            \
        try { return (A); }                                     \
        catch (const ::NUnitTest::TAssertException&) { throw; } \
        catch (...) {                                           \
            UNIT_FAIL_IMPL(                                     \
                "exception-free assertion failed",              \
                Sprintf("%s throws %s\nException message: %s",  \
                    #A, (::TStringBuilder() << C).data(),       \
                    CurrentExceptionMessage().data()));         \
            return decltype(A){};                               \
        }                                                       \
    }()

#define UNIT_ASSERT_NO_EXCEPTION_RESULT(A) UNIT_ASSERT_NO_EXCEPTION_RESULT_C(A, "")

#define UNIT_ASSERT_NO_EXCEPTION_C(A, C)                                                                                                                                 \
    do {                                                                                                                                                                 \
        try {                                                                                                                                                            \
            (void)(A);                                                                                                                                                   \
        } catch (const ::NUnitTest::TAssertException&) {                                                                                                                 \
            throw;                                                                                                                                                       \
        } catch (...) {                                                                                                                                                  \
            UNIT_FAIL_IMPL("exception-free assertion failed", Sprintf("%s throws %s\nException message: %s", #A, (::TStringBuilder() << C).data(), CurrentExceptionMessage().data())); \
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
            auto&& failMsg = Sprintf("(%s %s %s) failed: (%s %s %s) %s", #A, EQstr, #B, _as.data(), NEQstr, _bs.data(), (::TStringBuilder() << C).data()); \
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
                               << CurrentExceptionMessage());            \
        }                                                                \
        if (!checker.Failed()) {                                         \
            UNIT_FAIL_IMPL("fail test assertion failure",                \
                           "code is expected to generate test failure"); \
        }                                                                \
    } while (false)

#define UNIT_ASSERT_TEST_FAILS(A) UNIT_ASSERT_TEST_FAILS_C(A, "")

#define UNIT_ADD_METRIC(name, value) ut_context.Metrics[name] = value

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
        explicit TTestFactory(ITestSuiteProcessor* processor);

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

    struct TBaseTestCase {
        // NOTE: since EACH test case is instantiated for listing tests, its
        // ctor/dtor are not the best place to do heavy preparations in test fixtures.
        //
        // Consider using SetUp()/TearDown() methods instead

        inline TBaseTestCase()
            : TBaseTestCase(nullptr, nullptr, false)
        {
        }

        inline TBaseTestCase(const char* name, std::function<void(TTestContext&)> body, bool forceFork)
            : Name_(name)
            , Body_(std::move(body))
            , ForceFork_(forceFork)
        {
        }

        virtual ~TBaseTestCase() = default;

        // Each test case is executed in 3 steps:
        //
        // 1. SetUp() (from fixture)
        // 2. Execute_() (test body from Y_UNIT_TEST macro)
        // 3. TearDown() (from fixture)
        //
        // Both SetUp() and TearDown() may use UNIT_* check macros and are only
        // called when the test is executed.

        virtual void SetUp(TTestContext& /* context */) {
        }

        virtual void TearDown(TTestContext& /* context */) {
        }

        virtual void Execute_(TTestContext& context) {
            Body_(context);
        }

        const char* Name_;
        std::function<void(TTestContext&)> Body_;
        bool ForceFork_;
    };

    using TBaseFixture = TBaseTestCase;

    // Class for checking that code raises unittest failure
    class TUnitTestFailChecker {
    public:
        struct TInvokeGuard {
            explicit TInvokeGuard(TUnitTestFailChecker& parent)
                : Parent(&parent)
            {
                Parent->SetHandler();
            }

            TInvokeGuard(TInvokeGuard&& guard) noexcept
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
    static const ::NUnitTest::TTestBaseFactory<T> Y_GENERATE_UNIQUE_ID(UTREG_);

#define Y_UNIT_TEST_SUITE_IMPL_F(N, T, F)                                                                          \
    namespace NTestSuite##N {                                                                                           \
        class TCurrentTestCase: public F {                                                                              \
        };                                                                                                              \
        class TCurrentTest: public T {                                                                                  \
        private:                                                                                                        \
            typedef std::function<THolder<NUnitTest::TBaseTestCase>()> TTestCaseFactory;                                \
            typedef TVector<TTestCaseFactory> TTests;                                                                   \
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
            static void AddTest(const char* name,                                                                       \
                const std::function<void(NUnitTest::TTestContext&)>& body, bool forceFork)                              \
            {                                                                                                           \
                Tests().emplace_back([=]{ return MakeHolder<NUnitTest::TBaseTestCase>(name, body, forceFork); });       \
            }                                                                                                           \
                                                                                                                        \
            static void AddTest(TTestCaseFactory testCaseFactory) {                                                     \
                Tests().push_back(std::move(testCaseFactory));                                                          \
            }                                                                                                           \
                                                                                                                        \
            virtual void Execute() {                                                                                    \
                this->AtStart();                                                                                        \
                this->GlobalSuiteSetUp();                                                                               \
                for (TTests::iterator it = Tests().begin(), ie = Tests().end(); it != ie; ++it) {                       \
                    const auto i = (*it)();                                                                             \
                    if (!this->CheckAccessTest(i->Name_)) {                                                             \
                        continue;                                                                                       \
                    }                                                                                                   \
                    NUnitTest::TTestContext context(this->TTestBase::Processor());                                      \
                    try {                                                                                               \
                        this->BeforeTest(i->Name_);                                                                     \
                        {                                                                                               \
                            TCleanUp cleaner(this);                                                                     \
                            auto testCase = [this, &i, &context] {                                                      \
                                Y_DEFER {                                                                               \
                                    try {                                                                               \
                                        i->TearDown(context);                                                           \
                                    } catch (const ::NUnitTest::TAssertException&) {                                    \
                                    } catch (const yexception& e) {                                                     \
                                        CATCH_REACTION_BT(i->Name_, e, &context);                                       \
                                    } catch (const std::exception& e) {                                                 \
                                        CATCH_REACTION(i->Name_, e, &context);                                          \
                                    } catch (...) {                                                                     \
                                        this->AddError("non-std exception!", &context);                                 \
                                    }                                                                                   \
                                };                                                                                      \
                                i->SetUp(context);                                                                      \
                                i->Execute_(context);                                                                   \
                            };                                                                                          \
                            this->T::Run(testCase, StaticName(), i->Name_, i->ForceFork_);                              \
                        }                                                                                               \
                    } catch (const ::NUnitTest::TAssertException&) {                                                    \
                    } catch (const yexception& e) {                                                                     \
                        CATCH_REACTION_BT(i->Name_, e, &context);                                                       \
                    } catch (const std::exception& e) {                                                                 \
                        CATCH_REACTION(i->Name_, e, &context);                                                          \
                    } catch (...) {                                                                                     \
                        this->AddError("non-std exception!", &context);                                                 \
                    }                                                                                                   \
                    this->Finish(i->Name_, &context);                                                                   \
                }                                                                                                       \
                this->GlobalSuiteTearDown();                                                                            \
                this->AtEnd();                                                                                          \
            }                                                                                                           \
        };                                                                                                              \
        UNIT_TEST_SUITE_REGISTRATION(TCurrentTest)                                                                      \
    }                                                                                                                   \
    namespace NTestSuite##N

#define Y_UNIT_TEST_SUITE_IMPL(N, T) Y_UNIT_TEST_SUITE_IMPL_F(N, T, ::NUnitTest::TBaseTestCase)
#define Y_UNIT_TEST_SUITE(N) Y_UNIT_TEST_SUITE_IMPL(N, TTestBase)
#define Y_UNIT_TEST_SUITE_F(N, F) Y_UNIT_TEST_SUITE_IMPL_F(N, TTestBase, F)
#define RUSAGE_UNIT_TEST_SUITE(N) Y_UNIT_TEST_SUITE_IMPL(N, NUnitTest::TRusageTest, ::NUnitTest::TBaseTestCase)

#define Y_UNIT_TEST_IMPL_REGISTER(N, FF, F)            \
    struct TTestCase##N : public F {                        \
        TTestCase##N()                                      \
        {                                                   \
            Name_ = #N;                                     \
            ForceFork_ = FF;                                \
        }                                                   \
        static THolder<NUnitTest::TBaseTestCase> Create() { \
            return ::MakeHolder<TTestCase##N>();            \
        }                                                   \
        void Execute_(NUnitTest::TTestContext&) override;   \
    };                                                      \
    struct TTestRegistration##N {                           \
        TTestRegistration##N() {                            \
            TCurrentTest::AddTest(TTestCase##N::Create);    \
        }                                                   \
    };                                                      \
    static const TTestRegistration##N testRegistration##N;

#define Y_UNIT_TEST_IMPL(N, FF, F)      \
    Y_UNIT_TEST_IMPL_REGISTER(N, FF, F) \
    void TTestCase##N::Execute_(NUnitTest::TTestContext& ut_context Y_DECLARE_UNUSED)

#define Y_UNIT_TEST(N) Y_UNIT_TEST_IMPL(N, false, TCurrentTestCase)
#define Y_UNIT_TEST_F(N, F) Y_UNIT_TEST_IMPL(N, false, F)
#define SIMPLE_UNIT_FORKED_TEST(N) Y_UNIT_TEST_IMPL(N, true, TCurrentTestCase)

#define Y_UNIT_TEST_SUITE_IMPLEMENTATION(N) \
    namespace NTestSuite##N

#define Y_UNIT_TEST_DECLARE(N) \
    struct TTestCase##N

#define Y_UNIT_TEST_FRIEND(N, T) \
    friend NTestSuite##N::TTestCase##T \

    TString RandomString(size_t len, ui32 seed = 0);
}

using ::NUnitTest::TTestBase;
