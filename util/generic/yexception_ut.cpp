#include "yexception.h"

static inline void Throw1DontMove() {
    ythrow yexception() << "blabla"; // don't move this line
}

static inline void Throw2DontMove() {
    ythrow yexception() << 1 << " qw " << 12.1; // don't move this line
}

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/algorithm.h>
#include <util/memory/tempbuf.h>
#include <util/random/mersenne.h>
#include <util/stream/output.h>
#include <util/string/subst.h>
#include <util/string/split.h>

#include "yexception_ut.h"

#if defined(_MSC_VER)
    #pragma warning(disable : 4702) /*unreachable code*/
#endif

static void CallbackFun(int i) {
    throw i;
}

static IOutputStream* OUTS = nullptr;

namespace NOuter::NInner {
    void Compare10And20() {
        Y_ENSURE(10 > 20);
    }
} // namespace NOuter::NInner

class TExceptionTest: public TTestBase {
    UNIT_TEST_SUITE(TExceptionTest);
    UNIT_TEST_EXCEPTION(TestException, yexception)
    UNIT_TEST_EXCEPTION(TestLineInfo, yexception)
    UNIT_TEST(TestCurrentExceptionMessageWhenThereisNoException)
    UNIT_TEST(TestFormat1)
    UNIT_TEST(TestRaise1)
    UNIT_TEST(TestVirtuality)
    UNIT_TEST(TestVirtualInheritance)
    UNIT_TEST(TestMixedCode)
    UNIT_TEST(TestBackTrace)
    UNIT_TEST(TestEnsureWithBackTrace1)
    UNIT_TEST(TestEnsureWithBackTrace2)
#ifdef _YNDX_LIBUNWIND_ENABLE_EXCEPTION_BACKTRACE
    UNIT_TEST(TestFormatCurrentException)
#endif
    UNIT_TEST(TestFormatCurrentExceptionWithNoException)
#ifdef _YNDX_LIBUNWIND_ENABLE_EXCEPTION_BACKTRACE
    UNIT_TEST(TestFormatCurrentExceptionWithInvalidBacktraceFormatter)
#endif
    UNIT_TEST(TestRethrowAppend)
    UNIT_TEST(TestMacroOverload)
    UNIT_TEST(TestMessageCrop)
    UNIT_TEST(TestTIoSystemErrorSpecialMethods)
    UNIT_TEST(TestCurrentExceptionTypeNameMethod)
    UNIT_TEST_SUITE_END();

private:
    inline void TestRethrowAppend() {
        try {
            try {
                ythrow yexception() << "it";
            } catch (yexception& e) {
                e << "happens";

                throw;
            }
        } catch (...) {
            UNIT_ASSERT(CurrentExceptionMessage().Contains("ithappens"));
        }
    }

    inline void TestCurrentExceptionMessageWhenThereisNoException() {
        UNIT_ASSERT(CurrentExceptionMessage() == "(NO EXCEPTION)");
    }

    inline void TestBackTrace() {
        try {
            ythrow TWithBackTrace<TIoSystemError>() << "test";
        } catch (...) {
            UNIT_ASSERT(CurrentExceptionMessage().find('\n') != TString::npos);

            return;
        }

        UNIT_ASSERT(false);
    }

    template <typename TException>
    static void EnsureCurrentExceptionHasBackTrace() {
        auto exceptionPtr = std::current_exception();
        UNIT_ASSERT_C(exceptionPtr != nullptr, "No exception");
        try {
            std::rethrow_exception(exceptionPtr);
        } catch (const TException& e) {
            const TBackTrace* bt = e.BackTrace();
            UNIT_ASSERT(bt != nullptr);
        } catch (...) {
            UNIT_ASSERT_C(false, "Unexpected exception type");
        }
    }

    inline void TestEnsureWithBackTrace1() {
        try {
            Y_ENSURE_BT(4 > 6);
        } catch (...) {
            const TString msg = CurrentExceptionMessage();
            UNIT_ASSERT(msg.Contains("4 > 6"));
            UNIT_ASSERT(msg.Contains("\n"));
            EnsureCurrentExceptionHasBackTrace<yexception>();
            return;
        }
        UNIT_ASSERT(false);
    }

    inline void TestEnsureWithBackTrace2() {
        try {
            Y_ENSURE_BT(4 > 6, "custom "
                                   << "message");
        } catch (...) {
            const TString msg = CurrentExceptionMessage();
            UNIT_ASSERT(!msg.Contains("4 > 6"));
            UNIT_ASSERT(msg.Contains("custom message"));
            UNIT_ASSERT(msg.Contains("\n"));
            EnsureCurrentExceptionHasBackTrace<yexception>();
            return;
        }
        UNIT_ASSERT(false);
    }

    // TODO(svkrasnov): the output should be canonized after https://st.yandex-team.ru/YMAKE-103
#ifdef _YNDX_LIBUNWIND_ENABLE_EXCEPTION_BACKTRACE
    void TestFormatCurrentException() {
        try {
            throw std::logic_error("some exception"); // is instance of std::exception
            UNIT_ASSERT(false);
        } catch (...) {
            TString exceptionMessage = FormatCurrentException();
            UNIT_ASSERT(exceptionMessage.Contains("(std::logic_error) some exception"));
            TVector<TString> backtraceStrs = StringSplitter(exceptionMessage).Split('\n');
            UNIT_ASSERT(backtraceStrs.size() > 1);
        }
    }
#endif

    void TestFormatCurrentExceptionWithNoException() {
        UNIT_ASSERT_VALUES_EQUAL(FormatCurrentException(), "(NO EXCEPTION)\n");
    }

#ifdef _YNDX_LIBUNWIND_ENABLE_EXCEPTION_BACKTRACE
    void TestFormatCurrentExceptionWithInvalidBacktraceFormatter() {
        auto invalidFormatter = [](IOutputStream*, void* const*, size_t) {
            Throw2DontMove();
        };
        SetFormatBackTraceFn(invalidFormatter);

        try {
            Throw1DontMove();
            UNIT_ASSERT(false);
        } catch (...) {
            TString expected = "Caught:\n"
                               "(yexception) util/generic/yexception_ut.cpp:4: blabla\n"
                               "Failed to print backtrace: "
                               "(yexception) util/generic/yexception_ut.cpp:8: 1 qw 12.1";
            UNIT_ASSERT_EQUAL(FormatCurrentException(), expected);
        }
        try {
            throw std::logic_error("std exception");
            UNIT_ASSERT(false);
        } catch (...) {
            TString expected = "Caught:\n"
                               "(std::logic_error) std exception\n"
                               "Failed to print backtrace: "
                               "(yexception) util/generic/yexception_ut.cpp:8: 1 qw 12.1";
            UNIT_ASSERT_EQUAL(FormatCurrentException(), expected);
        }
    }
#endif

    inline void TestVirtualInheritance() {
        TStringStream ss;

        OUTS = &ss;

        class TA {
        public:
            inline TA() {
                *OUTS << "A";
            }
        };

        class TB {
        public:
            inline TB() {
                *OUTS << "B";
            }
        };

        class TC: public virtual TB, public virtual TA {
        public:
            inline TC() {
                *OUTS << "C";
            }
        };

        class TD: public virtual TA {
        public:
            inline TD() {
                *OUTS << "D";
            }
        };

        class TE: public TC, public TD {
        public:
            inline TE() {
                *OUTS << "E";
            }
        };

        TE e;

        UNIT_ASSERT_EQUAL(ss.Str(), "BACDE");
    }

    inline void TestVirtuality() {
        try {
            ythrow TFileError() << "1";
            UNIT_ASSERT(false);
        } catch (const TIoException&) {
        } catch (...) {
            UNIT_ASSERT(false);
        }

        try {
            ythrow TFileError() << 1;
            UNIT_ASSERT(false);
        } catch (const TSystemError&) {
        } catch (...) {
            UNIT_ASSERT(false);
        }

        try {
            ythrow TFileError() << '1';
            UNIT_ASSERT(false);
        } catch (const yexception&) {
        } catch (...) {
            UNIT_ASSERT(false);
        }

        try {
            ythrow TFileError() << 1.0;
            UNIT_ASSERT(false);
        } catch (const TFileError&) {
        } catch (...) {
            UNIT_ASSERT(false);
        }
    }

    inline void TestFormat1() {
        try {
            throw yexception() << 1 << " qw " << 12.1;
            UNIT_ASSERT(false);
        } catch (...) {
            const TString err = CurrentExceptionMessage();

            UNIT_ASSERT(err.Contains("1 qw 12.1"));
        }
    }

    static inline void CheckCurrentExceptionContains(const char* message) {
        TString err = CurrentExceptionMessage();
        SubstGlobal(err, '\\', '/'); // remove backslashes from path in message
        UNIT_ASSERT(err.Contains(message));
    }

    inline void TestRaise1() {
        try {
            Throw2DontMove();
            UNIT_ASSERT(false);
        } catch (...) {
            CheckCurrentExceptionContains("util/generic/yexception_ut.cpp:8: 1 qw 12.1");
        }
    }

    inline void TestException() {
        ythrow yexception() << "blablabla";
    }

    inline void TestLineInfo() {
        try {
            Throw1DontMove();
            UNIT_ASSERT(false);
        } catch (...) {
            CheckCurrentExceptionContains("util/generic/yexception_ut.cpp:4: blabla");

            throw;
        }
    }

    //! tests propagation of an exception through C code
    //! @note on some platforms, for example GCC on 32-bit Linux without -fexceptions option,
    //!       throwing an exception from a C++ callback through C code aborts program
    inline void TestMixedCode() {
        const int N = 26082009;
        try {
            TestCallback(&CallbackFun, N);
            UNIT_ASSERT(false);
        } catch (int i) {
            UNIT_ASSERT_VALUES_EQUAL(i, N);
        }
    }

    void TestMacroOverload() {
        try {
            Y_ENSURE(10 > 20);
        } catch (const yexception& e) {
            UNIT_ASSERT(e.AsStrBuf().Contains("10 > 20"));
        }

        try {
            Y_ENSURE(10 > 20, "exception message to search for");
        } catch (const yexception& e) {
            UNIT_ASSERT(e.AsStrBuf().Contains("exception message to search for"));
        }

        try {
            NOuter::NInner::Compare10And20();
        } catch (const yexception& e) {
            UNIT_ASSERT(e.AsStrBuf().Contains("10 > 20"));
        }
    }

    void TestMessageCrop() {
        TTempBuf tmp;
        size_t size = tmp.Size() * 1.5;
        TString s;
        s.reserve(size);
        TMersenne<ui64> generator(42);
        for (int j = 0; j < 50; ++j) {
            for (size_t i = 0; i < size; ++i) {
                s += static_cast<char>('a' + generator() % 26);
            }
            yexception e;
            e << s;
            UNIT_ASSERT_EQUAL(e.AsStrBuf(), s.substr(0, tmp.Size() - 1));
        }
    }

    void TestTIoSystemErrorSpecialMethods() {
        TString testStr{"systemError"};
        TIoSystemError err;
        err << testStr;
        UNIT_ASSERT(err.AsStrBuf().Contains(testStr));

        TIoSystemError errCopy{err};
        UNIT_ASSERT(err.AsStrBuf().Contains(testStr));
        UNIT_ASSERT(errCopy.AsStrBuf().Contains(testStr));

        TIoSystemError errAssign;
        errAssign = err;
        UNIT_ASSERT(err.AsStrBuf().Contains(testStr));
        UNIT_ASSERT(errAssign.AsStrBuf().Contains(testStr));

        TIoSystemError errMove{std::move(errCopy)};
        UNIT_ASSERT(errMove.AsStrBuf().Contains(testStr));

        TIoSystemError errMoveAssign;
        errMoveAssign = std::move(errMove);
        UNIT_ASSERT(errMoveAssign.AsStrBuf().Contains(testStr));
    }
    inline void TestCurrentExceptionTypeNameMethod() {
        // Basic test of getting the correct exception type name.
        try {
            throw std::runtime_error("Test Runtime Error Exception");
        } catch (...) {
            UNIT_ASSERT_STRING_CONTAINS(CurrentExceptionTypeName(), "std::runtime_error");
        }
        // Test when exception has an unusual type. Under Linux it should return "int" and under other OSs "unknown type".
        try {
            throw int(1);
        } catch (...) {
#if defined(_linux_) || defined(_darwin_)
            // On Linux and macOS we use libcxxrt which handles throw integers properly
            UNIT_ASSERT_VALUES_EQUAL(CurrentExceptionTypeName(), "int");
#else
            UNIT_ASSERT_VALUES_EQUAL(CurrentExceptionTypeName(), "unknown type");
#endif
        }
        // Test when the caught exception is rethrown with std::rethrow_exception.
        try {
            throw std::logic_error("Test Logic Error Exception");
        } catch (...) {
            try {
                std::rethrow_exception(std::current_exception());
            } catch (...) {
                UNIT_ASSERT_STRING_CONTAINS(CurrentExceptionTypeName(), "std::logic_error");
            }
        }
        // Test when the caught exception is rethrown with throw; .
        // This test is different from the previous one because of the interaction with cxxabi specifics.
        try {
            throw std::bad_alloc();
        } catch (...) {
            try {
                throw;
            } catch (...) {
                UNIT_ASSERT_STRING_CONTAINS(CurrentExceptionTypeName(), "std::bad_alloc");
            }
        }
        // For exceptions thrown by std::rethrow_exception() a nullptr will be returned by libcxxrt's __cxa_current_exception_type().
        // Adding an explicit test for the case.
        try {
            throw int(1);
        } catch (...) {
            try {
                std::rethrow_exception(std::current_exception());
            } catch (...) {
#ifdef __GLIBCXX__
                UNIT_ASSERT_VALUES_EQUAL(CurrentExceptionTypeName(), "int");
#else
                UNIT_ASSERT_VALUES_EQUAL(CurrentExceptionTypeName(), "unknown type");
#endif
            }
        }
        // Test when int is rethrown with throw; .
        try {
            throw int(1);
        } catch (...) {
            try {
                throw;
            } catch (...) {
#if defined(_linux_) || defined(_darwin_)
                UNIT_ASSERT_VALUES_EQUAL(CurrentExceptionTypeName(), "int");
#else
                UNIT_ASSERT_VALUES_EQUAL(CurrentExceptionTypeName(), "unknown type");
#endif
            }
        }
    }
};

UNIT_TEST_SUITE_REGISTRATION(TExceptionTest);
