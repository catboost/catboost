#include "stdout_interceptor.h"

#include <util/stream/str.h>

#include <library/cpp/testing/gtest/gtest.h>

#include <Python.h>

#include <thread>
#include <algorithm>

struct TSubinterpreters: ::testing::Test {
    static void SetUpTestSuite() {
        Py_InitializeEx(0);
        EXPECT_TRUE(TPyStdoutInterceptor::SetupInterceptionSupport());
    }
    static void TearDownTestSuite() {
        Py_Finalize();
    }

    static void ThreadPyRun(PyInterpreterState* interp, IOutputStream& pyout, const char* pycode) {
        PyThreadState* state = PyThreadState_New(interp);
        PyEval_RestoreThread(state);

        {
            TPyStdoutInterceptor interceptor{pyout};
            PyRun_SimpleString(pycode);
        }

        PyThreadState_Clear(state);
        PyThreadState_DeleteCurrent();
    }
};

TEST_F(TSubinterpreters, NonSubinterpreterFlowStillWorks) {
    TStringStream pyout;
    TPyStdoutInterceptor interceptor{pyout};

    PyRun_SimpleString("print('Hello World')");
    EXPECT_EQ(pyout.Str(), "Hello World\n");
}

TEST_F(TSubinterpreters, ThreadedSubinterpretersFlowWorks) {
    TStringStream pyout[2];

    PyInterpreterConfig cfg = {
        .use_main_obmalloc = 0,
        .allow_fork = 0,
        .allow_exec = 0,
        .allow_threads = 1,
        .allow_daemon_threads = 0,
        .check_multi_interp_extensions = 1,
        .gil = PyInterpreterConfig_OWN_GIL,
    };

    PyThreadState* mainState = PyThreadState_Get();
    PyThreadState *sub[2] = {nullptr, nullptr};
    Py_NewInterpreterFromConfig(&sub[0], &cfg);
    ASSERT_NE(sub[0], nullptr);
    Py_NewInterpreterFromConfig(&sub[1], &cfg);
    ASSERT_NE(sub[1], nullptr);
    PyThreadState_Swap(mainState);

    PyThreadState* savedState = PyEval_SaveThread();
    std::array<std::thread, 2> threads{
        std::thread{ThreadPyRun, sub[0]->interp, std::ref(pyout[0]), "print('Hello Thread 0')"},
        std::thread{ThreadPyRun, sub[1]->interp, std::ref(pyout[1]), "print('Hello Thread 1')"}
    };
    std::ranges::for_each(threads, &std::thread::join);
    PyEval_RestoreThread(savedState);

    PyThreadState_Swap(sub[0]);
    Py_EndInterpreter(sub[0]);

    PyThreadState_Swap(sub[1]);
    Py_EndInterpreter(sub[1]);

    PyThreadState_Swap(mainState);

    EXPECT_EQ(pyout[0].Str(), "Hello Thread 0\n");
    EXPECT_EQ(pyout[1].Str(), "Hello Thread 1\n");
}
