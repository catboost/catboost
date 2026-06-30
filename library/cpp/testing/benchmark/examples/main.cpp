#include <library/cpp/testing/benchmark/bench.h>

#include <util/generic/xrange.h>
#include <util/generic/algorithm.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/generic/yexception.h>

Y_CPU_BENCHMARK(F, iface) {
    TVector<size_t> x;

    x.reserve(iface.Iterations());

    for (size_t i = 0; i < iface.Iterations(); ++i) {
        x.push_back(i);
    }
}

Y_CPU_BENCHMARK(EmptyF, iface) {
    (void)iface;
}

Y_CPU_BENCHMARK(AlmostEmptyF, iface) {
    (void)iface;

    TVector<size_t> x;
    x.resize(1);
}

Y_CPU_BENCHMARK(TestThrow, iface) {
    for (size_t i = 0; i < iface.Iterations(); ++i) {
        try {
            ythrow yexception() << i;
        } catch (...) {
            //CurrentExceptionMessage();
        }
    }
}

Y_CPU_BENCHMARK(TestThrowBT, iface) {
    for (size_t i = 0; i < iface.Iterations(); ++i) {
        try {
            ythrow TWithBackTrace<yexception>() << i;
        } catch (...) {
            //CurrentExceptionMessage();
        }
    }
}

Y_CPU_BENCHMARK(TestThrowCatch, iface) {
    for (size_t i = 0; i < iface.Iterations(); ++i) {
        try {
            ythrow yexception() << i;
        } catch (...) {
            Y_DO_NOT_OPTIMIZE_AWAY(CurrentExceptionMessage());
        }
    }
}

Y_CPU_BENCHMARK(TestThrowCatchBT, iface) {
    for (size_t i = 0; i < iface.Iterations(); ++i) {
        try {
            ythrow TWithBackTrace<yexception>() << i;
        } catch (...) {
            Y_DO_NOT_OPTIMIZE_AWAY(CurrentExceptionMessage());
        }
    }
}

Y_CPU_BENCHMARK(TestRobust, iface) {
    if (iface.Iterations() % 100 == 0) {
        usleep(100000);
    }
}

Y_CPU_BENCHMARK(IterationSpeed, iface) {
    const auto n = iface.Iterations();

    for (size_t i = 0; i < n; ++i) {
        Y_DO_NOT_OPTIMIZE_AWAY(i);
    }
}

Y_CPU_BENCHMARK(XRangeSpeed, iface) {
    for (auto i : xrange<size_t>(0, iface.Iterations())) {
        Y_DO_NOT_OPTIMIZE_AWAY(i);
    }
}

Y_NO_INLINE int FFF() {
    return 0;
}

Y_NO_INLINE int FFF(int x) {
    return x;
}

Y_NO_INLINE int FFF(int x, int y) {
    return x + y;
}

Y_NO_INLINE size_t FS1(TStringBuf x) {
    return x.size();
}

Y_NO_INLINE size_t FS1_2(TStringBuf x, TStringBuf y) {
    return x.size() + y.size();
}

Y_NO_INLINE size_t FS2(const TStringBuf& x) {
    return x.size();
}

Y_NO_INLINE size_t FS2_2(const TStringBuf& x, const TStringBuf& y) {
    return x.size() + y.size();
}

Y_CPU_BENCHMARK(FunctionCallCost_StringBufVal1, iface) {
    TStringBuf x;

    for (auto i : xrange<size_t>(0, iface.Iterations())) {
        (void)i;
        NBench::Escape(&x);
        Y_DO_NOT_OPTIMIZE_AWAY(FS1(x));
        NBench::Clobber();
    }
}

Y_CPU_BENCHMARK(FunctionCallCost_StringBufRef1, iface) {
    TStringBuf x;

    for (auto i : xrange<size_t>(0, iface.Iterations())) {
        (void)i;
        NBench::Escape(&x);
        Y_DO_NOT_OPTIMIZE_AWAY(FS2(x));
        NBench::Clobber();
    }
}

Y_CPU_BENCHMARK(FunctionCallCost_StringBufVal2, iface) {
    TStringBuf x;
    TStringBuf y;

    for (auto i : xrange<size_t>(0, iface.Iterations())) {
        (void)i;
        NBench::Escape(&x);
        NBench::Escape(&y);
        Y_DO_NOT_OPTIMIZE_AWAY(FS1_2(x, y));
        NBench::Clobber();
    }
}

Y_CPU_BENCHMARK(FunctionCallCost_StringBufRef2, iface) {
    TStringBuf x;
    TStringBuf y;

    for (auto i : xrange<size_t>(0, iface.Iterations())) {
        (void)i;
        NBench::Escape(&x);
        NBench::Escape(&y);
        Y_DO_NOT_OPTIMIZE_AWAY(FS2_2(x, y));
        NBench::Clobber();
    }
}

Y_CPU_BENCHMARK(FunctionCallCost_NoArg, iface) {
    for (auto i : xrange<size_t>(0, iface.Iterations())) {
        (void)i;
        Y_DO_NOT_OPTIMIZE_AWAY(FFF());
    }
}

Y_CPU_BENCHMARK(FunctionCallCost_OneArg, iface) {
    for (auto i : xrange<size_t>(0, iface.Iterations())) {
        Y_DO_NOT_OPTIMIZE_AWAY(FFF(i));
    }
}

Y_CPU_BENCHMARK(FunctionCallCost_TwoArg, iface) {
    for (auto i : xrange<size_t>(0, iface.Iterations())) {
        Y_DO_NOT_OPTIMIZE_AWAY(FFF(i, i));
    }
}

/* An example of incorrect benchmark. As of r2581591 Clang 3.7 produced following assembly:
 * @code
 *        │      push   %rbp
 *        │      mov    %rsp,%rbp
 *        │      push   %rbx
 *        │      push   %rax
 *        │      mov    (%rdi),%rbx
 *        │      test   %rbx,%rbx
 *        │    ↓ je     25
 *        │      xor    %edi,%edi
 *        │      xor    %esi,%esi
 *        │    → callq  FS1(TBasicStringBuf<char, std::char_traits<char
 *        │      nop
 * 100.00 │20:┌─→dec    %rbx
 *        │   └──jne    20
 *        │25:   add    $0x8,%rsp
 *        │      pop    %rbx
 *        │      pop    %rbp
 *        │    ← retq
 * @endcode
 *
 * So, this benchmark is measuring empty loop!
 */
Y_CPU_BENCHMARK(Incorrect_FunctionCallCost_StringBufVal1, iface) {
    TStringBuf x;

    for (auto i : xrange<size_t>(0, iface.Iterations())) {
        (void)i;
        Y_DO_NOT_OPTIMIZE_AWAY(FS1(x));
    }
}
