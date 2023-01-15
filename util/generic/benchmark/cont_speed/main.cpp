#include <library/cpp/testing/benchmark/bench.h>

#include <util/generic/xrange.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/buffer.h>

template <class C>
Y_NO_INLINE void Run(const C& c) {
    for (size_t i = 0; i < c.size(); ++i) {
        Y_DO_NOT_OPTIMIZE_AWAY(c[i]);
    }
}

template <class C>
void Do(size_t len, auto& iface) {
    C c(len, 0);

    for (auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        Run(c);
    }
}

Y_CPU_BENCHMARK(TVector10, iface) {
    Do<TVector<char>>(10, iface);
}

Y_CPU_BENCHMARK(TVector100, iface) {
    Do<TVector<char>>(100, iface);
}

Y_CPU_BENCHMARK(TVector1000, iface) {
    Do<TVector<char>>(1000, iface);
}

Y_CPU_BENCHMARK(TString10, iface) {
    Do<TString>(10, iface);
}

Y_CPU_BENCHMARK(TString100, iface) {
    Do<TString>(100, iface);
}

Y_CPU_BENCHMARK(TString1000, iface) {
    Do<TString>(1000, iface);
}

Y_CPU_BENCHMARK(StdString10, iface) {
    Do<std::string>(10, iface);
}

Y_CPU_BENCHMARK(StdString100, iface) {
    Do<std::string>(100, iface);
}

Y_CPU_BENCHMARK(StdString1000, iface) {
    Do<std::string>(1000, iface);
}

struct TBuf: public TBuffer {
    TBuf(size_t len, char v) {
        for (size_t i = 0; i < len; ++i) {
            Append(v);
        }
    }

    inline const auto& operator[](size_t i) const noexcept {
        return *(data() + i);
    }
};

Y_CPU_BENCHMARK(TBuffer10, iface) {
    Do<TBuf>(10, iface);
}

Y_CPU_BENCHMARK(TBuffer100, iface) {
    Do<TBuf>(100, iface);
}

Y_CPU_BENCHMARK(TBuffer1000, iface) {
    Do<TBuf>(1000, iface);
}

struct TArr {
    inline TArr(size_t len, char ch)
        : A(new char[len])
        , L(len)
    {
        for (size_t i = 0; i < L; ++i) {
            A[i] = ch;
        }
    }

    inline const auto& operator[](size_t i) const noexcept {
        return A[i];
    }

    inline size_t size() const noexcept {
        return L;
    }

    char* A;
    size_t L;
};

Y_CPU_BENCHMARK(Pointer10, iface) {
    Do<TArr>(10, iface);
}

Y_CPU_BENCHMARK(Pointer100, iface) {
    Do<TArr>(100, iface);
}

Y_CPU_BENCHMARK(Pointer1000, iface) {
    Do<TArr>(1000, iface);
}
