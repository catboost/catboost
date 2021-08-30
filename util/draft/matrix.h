#pragma once

#include <util/generic/noncopyable.h>
#include <util/system/yassert.h>
#include <util/system/defaults.h>
#include <string.h>

template <typename T>
class TMatrix: TNonCopyable {
    // Constructor/Destructor
public:
    TMatrix()
        : Buf(nullptr)
        , Arr(nullptr)
        , M(0)
        , N(0)
        , BufSize(0)
    {
    }

    TMatrix(size_t m, size_t n)
        : Buf(new T[m * n])
        , Arr(Buf)
        , M(m)
        , N(n)
        , BufSize(m * n)
    {
    }

    TMatrix(size_t m, size_t n, T* buf)
        : Buf(nullptr)
        , Arr(buf)
        , M(m)
        , N(n)
        , BufSize(m * n)
    {
    }

    ~TMatrix() {
        delete[] Buf;
    }

    // Properties/Methods
public:
    void Clear() {
        M = N = 0;
    }

    void ReDim(size_t m, size_t n) {
        Y_ASSERT(m >= 1 && n >= 1);
        size_t newSize = m * n;
        if (newSize > BufSize) {
            T* newBuf = new T[newSize];
            delete[] Buf;
            Arr = Buf = newBuf;
            BufSize = newSize;
        }
        M = m;
        N = n;
    }

    size_t Width() const {
        return N;
    }

    size_t Height() const {
        return M;
    }

    // Access element matrix[i][j]
    T* operator[](size_t i) {
        Y_ASSERT(i >= 0 && i < M);
        return Arr + i * N;
    }

    // Access element matrix[i][j]
    const T* operator[](size_t i) const {
        Y_ASSERT(i >= 0 && i < M);
        return Arr + i * N;
    }

    // Access element matrix(i, j)
    T& operator()(size_t i, size_t j) {
        Y_ASSERT(i >= 0 && i < M && j >= 0 && j < N);
        return Arr[i * N + j];
    }

    // Access element matrix(i, j)
    const T& operator()(size_t i, size_t j) const {
        Y_ASSERT(i >= 0 && i < M && j >= 0 && j < N);
        return Arr[i * N + j];
    }

    void Zero() {
        memset((void*)Arr, 0, M * N * sizeof(T));
    }

    void Fill(T value) {
        for (T *p = Arr, *end = Arr + M * N; p < end; ++p)
            *p = value;
    }

private:
    T* Buf;
    T* Arr;
    size_t M, N;
    size_t BufSize;
};
