#pragma once

#include <util/system/yassert.h>
#include <util/system/defaults.h>
#include <string.h>

template <typename T>
class TMatrix {
    Y_DISABLE_COPY(TMatrix);

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

    TMatrix(int m, int n)
        : Buf(new T[m * n])
        , Arr(Buf)
        , M(m)
        , N(n)
        , BufSize(m * n)
    {
    }

    TMatrix(int m, int n, T* buf)
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

    void ReDim(int m, int n) {
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

    int Width() const {
        return N;
    }

    int Height() const {
        return M;
    }

    // Access element matrix[i][j]
    T* operator[](int i) {
        Y_ASSERT(i >= 0 && i < M);
        return Arr + i * N;
    }

    // Access element matrix[i][j]
    const T* operator[](int i) const {
        Y_ASSERT(i >= 0 && i < M);
        return Arr + i * N;
    }

    // Access element matrix(i, j)
    T& operator()(int i, int j) {
        Y_ASSERT(i >= 0 && i < M && j >= 0 && j < N);
        return Arr[i * N + j];
    }

    // Access element matrix(i, j)
    const T& operator()(int i, int j) const {
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
    int M, N;
    size_t BufSize;
};
