#pragma once

enum EWithHeader {
    HOne  /* "one" */,
    HTwo,
    HThree,
};


constexpr unsigned EvalValue(unsigned r, unsigned d) {
    while (r >= 50) {
        r *= d;
    }
    return r;
}

//  enumeration with values that depend on the preprocessor, architecture and constexpr function evaluation
enum class ENontrivialValues {
    A = __LINE__,
    B = EvalValue(1522858842, 13),
    C,
    D = sizeof(int*[A][C]),
};
