#pragma once

template <class T, class U>
inline T CeilDivide(T x, U y) {
    return (x + y - 1) / y;
}
