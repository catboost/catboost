#include "ymath.h"

double Exp2(double x) {
    return pow(2.0, x);
}

float Exp2f(float x) {
    return powf(2.0f, x);
}

#ifdef _MSC_VER

double Erf(double x) {
    static constexpr double _M_2_SQRTPI = 1.12837916709551257390;
    static constexpr double eps = 1.0e-7;
    if (fabs(x) >= 3.75) {
        return x > 0 ? 1.0 : -1.0;
    }
    double r = _M_2_SQRTPI * x;
    double f = r;
    for (int i = 1;; ++i) {
        r *= -x * x / i;
        f += r / (2 * i + 1);
        if (fabs(r) < eps * (2 * i + 1)) {
            break;
        }
    }
    return f;
}

#endif // _MSC_VER

double LogGammaImpl(double x) {
    static constexpr double lnSqrt2Pi = 0.91893853320467274178; // log(sqrt(2.0 * PI))
    static constexpr double coeff9 = 1.0 / 1188.0;
    static constexpr double coeff7 = -1.0 / 1680.0;
    static constexpr double coeff5 = 1.0 / 1260.0;
    static constexpr double coeff3 = -1.0 / 360.0;
    static constexpr double coeff1 = 1.0 / 12.0;

    if ((x == 1.0) || (x == 2.0)) {
        return 0.0; // 0! = 1
    }
    double bonus = 0.0;
    while (x < 3.0) {
        bonus -= log(x);
        x += 1.0;
    }
    double lnX = log(x);
    double sqrXInv = 1.0 / (x * x);
    double res = coeff9 * sqrXInv + coeff7;
    res = res * sqrXInv + coeff5;
    res = res * sqrXInv + coeff3;
    res = res * sqrXInv + coeff1;
    res /= x;
    res += x * lnX - x + lnSqrt2Pi - 0.5 * lnX;
    return res + bonus;
}
