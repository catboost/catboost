#pragma once

#include <util/random/mersenne.h>
#include <cmath>

class TRandom {
public:
    explicit TRandom(ui64 seed = 0)
        : Rng(seed)
    {
    }

    ptrdiff_t operator()(ptrdiff_t i) {
        return Rng.GenRand64() % i;
    }

    inline void Advance(ui32 n) {
        for (ui32 i = 0; i < n; ++i) {
            NextUniformL();
        }
    }

    unsigned long NextUniformL() {
        return Rng.GenRand64();
    }

    double NextUniform() {
        return Rng.GenRandReal1();
    }

    ui64 Uniform(ui64 size) {
        return Rng.Uniform(size);
    }

    double NextGaussian() {
        double a = NextUniform();
        double b = NextUniform();
        return sqrt(-2 * log(a)) * cos(2 * M_PI * b);
    }

    double NextGamma(double shape, double scale) {
        double d, c, x, xsquared, v, u;

        if (shape >= 1.0) {
            d = shape - 1.0 / 3.0;
            c = 1.0 / sqrt(9.0 * d);
            for (;;) {
                do {
                    x = NextGaussian();
                    v = 1.0 + c * x;
                } while (v <= 0.0);
                v = v * v * v;
                u = NextUniform();
                xsquared = x * x;
                if (u < 1.0 - .0331 * xsquared * xsquared || log(u) < 0.5 * xsquared + d * (1.0 - v + log(v)))
                    return scale * d * v;
            }
        } else {
            double g = NextGamma(shape + 1.0, 1.0);
            double w = NextUniform();
            return scale * g * pow(w, 1.0 / shape);
        }
    }

    double NextBeta(double a, double b) {
        double u = NextGamma(a, 1.0);
        double v = NextGamma(b, 1.0);
        return u / (u + v);
    }

    double NextPoisson(double alpha) {
        if (alpha > 20.0) {
            float a = sqrt(alpha) * NextGaussian() + alpha;
            while (a < 0) {
                a = sqrt(alpha) * NextGaussian() + alpha;
            }
            return a;
        }
        double p = 1, L = exp(-alpha);
        int k = 0;
        do {
            k++;
            p *= NextUniform();
        } while (p > L);
        return k - 1;
    }

    static ui64 GenerateSeed(const ui64 from) {
        const ui32 itr = 5;
        ui64 seed = from;
        for (ui32 i = 0; i < itr; ++i) {
            seed = 6364136223846793005 * seed + 1442695040888963407;
        }
        return seed;
    }

private:
    TMersenne<ui64> Rng;
};
