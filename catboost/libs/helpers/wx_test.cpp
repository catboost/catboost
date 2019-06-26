#include "wx_test.h"


static double ErrorFunction(const double x) {
    double coeffs[] = {
        -1.26551223,
        1.00002368,
        0.37409196,
        0.09678418,
        -0.18628806,
        0.27886807,
        -1.13520398,
        1.48851587,
        -0.82215223,
        0.17087277
    };

    double t = 1.0 / (1.0 + 0.5 * Abs(x));
    double sum = -x * x;
    double powT = 1.0;
    for (double coef : coeffs) {
        sum += coef * powT;
        powT *= t;
    }
    double tau = t * exp(sum);
    if (x > 0) {
        return 1.0 - tau;
    } else {
        return tau - 1.0;
    }
}

static double NormalCDF(double x) {
    return 0.5 + 0.5 * ErrorFunction(x / sqrt(2.0));
}

//w is Abs(wPlus-wMinus)
// wMinus/wPlus are sums of rank with appropriate sign
static double CalcLevelOfSignificanceWXMPSR(double w, int n) {
    Y_VERIFY(n < 20);
    // The total number of possible outcomes is 2**N
    int numberOfPossibilities = 1 << n;

    // Initialize and loop. The loop-interior will be run 2**N times.
    int countLarger = 0;
    // Generate all distributions of sign over ranks as bit-strings.
    for (int i = 0; i < numberOfPossibilities; i++) {
        double rankSum = 0;
        // Shift "sign" bits out of $i to determine the Sum of Ranks
        for (int j = 0; j < n; j++) {
            int sign = ((i >> j) & 1) ? 1 : - 1;
            rankSum += sign * (j + 1);
        }
        // Count the number of "samples" that have a Sum of Ranks larger or
        // equal to the one found (i.e., >= W).
        if (rankSum >= w) {
            countLarger += 1;
        }
    }

    // The level of significance is the number of outcomes with a
    // sum of ranks equal to or larger than the one found (W)
    // divided by the total number of possible outcomes.
    // The level is doubled to get the two-tailed result.
    return countLarger * 1.0 / numberOfPossibilities;
}


TWxTestResult WxTest(const TVector<double>& baseline, const TVector<double>& test) {
    TVector<double> diffs;

    for (size_t i = 0; i < baseline.size(); ++i) {
        const double i1 = baseline[i];
        const double i2 = test[i];
        const double diff = i1 - i2;
        if (diff != 0) {
            diffs.push_back(diff);
        }
    }

    if (diffs.size() < 2) {
        TWxTestResult result;
        result.PValue = 0.5;
        result.WMinus = result.WPlus = 0;
        return result;
    }

    Sort(diffs.begin(), diffs.end(), [&](double x, double y) { return Abs(x) < Abs(y); });

    double wPlus = 0;
    double wMinus = 0;
    double n = diffs.size();


    for (int i = 0; i < n; ++i) {
        double sum = 0;
        double weight = 0;
        double signPlus = 0;
        double signMinus = 0;

        int j = i;
        for (; j < n && Abs(diffs[j]) == Abs(diffs[i]); ++j) {
            sum += j;
            ++weight;
            signPlus += diffs[j] >= 0;
            signMinus += diffs[j] < 0;
        }

        const double meanRank = sum / weight + 1;
        wPlus += signPlus * meanRank;
        wMinus += signMinus * meanRank;

        i = j - 1;
    }

    TWxTestResult result;
    result.WPlus = wPlus;
    result.WMinus = wMinus;

    const double w = result.WPlus - result.WMinus;
    if (n > 16) {
        double z = w / sqrt(n * (n + 1) * (2 * n + 1) * 1.0 / 6);
        result.PValue = 2 * (1.0 - NormalCDF(Abs(z)));
    } else {
        result.PValue = 2 * CalcLevelOfSignificanceWXMPSR(Abs(w), (int) n);
    }
    return result;
}
