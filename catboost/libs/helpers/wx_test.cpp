#include "distribution_helpers.h"
#include "exception.h"
#include "wx_test.h"


using namespace NCB;


static double NormalCDF(double x) {
    return 0.5 + 0.5 * ErrorFunction(x / sqrt(2.0));
}

//w is Abs(wPlus-wMinus)
// wMinus/wPlus are sums of rank with appropriate sign
static double CalcLevelOfSignificanceWXMPSR(double w, int n) {
    CB_ENSURE(n < 20, "Size of sample is too large for CalcLevelOfSignificanceWXMPSR");
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
        if (IsFinite(diff) && diff != 0) {
            diffs.push_back(diff);
        }
    }

    if (diffs.size() < 2) {
        TWxTestResult result;
        result.PValue = 0.5;
        result.WMinus = result.WPlus = 0;
        return result;
    }

    StableSort(diffs.begin(), diffs.end(), [&](double x, double y) { return Abs(x) < Abs(y); });

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
