#include "distribution_helpers.h"

#include <library/cpp/fast_exp/fast_exp.h>

#include <util/generic/ymath.h>


namespace NCB {
    double TNormalDistribution::CalcPdf(double x) const {
        constexpr double INV_SQRT_2PI = 0.398942280401432677939946;
        auto zValue = fast_exp(-Sqr(x) / 2.0);
        return zValue * INV_SQRT_2PI;
    }

    double TNormalDistribution::CalcPdfDer1(double pdf, double x) const {
        return -x * pdf;
    }

    double TNormalDistribution::CalcPdfDer2(double pdf, double x) const {
        return (Sqr(x) - 1) * pdf;
    }

    double ErrorFunction(const double x) {
        return std::erf(x);
    }

    double TNormalDistribution::CalcCdf(double x) const {
        return 0.5 + 0.5 * ErrorFunction(x / sqrt(2.0));
    }

    EDistributionType TNormalDistribution::GetDistributionType() const {
        return EDistributionType::Normal;
    }

    double TExtremeDistribution::CalcPdf(double x) const {
        const double expX = fast_exp(x);
        return !IsFinite(expX) ? 0.0 : (expX * fast_exp(-expX));
    }

    double TExtremeDistribution::CalcPdfDer1(double pdf, double x) const {
        const double expX = fast_exp(x);
        return !IsFinite(expX) ? 0.0 : ((1 - expX) * pdf);
    }

    double TExtremeDistribution::CalcPdfDer2(double pdf, double x) const {
        const double expX = fast_exp(x);
        if (!IsFinite(expX) || !IsFinite(Sqr(expX))) {
            return 0.0;
        } else {
            return (Sqr(expX) - 3 * expX + 1) * pdf;
        }
    }

    double TExtremeDistribution::CalcCdf(double x) const {
        const double expX = fast_exp(x);
        return 1 - fast_exp(-expX);
    }

    EDistributionType TExtremeDistribution::GetDistributionType() const {
        return EDistributionType::Extreme;
    }

    double TLogisticDistribution::CalcPdf(double x) const {
        const double expX = fast_exp(x);
        const double sqrt_denominator = 1 + expX;
        if (!IsFinite(expX) || !IsFinite(Sqr(expX))) {
            return 0.0;
        } else {
            return expX / Sqr(sqrt_denominator);
        }
    }

    double TLogisticDistribution::CalcPdfDer1(double pdf, double x) const {
        const double expX = fast_exp(x);
        return !IsFinite(expX) ? 0.0 : (pdf * (1 - expX) / (1 + expX));
    }

    double TLogisticDistribution::CalcPdfDer2(double pdf, double x) const {
        const double expX = fast_exp(x);
        const double expXSquared = Sqr(expX);
        if (!IsFinite(expX) || !IsFinite(expXSquared)) {
            return 0.0;
        } else {
            return pdf * (expXSquared - 4 * expX + 1) / Sqr(1 + expX);
        }
    }

    double TLogisticDistribution::CalcCdf (double x) const {
        const double expX = fast_exp(x);
        return !IsFinite(expX) ? 1.0 : (expX / (1 + expX));
    }

    EDistributionType TLogisticDistribution::GetDistributionType() const {
        return EDistributionType::Logistic;
    }
}
