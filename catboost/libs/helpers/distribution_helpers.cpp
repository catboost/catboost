#include "distribution_helpers.h"

#include <library/cpp/fast_exp/fast_exp.h>


namespace NCB {
    double TNormalDistribution::CalcPdf(double x) const {
        auto zValue = fast_exp(-Sqr(x) / 2.0);
        return zValue * INV_SQRT_2PI;
    }

    double TNormalDistribution::CalcPdfDer1(double pdf, double x) const {
        return -x * pdf;
    }

    double TNormalDistribution::CalcPdfDer2(double pdf, double x) const {
        return (std::pow(x, 2) - 1) * pdf;
    }

    double ErrorFunction(const double x) {
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
        if (!IsFinite(expX) || !IsFinite(std::pow(expX, 2))) {
            return 0.0;
        } else {
            return (std::pow(expX, 2) - 3 * expX + 1) * pdf;
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
        if (!IsFinite(expX) || !IsFinite(std::pow(expX, 2))) {
            return 0.0;
        } else {
            return expX / (std::pow(sqrt_denominator, 2));
        }
    }

    double TLogisticDistribution::CalcPdfDer1(double pdf, double x) const {
        const double expX = fast_exp(x);
        return !IsFinite(expX) ? 0.0 : (pdf * (1 - expX) / (1 + expX));
    }

    double TLogisticDistribution::CalcPdfDer2(double pdf, double x) const {
        const double expX = fast_exp(x);
        const double expXSquared = std::pow(expX, 2);
        if (!IsFinite(expX) || !IsFinite(expXSquared)) {
            return 0.0;
        } else {
            return pdf * (expXSquared - 4 * expX + 1) / ((1 + expX) * (1 + expX));
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
