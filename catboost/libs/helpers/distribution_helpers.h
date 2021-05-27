#pragma once

#include <util/generic/string.h>
#include <util/generic/ymath.h>


namespace NCB {
    static constexpr double INV_SQRT_2PI = 0.398942280401432677939946;

    enum class EDistributionType {
        Normal /* "Normal" */,
        Logistic /* "Logistic" */,
        Extreme /* "Extreme" */,
    };

    class IDistribution {
    public:
        virtual double CalcPdf(double x) const = 0;
        virtual double CalcCdf(double x) const = 0;
        virtual double CalcPdfDer1(
            double pdf,
            double x) const = 0;
        virtual double CalcPdfDer2(
            double pdf,
            double x) const = 0;
        virtual EDistributionType GetDistributionType() const = 0;
        virtual ~IDistribution() = default;
    };

    class TNormalDistribution : public IDistribution {
    public:
        virtual double CalcPdf(double x) const override;

        virtual double CalcPdfDer1(
            double pdf,
            double x) const override;

        virtual double CalcPdfDer2(
            double pdf,
            double x) const override;

        virtual double CalcCdf (double x) const override;

        virtual EDistributionType GetDistributionType() const override;
    };

    double ErrorFunction(const double x);

    class TExtremeDistribution : public IDistribution {
    public:
        virtual double CalcPdf(double x) const override;

        virtual double CalcPdfDer1(
            double pdf,
            double x) const override;

        virtual double CalcPdfDer2(
            double pdf,
            double x) const override;

        virtual double CalcCdf (double x) const override;

        virtual EDistributionType GetDistributionType() const override;

        double ErrorFunction(const double x) const;
    };

    class TLogisticDistribution : public IDistribution {
    public:
        virtual double CalcPdf(double x) const override;

        virtual double CalcPdfDer1(
            double pdf,
            double x) const override;

        virtual double CalcPdfDer2(
            double pdf,
            double x) const override;

        virtual double CalcCdf (double x) const override;

        virtual EDistributionType GetDistributionType() const override;

        double ErrorFunction(const double x) const;
    };
}
