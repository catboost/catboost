#pragma once

namespace NCB {
    enum class EDistributionType {
        Normal,
        Logistic,
        Extreme,
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

    class TNormalDistribution final : public IDistribution {
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

    class TExtremeDistribution final : public IDistribution {
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

    class TLogisticDistribution final : public IDistribution {
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
