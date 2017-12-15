#pragma once

#include <catboost/libs/options/binarization_options.h>

#include <library/grid_creator/binarization.h>
#include <util/system/types.h>
#include <util/generic/algorithm.h>
#include <util/generic/set.h>
#include <util/generic/ymath.h>

namespace NCatboostCuda {
    inline TVector<float> CheckedCopyWithoutNans(const TVector<float>& values, ENanMode nanMode) {
        TVector<float> copy;
        copy.reserve(values.size());
        for (ui32 i = 0; i < values.size(); ++i) {
            const float val = values[i];
            if (IsNan(val)) {
                CB_ENSURE(nanMode != ENanMode::Forbidden, "Error: NaN in features, but NaNs are forbidden");
                continue;
            } else {
                copy.push_back(val);
            }
        }
        return copy;
    }

    class IGridBuilder {
    public:
        virtual ~IGridBuilder() {
        }

        virtual IGridBuilder& AddFeature(const TVector<float>& feature, ui32 borderCount, ENanMode nanMode) = 0;

        virtual const TVector<TVector<float>>& Borders() = 0;

        virtual TVector<float> BuildBorders(const TVector<float>& sortedFeature,
                                            ui32 borderCount) const = 0;
    };

    template <class T>
    class IFactory;

    template <>
    class IFactory<IGridBuilder> {
    public:
        virtual ~IFactory() {
        }

        virtual THolder<IGridBuilder> Create(EBorderSelectionType type) = 0;
    };

    template <class TBinarizer>
    class TGridBuilderBase: public IGridBuilder {
    public:
        TVector<float> BuildBorders(const TVector<float>& sortedFeature, ui32 borderCount) const override {
            TVector<float> copy = CheckedCopyWithoutNans(sortedFeature, ENanMode::Forbidden);
            auto bordersSet = Binarizer.BestSplit(copy, borderCount, true);
            TVector<float> borders(bordersSet.begin(), bordersSet.end());
            Sort(borders.begin(), borders.end());
            return borders;
        }

    private:
        TBinarizer Binarizer;
    };

    template <class TBinarizer>
    class TCpuGridBuilder: public TGridBuilderBase<TBinarizer> {
    public:
        IGridBuilder& AddFeature(const TVector<float>& feature,
                                 ui32 borderCount,
                                 ENanMode nanMode) override {
            TVector<float> sortedFeature = CheckedCopyWithoutNans(feature, nanMode);
            Sort(sortedFeature.begin(), sortedFeature.end());
            auto borders = TGridBuilderBase<TBinarizer>::BuildBorders(sortedFeature, borderCount);
            Result.push_back(std::move(borders));
            return *this;
        }

        const TVector<TVector<float>>& Borders() override {
            return Result;
        }

    private:
        TVector<TVector<float>> Result;
    };

    template <template <class T> class TGridBuilder>
    class TGridBuilderFactory: public IFactory<IGridBuilder> {
    public:
        THolder<IGridBuilder> Create(EBorderSelectionType type) override {
            THolder<IGridBuilder> builder;
            switch (type) {
                case EBorderSelectionType::UniformAndQuantiles: {
                    builder.Reset(new TGridBuilder<NSplitSelection::TMedianPlusUniformBinarizer>());
                    break;
                }
                case EBorderSelectionType::GreedyLogSum: {
                    builder.Reset(new TGridBuilder<NSplitSelection::TMedianInBinBinarizer>());
                    break;
                }
                case EBorderSelectionType::MinEntropy: {
                    builder.Reset(new TGridBuilder<NSplitSelection::TMinEntropyBinarizer>());
                    break;
                }
                case EBorderSelectionType::MaxLogSum: {
                    builder.Reset(new TGridBuilder<NSplitSelection::TMaxSumLogBinarizer>());
                    break;
                }
                case EBorderSelectionType::Median: {
                    builder.Reset(new TGridBuilder<NSplitSelection::TMedianBinarizer>());
                    break;
                }
                case EBorderSelectionType::Uniform: {
                    builder.Reset(new TGridBuilder<NSplitSelection::TUniformBinarizer>());
                    break;
                }
                default: {
                    ythrow yexception() << "Invalid grid builder type!";
                }
            }
            return builder;
        }
    };

    class TBordersBuilder {
    public:
        TBordersBuilder(IFactory<IGridBuilder>& builderFactory,
                        const TVector<float>& values)
            : BuilderFactory(builderFactory)
            , Values(values)
        {
        }

        TVector<float> operator()(const NCatboostOptions::TBinarizationOptions& description) {
            auto builder = BuilderFactory.Create(description.BorderSelectionType);
            const ui32 borderCount = description.NanMode == ENanMode::Forbidden ? description.BorderCount : description.BorderCount - 1;
            CB_ENSURE(borderCount > 0, "Error: border count should be greater than 0. If you have nan-features, border count should be > 1. Got " << description.BorderCount);
            builder->AddFeature(Values, description.BorderCount, description.NanMode);
            return builder->Borders()[0];
        }

    private:
        IFactory<IGridBuilder>& BuilderFactory;
        const TVector<float>& Values;
    };

    using TOnCpuGridBuilderFactory = TGridBuilderFactory<TCpuGridBuilder>;
}
