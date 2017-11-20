#pragma once

#include "binarization_config.h"

#include <library/grid_creator/binarization.h>
#include <util/system/types.h>
#include <util/generic/algorithm.h>
#include <util/generic/set.h>
namespace NCatboostCuda
{
    class IGridBuilder
    {
    public:
        virtual ~IGridBuilder()
        {
        }

        virtual IGridBuilder& AddFeature(const TVector<float>& feature, ui32 borderCount) = 0;

        virtual const TVector<TVector<float>>& Borders() = 0;

        virtual TVector<float> BuildBorders(const TVector<float>& sortedFeature,
                                            ui32 borderCount) const = 0;
    };

    template<class T>
    class IFactory;

    template<>
    class IFactory<IGridBuilder>
    {
    public:
        virtual ~IFactory()
        {
        }

        virtual THolder<IGridBuilder> Create(EBorderSelectionType type) = 0;
    };

    template<class TBinarizer>
    class TGridBuilderBase: public IGridBuilder
    {
    public:
        TVector<float> BuildBorders(const TVector<float>& sortedFeature, ui32 borderCount) const override
        {
            TVector<float> copy(sortedFeature.begin(), sortedFeature.end());
            auto bordersSet = Binarizer.BestSplit(copy, borderCount, true);
            TVector<float> borders(bordersSet.begin(), bordersSet.end());
            Sort(borders.begin(), borders.end());
            return borders;
        }

    private:
        TBinarizer Binarizer;
    };

    template<class TBinarizer>
    class TCpuGridBuilder: public TGridBuilderBase<TBinarizer>
    {
    public:
        IGridBuilder& AddFeature(const TVector<float>& feature,
                                 ui32 borderCount) override
        {
            TVector<float> sortedFeature(feature.begin(), feature.end());
            Sort(sortedFeature.begin(), sortedFeature.end());
            auto borders = TGridBuilderBase<TBinarizer>::BuildBorders(sortedFeature, borderCount);
            Result.push_back(std::move(borders));
            return *this;
        }

        const TVector<TVector<float>>& Borders() override
        {
            return Result;
        }

    private:
        TVector<TVector<float>> Result;
    };

    template<template<class T> class TGridBuilder>
    class TGridBuilderFactory: public IFactory<IGridBuilder>
    {
    public:
        THolder<IGridBuilder> Create(EBorderSelectionType type) override
        {
            THolder<IGridBuilder> builder;
            switch (type)
            {
                case EBorderSelectionType::UniformAndQuantiles:
                {
                    builder.Reset(new TGridBuilder<NSplitSelection::TMedianPlusUniformBinarizer>());
                    break;
                }
                case EBorderSelectionType::GreedyLogSum:
                {
                    builder.Reset(new TGridBuilder<NSplitSelection::TMedianInBinBinarizer>());
                    break;
                }
                case EBorderSelectionType::MinEntropy:
                {
                    builder.Reset(new TGridBuilder<NSplitSelection::TMinEntropyBinarizer>());
                    break;
                }
                case EBorderSelectionType::MaxLogSum:
                {
                    builder.Reset(new TGridBuilder<NSplitSelection::TMaxSumLogBinarizer>());
                    break;
                }
                case EBorderSelectionType::Median:
                {
                    builder.Reset(new TGridBuilder<NSplitSelection::TMedianBinarizer>());
                    break;
                }
                case EBorderSelectionType::Uniform:
                {
                    builder.Reset(new TGridBuilder<NSplitSelection::TUniformBinarizer>());
                    break;
                }
                default:
                {
                    ythrow yexception() << "Invalid grid builder type!";
                }
            }
            return builder;
        }
    };

    class TBordersBuilder
    {
    public:
        TBordersBuilder(IFactory<IGridBuilder>& builderFactory,
                        const TVector<float>& values)
                : BuilderFactory(builderFactory)
                  , Values(values)
        {
        }

        TVector<float> operator()(const TBinarizationDescription& description)
        {
            auto builder = BuilderFactory.Create(description.BorderSelectionType);
            builder->AddFeature(Values, description.Discretization);
            return builder->Borders()[0];
        }

    private:
        IFactory<IGridBuilder>& BuilderFactory;
        const TVector<float>& Values;
    };

    using TOnCpuGridBuilderFactory = TGridBuilderFactory<TCpuGridBuilder>;
}

