#pragma once

#include <catboost/libs/options/binarization_options.h>

#include <library/grid_creator/binarization.h>

#include <util/generic/algorithm.h>
#include <util/generic/hash_set.h>
#include <util/generic/ptr.h>
#include <util/generic/set.h>
#include <util/generic/ymath.h>
#include <util/system/types.h>

namespace NCatboostCuda {

    TVector<float> CheckedCopyWithoutNans(const TVector<float>& values, ENanMode nanMode);

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


    class TGridBuilderFactory: public IFactory<IGridBuilder> {
    public:
        THolder<IGridBuilder> Create(EBorderSelectionType type) override;
    };

    class TBordersBuilder {
    public:
        TBordersBuilder(IFactory<IGridBuilder>& builderFactory,
                        const TVector<float>& values)
            : BuilderFactory(builderFactory)
            , Values(values)
        {
        }

        TVector<float> operator()(const NCatboostOptions::TBinarizationOptions& description);
    private:
        IFactory<IGridBuilder>& BuilderFactory;
        const TVector<float>& Values;
    };

    using TOnCpuGridBuilderFactory = TGridBuilderFactory;
}
