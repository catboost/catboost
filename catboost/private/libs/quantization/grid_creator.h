#pragma once

#include <catboost/private/libs/options/binarization_options.h>

#include <library/cpp/grid_creator/binarization.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/hash_set.h>
#include <util/generic/ptr.h>
#include <util/generic/set.h>
#include <util/generic/ymath.h>
#include <util/system/types.h>

namespace NCB {

    TVector<float> CheckedCopyWithoutNans(TConstArrayRef<float> values, ENanMode nanMode);

    class IGridBuilder {
    public:
        virtual ~IGridBuilder() {
        }

        virtual IGridBuilder& AddFeature(TConstArrayRef<float> feature, ui32 borderCount, ENanMode nanMode) = 0;

        virtual const TVector<TVector<float>>& Borders() = 0;

        virtual TVector<float> BuildBorders(TConstArrayRef<float> sortedFeature,
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
                        TConstArrayRef<float> values)
            : BuilderFactory(builderFactory)
            , Values(values)
        {
        }

        TVector<float> operator()(const NCatboostOptions::TBinarizationOptions& description);
    private:
        IFactory<IGridBuilder>& BuilderFactory;
        TConstArrayRef<float> Values;
    };

    using TOnCpuGridBuilderFactory = TGridBuilderFactory;
}
