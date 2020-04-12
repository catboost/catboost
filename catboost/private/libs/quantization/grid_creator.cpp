#include "grid_creator.h"

#include <util/generic/vector.h>

namespace NCB {

    namespace {
        template <EBorderSelectionType type>
        class TGridBuilderBase: public IGridBuilder {
        public:
            TVector<float> BuildBorders(TConstArrayRef<float> sortedFeature, ui32 borderCount) const override {
                TVector<float> copy = CheckedCopyWithoutNans(sortedFeature, ENanMode::Forbidden);
                auto quantization
                    = Binarizer->BestSplit(NSplitSelection::TFeatureValues(std::move(copy), true), borderCount);

                return std::move(quantization.Borders);
            }

        private:
            THolder<const NSplitSelection::IBinarizer> Binarizer{NSplitSelection::MakeBinarizer(type)};
        };

        template <EBorderSelectionType type>
        class TCpuGridBuilder: public TGridBuilderBase<type> {
        public:
            IGridBuilder& AddFeature(TConstArrayRef<float> feature,
                                     ui32 borderCount,
                                     ENanMode nanMode) override {
                TVector<float> sortedFeature = CheckedCopyWithoutNans(feature, nanMode);
                Sort(sortedFeature.begin(), sortedFeature.end());
                auto borders = TGridBuilderBase<type>::BuildBorders(sortedFeature, borderCount);
                Result.push_back(std::move(borders));
                return *this;
            }

            const TVector<TVector<float>>& Borders() override {
                return Result;
            }

        private:
            TVector<TVector<float>> Result;
        };
    }

    TVector<float> CheckedCopyWithoutNans(TConstArrayRef<float> values, ENanMode nanMode) {
        TVector<float> copy;
        copy.reserve(values.size());
        for (ui32 i = 0; i < values.size(); ++i) {
            const float val = values[i];
            if (std::isnan(val)) {
                CB_ENSURE(nanMode != ENanMode::Forbidden, "Error: NaN in features, but NaNs are forbidden");
                continue;
            } else {
                copy.push_back(val);
            }
        }
        return copy;
    }


    THolder<IGridBuilder> TGridBuilderFactory::Create(EBorderSelectionType type)  {
        switch (type) {
            case EBorderSelectionType::UniformAndQuantiles:
                return MakeHolder<TCpuGridBuilder<EBorderSelectionType::UniformAndQuantiles>>();
            case EBorderSelectionType::GreedyLogSum:
                return MakeHolder<TCpuGridBuilder<EBorderSelectionType::GreedyLogSum>>();
            case EBorderSelectionType::GreedyMinEntropy:
                return MakeHolder<TCpuGridBuilder<EBorderSelectionType::GreedyMinEntropy>>();
            case EBorderSelectionType::MinEntropy:
                return MakeHolder<TCpuGridBuilder<EBorderSelectionType::MinEntropy>>();
            case EBorderSelectionType::MaxLogSum:
                return MakeHolder<TCpuGridBuilder<EBorderSelectionType::MaxLogSum>>();
            case EBorderSelectionType::Median:
                return MakeHolder<TCpuGridBuilder<EBorderSelectionType::Median>>();
            case EBorderSelectionType::Uniform:
                return MakeHolder<TCpuGridBuilder<EBorderSelectionType::Uniform>>();
        }
        ythrow yexception() << "Invalid grid builder type!";
    }

    TVector<float> TBordersBuilder::operator()(const NCatboostOptions::TBinarizationOptions& description) {
        auto builder = BuilderFactory.Create(description.BorderSelectionType);
        const ui32 borderCount = description.NanMode == ENanMode::Forbidden ? description.BorderCount : description.BorderCount - 1;
        CB_ENSURE(borderCount > 0, "Error: border count should be greater than 0. If you have nan-features, border count should be > 1. Got " << description.BorderCount);
        builder->AddFeature(Values, description.BorderCount, description.NanMode);
        return builder->Borders()[0];
    }

}
