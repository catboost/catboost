#pragma once

namespace NCatboostCuda {
    template <class TInner>
    class TScaledModel {
    public:
        TInner Inner;
        double Scale;

        TScaledModel(const TInner& inner,
                     double scale)
            : Inner(inner)
            , Scale(scale)
        {
        }

        template <class TDataSet, class TCursor>
        void Append(TDataSet& ds,
                    TCursor& cursor) {
            Inner.Append(ds, cursor, Scale);
        }

        TScaledModel Rescale(double scale) const {
            double newScale = Scale * scale;
            return TScaledModel(Inner, newScale);
        }
    };
}
