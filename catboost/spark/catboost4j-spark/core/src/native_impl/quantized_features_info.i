%{
#include <catboost/spark/catboost4j-spark/core/src/native_impl/quantized_features_info.h>
#include <catboost/libs/data/feature_index.h>
#include <catboost/libs/data/quantized_features_info.h>
#include <library/cpp/grid_creator/binarization.h>
#include <util/generic/cast.h>
%}

%include "catboost_enums.i"
%include "defaults.i"
%include "features_layout.i"
%include "java_helpers.i"
%include "intrusive_ptr.i"


namespace NSplitSelection {
    struct TDefaultQuantizedBin {
        ui32 Idx; // if for splits: bin borders are [Border[Idx - 1], Borders[Idx])
        float Fraction;

        %typemap(javaimports) TDefaultQuantizedBin "import java.io.*;"
        %typemap(javainterfaces) TDefaultQuantizedBin "Serializable"

        %proxycode %{
            private void writeObject(ObjectOutputStream out) throws IOException {
                out.writeLong(this.getIdx());
                out.writeFloat(this.getFraction());
            }

            private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
                this.swigCPtr = native_implJNI.new_TDefaultQuantizedBin();
                this.swigCMemOwn = true;

                this.setIdx(in.readLong());
                this.setFraction(in.readFloat());
            }
        %}
    };
}

%include "cpointer.i"
%pointer_class(bool, boolp)


%catches(std::exception) TQuantizedFeaturesInfo::Init(TFeaturesLayout* featuresLayout);

%catches(yexception) TQuantizedFeaturesInfo::GetNanMode(int floatFeatureIdx) const;
%catches(yexception) TQuantizedFeaturesInfo::SetNanMode(int floatFeatureIdx, ENanMode nanMode);

%catches(yexception) TQuantizedFeaturesInfo::GetQuantization(
    int floatFeatureIdx,
    TVector<float>* borders,
    bool* hasDefaultQuantizedBin,
    NSplitSelection::TDefaultQuantizedBin* defaultQuantizedBin
) const;

%catches(yexception) TQuantizedFeaturesInfo::SetQuantization(
    int floatFeatureIdx,
    // moved into
    TVector<float>* borders,
    // optional, poor man's TMaybe substitute
    const NSplitSelection::TDefaultQuantizedBin* defaultQuantizedBin = nullptr
);

%catches(std::exception) TQuantizedFeaturesInfo::equalsImpl(const TQuantizedFeaturesInfo& rhs) const;

namespace NCB {
    class TQuantizedFeaturesInfo {
    public:
        const TFeaturesLayoutPtr GetFeaturesLayout() const;

        /* for Java deserialization & testing
         *  ignored features are already set in featuresLayout
         */
        void Init(TFeaturesLayout* featuresLayout); // featuresLayout is moved into

        bool EqualWithoutOptionsTo(const TQuantizedFeaturesInfo& rhs, bool ignoreSparsity = false) const;

        %extend {
            ENanMode GetNanMode(int floatFeatureIdx) const {
                return self->GetNanMode(NCB::TFloatFeatureIdx(SafeIntegerCast<int>(floatFeatureIdx)));
            }

            void SetNanMode(int floatFeatureIdx, ENanMode nanMode) {
                self->SetNanMode(NCB::TFloatFeatureIdx(SafeIntegerCast<int>(floatFeatureIdx)), nanMode);
            }

            void GetQuantization(
                int floatFeatureIdx,
                TVector<float>* borders,
                bool* hasDefaultQuantizedBin,
                NSplitSelection::TDefaultQuantizedBin* defaultQuantizedBin
            ) const {
                const auto& quantization = self->GetQuantization(
                    NCB::TFloatFeatureIdx(SafeIntegerCast<ui32>(floatFeatureIdx))
                );
                *borders = quantization.Borders;
                if (quantization.DefaultQuantizedBin) {
                    *hasDefaultQuantizedBin = true;
                    *defaultQuantizedBin = *quantization.DefaultQuantizedBin;
                } else {
                    *hasDefaultQuantizedBin = false;
                }
            }

            void SetQuantization(
                int floatFeatureIdx,
                // moved into
                TVector<float>* borders,
                // optional, poor man's TMaybe substitute
                const NSplitSelection::TDefaultQuantizedBin* defaultQuantizedBin = nullptr
            ) {
                NSplitSelection::TQuantization quantization(
                    std::move(*borders),
                    defaultQuantizedBin ?
                        TMaybe<NSplitSelection::TDefaultQuantizedBin>(*defaultQuantizedBin)
                        : Nothing()
                );
                self->SetQuantization(
                    NCB::TFloatFeatureIdx(SafeIntegerCast<ui32>(floatFeatureIdx)),
                    std::move(quantization)
                );
            }
        }

        /**
         * Note: serialization only supports featuresLayout and already computed quantization and nanModes
         *  for now
         */
        %typemap(javaimports) TQuantizedFeaturesInfo "import java.io.*;"
        %typemap(javainterfaces) TQuantizedFeaturesInfo "Serializable"

        %proxycode %{
            private void writeObject(ObjectOutputStream out) throws IOException {
                try {
                    TFeaturesLayout featuresLayout = this.GetFeaturesLayout().Get();
                    out.writeObject(featuresLayout);

                    int[] availableFloatFeatures
                        = native_impl.GetAvailableFeatures_Float(featuresLayout).toPrimitiveArray();

                    ENanMode nanMode;
                    TVector_float borders = new TVector_float();
                    boolp hasDefaultQuantizedBin = new boolp();
                    TDefaultQuantizedBin defaultQuantizedBin = new TDefaultQuantizedBin();

                    for (int i : availableFloatFeatures) {
                        nanMode = this.GetNanMode(i);
                        out.writeUnshared(nanMode);
                        this.GetQuantization(i, borders, hasDefaultQuantizedBin.cast(), defaultQuantizedBin);
                        out.writeUnshared(borders);
                        boolean hasDefaultQuantizedBinValue = hasDefaultQuantizedBin.value();
                        out.writeBoolean(hasDefaultQuantizedBinValue);
                        if (hasDefaultQuantizedBinValue) {
                            out.writeUnshared(defaultQuantizedBin);
                        }
                    }

                    out.writeUnshared(native_impl.GetCategoricalFeaturesUniqueValuesCounts(this));
                } catch (Exception e) {
                    throw new IOException(e);
                }
            }

            private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
                try {
                    this.swigCPtr = native_implJNI.new_TQuantizedFeaturesInfo();
                    this.swigCMemOwn = true;

                    TFeaturesLayout featuresLayout = (TFeaturesLayout)in.readObject();

                    int[] availableFloatFeatures
                        = native_impl.GetAvailableFeatures_Float(featuresLayout).toPrimitiveArray();

                    Init(featuresLayout);

                    for (int i : availableFloatFeatures) {
                        this.SetNanMode(i, (ENanMode)in.readUnshared());
                        TVector_float borders = (TVector_float)in.readUnshared();
                        boolean hasDefaultQuantizedBinValue = in.readBoolean();
                        TDefaultQuantizedBin defaultQuantizedBin = null;
                        if (hasDefaultQuantizedBinValue) {
                            defaultQuantizedBin = (TDefaultQuantizedBin)in.readUnshared();
                        }
                        this.SetQuantization(i, borders, defaultQuantizedBin);
                    }

                    TVector_i32 catFeaturesUniqueValuesCounts = (TVector_i32)in.readUnshared();
                    native_impl.UpdateCatFeaturesInfo(
                        catFeaturesUniqueValuesCounts.toPrimitiveArray(),
                        /*isInitialization*/ false,
                        this
                    );
                } catch (Exception e) {
                    throw new IOException(e);
                }
            }
        %}

        ADD_EQUALS_WITH_IMPL_AND_HASH_CODE_METHODS(TQuantizedFeaturesInfo)
        ADD_RELEASE_MEM()
    };

    using TQuantizedFeaturesInfoPtr = TIntrusivePtr<TQuantizedFeaturesInfo>;
}

%template(QuantizedFeaturesInfoPtr) TIntrusivePtr<NCB::TQuantizedFeaturesInfo>;


%catches(yexception) MakeQuantizedFeaturesInfo(
    const NCB::TFeaturesLayout& featuresLayout
);

%catches(yexception) MakeEstimatedQuantizedFeaturesInfo(i32 featureCount);

%catches(yexception) UpdateCatFeaturesInfo(
    TConstArrayRef<i32> catFeaturesUniqValueCounts, // [flatFeatureIdx]
    bool isInitialization,
    NCB::TQuantizedFeaturesInfo* quantizedFeaturesInfo
);

%catches(yexception) CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn(
    const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo
);

%catches(yexception) GetCategoricalFeaturesUniqueValuesCounts(
    const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo
);

%include "quantized_features_info.h"
