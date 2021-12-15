%{
#include <catboost/spark/catboost4j-spark/core/src/native_impl/features_layout.h>
#include <catboost/libs/data/features_layout.h>
#include <util/generic/yexception.h>
%}

%include "catboost_enums.i"
%include "defaults.i"
%include "intrusive_ptr.i"
%include "java_helpers.i"
%include "tvector.i"


%catches(std::exception) NCB::TFeatureMetaInfo::equalsImpl(const NCB::TFeatureMetaInfo& rhs) const;

namespace NCB {
    struct TFeatureMetaInfo {
        EFeatureType Type;
        TString Name;

        /* Means that the distribution of values of this feature contains a large
         * probability for a single value (often called default value).
         * Note that IsSparse == true does not mean that data for this feature will always be stored
         * as sparse. For some applications dense storage might be necessary/more effective.
         */
        bool IsSparse = false;

        bool IsIgnored = false;

        /* some datasets can contain only part of all features present in the whole dataset
         * (e.g. workers in distributed processing)
         * ignored features are always unavailable
         */
        bool IsAvailable = true;

    public:
        TFeatureMetaInfo() = default;

        %typemap(javaimports) TFeatureMetaInfo "import java.io.*;"
        %typemap(javainterfaces) TFeatureMetaInfo "Serializable"

        %proxycode %{
            private void writeObject(ObjectOutputStream out) throws IOException {
                out.writeUnshared(this.getType());
                out.writeUTF(this.getName());
                out.writeBoolean(this.getIsSparse());
                out.writeBoolean(this.getIsIgnored());
                out.writeBoolean(this.getIsAvailable());
            }

            private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
                this.swigCPtr = native_implJNI.new_TFeatureMetaInfo();
                this.swigCMemOwn = true;

                this.setType((EFeatureType)in.readUnshared());
                this.setName(in.readUTF());
                this.setIsSparse(in.readBoolean());
                this.setIsIgnored(in.readBoolean());
                this.setIsAvailable(in.readBoolean());
            }
        %}

        ADD_EQUALS_WITH_IMPL_AND_HASH_CODE_METHODS(TFeatureMetaInfo)
    };
}

DECLARE_TVECTOR(TVector_TFeatureMetaInfo, NCB::TFeatureMetaInfo)


%catches(yexception) NCB::TFeaturesLayout::Init(TVector<TFeatureMetaInfo>* data);
%catches(yexception) NCB::TFeaturesLayout::GetExternalFeatureIds() const;
%catches(yexception) NCB::TFeaturesLayout::GetExternalFeaturesMetaInfoAsVector() const;
%catches(std::exception) NCB::TFeaturesLayout::equalsImpl(const NCB::TFeaturesLayout& rhs) const;

namespace NCB {
    class TFeaturesLayout {
    public:
        TFeaturesLayout() = default;

        // needed for SWIG wrapper deserialization
        // data is moved into - poor substitute to && because SWIG does not support it
        void Init(TVector<TFeatureMetaInfo>* data);

        TVector<TString> GetExternalFeatureIds() const;

        ui32 GetExternalFeatureCount() const;

        %extend {
            TVector<NCB::TFeatureMetaInfo> GetExternalFeaturesMetaInfoAsVector() const {
                TConstArrayRef<NCB::TFeatureMetaInfo> externalFeaturesMetaInfo
                    = self->GetExternalFeaturesMetaInfo();
                return TVector<NCB::TFeatureMetaInfo>(
                    externalFeaturesMetaInfo.begin(),
                    externalFeaturesMetaInfo.end()
                );
            }
        }

        %typemap(javaimports) TFeaturesLayout "import java.io.*;"
        %typemap(javainterfaces) TFeaturesLayout "Serializable"

        %proxycode %{
            private void writeObject(ObjectOutputStream out) throws IOException {
                TVector_TFeatureMetaInfo data = null;
                try {
                    data = this.GetExternalFeaturesMetaInfoAsVector();
                } catch (Exception e) {
                    throw new IOException(
                        "Error in TFeaturesLayout::GetExternalFeaturesMetaInfoAsVector: " + e.getMessage()
                    );
                }
                out.writeUnshared(data);
            }

            private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
                this.swigCPtr = native_implJNI.new_TFeaturesLayout();
                this.swigCMemOwn = true;

                TVector_TFeatureMetaInfo data = (TVector_TFeatureMetaInfo)in.readUnshared();
                try {
                    Init(data);
                } catch (Exception e) {
                    throw new IOException(
                        "Error in TFeaturesLayout::Init: " + e.getMessage()
                    );
                }
            }
        %}

        ADD_EQUALS_WITH_IMPL_AND_HASH_CODE_METHODS(TFeaturesLayout)
        ADD_RELEASE_MEM()

    private:
        TVector<TFeatureMetaInfo> ExternalIdxToMetaInfo;
    };

    using TFeaturesLayoutPtr = TIntrusivePtr<TFeaturesLayout>;
}

%template(TFeaturesLayoutPtr) TIntrusivePtr<NCB::TFeaturesLayout>;


%catches(yexception) MakeFeatureMetaInfo(
    EFeatureType type,
    const TString& name,
    bool isSparse = false,
    bool isIgnored = false,
    bool isAvailable = true // isIgnored = true overrides this parameter
);

%catches(yexception) MakeFeaturesLayout(TVector<NCB::TFeatureMetaInfo>* data);
%catches(yexception) MakeFeaturesLayout(
    const int featureCount,
    const TVector<TString>& featureNames,
    const TVector<i32>& ignoredFeatures
);

%catches(yexception) CloneWithSelectedFeatures(
    const NCB::TFeaturesLayout& featuresLayout,
    TConstArrayRef<i32> selectedFeatures
);

%include "features_layout.h"


%catches(yexception) GetAvailableFeatures_Float(const NCB::TFeaturesLayout& featuresLayout);
%template(GetAvailableFeatures_Float) GetAvailableFeatures<EFeatureType::Float>;

%catches(yexception) GetAvailableFeatures_Categorical(const NCB::TFeaturesLayout& featuresLayout);
%template(GetAvailableFeatures_Categorical) GetAvailableFeatures<EFeatureType::Categorical>;

%catches(yexception) GetAvailableFeaturesFlatIndices_Float(const NCB::TFeaturesLayout& featuresLayout);
%template(GetAvailableFeaturesFlatIndices_Float) GetAvailableFeaturesFlatIndices<EFeatureType::Float>;

%catches(yexception) GetAvailableFeaturesFlatIndices_Categorical(const NCB::TFeaturesLayout& featuresLayout);
%template(GetAvailableFeaturesFlatIndices_Categorical) GetAvailableFeaturesFlatIndices<EFeatureType::Categorical>;
