%{
#include <catboost/spark/catboost4j-spark/core/src/native_impl/meta_info.h>
#include <catboost/libs/data/meta_info.h>
%}

%include "catboost_enums.i"
%include "column.i"
%include "defaults.i"
%include "java_helpers.i"
%include "maybe.i"
%include "tvector.i"
%include "features_layout.i"
%include "cd_parser.i" // for TVector_TColumn


%catches(std::exception) NCB::TDataColumnsMetaInfo::equalsImpl(const NCB::TDataColumnsMetaInfo& rhs) const;

namespace NCB {

    struct TDataColumnsMetaInfo {
        TVector<TColumn> Columns;

        %typemap(javaimports) TDataColumnsMetaInfo "import java.io.*;"
        %typemap(javainterfaces) TDataColumnsMetaInfo "Serializable"

        %proxycode %{
            private void writeObject(ObjectOutputStream out) throws IOException {
                out.writeUnshared(this.getColumns());
            }

            private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
                this.swigCPtr = native_implJNI.new_$javaclassname();
                this.swigCMemOwn = true;

                this.setColumns((TVector_TColumn)in.readUnshared());
            }
        %}

        ADD_EQUALS_WITH_IMPL_AND_HASH_CODE_METHODS(TDataColumnsMetaInfo)
    };
}

%template(TMaybe_TDataColumnsMetaInfo) TMaybe<NCB::TDataColumnsMetaInfo>;

namespace NCB {
    // Some fields not used in Java wrapper are omitted for simplicity
    struct TDataMetaInfo {
        ui64 ObjectCount = 0;

        TFeaturesLayoutPtr FeaturesLayout;

        ERawTargetType TargetType = ERawTargetType::None;
        ui32 TargetCount = 0;

        ui32 BaselineCount = 0;

        bool HasGroupId = false;
        bool HasGroupWeight = false;
        bool HasSubgroupIds = false;
        bool HasWeights = false;
        bool HasTimestamp = false;
        bool HasPairs = false;

        // set only for dsv format pools
        // TODO(akhropov): temporary, serialization details shouldn't be here
        TMaybe<NCB::TDataColumnsMetaInfo> ColumnsInfo;

    public:
        ui32 GetFeatureCount() const {
            return FeaturesLayout ? FeaturesLayout->GetExternalFeatureCount() : 0;
        }

         %proxycode %{
            // Needed for deserialization
            void setSwigCPtr(long swigCPtr) {
                this.swigCPtr = swigCPtr;
            }
         %}
    };
}


%catches(yexception) TIntermediateDataMetaInfo::SetAvailableFeatures(TConstArrayRef<i32> selectedFeatures);
%catches(std::exception) TIntermediateDataMetaInfo::equalsImpl(const TIntermediateDataMetaInfo& rhs) const;

class TIntermediateDataMetaInfo : public NCB::TDataMetaInfo {
public:
    TIntermediateDataMetaInfo() = default;

    TIntermediateDataMetaInfo(
        const NCB::TDataMetaInfo& dataMetaInfo,
        bool hasUnknownNumberOfSparseFeatures
    )
        : NCB::TDataMetaInfo(dataMetaInfo)
        , HasUnknownNumberOfSparseFeatures(hasUnknownNumberOfSparseFeatures)
    {}

    bool HasSparseFeatures() const;

public:
    bool HasUnknownNumberOfSparseFeatures = false;
    
    %extend {
        TIntermediateDataMetaInfo Clone() const {
            return *self;
        }
    
        TIntermediateDataMetaInfo SetAvailableFeatures(TConstArrayRef<i32> selectedFeatures) {
            TIntermediateDataMetaInfo selfWithSelectedFeatures = *self;
            selfWithSelectedFeatures.FeaturesLayout = CloneWithSelectedFeatures(
                *(self->FeaturesLayout), 
                selectedFeatures
            );
            return selfWithSelectedFeatures;
        }
    }

    %typemap(javaimports) TIntermediateDataMetaInfo "import java.io.*;"
    %typemap(javainterfaces) TIntermediateDataMetaInfo "Serializable"

    %proxycode %{
        private void writeObject(ObjectOutputStream out) throws IOException {
            out.writeUnshared(this.getObjectCount());

            out.writeUnshared(this.getFeaturesLayout());

            out.writeUnshared(this.getTargetType());
            out.writeLong(this.getTargetCount());

            out.writeLong(this.getBaselineCount());

            out.writeBoolean(this.getHasGroupId());
            out.writeBoolean(this.getHasGroupWeight());
            out.writeBoolean(this.getHasSubgroupIds());
            out.writeBoolean(this.getHasWeights());
            out.writeBoolean(this.getHasTimestamp());
            out.writeBoolean(this.getHasPairs());

            out.writeUnshared(this.getColumnsInfo());

            out.writeBoolean(this.getHasUnknownNumberOfSparseFeatures());
        }

        private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
            this.swigCPtr = native_implJNI.new_$javaclassname__SWIG_0();
            super.setSwigCPtr(native_implJNI.$javaclassname_SWIGUpcast(this.swigCPtr));
            this.swigCMemOwn = true;

            this.setObjectCount((java.math.BigInteger)in.readUnshared());

            this.setFeaturesLayout((TFeaturesLayoutPtr)in.readUnshared());

            this.setTargetType((ERawTargetType)in.readUnshared());
            this.setTargetCount(in.readLong());

            this.setBaselineCount(in.readLong());

            this.setHasGroupId(in.readBoolean());
            this.setHasGroupWeight(in.readBoolean());
            this.setHasSubgroupIds(in.readBoolean());
            this.setHasWeights(in.readBoolean());
            this.setHasTimestamp(in.readBoolean());
            this.setHasPairs(in.readBoolean());

            this.setColumnsInfo((TMaybe_TDataColumnsMetaInfo)in.readUnshared());

            this.setHasUnknownNumberOfSparseFeatures(in.readBoolean());
        }
    %}

    ADD_EQUALS_WITH_IMPL_AND_HASH_CODE_METHODS(TIntermediateDataMetaInfo)
};


%catches(yexception) GetIntermediateDataMetaInfo(
    const TString& schema,
    const TString& columnDescriptionPathWithScheme, // can be empty
    const TString& plainJsonParamsAsString,
    const TMaybe<TString>& dsvHeader,
    const TString& firstDataLine
);

%include "meta_info.h"
