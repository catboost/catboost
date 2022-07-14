%{
#include <catboost/spark/catboost4j-spark/core/src/native_impl/calc_fstr.h>
#include <catboost/spark/catboost4j-spark/core/src/native_impl/vector_output.h>
#include <catboost/libs/fstr/calc_fstr.h>
#include <catboost/libs/fstr/shap_prepared_trees.h>
#include <catboost/libs/fstr/util.h>
#include <catboost/libs/helpers/exception.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/stream/mem.h>
%}

%include "defaults.i"
%include "local_executor.i"
%include "model.i"
%include "tvector.i"



%catches(yexception) GetMaybeGeneratedModelFeatureIds(
    const TFullModel& model,
    const NCB::TFeaturesLayoutPtr datasetFeaturesLayout
);

TVector<TString> GetMaybeGeneratedModelFeatureIds(
    const TFullModel& model,
    const NCB::TFeaturesLayoutPtr datasetFeaturesLayout
);


%catches(yexception) GetDefaultFstrType(const TFullModel& model);

%catches(yexception) PreparedTreesNeedLeavesWeightsFromDataset(const TFullModel& model);

%catches(yexception) CollectLeavesStatisticsWrapper(
    const NCB::TDataProviderPtr dataset,
    const TFullModel& model,
    NPar::TLocalExecutor* localExecutor
);

%catches(yexception) PrepareTreesWithoutIndependent(
    const TFullModel& model,
    i64 datasetObjectCount,
    bool needSumModelAndDatasetWeights,
    TConstArrayRef<double> leafWeightsFromDataset,
    EPreCalcShapValues mode,
    bool calcInternalValues,
    ECalcTypeShapValues calcType,
    bool calcShapValuesByLeaf,
    NPar::TLocalExecutor* localExecutor
);

%catches(yexception) CalcFeatureEffectLossChangeMetricStatsWrapper(
    const TFullModel& model,
    const int featuresCount,
    const TShapPreparedTrees& preparedTrees,
    const NCB::TDataProviderPtr dataset,
    ECalcTypeShapValues calcType,
    NPar::TLocalExecutor* localExecutor
);

%catches(yexception) CalcFeatureEffectLossChangeFromScores(
    const TFullModel& model,
    const TCombinationClassFeatures& combinationClassFeatures,
    TConstArrayRef<double> scoresMatrix // row-major matrix representation of Stats[featureIdx][metricIdx]
);

%catches(yexception) CalcFeatureEffectAverageChangeWrapper(
    const TFullModel& model,
    TConstArrayRef<double> leafWeightsFromDataset // can be empty
);

%catches(yexception) GetPredictionDiffWrapper(
    const TFullModel& model,
    const NCB::TRawObjectsDataProviderPtr objectsDataProvider,
    NPar::TLocalExecutor* localExecutor
);

%catches(yexception) TShapValuesResult::GetObjectCount() const;
%catches(yexception) TShapValuesResult::GetShapValuesCount() const;
%catches(yexception) TShapValuesResult::Get(i32 objectIdx) const;

%catches(yexception) CalcShapValuesWithPreparedTreesWrapper(
    const TFullModel& model,
    const NCB::TDataProviderPtr dataset,
    const TShapPreparedTrees& preparedTrees,
    ECalcTypeShapValues calcType,
    NPar::TLocalExecutor* localExecutor
);

%catches(yexception) GetSelectedFeaturesIndices(
    const TFullModel& model,
    const TString& feature1Name,
    const TString& feature2Name,
    TArrayRef<i32> featureIndices // out param
);

%catches(yexception) TShapInteractionValuesResult::GetObjectCount() const;
%catches(yexception) TShapInteractionValuesResult::GetShapInteractionValuesCount() const;
%catches(yexception) TShapInteractionValuesResult::Get(i32 objectIdx, i32 dimensionIdx = 0) const;

%catches(yexception) CalcShapInteractionValuesWithPreparedTreesWrapper(
    const TFullModel& model,
    const NCB::TDataProviderPtr dataset,
    TConstArrayRef<i32> selectedFeatureIndices, // -1 if not selected
    ECalcTypeShapValues calcType,
    NPar::TLocalExecutor* localExecutor,
    TShapPreparedTrees* preparedTrees
);

%catches(yexception) CalcInteraction(
    const TFullModel& model,
    TVector<i32>* firstIndices,
    TVector<i32>* secondIndices,
    TVector<double>* scores
);

%include "calc_fstr.h"


%catches(yexception) HasNonZeroApproxForZeroWeightLeaf(const TFullModel& model);
bool HasNonZeroApproxForZeroWeightLeaf(const TFullModel& model);

%catches(yexception) GetMaxObjectCountForFstrCalc(i64 objectCount, i32 featureCount);
i64 GetMaxObjectCountForFstrCalc(i64 objectCount, i32 featureCount);


%catches(yexception) TShapPreparedTrees::Serialize() const;
%catches(yexception) TShapPreparedTrees::Deserialize(TConstArrayRef<i8> binaryBuffer);

struct TShapPreparedTrees {

    /**
     * Note: Max size is limited for now to 2Gb due to Java arrays size limitation
     */

    %typemap(javaimports) TShapPreparedTrees "import java.io.*;"
    %typemap(javainterfaces) TShapPreparedTrees "Externalizable"
    
    %extend {
        TVector<i8> Serialize() const {
            TVector<i8> result;
            TVectorOutput out(&result);
            self->Save(&out);
            out.Finish();
            CB_ENSURE(result.size() <= Max<i32>(), "TShapPreparedTrees size is too big (>2Gb) to be passed to JVM");
            return result;
        }

        void Deserialize(TConstArrayRef<i8> binaryBuffer) {
            TMemoryInput in(binaryBuffer.data(), binaryBuffer.size());
            self->Load(&in);
        }
    }
    
    %proxycode %{
        private void writeSerialized(java.nio.ByteBuffer serializedData, DataOutput out) throws IOException {
            // Transfer data by blocks because there's no way to copy data to out directly
            final int BUFFER_SIZE = 16384;

            byte[] javaBuffer = new byte[BUFFER_SIZE];
            while (serializedData.remaining() >= BUFFER_SIZE) {
                serializedData.get(javaBuffer, 0, BUFFER_SIZE);
                out.write(javaBuffer);
            }
            if (serializedData.hasRemaining()) {
                final int remaining = serializedData.remaining();
                serializedData.get(javaBuffer, 0, remaining);
                out.write(javaBuffer, 0, remaining);
            }
        }

        public void writeExternal(ObjectOutput out) throws IOException {
            TVector_i8 data;
            try {
                data = Serialize();
            } catch (Exception e) {
                throw new IOException("Error in TShapPreparedTrees::Serialize: " + e.getMessage());
            }
            java.nio.ByteBuffer byteBuffer = data.asDirectByteBuffer();
            out.writeInt(byteBuffer.remaining());
            writeSerialized(byteBuffer, out);
        }

        // without writing size to the stream, can be used to save to local or HDFS file
        public void write(DataOutput out) throws IOException {
            TVector_i8 data;
            try {
                data = Serialize();
            } catch (Exception e) {
                throw new IOException("Error in TShapPreparedTrees::Serialize: " + e.getMessage());
            }
            java.nio.ByteBuffer byteBuffer = data.asDirectByteBuffer();
            writeSerialized(byteBuffer, out);
        }

        public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
            read(in.readInt(), in);
        }

        // without reading size from the stream, can be used to save to local or HDFS file
        public void read(int size, DataInput in) throws IOException {
            byte[] data = new byte[size];
            in.readFully(data);
            try {
                Deserialize(data);
            } catch (Exception e) {
                throw new IOException("Error in TShapPreparedTrees::Deserialize: " + e.getMessage());
            }
        }
    %}

};


struct TCombinationClassFeatures {
    size_t size() const;
};

%catches(yexception) GetCombinationClassFeatures(const TFullModel& model);
TCombinationClassFeatures GetCombinationClassFeatures(const TFullModel& model);
