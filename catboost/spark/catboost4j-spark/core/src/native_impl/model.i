%{
#include <catboost/spark/catboost4j-spark/core/src/native_impl/model.h>
#include <catboost/spark/catboost4j-spark/core/src/native_impl/vector_output.h>
#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/enums.h>
#include <catboost/libs/model/eval_processing.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/model_export/model_exporter.h>
#include <util/generic/algorithm.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/ylimits.h>
%}

%include <bindings/swiglib/stroka.swg>

%include "catboost_enums.i"

%include "defaults.i"
%include "java_helpers.i"
%include "primitive_arrays.i"
%include "tvector.i"

/*
 * Use a separate binding instead of catboost4j-prediction for now because some extra functionality
 *  is needed.
 */

%catches(std::exception) TFullModel::Calc(
    TConstArrayRef<double> featureValuesFromSpark,
    TArrayRef<double> result
) const;

%catches(std::exception) TFullModel::CalcSparse(
    i32 size,
    TConstArrayRef<i32> featureIndicesFromSpark,
    TConstArrayRef<double> featureValuesFromSpark,
    TArrayRef<double> result
) const;

%catches(yexception, std::exception) TFullModel::Save(
    const TString& fileName,
    EModelType format,
    const TString& exportParametersJsonString,
    i32 poolCatFeaturesMaxUniqValueCount
);

%catches(std::exception) TFullModel::equalsImpl(const TFullModel& rhs) const;

class TFullModel {
public:

    /**
     * @return Number of dimensions in model.
     */
    size_t GetDimensionsCount() const;

    %extend {
        i32 GetLeafCount() const {
            const int approxDimension = self->ModelTrees->GetDimensionsCount();
            return i32(self->ModelTrees->GetModelTreeData()->GetLeafValues().size() / approxDimension);
        }

        bool HasLeafWeights() const {
            return !self->ModelTrees->GetModelTreeData()->GetLeafWeights().empty();
        }

        void Calc(TConstArrayRef<double> featureValuesFromSpark, TArrayRef<double> result) const {
            CalcOnSparkFeatureVector(*self, featureValuesFromSpark, result);
        }

        void CalcSparse(
            i32 size,
            TConstArrayRef<i32> featureIndicesFromSpark,
            TConstArrayRef<double> featureValuesFromSpark,
            TArrayRef<double> result
        ) const {
            TVector<float> denseFeaturesValues(size, 0.0f);
            for (auto i : xrange(featureIndicesFromSpark.size())) {
                denseFeaturesValues[featureIndicesFromSpark[i]] = featureValuesFromSpark[i];
            }
            CalcOnSparkFeatureVector<float>(*self, denseFeaturesValues, result);
        }
        
        void Save(
            const TString& fileName,
            EModelType format,
            const TString& exportParametersJsonString,
            i32 poolCatFeaturesMaxUniqValueCount
        ) {
            THashMap<ui32, TString> catFeaturesHashToString;
            for (auto v : xrange(SafeIntegerCast<ui32>(poolCatFeaturesMaxUniqValueCount))) {
                const TString vAsString = ToString(v);
                catFeaturesHashToString.emplace(CalcCatFeatureHash(vAsString), vAsString);
            }
        
            NCB::ExportModel(
                *self, 
                fileName, 
                format, 
                exportParametersJsonString,
                /*addFileFormatExtension*/ false,
                /*featureId*/ nullptr,
                &catFeaturesHashToString
            );
        }
    }

    /**
     * Note: Max model size is limited for now to 2Gb due to Java arrays size limitation
     */

    %typemap(javaimports) TFullModel "import java.io.*;"
    %typemap(javainterfaces) TFullModel "Externalizable"

    %extend {
        TVector<i8> Serialize() const {
            TVector<i8> result;
            TVectorOutput out(&result);
            OutputModel(*self, &out);
            out.Finish();
            CB_ENSURE(result.size() <= Max<i32>(), "Model size is too big (>2Gb) to be passed to JVM");
            return result;
        }

        void Deserialize(TConstArrayRef<i8> binaryBuffer) {
            (*self) = ReadModel(binaryBuffer.data(), binaryBuffer.size());
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
                throw new IOException("Error in TFullModel::Serialize: " + e.getMessage());
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
                throw new IOException("Error in TFullModel::Serialize: " + e.getMessage());
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
                throw new IOException("Error in TFullModel::Deserialize: " + e.getMessage());
            }
        }
    %}

    ADD_EQUALS_WITH_IMPL_AND_HASH_CODE_METHODS(TFullModel)
};



%catches(yexception) ReadModel(const TString& modelFile, EModelType format = EModelType::CatboostBinary);

TFullModel ReadModel(const TString& modelFile, EModelType format = EModelType::CatboostBinary);

DECLARE_TVECTOR(TVector_const_TFullModel_ptr, const TFullModel*)

%catches(yexception) SumModels(
    const TVector<const TFullModel*> modelVector,
    const TVector<double>& weights,
    const TVector<TString>& modelParamsPrefixes,
    ECtrTableMergePolicy ctrMergePolicy = ECtrTableMergePolicy::IntersectingCountersAverage);

TFullModel SumModels(
    const TVector<const TFullModel*> modelVector,
    const TVector<double>& weights,
    const TVector<TString>& modelParamsPrefixes,
    ECtrTableMergePolicy ctrMergePolicy = ECtrTableMergePolicy::IntersectingCountersAverage);


void CalcSoftmax(const TConstArrayRef<double> approx, TArrayRef<double> softmax);
