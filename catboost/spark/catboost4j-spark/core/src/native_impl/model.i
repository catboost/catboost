%{
#include <catboost/spark/catboost4j-spark/core/src/native_impl/model.h>
#include <catboost/spark/catboost4j-spark/core/src/native_impl/vector_output.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/eval_processing.h>
#include <catboost/libs/model/model.h>
#include <util/generic/algorithm.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/ylimits.h>
%}

%include <bindings/swiglib/stroka.swg>

%include "defaults.i"
%include "java_helpers.i"
%include "primitive_arrays.i"
%include "tvector.i"

/*
 * Use a separate binding instead of catboost4j-prediction for now because some extra functionality
 *  is needed.
 */

class TFullModel {
public:

    /**
     * @return Number of dimensions in model.
     */
    size_t GetDimensionsCount() const;

    %extend {
        void Calc(TConstArrayRef<double> numericFeatures, TArrayRef<double> result) const {
            TVector<float> featuresAsFloat;
            featuresAsFloat.yresize(numericFeatures.size());
            Copy(numericFeatures.begin(), numericFeatures.end(), featuresAsFloat.begin());
            self->Calc(featuresAsFloat, TConstArrayRef<int>(), result);
        }

        void CalcSparse(
            i32 size,
            TConstArrayRef<i32> numericFeaturesIndices,
            TConstArrayRef<double> numericFeaturesValues,
            TArrayRef<double> result
        ) const {
            TVector<float> featuresAsFloat(size, 0.0f);
            for (auto i : xrange(numericFeaturesIndices.size())) {
                featuresAsFloat[numericFeaturesIndices[i]] = numericFeaturesValues[i];
            }
            self->Calc(featuresAsFloat, TConstArrayRef<int>(), result);
        }
    }

    /**
     * Note: Max model size is limited for now to 2Gb due to Java arrays size limitation
     */

    %typemap(javaimports) TFullModel "import java.io.*;"
    %typemap(javainterfaces) TFullModel "Externalizable"

    %extend {
        TVector<i8> Serialize() const throw (yexception) {
            TVector<i8> result;
            TVectorOutput out(&result);
            OutputModel(*self, &out);
            out.Finish();
            CB_ENSURE(result.size() <= Max<i32>(), "Model size is too big (>2Gb) to be passed to JVM");
            return result;
        }

        void Deserialize(TConstArrayRef<i8> binaryBuffer) throw (yexception) {
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

void CalcSoftmax(const TConstArrayRef<double> approx, TArrayRef<double> softmax);

%include "model.h"
