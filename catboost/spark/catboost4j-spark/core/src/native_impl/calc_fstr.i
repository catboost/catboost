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

%include "calc_fstr.h"


bool HasNonZeroApproxForZeroWeightLeaf(const TFullModel& model);

i64 GetMaxObjectCountForFstrCalc(i64 objectCount, i32 featureCount);


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

TCombinationClassFeatures GetCombinationClassFeatures(const TFullModel& model);
