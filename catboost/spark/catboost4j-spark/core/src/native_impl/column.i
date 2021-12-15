%{
#include <catboost/libs/column_description/column.h>
%}

%include "enums.swg"

%include "java_helpers.i"

%javaconst(1);
enum class EColumn {
    Num,
    Categ,
    Label,
    Auxiliary,
    Baseline,
    Weight,
    SampleId,
    GroupId,
    GroupWeight,
    SubgroupId,
    Timestamp,
    Sparse,
    Prediction,
    Text,
    NumVector
};


%catches(std::exception) TColumn::equalsImpl(const TColumn& rhs) const;

struct TColumn {
    EColumn Type;
    TString Id;

    %typemap(javaimports) TColumn "import java.io.*;"
    %typemap(javainterfaces) TColumn "Serializable"

    %proxycode %{
        private void writeObject(ObjectOutputStream out) throws IOException {
            out.writeUnshared(this.getType());
            out.writeUnshared(this.getId());
        }

        private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
            this.swigCPtr = native_implJNI.new_$javaclassname();
            this.swigCMemOwn = true;

            this.setType((EColumn)in.readUnshared());
            this.setId((String)in.readUnshared());
        }
    %}

    ADD_EQUALS_WITH_IMPL_AND_HASH_CODE_METHODS(TColumn)
};

