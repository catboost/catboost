%{
#include <catboost/spark/catboost4j-spark/core/src/native_impl/dataset_rows_reader.h>
%}

%include "line_data_reader.i"

%feature("director", assumeoverride=1) IJVMLineDataReader;


%catches(yexception) TStringOutWrapper::Assign(TConstArrayRef<i8> data, i32 length);

%catches(yexception, Swig::DirectorException) TRawDatasetRowsReader::TRawDatasetRowsReader(
    const TString& schema,

    // takes ownership
    // call swigReleaseOwnership() on IJVMLineDataReader-derived class before calling this method
    NCB::ILineDataReader* lineReader,
    const TString& columnDescriptionPathWithScheme, // if non-empty prefer it to columnDescription
    const TVector<TColumn>& columnsDescription,
    const TString& plainJsonParamsAsString,

    // specified here, not in params json because it will be different for different partitions
    bool hasHeader,
    i32 blockSize,
    i32 threadCount
);

%catches(yexception, Swig::DirectorException) TRawDatasetRowsReader::ReadNextBlock();
%catches(yexception, Swig::DirectorException) TRawDatasetRowsReader::GetRow(i32 objectIdx);

%include "dataset_rows_reader.h"
