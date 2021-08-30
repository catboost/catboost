%{
#include <catboost/spark/catboost4j-spark/core/src/native_impl/dataset_rows_reader.h>
%}

%include "line_data_reader.i"

%feature("director", assumeoverride=1) IJVMLineDataReader;

%include "dataset_rows_reader.h"
