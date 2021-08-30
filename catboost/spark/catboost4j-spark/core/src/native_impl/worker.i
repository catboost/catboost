%{
#include <catboost/spark/catboost4j-spark/core/src/native_impl/worker.h>
%}

%include <bindings/swiglib/stroka.swg>

%include "defaults.i"
%include "data_provider.i"
%include "quantized_features_info.i"

%include "worker.h"