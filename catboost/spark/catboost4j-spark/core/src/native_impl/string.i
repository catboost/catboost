%{
#include <catboost/spark/catboost4j-spark/core/src/native_impl/string.h>
#include <util/generic/string.h>
%}

%include <bindings/swiglib/stroka.swg>

%include "primitive_arrays.i"
%include "defaults.i"
%include "maybe.i"

%template(TMaybe_TString) TMaybe<TString>;


%catches(std::exception) MakeMaybeUtf8String(TConstArrayRef<i8> data, i32 length);

%include "string.h"
