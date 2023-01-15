%module(directors="1") native_impl

%{
#include <util/generic/yexception.h>
%}

%include <typemaps.i>

%typemap(throws, throws="Exception") yexception {
    jclass excep = jenv->FindClass("ai/catboost/CatBoostError");
    if (excep)
        jenv->ThrowNew(excep, $1.what());
    return $null;
}

%include "primitive_arrays.i"
%include "maybe_owning_array_holder.i"
%include "tvector.i"

%include <bindings/swiglib/stroka.swg>

%template(TVector_TString) TVector<TString>;

%include "maybe.i"
%include "string.i"

%include "column.i"
%include "cd_parser.i"
%include "features_layout.i"
%include "meta_info.i"
%include "dataset_rows_reader.i"

%include "quantized_features_info.i"
%include "data_provider.i"
%include "data_provider_builders.i"
%include "quantization.i"
