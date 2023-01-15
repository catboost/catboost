%module native_impl

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

%include "features_layout.i"
%include "quantized_features_info.i"
%include "data_provider.i"
%include "data_provider_builders.i"
%include "quantization.i"
