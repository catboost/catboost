%{
#include <catboost/spark/catboost4j-spark/core/src/native_impl/data_provider_builders.h>
%}

%include "data_provider.i"
%include "maybe_owning_array_holder.i"
%include "tvector.i"

%template(TVector_TMaybeOwningConstArrayHolder_float) TVector<NCB::TMaybeOwningConstArrayHolder<float>>;

%include "data_provider_builders.h"
