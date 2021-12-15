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

DECLARE_TVECTOR(TVector_TString, TString)

%include "maybe.i"
%include "string.i"

%include "groupid.i"

%include "local_executor.i"

%include "column.i"
%include "cd_parser.i"
%include "features_layout.i"
%include "meta_info.i"
%include "dataset_rows_reader.i"

%include "quantized_features_info.i"
%include "data_provider.i"
%include "data_provider_builders.i"
%include "pairs.i"
%include "quantization.i"
%include "target.i"

%include "ctrs.i"

%include "quantized_pool_serialization.i"

%include "options_helper.i"

%include "model.i"

%include "master.i"
%include "worker.i"

%include "model_application.i"

%include "calc_fstr.i"

%pragma(java) jniclasscode=%{
  static {
    try {
      //ai.catboost.common.NativeLib.smartLoad("catboost4j-spark-impl");

      Class cls = Class.forName("ai.catboost.common.NativeLib");
      Object obj = cls.newInstance();

      Class[] paramString = new Class[1];
      paramString[0] = String.class;

      cls.getDeclaredMethod("smartLoad", paramString).invoke(obj, new String("catboost4j-spark-impl"));

    } catch (Exception e) {
      System.err.println("Failed to load catboost4j-spark-impl native library: " + e);
      System.exit(1);
    }
  }
%}
