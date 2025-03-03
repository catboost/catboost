/* -----------------------------------------------------------------------------
 * guile_scm.swg
 *
 * This SWIG interface file is processed if the Guile module is run
 * with SCM_ flavor.
 * ----------------------------------------------------------------------------- */

#define SWIGGUILE_SCM

%runtime "swigrun.swg"       // Common C API type-checking code
%runtime "swigerrors.swg"    // SWIG errors

%runtime "guile_scm_run.swg"
%include <guile.i>

%runtime %{

#define GUILE_MAYBE_VALUES \
      if (gswig_list_p) gswig_result = scm_values(gswig_result);

#define GUILE_MAYBE_VECTOR \
      if (gswig_list_p) gswig_result = scm_vector(gswig_result);

#define SWIG_APPEND_VALUE(object)						\
    if (gswig_result == SCM_UNSPECIFIED)						\
        gswig_result = object;							\
    else {									\
      if (!gswig_list_p) {							\
	      gswig_list_p = 1;							\
	      gswig_result = scm_list_n(gswig_result, object, SCM_UNDEFINED);	\
      }									\
      else									\
            gswig_result = scm_append(scm_list_n(gswig_result, scm_list_n(object, SCM_UNDEFINED), SCM_UNDEFINED));		\
    }

%}

%insert(init) "swiginit.swg"

%init %{
SWIG_GUILE_INIT_STATIC void
SWIG_init(void)
{
  SWIG_InitializeModule(0);
  SWIG_PropagateClientData();
%}
