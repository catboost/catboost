/* -*- mode: c -*- */

%module swigrun

#ifdef SWIGGUILE_SCM

/* Hook the runtime module initialization
   into the shared initialization function SWIG_Guile_Init. */
%runtime %{
/* Hook the runtime module initialization
   into the shared initialization function SWIG_Guile_Init. */
#include <libguile.h>
#ifdef __cplusplus
extern "C"
#endif
SCM scm_init_Swig_swigrun_module (void);
#define SWIG_INIT_RUNTIME_MODULE scm_init_Swig_swigrun_module();
%}

/* The runtime type system from common.swg */

typedef struct swig_type_info swig_type_info;

const char *
SWIG_TypeName(const swig_type_info *type);

const char *
SWIG_TypePrettyName(const swig_type_info *type);

swig_type_info *
SWIG_TypeQuery(const char *);

/* Language-specific stuff */

%apply bool { int };

int
SWIG_IsPointer(SCM object);

int
SWIG_IsPointerOfType(SCM object, swig_type_info *type);

unsigned long
SWIG_PointerAddress(SCM object);

swig_type_info *
SWIG_PointerType(SCM object);

#endif
