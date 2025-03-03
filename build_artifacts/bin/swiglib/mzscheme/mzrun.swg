/* -----------------------------------------------------------------------------
 * mzrun.swg
 * ----------------------------------------------------------------------------- */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <escheme.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Common SWIG API */
  
#define SWIG_ConvertPtr(s, result, type, flags) \
  SWIG_MzScheme_ConvertPtr(s, result, type, flags)
#define SWIG_NewPointerObj(ptr, type, owner) \
  SWIG_MzScheme_NewPointerObj((void *)ptr, type, owner)
#define SWIG_MustGetPtr(s, type, argnum, flags) \
  SWIG_MzScheme_MustGetPtr(s, type, argnum, flags, FUNC_NAME, argc, argv)

#define SWIG_contract_assert(expr,msg) \
 if (!(expr)) { \
    char *m=(char *) scheme_malloc(strlen(msg)+1000); \
    sprintf(m,"SWIG contract, assertion failed: function=%s, message=%s", \
            (char *) FUNC_NAME,(char *) msg); \
    scheme_signal_error(m); \
 }

/* Runtime API */
#define SWIG_GetModule(clientdata) SWIG_MzScheme_GetModule((Scheme_Env *)(clientdata))
#define SWIG_SetModule(clientdata, pointer) SWIG_MzScheme_SetModule((Scheme_Env *) (clientdata), pointer)
#define SWIG_MODULE_CLIENTDATA_TYPE Scheme_Env *

/* MzScheme-specific SWIG API */
  
#define SWIG_malloc(size) SWIG_MzScheme_Malloc(size, FUNC_NAME)
#define SWIG_free(mem) free(mem)
#define SWIG_NewStructFromPtr(ptr,type) \
        _swig_convert_struct_##type##(ptr)

#define MAXVALUES 6
#define swig_make_boolean(b) (b ? scheme_true : scheme_false)

static long
SWIG_convert_integer(Scheme_Object *o,
		     long lower_bound, long upper_bound, 
		     const char *func_name, int argnum, int argc,
		     Scheme_Object **argv)
{
  long value;
  int status = scheme_get_int_val(o, &value);
  if (!status)
    scheme_wrong_type(func_name, "integer", argnum, argc, argv);
  if (value < lower_bound || value > upper_bound)
    scheme_wrong_type(func_name, "integer", argnum, argc, argv);
  return value;
}

static int
SWIG_is_integer(Scheme_Object *o)
{
  long value;
  return scheme_get_int_val(o, &value);
}

static unsigned long
SWIG_convert_unsigned_integer(Scheme_Object *o,
			      unsigned long lower_bound, unsigned long upper_bound, 
			      const char *func_name, int argnum, int argc,
			      Scheme_Object **argv)
{
  unsigned long value;
  int status = scheme_get_unsigned_int_val(o, &value);
  if (!status)
    scheme_wrong_type(func_name, "integer", argnum, argc, argv);
  if (value < lower_bound || value > upper_bound)
    scheme_wrong_type(func_name, "integer", argnum, argc, argv);
  return value;
}

static int
SWIG_is_unsigned_integer(Scheme_Object *o)
{
  unsigned long value;
  return scheme_get_unsigned_int_val(o, &value);
}
  
/* ----------------------------------------------------------------------- 
 * mzscheme 30X support code
 * ----------------------------------------------------------------------- */

#ifndef SCHEME_STR_VAL
#define MZSCHEME30X 1
#endif

#ifdef MZSCHEME30X 
/* 
 * This is MZSCHEME 299.100 or higher (30x).  From version 299.100 of
 * mzscheme upwards, strings are in unicode. These functions convert
 * to and from utf8 encodings of these strings.  NB! strlen(s) will be
 * the size in bytes of the string, not the actual length.
 */
#define SCHEME_STR_VAL(obj)  	       SCHEME_BYTE_STR_VAL(scheme_char_string_to_byte_string(obj))
#define SCHEME_STRLEN_VAL(obj)         SCHEME_BYTE_STRLEN_VAL(scheme_char_string_to_byte_string(obj))
#define SCHEME_STRINGP(obj)            SCHEME_CHAR_STRINGP(obj)
#define scheme_make_string(s)          scheme_make_utf8_string(s)
#define scheme_make_sized_string(s,l)  scheme_make_sized_utf8_string(s,l)
#define scheme_make_sized_offset_string(s,d,l) \
                   scheme_make_sized_offset_utf8_string(s,d,l)
#define SCHEME_MAKE_STRING(s) scheme_make_utf8_string(s)
#else
#define SCHEME_MAKE_STRING(s) scheme_make_string_without_copying(s)
#endif
/* ----------------------------------------------------------------------- 
 * End of mzscheme 30X support code 
 * ----------------------------------------------------------------------- */
  
struct swig_mz_proxy {
  Scheme_Type mztype;
  swig_type_info *type;
  void *object;
};

static Scheme_Type swig_type;

static void 
mz_free_swig(void *p, void *data) {
  struct swig_mz_proxy *proxy = (struct swig_mz_proxy *) p;
  if (SCHEME_NULLP((Scheme_Object*)p) || SCHEME_TYPE((Scheme_Object*)p) != swig_type)
    return;
  if (proxy->type) {
    if (proxy->type->clientdata) {
      ((Scheme_Prim *)proxy->type->clientdata)(1, (Scheme_Object **)&proxy);
    }
  }
}

static Scheme_Object *
SWIG_MzScheme_NewPointerObj(void *ptr, swig_type_info *type, int owner) {
  struct swig_mz_proxy *new_proxy;
  new_proxy = (struct swig_mz_proxy *) scheme_malloc(sizeof(struct swig_mz_proxy));
  new_proxy->mztype = swig_type;
  new_proxy->type = type;
  new_proxy->object = ptr;
  if (owner) {
    scheme_add_finalizer(new_proxy, mz_free_swig, NULL);
  }
  return (Scheme_Object *) new_proxy;
}

static int
SWIG_MzScheme_ConvertPtr(Scheme_Object *s, void **result, swig_type_info *type, int flags) {
  swig_cast_info *cast;

  if (SCHEME_NULLP(s)) {
    *result = NULL;
    return (flags & SWIG_POINTER_NO_NULL) ? SWIG_NullReferenceError : SWIG_OK;
  } else if (SCHEME_TYPE(s) == swig_type) {
    struct swig_mz_proxy *proxy = (struct swig_mz_proxy *) s;
    if (type) {
      cast = SWIG_TypeCheckStruct(proxy->type, type);
      if (cast) {
        int newmemory = 0;
        *result = SWIG_TypeCast(cast, proxy->object, &newmemory);
        assert(!newmemory); /* newmemory handling not yet implemented */
        return 0;
      } else {
        return 1;
      }
    } else {
      *result = proxy->object;
      return 0;
    }
  }
  return 1;
}

static SWIGINLINE void *
SWIG_MzScheme_MustGetPtr(Scheme_Object *s, swig_type_info *type, 
                         int argnum, int flags, const char *func_name,
                         int argc, Scheme_Object **argv) {
  void *result;
  if (SWIG_MzScheme_ConvertPtr(s, &result, type, flags)) {
    scheme_wrong_type(func_name, type->str ? type->str : "void *", argnum - 1, argc, argv);
  }
  return result;
}

static SWIGINLINE void *
SWIG_MzScheme_Malloc(size_t size, const char *func_name) {
  void *p = malloc(size);
  if (p == NULL) {
    scheme_signal_error("swig-memory-error");
  } else return p;
}

static Scheme_Object *
SWIG_MzScheme_PackageValues(int num, Scheme_Object **values) {
    /* ignore first value if void */
    if (num > 0 && SCHEME_VOIDP(values[0]))
	num--, values++;
    if (num == 0) return scheme_void;
    else if (num == 1) return values[0];
    else return scheme_values(num, values);
}

#ifndef scheme_make_inspector
#define scheme_make_inspector(x,y) \
        _scheme_apply(scheme_builtin_value("make-inspector"), x, y)
#endif

/* Function to create a new struct. */
static Scheme_Object *
SWIG_MzScheme_new_scheme_struct (Scheme_Env* env, const char* basename, 
				 int num_fields, char** field_names)
{
    Scheme_Object *new_type;
    int count_out, i;
    Scheme_Object **struct_names;
    Scheme_Object **vals;
    Scheme_Object **a = (Scheme_Object**) \
        scheme_malloc(num_fields*sizeof(Scheme_Object*));
    
    for (i=0; i<num_fields; ++i) {
        a[i] = (Scheme_Object*) scheme_intern_symbol(field_names[i]);
    }

    new_type = scheme_make_struct_type(scheme_intern_symbol(basename),
                                       NULL /*super_type*/,
                                       scheme_make_inspector(0, NULL),
                                       num_fields,
                                       0 /* auto_fields */,
                                       NULL /* auto_val */,
                                       NULL /* properties */
#ifdef MZSCHEME30X
				       ,NULL /* Guard */
#endif
				       );
    struct_names = scheme_make_struct_names(scheme_intern_symbol(basename),
                                            scheme_build_list(num_fields,a),
                                            0 /*flags*/, &count_out);
    vals = scheme_make_struct_values(new_type, struct_names, count_out, 0);

    for (i = 0; i < count_out; i++)
        scheme_add_global_symbol(struct_names[i], vals[i],env);

    return new_type;
}

#if defined(_WIN32) || defined(__WIN32__)
#define __OS_WIN32
#endif

#ifdef __OS_WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

  static char **mz_dlopen_libraries=NULL;
  static void **mz_libraries=NULL;
  static char **mz_dynload_libpaths=NULL;

  static void mz_set_dlopen_libraries(const char *_libs)
  {
    int   i,k,n;
    int   mz_dynload_debug=(1==0);
    char *extra_paths[1000];
    char *EP;
    
    {
      char *dbg=getenv("MZ_DYNLOAD_DEBUG");
      if (dbg!=NULL) {
	mz_dynload_debug=atoi(dbg);
      }
    }

    {
      char *ep=getenv("MZ_DYNLOAD_LIBPATH");
      int   i,k,j;
      k=0;
      if (ep!=NULL) {
	EP=strdup(ep);
	for(i=0,j=0;EP[i]!='\0';i++) {
	  if (EP[i]==':') {
	    EP[i]='\0';
	    extra_paths[k++]=&EP[j];
	    j=i+1;
	  }
	}
	if (j!=i) {
	  extra_paths[k++]=&EP[j];
	}
      }
      else {
	EP=strdup("");
      }
      extra_paths[k]=NULL;
      k+=1;

      if (mz_dynload_debug) {
	fprintf(stderr,"SWIG:mzscheme:MZ_DYNLOAD_LIBPATH=%s\n",(ep==NULL) ? "(null)" : ep);
	fprintf(stderr,"SWIG:mzscheme:extra_paths[%d]\n",k-1);
	for(i=0;i<k-1;i++) {
	  fprintf(stderr,"SWIG:mzscheme:extra_paths[%d]=%s\n",i,extra_paths[i]);
	}
      }

      mz_dynload_libpaths=(char **) malloc(sizeof(char *)*k);
      for(i=0;i<k;i++) {
	if (extra_paths[i]!=NULL) {
	  mz_dynload_libpaths[i]=strdup(extra_paths[i]);
	}
	else {
	  mz_dynload_libpaths[i]=NULL;
	}
      }

      if (mz_dynload_debug) {
	int i;
	for(i=0;extra_paths[i]!=NULL;i++) {
	  fprintf(stderr,"SWIG:mzscheme:%s\n",extra_paths[i]);
	}
      }
    }

    {
#ifdef MZ_DYNLOAD_LIBS
      char *libs=(char *) malloc((strlen(MZ_DYNLOAD_LIBS)+1)*sizeof(char));
      strcpy(libs,MZ_DYNLOAD_LIBS);
#else
      char *libs=(char *) malloc((strlen(_libs)+1)*sizeof(char));
      strcpy(libs,_libs);
#endif
      
      for(i=0,n=strlen(libs),k=0;i<n;i++) {
	if (libs[i]==',') { k+=1; }
      }
      k+=1;
      mz_dlopen_libraries=(char **) malloc(sizeof(char *)*(k+1));
      mz_dlopen_libraries[0]=libs;
      for(i=0,k=1,n=strlen(libs);i<n;i++) {
	if (libs[i]==',') {
	  libs[i]='\0';
	  mz_dlopen_libraries[k++]=&libs[i+1];
	  i+=1;
	}
      }
      
      if (mz_dynload_debug) {
	fprintf(stderr,"k=%d\n",k);
      }
      mz_dlopen_libraries[k]=NULL;
      
      free(EP);
    }
  }

  static void *mz_load_function(char *function)
  {
    int mz_dynload_debug=(1==0);
    
    {
      char *dbg=getenv("MZ_DYNLOAD_DEBUG");
      if (dbg!=NULL) {
	mz_dynload_debug=atoi(dbg);
      }
    }

    if (mz_dlopen_libraries==NULL) {
      return NULL;
    }
    else {
      if (mz_libraries==NULL) {
        int i,n;
        for(n=0;mz_dlopen_libraries[n]!=NULL;n++);
	if (mz_dynload_debug) {
	  fprintf(stderr,"SWIG:mzscheme:n=%d\n",n);
	}
        mz_libraries=(void **) malloc(sizeof(void*)*n);
        for(i=0;i<n;i++) { 
	  if (mz_dynload_debug) {
	   fprintf(stderr,"SWIG:mzscheme:loading %s\n",mz_dlopen_libraries[i]);
	  }
#ifdef __OS_WIN32
	  mz_libraries[i]=(void *) LoadLibrary(mz_dlopen_libraries[i]); 
#else
	  mz_libraries[i]=(void *) dlopen(mz_dlopen_libraries[i],RTLD_LAZY); 
#endif
	  if (mz_libraries[i]==NULL) {
	    int k;
	    char *libp;
	    for(k=0;mz_dynload_libpaths[k]!=NULL && mz_libraries[i]==NULL;k++) {
	      int L=strlen(mz_dynload_libpaths[k])+strlen("\\")+strlen(mz_dlopen_libraries[i])+1;
	      libp=(char *) malloc(L*sizeof(char));
#ifdef __OS_WIN32
	      sprintf(libp,"%s\\%s",mz_dynload_libpaths[k],mz_dlopen_libraries[i]);
	      mz_libraries[i]=(void *) LoadLibrary(libp); 
#else
	      sprintf(libp,"%s/%s",mz_dynload_libpaths[k],mz_dlopen_libraries[i]);
	      mz_libraries[i]=(void *) dlopen(libp,RTLD_LAZY); 
#endif
	      if (mz_dynload_debug) {
		fprintf(stderr,"SWIG:mzscheme:trying %s --> %p\n",libp,mz_libraries[i]);
	      }
	      free(libp);
	    }
	  }
        }
      }
      {
        int i;
        void *func=NULL;

        for(i=0;mz_dlopen_libraries[i]!=NULL && func==NULL;i++) {
          if (mz_libraries[i]!=NULL) {
#ifdef __OS_WIN32
            func=GetProcAddress(mz_libraries[i],function);
#else
            func=dlsym(mz_libraries[i],function);
#endif
          }
	  if (mz_dynload_debug) {
	    fprintf(stderr,
		    "SWIG:mzscheme:library:%s;dlopen=%p,function=%s,func=%p\n",
		    mz_dlopen_libraries[i],mz_libraries[i],function,func
		    );
	  }
        }

        return func;
      }
    }
  }

/* The interpreter will store a pointer to this structure in a global
   variable called swig-runtime-data-type-pointer.  The instance of this
   struct is only used if no other module has yet been loaded */
struct swig_mzscheme_runtime_data {
  swig_module_info *module_head;
  Scheme_Type type;
};
static struct swig_mzscheme_runtime_data swig_mzscheme_runtime_data;


static swig_module_info *
SWIG_MzScheme_GetModule(Scheme_Env *env) {
  Scheme_Object *pointer, *symbol;
  struct swig_mzscheme_runtime_data *data;

  /* first check if pointer already created */
  symbol = scheme_intern_symbol("swig-runtime-data-type-pointer" SWIG_RUNTIME_VERSION SWIG_TYPE_TABLE_NAME);
  pointer = scheme_lookup_global(symbol, env);
  if (pointer && SCHEME_CPTRP(pointer)) {
      data = (struct swig_mzscheme_runtime_data *) SCHEME_CPTR_VAL(pointer);
      swig_type = data->type;
      return data->module_head;
  } else {
      return NULL;
  }
}

static void
SWIG_MzScheme_SetModule(Scheme_Env *env, swig_module_info *module) {
  Scheme_Object *pointer, *symbol;
  struct swig_mzscheme_runtime_data *data;

  /* first check if pointer already created */
  symbol = scheme_intern_symbol("swig-runtime-data-type-pointer" SWIG_RUNTIME_VERSION SWIG_TYPE_TABLE_NAME);
  pointer = scheme_lookup_global(symbol, env);
  if (pointer && SCHEME_CPTRP(pointer)) {
    data = (struct swig_mzscheme_runtime_data *) SCHEME_CPTR_VAL(pointer);
    swig_type = data->type;
    data->module_head = module;
  } else {
    /* create a new type for wrapped pointer values */
    swig_type = scheme_make_type((char *)"swig");
    swig_mzscheme_runtime_data.module_head = module;
    swig_mzscheme_runtime_data.type = swig_type;
    
    /* create a new pointer */
#ifndef MZSCHEME30X
    pointer = scheme_make_cptr((void *) &swig_mzscheme_runtime_data, "swig_mzscheme_runtime_data");
#else
    pointer = scheme_make_cptr((void *) &swig_mzscheme_runtime_data,
			       scheme_make_byte_string("swig_mzscheme_runtime_data"));
#endif
    scheme_add_global_symbol(symbol, pointer, env);
  }
}

#ifdef __cplusplus
}
#endif

