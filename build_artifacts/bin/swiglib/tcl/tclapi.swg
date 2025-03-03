/* -----------------------------------------------------------------------------
 * SWIG API. Portion that goes into the runtime
 * ----------------------------------------------------------------------------- */
#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------------
 * Constant declarations
 * ----------------------------------------------------------------------------- */

/* Constant Types */
#define SWIG_TCL_POINTER 4
#define SWIG_TCL_BINARY  5

/* Constant information structure */
typedef struct swig_const_info {
    int type;
    const char *name;
    long lvalue;
    double dvalue;
    void   *pvalue;
    swig_type_info **ptype;
} swig_const_info;

typedef int   (*swig_wrapper)(ClientData, Tcl_Interp *, int, Tcl_Obj *CONST []);
typedef int   (*swig_wrapper_func)(ClientData, Tcl_Interp *, int, Tcl_Obj *CONST []);
typedef char *(*swig_variable_func)(ClientData, Tcl_Interp *, char *, char *, int);
typedef void  (*swig_delete_func)(ClientData);

typedef struct swig_method {
  const char     *name;
  swig_wrapper   method;
} swig_method;

typedef struct swig_attribute {
  const char     *name;
  swig_wrapper   getmethod;
  swig_wrapper   setmethod;
} swig_attribute;

typedef struct swig_class {
  const char         *name;
  swig_type_info   **type;
  swig_wrapper       constructor;
  void              (*destructor)(void *);
  swig_method        *methods;
  swig_attribute     *attributes;
  struct swig_class **bases;
  const char              **base_names;
  swig_module_info   *module;
  Tcl_HashTable       hashtable;
} swig_class;

typedef struct swig_instance {
  Tcl_Obj       *thisptr;
  void          *thisvalue;
  swig_class   *classptr;
  int            destroy;
  Tcl_Command    cmdtok;
} swig_instance;

/* Structure for command table */
typedef struct {
  const char *name;
  int       (*wrapper)(ClientData, Tcl_Interp *, int, Tcl_Obj *CONST []);
  ClientData  clientdata;
} swig_command_info;

/* Structure for variable linking table */
typedef struct {
  const char *name;
  void *addr;
  char * (*get)(ClientData, Tcl_Interp *, char *, char *, int);
  char * (*set)(ClientData, Tcl_Interp *, char *, char *, int);
} swig_var_info;


/* -----------------------------------------------------------------------------*
 *  Install a constant object 
 * -----------------------------------------------------------------------------*/

static Tcl_HashTable   swigconstTable;
static int             swigconstTableinit = 0;

SWIGINTERN void
SWIG_Tcl_SetConstantObj(Tcl_Interp *interp, const char* name, Tcl_Obj *obj) {
  int newobj;
  Tcl_ObjSetVar2(interp,Tcl_NewStringObj(name,-1), NULL, obj, TCL_GLOBAL_ONLY);
  Tcl_SetHashValue(Tcl_CreateHashEntry(&swigconstTable, name, &newobj), (ClientData) obj);
}

SWIGINTERN Tcl_Obj *
SWIG_Tcl_GetConstantObj(const char *key) {
  Tcl_HashEntry *entryPtr;
  if (!swigconstTableinit) return 0;
  entryPtr = Tcl_FindHashEntry(&swigconstTable, key);
  if (entryPtr) {
    return (Tcl_Obj *) Tcl_GetHashValue(entryPtr);
  }
  return 0;
}

#ifdef __cplusplus
}
#endif


