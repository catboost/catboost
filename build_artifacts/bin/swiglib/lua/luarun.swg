/* -----------------------------------------------------------------------------
 * luarun.swg
 *
 * This file contains the runtime support for Lua modules
 * and includes code for managing global variables and pointer
 * type checking.
 * ----------------------------------------------------------------------------- */

#ifdef __cplusplus
extern "C" {
#endif

#include "lua.h"
#include "lauxlib.h"
#include <stdlib.h>  /* for malloc */
#include <assert.h>  /* for a few sanity tests */

/* -----------------------------------------------------------------------------
 * Lua flavors
 * ----------------------------------------------------------------------------- */

#define SWIG_LUA_FLAVOR_LUA 1
#define SWIG_LUA_FLAVOR_ELUA 2
#define SWIG_LUA_FLAVOR_ELUAC 3

#if !defined(SWIG_LUA_TARGET)
# error SWIG_LUA_TARGET not defined
#endif

#if defined(SWIG_LUA_ELUA_EMULATE)

struct swig_elua_entry;

typedef struct swig_elua_key {
  int type;
  union {
    const char* strkey;
    lua_Number numkey;
  } key;
} swig_elua_key;

typedef struct swig_elua_val {
  int type;
  union {
    lua_Number number;
    const struct swig_elua_entry *table;
    const char *string;
    lua_CFunction function;
    struct {
      char member;
      long lvalue;
      void *pvalue;
      swig_type_info **ptype;
    } userdata;
  } value;
} swig_elua_val;

typedef struct swig_elua_entry {
  swig_elua_key key;
  swig_elua_val value;
} swig_elua_entry;

#define LSTRKEY(x) {LUA_TSTRING, {.strkey = x} }
#define LNUMKEY(x) {LUA_TNUMBER, {.numkey = x} }
#define LNILKEY {LUA_TNIL, {.strkey = 0} }

#define LNUMVAL(x) {LUA_TNUMBER, {.number = x} }
#define LFUNCVAL(x) {LUA_TFUNCTION, {.function = x} }
#define LROVAL(x) {LUA_TTABLE, {.table = x} }
#define LNILVAL {LUA_TNIL, {.string = 0} }
#define LSTRVAL(x) {LUA_TSTRING, {.string = x} }

#define LUA_REG_TYPE swig_elua_entry

#define SWIG_LUA_ELUA_EMUL_METATABLE_KEY "__metatable"

#define lua_pushrotable(L,p)\
  lua_newtable(L);\
  assert(p);\
  SWIG_Lua_elua_emulate_register(L,(swig_elua_entry*)(p));

#define SWIG_LUA_CONSTTAB_POINTER(B,C,D)\
  LSTRKEY(B), {LUA_TUSERDATA, { .userdata={0,0,(void*)(C),&D} } }

#define SWIG_LUA_CONSTTAB_BINARY(B,S,C,D)\
  LSTRKEY(B), {LUA_TUSERDATA, { .userdata={1,S,(void*)(C),&D} } }
#endif

#if (SWIG_LUA_TARGET == SWIG_LUA_FLAVOR_ELUA) || (SWIG_LUA_TARGET == SWIG_LUA_FLAVOR_ELUAC)
#  define SWIG_LUA_CONSTTAB_INT(B, C) LSTRKEY(B), LNUMVAL(C)
#  define SWIG_LUA_CONSTTAB_FLOAT(B, C) LSTRKEY(B), LNUMVAL(C)
#  define SWIG_LUA_CONSTTAB_STRING(B, C) LSTRKEY(B), LSTRVAL(C)
#  define SWIG_LUA_CONSTTAB_CHAR(B, C) LSTRKEY(B), LNUMVAL(C)
    /* Those two types of constants are not supported in elua */

#ifndef SWIG_LUA_CONSTTAB_POINTER
#warning eLua does not support pointers as constants. By default, nil will be used as value
#define SWIG_LUA_CONSTTAB_POINTER(B,C,D) LSTRKEY(B), LNILVAL
#endif

#ifndef SWIG_LUA_CONSTTAB_BINARY
#warning eLua does not support pointers to member as constants. By default, nil will be used as value
#define SWIG_LUA_CONSTTAB_BINARY(B, S, C, D) LSTRKEY(B), LNILVAL
#endif
#else /* SWIG_LUA_FLAVOR_LUA */
#  define SWIG_LUA_CONSTTAB_INT(B, C) SWIG_LUA_INT, (char *)B, (long)C, 0, 0, 0
#  define SWIG_LUA_CONSTTAB_FLOAT(B, C) SWIG_LUA_FLOAT, (char *)B, 0, (double)C, 0, 0
#  define SWIG_LUA_CONSTTAB_STRING(B, C) SWIG_LUA_STRING, (char *)B, 0, 0, (void *)C, 0
#  define SWIG_LUA_CONSTTAB_CHAR(B, C) SWIG_LUA_CHAR, (char *)B, (long)C, 0, 0, 0
#  define SWIG_LUA_CONSTTAB_POINTER(B,C,D)\
       SWIG_LUA_POINTER, (char *)B, 0, 0, (void *)C, &D
#  define SWIG_LUA_CONSTTAB_BINARY(B, S, C, D)\
       SWIG_LUA_BINARY,  (char *)B, S, 0, (void *)C, &D
#endif

#ifndef SWIG_LUA_ELUA_EMULATE
#if (SWIG_LUA_TARGET == SWIG_LUA_FLAVOR_ELUA) || (SWIG_LUA_TARGET == SWIG_LUA_FLAVOR_ELUAC)
#  define LRO_STRVAL(v) {{.p = (char *) v}, LUA_TSTRING}
#  define LSTRVAL LRO_STRVAL
#endif
#endif /* SWIG_LUA_ELUA_EMULATE*/

#ifndef SWIG_LUA_ELUA_EMULATE
#if (SWIG_LUA_TARGET == SWIG_LUA_FLAVOR_ELUA) || (SWIG_LUA_TARGET == SWIG_LUA_FLAVOR_ELUAC)

#ifndef MIN_OPT_LEVEL
#define MIN_OPT_LEVEL 2
#endif

#include "lrodefs.h"
#include "lrotable.h"
#endif
#endif /* SWIG_LUA_ELUA_EMULATE*/
/* -----------------------------------------------------------------------------
 * compatibility defines
 * ----------------------------------------------------------------------------- */

/* History of Lua C API length functions:  In Lua 5.0 (and before?)
   there was "lua_strlen".  In Lua 5.1, this was renamed "lua_objlen",
   but a compatibility define of "lua_strlen" was added.  In Lua 5.2,
   this function was again renamed, to "lua_rawlen" (to emphasize that
   it doesn't call the "__len" metamethod), and the compatibility
   define of lua_strlen was removed.  All SWIG uses have been updated
   to "lua_rawlen", and we add our own defines of that here for older
   versions of Lua.  */
#if !defined(LUA_VERSION_NUM) || LUA_VERSION_NUM < 501
# define lua_rawlen lua_strlen
#elif LUA_VERSION_NUM == 501
# define lua_rawlen lua_objlen
#endif


/* lua_pushglobaltable is the recommended "future-proof" way to get
   the global table for Lua 5.2 and later.  Here we define
   lua_pushglobaltable ourselves for Lua versions before 5.2.  */
#if !defined(LUA_VERSION_NUM) || LUA_VERSION_NUM < 502
# define lua_pushglobaltable(L) lua_pushvalue(L, LUA_GLOBALSINDEX)
#endif

/* lua_absindex was introduced in Lua 5.2 */
#if !defined(LUA_VERSION_NUM) || LUA_VERSION_NUM < 502
# define lua_absindex(L,i) ((i)>0 || (i) <= LUA_REGISTRYINDEX ? (i) : lua_gettop(L) + (i) + 1)
#endif

/* lua_rawsetp was introduced in Lua 5.2 */
#if !defined(LUA_VERSION_NUM) || LUA_VERSION_NUM < 502
#define lua_rawsetp(L,index,ptr)\
  lua_pushlightuserdata(L,(void*)(ptr));\
  lua_insert(L,-2);\
  lua_rawset(L,index);

#define lua_rawgetp(L,index,ptr)\
  lua_pushlightuserdata(L,(void*)(ptr));\
  lua_rawget(L,index);

#endif

/* --------------------------------------------------------------------------
 * Helper functions for error handling
 * -------------------------------------------------------------------------- */

/* Push the string STR on the Lua stack, like lua_pushstring, but
   prefixed with the location of the innermost Lua call-point
   (as formatted by luaL_where).  */
SWIGRUNTIME void
SWIG_Lua_pusherrstring (lua_State *L, const char *str)
{
  luaL_where (L, 1);
  lua_pushstring (L, str);
  lua_concat (L, 2);
}

/* Push a formatted string generated from FMT and following args on
   the Lua stack, like lua_pushfstring, but prefixed with the
   location of the innermost Lua call-point (as formatted by luaL_where).  */
SWIGRUNTIME void
SWIG_Lua_pushferrstring (lua_State *L, const char *fmt, ...)
{
  va_list argp;
  va_start(argp, fmt);
  luaL_where(L, 1);
  lua_pushvfstring(L, fmt, argp);
  va_end(argp);
  lua_concat(L, 2);
}


/* -----------------------------------------------------------------------------
 * global swig types
 * ----------------------------------------------------------------------------- */
/* Constant table */
#define SWIG_LUA_INT     1
#define SWIG_LUA_FLOAT   2
#define SWIG_LUA_STRING  3
#define SWIG_LUA_POINTER 4
#define SWIG_LUA_BINARY  5
#define SWIG_LUA_CHAR    6

/* Structure for variable linking table */
typedef struct {
  const char *name;
  lua_CFunction get;
  lua_CFunction set;
} swig_lua_var_info;

#if (SWIG_LUA_TARGET == SWIG_LUA_FLAVOR_ELUA) || (SWIG_LUA_TARGET == SWIG_LUA_FLAVOR_ELUAC)
typedef const LUA_REG_TYPE swig_lua_method;
typedef const LUA_REG_TYPE swig_lua_const_info;
#else /* Normal lua */
typedef luaL_Reg swig_lua_method;

/* Constant information structure */
typedef struct {
    int type;
    char *name;
    long lvalue;
    double dvalue;
    void   *pvalue;
    swig_type_info **ptype;
} swig_lua_const_info;

#endif

typedef struct {
  const char     *name;
  lua_CFunction   getmethod;
  lua_CFunction   setmethod;
} swig_lua_attribute;


struct swig_lua_class;
/* Can be used to create namespaces. Currently used to wrap class static methods/variables/constants */
typedef struct swig_lua_namespace {
  const char            *name;
  swig_lua_method       *ns_methods;
  swig_lua_attribute    *ns_attributes;
  swig_lua_const_info   *ns_constants;
  struct swig_lua_class        **ns_classes;
  struct swig_lua_namespace    **ns_namespaces;
} swig_lua_namespace;

typedef struct swig_lua_class {
  const char    *name; /* Name that this class has in Lua */
  const char    *fqname; /* Fully qualified name - Scope + class name */
  swig_type_info   **type;
  lua_CFunction  constructor;
  void    (*destructor)(void *);
  swig_lua_method   *methods;
  swig_lua_attribute     *attributes;
  swig_lua_namespace    *cls_static;
  swig_lua_method   *metatable; /* 0 for -eluac */
  struct swig_lua_class **bases;
  const char **base_names;
} swig_lua_class;

/* this is the struct for wrapping all pointers in SwigLua
*/
typedef struct {
  swig_type_info   *type;
  int     own;  /* 1 if owned & must be destroyed */
  void        *ptr;
} swig_lua_userdata;

/* this is the struct for wrapping arbitrary packed binary data
(currently it is only used for member function pointers)
the data ordering is similar to swig_lua_userdata, but it is currently not possible
to tell the two structures apart within SWIG, other than by looking at the type
*/
typedef struct {
  swig_type_info   *type;
  int     own;  /* 1 if owned & must be destroyed */
  char data[1];       /* arbitrary amount of data */
} swig_lua_rawdata;

/* Common SWIG API */
#define SWIG_NewPointerObj(L, ptr, type, owner)       SWIG_Lua_NewPointerObj(L, (void *)ptr, type, owner)
#define SWIG_ConvertPtr(L,idx, ptr, type, flags)    SWIG_Lua_ConvertPtr(L,idx,ptr,type,flags)
#define SWIG_MustGetPtr(L,idx, type,flags, argnum,fnname)  SWIG_Lua_MustGetPtr(L,idx, type,flags, argnum,fnname)
/* for C++ member pointers, ie, member methods */
#define SWIG_ConvertMember(L, idx, ptr, sz, ty)       SWIG_Lua_ConvertPacked(L, idx, ptr, sz, ty)
#define SWIG_NewMemberObj(L, ptr, sz, type)      SWIG_Lua_NewPackedObj(L, ptr, sz, type)

/* Runtime API */
#define SWIG_GetModule(clientdata) SWIG_Lua_GetModule((lua_State*)(clientdata))
#define SWIG_SetModule(clientdata, pointer) SWIG_Lua_SetModule((lua_State*) (clientdata), pointer)
#define SWIG_MODULE_CLIENTDATA_TYPE lua_State*

/* Contract support */
#define SWIG_contract_assert(expr, msg)  \
  if (!(expr)) { SWIG_Lua_pusherrstring(L, (char *) msg); goto fail; } else


/* helper #defines */
#define SWIG_fail {goto fail;}
#define SWIG_fail_arg(func_name,argnum,type) \
  {SWIG_Lua_pushferrstring(L,"Error in %s (arg %d), expected '%s' got '%s'",\
  func_name,argnum,type,SWIG_Lua_typename(L,argnum));\
  goto fail;}
#define SWIG_fail_ptr(func_name,argnum,type) \
  SWIG_fail_arg(func_name,argnum,(type && type->str)?type->str:"void*")
#define SWIG_check_num_args(func_name,a,b) \
  if (lua_gettop(L)<a || lua_gettop(L)>b) \
  {SWIG_Lua_pushferrstring(L,"Error in %s expected %d..%d args, got %d",func_name,a,b,lua_gettop(L));\
  goto fail;}


#define SWIG_Lua_get_table(L,n) \
  (lua_pushstring(L, n), lua_rawget(L,-2))

#define SWIG_Lua_add_function(L,n,f) \
  (lua_pushstring(L, n), \
      lua_pushcfunction(L, f), \
      lua_rawset(L,-3))

#define SWIG_Lua_add_boolean(L,n,b) \
  (lua_pushstring(L, n), \
      lua_pushboolean(L, b), \
      lua_rawset(L,-3))

/* special helper for allowing 'nil' for usertypes */
#define SWIG_isptrtype(L,I) (lua_isuserdata(L,I) || lua_isnil(L,I))

#ifdef __cplusplus
/* Special helper for member function pointers
it gets the address, casts it, then dereferences it */
/*#define SWIG_mem_fn_as_voidptr(a)  (*((char**)&(a))) */
#endif

/* storing/access of swig_module_info */
SWIGRUNTIME swig_module_info *
SWIG_Lua_GetModule(lua_State *L) {
  swig_module_info *ret = 0;
  lua_pushstring(L,"swig_runtime_data_type_pointer" SWIG_RUNTIME_VERSION SWIG_TYPE_TABLE_NAME);
  lua_rawget(L,LUA_REGISTRYINDEX);
  if (lua_islightuserdata(L,-1))
    ret=(swig_module_info*)lua_touserdata(L,-1);
  lua_pop(L,1);  /* tidy */
  return ret;
}

SWIGRUNTIME void
SWIG_Lua_SetModule(lua_State *L, swig_module_info *module) {
  /* add this all into the Lua registry: */
  lua_pushstring(L,"swig_runtime_data_type_pointer" SWIG_RUNTIME_VERSION SWIG_TYPE_TABLE_NAME);
  lua_pushlightuserdata(L,(void*)module);
  lua_rawset(L,LUA_REGISTRYINDEX);
}

/* -----------------------------------------------------------------------------
 * global variable support code: modules
 * ----------------------------------------------------------------------------- */

/* this function is called when trying to set an immutable.
default action is to print an error.
This can removed with a compile flag SWIGLUA_IGNORE_SET_IMMUTABLE */
SWIGINTERN int SWIG_Lua_set_immutable(lua_State *L)
{
/*  there should be 1 param passed in: the new value */
#ifndef SWIGLUA_IGNORE_SET_IMMUTABLE
  lua_pop(L,1);  /* remove it */
  luaL_error(L,"This variable is immutable");
#endif
    return 0;   /* should not return anything */
}

#ifdef SWIG_LUA_ELUA_EMULATE

SWIGRUNTIME void SWIG_Lua_NewPointerObj(lua_State *L,void *ptr,swig_type_info *type, int own);
SWIGRUNTIME void SWIG_Lua_NewPackedObj(lua_State *L,void *ptr,size_t size,swig_type_info *type);
static int swig_lua_elua_emulate_unique_key;

/* This function emulates eLua rotables behaviour. It loads a rotable definition into the usual lua table. */
SWIGINTERN void SWIG_Lua_elua_emulate_register(lua_State *L, const swig_elua_entry *table)
{
  int i, table_parsed, parsed_tables_array, target_table;
  assert(lua_istable(L,-1));
  target_table = lua_gettop(L);
  /* Get the registry where we put all parsed tables to avoid loops */
  lua_rawgetp(L, LUA_REGISTRYINDEX, &swig_lua_elua_emulate_unique_key);
  if(lua_isnil(L,-1)) {
    lua_pop(L,1);
    lua_newtable(L);
    lua_pushvalue(L,-1);
    lua_rawsetp(L,LUA_REGISTRYINDEX,(void*)(&swig_lua_elua_emulate_unique_key));
  }
  parsed_tables_array = lua_gettop(L);
  lua_pushvalue(L,target_table);
  lua_rawsetp(L, parsed_tables_array, table);
  table_parsed = 0;
  const int SWIGUNUSED pairs_start = lua_gettop(L);
  for(i = 0;table[i].key.type != LUA_TNIL || table[i].value.type != LUA_TNIL;i++)
  {
    const swig_elua_entry *entry = table + i;
    int is_metatable = 0;
    switch(entry->key.type) {
      case LUA_TSTRING:
        lua_pushstring(L,entry->key.key.strkey);
        if(strcmp(entry->key.key.strkey, SWIG_LUA_ELUA_EMUL_METATABLE_KEY) == 0)
          is_metatable = 1;
        break;
      case  LUA_TNUMBER:
        lua_pushnumber(L,entry->key.key.numkey);
        break;
      case LUA_TNIL:
        lua_pushnil(L);
        break;
      default:
        assert(0);
    }
    switch(entry->value.type) {
      case LUA_TSTRING:
        lua_pushstring(L,entry->value.value.string);
        break;
      case  LUA_TNUMBER:
        lua_pushnumber(L,entry->value.value.number);
        break;
      case LUA_TFUNCTION:
        lua_pushcfunction(L,entry->value.value.function);
        break;
      case LUA_TTABLE:
        lua_rawgetp(L,parsed_tables_array, entry->value.value.table);
        table_parsed = !lua_isnil(L,-1);
        if(!table_parsed) {
          lua_pop(L,1); /*remove nil */
          lua_newtable(L);
          SWIG_Lua_elua_emulate_register(L,entry->value.value.table);
        }
        if(is_metatable) {
          assert(lua_istable(L,-1));
          lua_pushvalue(L,-1);
          lua_setmetatable(L,target_table);
        }

        break;
      case LUA_TUSERDATA:
        if(entry->value.value.userdata.member)
          SWIG_NewMemberObj(L,entry->value.value.userdata.pvalue,
              entry->value.value.userdata.lvalue,
              *(entry->value.value.userdata.ptype));
        else
          SWIG_NewPointerObj(L,entry->value.value.userdata.pvalue,
              *(entry->value.value.userdata.ptype),0);
        break;
      case LUA_TNIL:
        lua_pushnil(L);
        break;
      default:
        assert(0);
    }
    assert(lua_gettop(L) == pairs_start + 2);
    lua_rawset(L,target_table);
  }
  lua_pop(L,1); /* Removing parsed tables storage */
  assert(lua_gettop(L) == target_table);
}

SWIGINTERN void SWIG_Lua_elua_emulate_register_clear(lua_State *L)
{
  lua_pushnil(L);
  lua_rawsetp(L, LUA_REGISTRYINDEX, &swig_lua_elua_emulate_unique_key);
}

SWIGINTERN void SWIG_Lua_get_class_registry(lua_State *L);

SWIGINTERN int SWIG_Lua_emulate_elua_getmetatable(lua_State *L)
{
  SWIG_check_num_args("getmetatable(SWIG eLua emulation)", 1, 1);
  SWIG_Lua_get_class_registry(L);
  lua_getfield(L,-1,"lua_getmetatable");
  lua_remove(L,-2); /* remove the registry*/
  assert(!lua_isnil(L,-1));
  lua_pushvalue(L,1);
  assert(lua_gettop(L) == 3); /* object | function | object again */
  lua_call(L,1,1);
  if(!lua_isnil(L,-1)) /*There is an ordinary metatable */
    return 1;
  /*if it is a table, then emulate elua behaviour - check for __metatable attribute of a table*/
  assert(lua_gettop(L) == 2);
  if(lua_istable(L,-2)) {
    lua_pop(L,1); /*remove the nil*/
    lua_getfield(L,-1, SWIG_LUA_ELUA_EMUL_METATABLE_KEY);
  }
  assert(lua_gettop(L) == 2);
  return 1;

fail:
  lua_error(L);
  return 0;
}

SWIGINTERN void SWIG_Lua_emulate_elua_swap_getmetatable(lua_State *L)
{
  SWIG_Lua_get_class_registry(L);
  lua_pushglobaltable(L);
  lua_pushstring(L,"lua_getmetatable");
  lua_getfield(L,-2,"getmetatable");
  assert(!lua_isnil(L,-1));
  lua_rawset(L,-4);
  lua_pushstring(L, "getmetatable");
  lua_pushcfunction(L, SWIG_Lua_emulate_elua_getmetatable);
  lua_rawset(L,-3);
  lua_pop(L,2);

}
/* END OF REMOVE */

#endif
/* -----------------------------------------------------------------------------
 * global variable support code: namespaces and modules (which are the same thing)
 * ----------------------------------------------------------------------------- */

SWIGINTERN int SWIG_Lua_namespace_get(lua_State *L)
{
/*  there should be 2 params passed in
  (1) table (not the meta table)
  (2) string name of the attribute
*/
  assert(lua_istable(L,-2));  /* just in case */
  lua_getmetatable(L,-2);
  assert(lua_istable(L,-1));
  SWIG_Lua_get_table(L,".get"); /* find the .get table */
  assert(lua_istable(L,-1));
  /* look for the key in the .get table */
  lua_pushvalue(L,2);  /* key */
  lua_rawget(L,-2);
  lua_remove(L,-2); /* stack tidy, remove .get table */
  if (lua_iscfunction(L,-1))
  {  /* found it so call the fn & return its value */
    lua_call(L,0,1);  /* 1 value in (userdata),1 out (result) */
    lua_remove(L,-2); /* stack tidy, remove metatable */
    return 1;
  }
  lua_pop(L,1);  /* remove whatever was there */
  /* ok, so try the .fn table */
  SWIG_Lua_get_table(L,".fn"); /* find the .get table */
  assert(lua_istable(L,-1));  /* just in case */
  lua_pushvalue(L,2);  /* key */
  lua_rawget(L,-2);  /* look for the fn */
  lua_remove(L,-2); /* stack tidy, remove .fn table */
  if (lua_isfunction(L,-1)) /* note: whether it's a C function or lua function */
  {  /* found it so return the fn & let lua call it */
    lua_remove(L,-2); /* stack tidy, remove metatable */
    return 1;
  }
  lua_pop(L,1);  /* remove whatever was there */
  return 0;
}

SWIGINTERN int SWIG_Lua_namespace_set(lua_State *L)
{
/*  there should be 3 params passed in
  (1) table (not the meta table)
  (2) string name of the attribute
  (3) any for the new value
*/

  assert(lua_istable(L,1));
  lua_getmetatable(L,1);    /* get the meta table */
  assert(lua_istable(L,-1));

  SWIG_Lua_get_table(L,".set"); /* find the .set table */
  if (lua_istable(L,-1))
  {
    /* look for the key in the .set table */
    lua_pushvalue(L,2);  /* key */
    lua_rawget(L,-2);
    if (lua_iscfunction(L,-1))
    {  /* found it so call the fn & return its value */
      lua_pushvalue(L,3);  /* value */
      lua_call(L,1,0);
      return 0;
    }
    lua_pop(L,1);  /* remove the value */
  }
  lua_pop(L,1);  /* remove the value .set table */
  lua_pop(L,1); /* remote metatable */
  lua_rawset(L,-3);
  return 0;
}

#if (SWIG_LUA_TARGET == SWIG_LUA_FLAVOR_LUA) /* In elua this is useless */
SWIGINTERN void SWIG_Lua_InstallConstants(lua_State *L, swig_lua_const_info constants[]); /* forward declaration */
SWIGINTERN void  SWIG_Lua_add_variable(lua_State *L,const char *name,lua_CFunction getFn,lua_CFunction setFn); /* forward declaration */
SWIGINTERN void  SWIG_Lua_class_register(lua_State *L,swig_lua_class *clss);

/* helper function - register namespace methods and attributes into namespace */
SWIGINTERN int SWIG_Lua_add_namespace_details(lua_State *L, swig_lua_namespace *ns)
{
  int i;
  /* There must be namespace table (not metatable) at the top of the stack */
  assert(lua_istable(L,-1));
  SWIG_Lua_InstallConstants(L, ns->ns_constants);

  /* add methods to the namespace/module table */
  for(i=0;ns->ns_methods[i].name;i++){
    SWIG_Lua_add_function(L,ns->ns_methods[i].name,ns->ns_methods[i].func);
  }
  lua_getmetatable(L,-1);

  /* add fns */
  for(i=0;ns->ns_attributes[i].name;i++){
    SWIG_Lua_add_variable(L,ns->ns_attributes[i].name,ns->ns_attributes[i].getmethod,ns->ns_attributes[i].setmethod);
  }

  /* clear stack - remove metatble */
  lua_pop(L,1);
  return 0;
}

/* Register all classes in the namespace */
SWIGINTERN void SWIG_Lua_add_namespace_classes(lua_State *L, swig_lua_namespace *ns)
{
  swig_lua_class **classes;

  /* There must be a module/namespace table at the top of the stack */
  assert(lua_istable(L,-1));

  classes = ns->ns_classes;

  if( classes != 0 ) {
    while(*classes != 0) {
      SWIG_Lua_class_register(L, *classes);
      classes++;
    }
  }
}

/* Helper function. Creates namespace table and adds it to module table
   if 'reg' is true, then will register namespace table to parent one (must be on top of the stack
   when function is called).
   Function always returns newly registered table on top of the stack.
*/
SWIGINTERN void SWIG_Lua_namespace_register(lua_State *L, swig_lua_namespace *ns, int reg)
{
  swig_lua_namespace **sub_namespace;
  /* 1 argument - table on the top of the stack */
  const int SWIGUNUSED begin = lua_gettop(L);
  assert(lua_istable(L,-1)); /* just in case. This is supposed to be module table or parent namespace table */
  lua_checkstack(L,5);
  lua_newtable(L); /* namespace itself */
  lua_newtable(L); /* metatable for namespace */

  /* add a table called ".get" */
  lua_pushstring(L,".get");
  lua_newtable(L);
  lua_rawset(L,-3);
  /* add a table called ".set" */
  lua_pushstring(L,".set");
  lua_newtable(L);
  lua_rawset(L,-3);
  /* add a table called ".fn" */
  lua_pushstring(L,".fn");
  lua_newtable(L);
  lua_rawset(L,-3);

  /* add accessor fns for using the .get,.set&.fn */
  SWIG_Lua_add_function(L,"__index",SWIG_Lua_namespace_get);
  SWIG_Lua_add_function(L,"__newindex",SWIG_Lua_namespace_set);

  lua_setmetatable(L,-2); /* set metatable */

  /* Register all functions, variables etc */
  SWIG_Lua_add_namespace_details(L,ns);
  /* Register classes */
  SWIG_Lua_add_namespace_classes(L,ns);

  sub_namespace = ns->ns_namespaces;
  if( sub_namespace != 0) {
    while(*sub_namespace != 0) {
      SWIG_Lua_namespace_register(L, *sub_namespace, 1);
      lua_pop(L,1); /* removing sub-namespace table */
      sub_namespace++;
    }
  }

  if (reg) {
    lua_pushstring(L,ns->name);
    lua_pushvalue(L,-2);
    lua_rawset(L,-4); /* add namespace to module table */
  }
  assert(lua_gettop(L) == begin+1);
}
#endif /* SWIG_LUA_TARGET == SWIG_LUA_FLAVOR_LUA */

/* -----------------------------------------------------------------------------
 * global variable support code: classes
 * ----------------------------------------------------------------------------- */

SWIGINTERN void  SWIG_Lua_get_class_metatable(lua_State *L,const char *cname);

typedef int (*swig_lua_base_iterator_func)(lua_State*,swig_type_info*, int, int *ret);

SWIGINTERN int SWIG_Lua_iterate_bases(lua_State *L, swig_type_info * SWIGUNUSED swig_type,
  int first_arg, swig_lua_base_iterator_func func, int  *const ret)
{
    /* first_arg - position of the object in stack. Everything that is above are arguments
     * and is passed to every evocation of the func */
    int last_arg = lua_gettop(L);/* position of last argument */
    int original_metatable = last_arg + 1;
    size_t bases_count;
    int result = SWIG_ERROR;
    int bases_table;
    (void)swig_type;
    lua_getmetatable(L,first_arg);

    /* initialise base search */
#if (SWIG_LUA_TARGET == SWIG_LUA_FLAVOR_LUA)
    SWIG_Lua_get_table(L,".bases");
    assert(lua_istable(L,-1));
    bases_count = lua_rawlen(L,-1);
    bases_table = lua_gettop(L);
#else
    /* In elua .bases table doesn't exist. Use table from swig_lua_class */
    (void)bases_table;
    assert(swig_type!=0);
    swig_module_info *module=SWIG_GetModule(L);
    swig_lua_class **bases= ((swig_lua_class*)(swig_type->clientdata))->bases;
    const char **base_names= ((swig_lua_class*)(swig_type->clientdata))->base_names;
    bases_count = 0;
    for(;base_names[bases_count];
      bases_count++);/* get length of bases */
#endif

    if(ret)
      *ret = 0;
    if(bases_count>0)
    {
      int to_remove;
      size_t i;
      int j;
      int subcall_last_arg;
      int subcall_first_arg = lua_gettop(L) + 1;/* Here a copy of first_arg and arguments begin */
      int valid = 1;
      swig_type_info *base_swig_type = 0;
      for(j=first_arg;j<=last_arg;j++)
        lua_pushvalue(L,j);
      subcall_last_arg = lua_gettop(L);

      /* Trick: temporarily replacing original metatable with metatable for base class and call getter */
      for(i=0;i<bases_count;i++) {
        /* Iteration through class bases */
#if (SWIG_LUA_TARGET == SWIG_LUA_FLAVOR_LUA)
        lua_rawgeti(L,bases_table,i+1);
        base_swig_type = 0;
        if(lua_isnil(L,-1)) {
          valid = 0;
          lua_pop(L,1);
        } else {
          valid = 1;
        }
#else /* In elua .bases table doesn't exist. Use table from swig_lua_class */
        swig_lua_class *base_class = bases[i];
        if(!base_class) {
          valid = 0;
        } else {
          valid = 1;
          SWIG_Lua_get_class_metatable(L,base_class->fqname);
          base_swig_type = SWIG_TypeQueryModule(module,module,base_names[i]);
          assert(base_swig_type != 0);
        }
#endif

        if(!valid)
          continue;
        assert(lua_isuserdata(L, subcall_first_arg));
        assert(lua_istable(L,-1));
        lua_setmetatable(L,subcall_first_arg); /* Set new metatable */
        assert(lua_gettop(L) == subcall_last_arg);
        result = func(L, base_swig_type,subcall_first_arg, ret); /* Forward call */
        if(result != SWIG_ERROR) {
          break;
        }
      }
      /* Restore original metatable */
      lua_pushvalue(L,original_metatable);
      lua_setmetatable(L,first_arg);
      /* Clear - remove everything between last_arg and subcall_last_arg including */
      to_remove = subcall_last_arg - last_arg;
      for(j=0;j<to_remove;j++)
        lua_remove(L,last_arg+1);
    } else {
      /* Remove everything after last_arg */
      lua_pop(L, lua_gettop(L) - last_arg);
    }
    if(ret) assert(lua_gettop(L) == last_arg + *ret);
    return result;
}

/* The class.get method helper, performs the lookup of class attributes.
 * It returns an error code. Number of function return values is passed inside 'ret'.
 * first_arg is not used in this function because function always has 2 arguments.
 */
SWIGINTERN int  SWIG_Lua_class_do_get_item(lua_State *L, swig_type_info *type, int SWIGUNUSED first_arg, int *ret)
{
/*  there should be 2 params passed in
  (1) userdata (not the meta table)
  (2) string name of the attribute
*/
  int bases_search_result;
  int substack_start = lua_gettop(L)-2;
  assert(first_arg == substack_start+1);
  lua_checkstack(L,5);
  assert(lua_isuserdata(L,-2));  /* just in case */
  lua_getmetatable(L,-2);    /* get the meta table */
  assert(lua_istable(L,-1));  /* just in case */
  /* NEW: looks for the __getitem() fn
  this is a user provided get fn */
  SWIG_Lua_get_table(L,"__getitem"); /* find the __getitem fn */
  if (lua_iscfunction(L,-1))  /* if its there */
  {  /* found it so call the fn & return its value */
    lua_pushvalue(L,substack_start+1);  /* the userdata */
    lua_pushvalue(L,substack_start+2);  /* the parameter */
    lua_call(L,2,1);  /* 2 value in (userdata),1 out (result) */
    lua_remove(L,-2); /* stack tidy, remove metatable */
    if(ret) *ret = 1;
    return SWIG_OK;
  }
  lua_pop(L,1);
  /* Remove the metatable */
  lua_pop(L,1);
  /* Search in base classes */
  bases_search_result = SWIG_Lua_iterate_bases(L,type,substack_start+1,SWIG_Lua_class_do_get_item,ret);
  return bases_search_result;  /* sorry not known */
}


/* The class.get method helper, performs the lookup of class attributes.
 * It returns an error code. Number of function return values is passed inside 'ret'.
 * first_arg is not used in this function because function always has 2 arguments.
 */
SWIGINTERN int  SWIG_Lua_class_do_get(lua_State *L, swig_type_info *type, int SWIGUNUSED first_arg, int *ret)
{
/*  there should be 2 params passed in
  (1) userdata (not the meta table)
  (2) string name of the attribute
*/
  int bases_search_result;
  int substack_start = lua_gettop(L)-2;
  assert(first_arg == substack_start+1);
  lua_checkstack(L,5);
  assert(lua_isuserdata(L,-2));  /* just in case */
  lua_getmetatable(L,-2);    /* get the meta table */
  assert(lua_istable(L,-1));  /* just in case */
  SWIG_Lua_get_table(L,".get"); /* find the .get table */
  assert(lua_istable(L,-1));  /* just in case */
  /* look for the key in the .get table */
  lua_pushvalue(L,substack_start+2);  /* key */
  lua_rawget(L,-2);
  lua_remove(L,-2); /* stack tidy, remove .get table */
  if (lua_iscfunction(L,-1))
  {  /* found it so call the fn & return its value */
    lua_pushvalue(L,substack_start+1);  /* the userdata */
    lua_call(L,1,1);  /* 1 value in (userdata),1 out (result) */
    lua_remove(L,-2); /* stack tidy, remove metatable */
    if(ret)
      *ret = 1;
    return SWIG_OK;
  }
  lua_pop(L,1);  /* remove whatever was there */
  /* ok, so try the .fn table */
  SWIG_Lua_get_table(L,".fn"); /* find the .fn table */
  assert(lua_istable(L,-1));  /* just in case */
  lua_pushvalue(L,substack_start+2);  /* key */
  lua_rawget(L,-2);  /* look for the fn */
  lua_remove(L,-2); /* stack tidy, remove .fn table */
  if (lua_isfunction(L,-1)) /* note: if its a C function or lua function */
  {  /* found it so return the fn & let lua call it */
    lua_remove(L,-2); /* stack tidy, remove metatable */
    if(ret)
      *ret = 1;
    return SWIG_OK;
  }
  lua_pop(L,1);  /* remove whatever was there */
  /* Remove the metatable */
  lua_pop(L,1);
  /* Search in base classes */
  bases_search_result = SWIG_Lua_iterate_bases(L,type,substack_start+1,SWIG_Lua_class_do_get,ret);
  return bases_search_result;  /* sorry not known */
}

/* the class.get method, performs the lookup of class attributes
 */
SWIGINTERN int  SWIG_Lua_class_get(lua_State *L)
{
/*  there should be 2 params passed in
  (1) userdata (not the meta table)
  (2) string name of the attribute
*/
  int result;
  swig_lua_userdata *usr;
  swig_type_info *type;
  int ret = 0;
  assert(lua_isuserdata(L,1));
  usr=(swig_lua_userdata*)lua_touserdata(L,1);  /* get data */
  type = usr->type;
  result = SWIG_Lua_class_do_get(L,type,1,&ret);
  if(result == SWIG_OK)
    return ret;

  result = SWIG_Lua_class_do_get_item(L,type,1,&ret);
  if(result == SWIG_OK)
    return ret;

  return 0;
}

/* helper for the class.set method, performs the lookup of class attributes
 * It returns error code. Number of function return values is passed inside 'ret'
 */
SWIGINTERN int  SWIG_Lua_class_do_set(lua_State *L, swig_type_info *type, int first_arg, int *ret)
{
/*  there should be 3 params passed in
  (1) table (not the meta table)
  (2) string name of the attribute
  (3) any for the new value
  */

  int bases_search_result;
  int substack_start = lua_gettop(L) - 3;
  lua_checkstack(L,5);
  assert(lua_isuserdata(L,substack_start+1));  /* just in case */
  lua_getmetatable(L,substack_start+1);    /* get the meta table */
  assert(lua_istable(L,-1));  /* just in case */
  if(ret)
    *ret = 0; /* it is setter - number of return values is always 0 */

  SWIG_Lua_get_table(L,".set"); /* find the .set table */
  if (lua_istable(L,-1))
  {
    /* look for the key in the .set table */
    lua_pushvalue(L,substack_start+2);  /* key */
    lua_rawget(L,-2);
    lua_remove(L,-2); /* tidy stack, remove .set table */
    if (lua_iscfunction(L,-1))
    {  /* found it so call the fn & return its value */
      lua_pushvalue(L,substack_start+1);  /* userdata */
      lua_pushvalue(L,substack_start+3);  /* value */
      lua_call(L,2,0);
      lua_remove(L,substack_start+4); /*remove metatable*/
      return SWIG_OK;
    }
    lua_pop(L,1);  /* remove the value */
  } else {
    lua_pop(L,1);  /* remove the answer for .set table request*/
  }
  /* NEW: looks for the __setitem() fn
  this is a user provided set fn */
  SWIG_Lua_get_table(L,"__setitem"); /* find the fn */
  if (lua_iscfunction(L,-1))  /* if its there */
  {  /* found it so call the fn & return its value */
    lua_pushvalue(L,substack_start+1);  /* the userdata */
    lua_pushvalue(L,substack_start+2);  /* the parameter */
    lua_pushvalue(L,substack_start+3);  /* the value */
    lua_call(L,3,0);  /* 3 values in ,0 out */
    lua_remove(L,-2); /* stack tidy, remove metatable */
    return SWIG_OK;
  }
  lua_pop(L,1); /* remove value */

  lua_pop(L,1); /* remove metatable */
  /* Search among bases */
  bases_search_result = SWIG_Lua_iterate_bases(L,type,first_arg,SWIG_Lua_class_do_set,ret);
  if(ret)
    assert(*ret == 0);
  assert(lua_gettop(L) == substack_start + 3);
  return bases_search_result;
}

/* This is the actual method exported to Lua. It calls SWIG_Lua_class_do_set and correctly
 * handles return values.
 */
SWIGINTERN int  SWIG_Lua_class_set(lua_State *L)
{
/*  There should be 3 params passed in
  (1) table (not the meta table)
  (2) string name of the attribute
  (3) any for the new value
  */
  int ret = 0;
  int result;
  swig_lua_userdata *usr;
  swig_type_info *type;
  assert(lua_isuserdata(L,1));
  usr=(swig_lua_userdata*)lua_touserdata(L,1);  /* get data */
  type = usr->type;
  result = SWIG_Lua_class_do_set(L,type,1,&ret);
  if(result != SWIG_OK) {
   SWIG_Lua_pushferrstring(L,"Assignment not possible. No setter/member with this name. For custom assignments implement __setitem method.");
   lua_error(L);
  } else {
    assert(ret==0);
  }
  return 0;
}

/* the class.destruct method called by the interpreter */
SWIGINTERN int  SWIG_Lua_class_destruct(lua_State *L)
{
/*  there should be 1 params passed in
  (1) userdata (not the meta table) */
  swig_lua_userdata *usr;
  swig_lua_class *clss;
  assert(lua_isuserdata(L,-1));  /* just in case */
  usr=(swig_lua_userdata*)lua_touserdata(L,-1);  /* get it */
  /* if must be destroyed & has a destructor */
  if (usr->own) /* if must be destroyed */
  {
    clss=(swig_lua_class*)usr->type->clientdata;  /* get the class */
    if (clss && clss->destructor)  /* there is a destroy fn */
    {
      clss->destructor(usr->ptr);  /* bye bye */
    }
  }
  return 0;
}

/* the class.__tostring method called by the interpreter and print */
SWIGINTERN int  SWIG_Lua_class_tostring(lua_State *L)
{
/*  there should be 1 param passed in
  (1) userdata (not the metatable) */
  swig_lua_userdata* userData;
  assert(lua_isuserdata(L,1));  /* just in case */
  userData = (swig_lua_userdata*)lua_touserdata(L,1); /* get the userdata address */

  lua_pushfstring(L, "<userdata of type '%s' at %p>", userData->type->str, userData->ptr);
  return 1;
}

/* to manually disown some userdata */
SWIGINTERN int  SWIG_Lua_class_disown(lua_State *L)
{
/*  there should be 1 params passed in
  (1) userdata (not the meta table) */
  swig_lua_userdata *usr;
  assert(lua_isuserdata(L,-1));  /* just in case */
  usr=(swig_lua_userdata*)lua_touserdata(L,-1);  /* get it */

  usr->own = 0; /* clear our ownership */
  return 0;
}

/* lua callable function to compare userdata's value
the issue is that two userdata may point to the same thing
but to lua, they are different objects */
SWIGRUNTIME int SWIG_Lua_class_equal(lua_State *L)
{
  int result;
  swig_lua_userdata *usr1,*usr2;
  if (!lua_isuserdata(L,1) || !lua_isuserdata(L,2))  /* just in case */
    return 0;  /* nil reply */
  usr1=(swig_lua_userdata*)lua_touserdata(L,1);  /* get data */
  usr2=(swig_lua_userdata*)lua_touserdata(L,2);  /* get data */
  /*result=(usr1->ptr==usr2->ptr && usr1->type==usr2->type); only works if type is the same*/
  result=(usr1->ptr==usr2->ptr);
   lua_pushboolean(L,result);
  return 1;
}

/* populate table at the top of the stack with metamethods that ought to be inherited */
SWIGINTERN void SWIG_Lua_populate_inheritable_metamethods(lua_State *L)
{
  SWIG_Lua_add_boolean(L, "__add", 1);
  SWIG_Lua_add_boolean(L, "__sub", 1);
  SWIG_Lua_add_boolean(L, "__mul", 1);
  SWIG_Lua_add_boolean(L, "__div", 1);
  SWIG_Lua_add_boolean(L, "__mod", 1);
  SWIG_Lua_add_boolean(L, "__pow", 1);
  SWIG_Lua_add_boolean(L, "__unm", 1);
  SWIG_Lua_add_boolean(L, "__len", 1 );
  SWIG_Lua_add_boolean(L, "__concat", 1 );
  SWIG_Lua_add_boolean(L, "__eq", 1);
  SWIG_Lua_add_boolean(L, "__lt", 1);
  SWIG_Lua_add_boolean(L, "__le", 1);
  SWIG_Lua_add_boolean(L, "__call", 1);
  SWIG_Lua_add_boolean(L, "__tostring", 1);
  SWIG_Lua_add_boolean(L, "__gc", 0);
}

/* creates the swig registry */
SWIGINTERN void SWIG_Lua_create_class_registry(lua_State *L)
{
  /* create main SWIG registry table */
  lua_pushstring(L,"SWIG");
  lua_newtable(L);
  /* populate it with some predefined data */

  /* .library table. Placeholder */
  lua_pushstring(L,".library");
  lua_newtable(L);
  {
    /* list of metamethods that class inherits from its bases */
    lua_pushstring(L,"inheritable_metamethods");
    lua_newtable(L);
    /* populate with list of metamethods */
    SWIG_Lua_populate_inheritable_metamethods(L);
    lua_rawset(L,-3);
  }
  lua_rawset(L,-3);

  lua_rawset(L,LUA_REGISTRYINDEX);
}

/* gets the swig registry (or creates it) */
SWIGINTERN void  SWIG_Lua_get_class_registry(lua_State *L)
{
  /* add this all into the swig registry: */
  lua_pushstring(L,"SWIG");
  lua_rawget(L,LUA_REGISTRYINDEX);  /* get the registry */
  if (!lua_istable(L,-1))  /* not there */
  {  /* must be first time, so add it */
    lua_pop(L,1);  /* remove the result */
    SWIG_Lua_create_class_registry(L);
    /* then get it */
    lua_pushstring(L,"SWIG");
    lua_rawget(L,LUA_REGISTRYINDEX);
  }
}

SWIGINTERN void SWIG_Lua_get_inheritable_metamethods(lua_State *L)
{
  SWIG_Lua_get_class_registry(L);
  lua_pushstring(L, ".library");
  lua_rawget(L,-2);
  assert( !lua_isnil(L,-1) );
  lua_pushstring(L, "inheritable_metamethods");
  lua_rawget(L,-2);

  /* Remove class registry and library table */
  lua_remove(L,-2);
  lua_remove(L,-2);
}

/* Helper function to get the classes metatable from the register */
SWIGINTERN void  SWIG_Lua_get_class_metatable(lua_State *L,const char *cname)
{
  SWIG_Lua_get_class_registry(L);  /* get the registry */
  lua_pushstring(L,cname);  /* get the name */
  lua_rawget(L,-2);    /* get it */
  lua_remove(L,-2);    /* tidy up (remove registry) */
}

/* Set up the base classes pointers.
Each class structure has a list of pointers to the base class structures.
This function fills them.
It cannot be done at compile time, as this will not work with hireachies
spread over more than one swig file.
Therefore it must be done at runtime, querying the SWIG type system.
*/
SWIGINTERN void SWIG_Lua_init_base_class(lua_State *L,swig_lua_class *clss)
{
  int i=0;
  swig_module_info *module=SWIG_GetModule(L);
  for(i=0;clss->base_names[i];i++)
  {
    if (clss->bases[i]==0) /* not found yet */
    {
      /* lookup and cache the base class */
      swig_type_info *info = SWIG_TypeQueryModule(module,module,clss->base_names[i]);
      if (info) clss->bases[i] = (swig_lua_class *) info->clientdata;
    }
  }
}

#if defined(SWIG_LUA_SQUASH_BASES) && (SWIG_LUA_TARGET == SWIG_LUA_FLAVOR_LUA)
/* Merges two tables  */
SWIGINTERN void SWIG_Lua_merge_tables_by_index(lua_State *L, int target, int source)
{
  /* iterating */
  lua_pushnil(L);
  while (lua_next(L,source) != 0) {
    /* -1 - value, -2 - index */
    /* have to copy to assign */
    lua_pushvalue(L,-2); /* copy of index */
    lua_pushvalue(L,-2); /* copy of value */
    lua_rawset(L, target);
    lua_pop(L,1);
    /* only key is left */
  }
}

/* Merges two tables with given name. original - index of target metatable, base - index of source metatable */
SWIGINTERN void SWIG_Lua_merge_tables(lua_State *L, const char* name, int original, int base)
{
  /* push original[name], then base[name] */
  lua_pushstring(L,name);
  lua_rawget(L,original);
  int original_table = lua_gettop(L);
  lua_pushstring(L,name);
  lua_rawget(L,base);
  int base_table = lua_gettop(L);
  SWIG_Lua_merge_tables_by_index(L, original_table, base_table);
  /* clearing stack */
  lua_pop(L,2);
}

/* Function takes all symbols from base and adds it to derived class. It's just a helper. */
SWIGINTERN void SWIG_Lua_class_squash_base(lua_State *L, swig_lua_class *base_cls)
{
  /* There is one parameter - original, i.e. 'derived' class metatable */
  assert(lua_istable(L,-1));
  int original = lua_gettop(L);
  SWIG_Lua_get_class_metatable(L,base_cls->fqname);
  int base = lua_gettop(L);
  SWIG_Lua_merge_tables(L, ".fn", original, base );
  SWIG_Lua_merge_tables(L, ".set", original, base );
  SWIG_Lua_merge_tables(L, ".get", original, base );
  lua_pop(L,1);
}

/* Function squashes all symbols from 'clss' bases into itself */
SWIGINTERN void  SWIG_Lua_class_squash_bases(lua_State *L, swig_lua_class *clss)
{
  int i;
  SWIG_Lua_get_class_metatable(L,clss->fqname);
  for(i=0;clss->base_names[i];i++)
  {
    if (clss->bases[i]==0) /* Somehow it's not found. Skip it */
      continue;
    /* Thing is: all bases are already registered. Thus they have already executed
     * this function. So we just need to squash them into us, because their bases
     * are already squashed into them. No need for recursion here!
     */
    SWIG_Lua_class_squash_base(L, clss->bases[i]);
  }
  lua_pop(L,1); /*tidy stack*/
}
#endif

#if (SWIG_LUA_TARGET == SWIG_LUA_FLAVOR_LUA) /* In elua this is useless */
/* helper add a variable to a registered class */
SWIGINTERN void  SWIG_Lua_add_variable(lua_State *L,const char *name,lua_CFunction getFn,lua_CFunction setFn)
{
  assert(lua_istable(L,-1));  /* just in case */
  SWIG_Lua_get_table(L,".get"); /* find the .get table */
  assert(lua_istable(L,-1));  /* just in case */
  SWIG_Lua_add_function(L,name,getFn);
  lua_pop(L,1);       /* tidy stack (remove table) */
  if (setFn)
  {
    SWIG_Lua_get_table(L,".set"); /* find the .set table */
    assert(lua_istable(L,-1));  /* just in case */
    SWIG_Lua_add_function(L,name,setFn);
    lua_pop(L,1);       /* tidy stack (remove table) */
  }
}

/* helper to recursively add class static details (static attributes, operations and constants) */
SWIGINTERN void SWIG_Lua_add_class_static_details(lua_State *L, swig_lua_class *clss)
{
  int i = 0;
  /* The class namespace table must be on the top of the stack */
  assert(lua_istable(L,-1));
  /* call all the base classes first: we can then override these later: */
  for(i=0;clss->bases[i];i++)
  {
    SWIG_Lua_add_class_static_details(L,clss->bases[i]);
  }

  SWIG_Lua_add_namespace_details(L, clss->cls_static);
}

SWIGINTERN void SWIG_Lua_add_class_user_metamethods(lua_State *L, swig_lua_class *clss); /* forward declaration */

/* helper to recursively add class details (attributes & operations) */
SWIGINTERN void  SWIG_Lua_add_class_instance_details(lua_State *L, swig_lua_class *clss)
{
  int i;
  size_t bases_count = 0;
  /* Add bases to .bases table */
  SWIG_Lua_get_table(L,".bases");
  assert(lua_istable(L,-1));  /* just in case */
  for(i=0;clss->bases[i];i++)
  {
    SWIG_Lua_get_class_metatable(L,clss->bases[i]->fqname);
    /* Base class must be already registered */
    assert(lua_istable(L,-1));
    lua_rawseti(L,-2,i+1); /* In lua indexing starts from 1 */
    bases_count++;
  }
  assert(lua_rawlen(L,-1) == bases_count);
  lua_pop(L,1); /* remove .bases table */
  /* add attributes */
  for(i=0;clss->attributes[i].name;i++){
    SWIG_Lua_add_variable(L,clss->attributes[i].name,clss->attributes[i].getmethod,clss->attributes[i].setmethod);
  }
  /* add methods to the metatable */
  SWIG_Lua_get_table(L,".fn"); /* find the .fn table */
  assert(lua_istable(L,-1));  /* just in case */
  for(i=0;clss->methods[i].name;i++){
    SWIG_Lua_add_function(L,clss->methods[i].name,clss->methods[i].func);
  }
  lua_pop(L,1);       /* tidy stack (remove table) */
  /* add operator overloads
    This adds methods from metatable array to metatable. Can mess up garbage
    collectind if someone defines __gc method
    */
  if(clss->metatable) {
    for(i=0;clss->metatable[i].name;i++) {
      SWIG_Lua_add_function(L,clss->metatable[i].name,clss->metatable[i].func);
    }
  }

#if !defined(SWIG_LUA_SQUASH_BASES)
  /* Adding metamethods that are defined in base classes. If bases were squashed
   * then it is obviously unnecessary
   */
  SWIG_Lua_add_class_user_metamethods(L, clss);
#endif
}

/* Helpers to add user defined class metamedhods - __add, __sub etc. The helpers are needed
   for the following issue: Lua runtime checks for metamethod existence with rawget function
   ignoring our SWIG-provided __index and __newindex functions. Thus our inheritance-aware method
   search algorithm doesn't work in such case. (Not to say that Lua runtime queries metamethod directly
   in metatable and not in object).
   Current solution is this: if somewhere in hierarchy metamethod __x is defined, then all descendants
   are automatically given a special proxy __x that calls the real __x method.
   Obvious idea - to copy __x instead of creating __x-proxy is wrong because if someone changes __x in runtime,
   those changes must be reflected in all descendants.
*/

SWIGRUNTIME int SWIG_Lua_resolve_metamethod(lua_State *L); /*forward declaration*/

/* The real function that resolves a metamethod.
 * Function searches given class and all it's bases(recursively) for first instance of something that is
 * not equal to SWIG_Lua_resolve_metatmethod. (Almost always this 'something' is actual metamethod implementation
 * and it is a SWIG-generated C function.). It returns value on the top of the L and there is no garbage below the
 * answer.
 * Returns 1 if found, 0 otherwise.
 * clss is class which metatable we will search for method
 * metamethod_name_idx is index in L where metamethod name (as string) lies
 * skip_check allows to skip searching metamethod in givel clss and immideatelly go to searching in bases. skip_check
 * is not caried to subsequent recursive calls - false is always passed. It is set to true only at first call from
 * SWIG_Lua_resolve_metamethod
 * */
SWIGINTERN int SWIG_Lua_do_resolve_metamethod(lua_State *L, const swig_lua_class *clss, int metamethod_name_idx,
    int skip_check)
{
  /* This function is called recursively */
  int result = 0;
  int i = 0;

  if (!skip_check) {
    SWIG_Lua_get_class_metatable(L, clss->fqname);
    lua_pushvalue(L, metamethod_name_idx);
    lua_rawget(L,-2);
    /* If this is cfunction and it is equal to SWIG_Lua_resolve_metamethod then
     * this isn't the function we are looking for :)
     * lua_tocfunction will return NULL if not cfunction
     */
    if (!lua_isnil(L,-1) && lua_tocfunction(L,-1) != SWIG_Lua_resolve_metamethod ) {
      lua_remove(L,-2); /* removing class metatable */
      return 1;
    }
    lua_pop(L,2); /* remove class metatable and query result */
  }

  /* Forwarding calls to bases */
  for(i=0;clss->bases[i];i++)
  {
    result = SWIG_Lua_do_resolve_metamethod(L, clss->bases[i], metamethod_name_idx, 0);
    if (result)
      break;
  }

  return result;
}

/* The proxy function for metamethod. All parameters are passed as cclosure. Searches for actual method
 * and calls it */
SWIGRUNTIME int SWIG_Lua_resolve_metamethod(lua_State *L)
{
  int numargs;
  int metamethod_name_idx;
  const swig_lua_class* clss;
  int result;

  lua_checkstack(L,5);
  numargs = lua_gettop(L); /* number of arguments to pass to actual metamethod */

  /* Get upvalues from closure */
  lua_pushvalue(L, lua_upvalueindex(1)); /*Get function name*/
  metamethod_name_idx = lua_gettop(L);

  lua_pushvalue(L, lua_upvalueindex(2));
  clss = (const swig_lua_class*)(lua_touserdata(L,-1));
  lua_pop(L,1); /* remove lightuserdata with clss from stack */

  /* Actual work */
  result = SWIG_Lua_do_resolve_metamethod(L, clss, metamethod_name_idx, 1);
  if (!result) {
   SWIG_Lua_pushferrstring(L,"The metamethod proxy is set, but it failed to find actual metamethod. Memory corruption is most likely explanation.");
   lua_error(L);
   return 0;
  }

  lua_remove(L,-2); /* remove metamethod key */
  lua_insert(L,1); /* move function to correct position */
  lua_call(L, numargs, LUA_MULTRET);
  return lua_gettop(L); /* return all results */
}


/* If given metamethod must be present in given class, then creates appropriate proxy
 * Returns 1 if successfully added, 0 if not added because no base class has it, -1
 * if method is defined in the class metatable itself
 */
SWIGINTERN int SWIG_Lua_add_class_user_metamethod(lua_State *L, swig_lua_class *clss, const int metatable_index)
{
  int key_index;
  int success = 0;
  int i = 0;

  /* metamethod name - on the top of the stack */
  assert(lua_isstring(L,-1));

  key_index = lua_gettop(L);

  /* Check whether method is already defined in metatable */
  lua_pushvalue(L,key_index); /* copy of the key */
  lua_gettable(L,metatable_index);
  if( !lua_isnil(L,-1) ) {
    lua_pop(L,1);
    return -1;
  }
  lua_pop(L,1);

  /* Iterating over immediate bases */
  for(i=0;clss->bases[i];i++)
  {
    const swig_lua_class *base = clss->bases[i];
    SWIG_Lua_get_class_metatable(L, base->fqname);
    lua_pushvalue(L, key_index);
    lua_rawget(L, -2);
    if( !lua_isnil(L,-1) ) {
      lua_pushvalue(L, key_index);

      /* Add proxy function */
      lua_pushvalue(L, key_index); /* first closure value is function name */
      lua_pushlightuserdata(L, clss); /* second closure value is swig_lua_class structure */
      lua_pushcclosure(L, SWIG_Lua_resolve_metamethod, 2);

      lua_rawset(L, metatable_index);
      success = 1;
    }
    lua_pop(L,1); /* remove function or nil */
    lua_pop(L,1); /* remove base class metatable */

    if( success )
      break;
  }

  return success;
}

SWIGINTERN void SWIG_Lua_add_class_user_metamethods(lua_State *L, swig_lua_class *clss)
{
  int metatable_index;
  int metamethods_info_index;
  int tostring_undefined;
  int eq_undefined = 0;

  SWIG_Lua_get_class_metatable(L, clss->fqname);
  metatable_index = lua_gettop(L);
  SWIG_Lua_get_inheritable_metamethods(L);
  assert(lua_istable(L,-1));
  metamethods_info_index = lua_gettop(L);
  lua_pushnil(L); /* first key */
  while(lua_next(L, metamethods_info_index) != 0 ) {
    /* key at index -2, value at index -1 */
    const int is_inheritable = lua_toboolean(L,-2);
    lua_pop(L,1); /* remove value - we don't need it anymore */

    if(is_inheritable) { /* if metamethod is inheritable */
      SWIG_Lua_add_class_user_metamethod(L,clss,metatable_index);
    }
  }

  lua_pop(L,1); /* remove inheritable metatmethods table */

  /* Special handling for __tostring method */
  lua_pushstring(L, "__tostring");
  lua_pushvalue(L,-1);
  lua_rawget(L,metatable_index);
  tostring_undefined = lua_isnil(L,-1);
  lua_pop(L,1);
  if( tostring_undefined ) {
    lua_pushcfunction(L, SWIG_Lua_class_tostring);
    lua_rawset(L, metatable_index);
  } else {
    lua_pop(L,1); /* remove copy of the key */
  }

  /* Special handling for __eq method */
  lua_pushstring(L, "__eq");
  lua_pushvalue(L,-1);
  lua_rawget(L,metatable_index);
  eq_undefined = lua_isnil(L,-1);
  lua_pop(L,1);
  if( eq_undefined ) {
    lua_pushcfunction(L, SWIG_Lua_class_equal);
    lua_rawset(L, metatable_index);
  } else {
    lua_pop(L,1); /* remove copy of the key */
  }
  /* Warning: __index and __newindex are SWIG-defined. For user-defined operator[]
   * a __getitem/__setitem method should be defined
   */
  lua_pop(L,1); /* pop class metatable */
}

/* Register class static methods,attributes etc as well as constructor proxy */
SWIGINTERN void SWIG_Lua_class_register_static(lua_State *L, swig_lua_class *clss)
{
  const int SWIGUNUSED begin = lua_gettop(L);
  lua_checkstack(L,5); /* just in case */
  assert(lua_istable(L,-1));  /* just in case */
  assert(strcmp(clss->name, clss->cls_static->name) == 0); /* in class those 2 must be equal */

  SWIG_Lua_namespace_register(L,clss->cls_static, 1);

  assert(lua_istable(L,-1)); /* just in case */

  /*  add its constructor to module with the name of the class
  so you can do MyClass(...) as well as new_MyClass(...)
  BUT only if a constructor is defined
  (this overcomes the problem of pure virtual classes without constructors)*/
  if (clss->constructor)
  {
    lua_getmetatable(L,-1);
    assert(lua_istable(L,-1)); /* just in case */
    SWIG_Lua_add_function(L,"__call", clss->constructor);
    lua_pop(L,1);
  }

  assert(lua_istable(L,-1)); /* just in case */
  SWIG_Lua_add_class_static_details(L, clss);

  /* clear stack */
  lua_pop(L,1);
  assert( lua_gettop(L) == begin );
}

/* Performs the instance (non-static) class registration process. Metatable for class is created
 * and added to the class registry.
 */
SWIGINTERN void  SWIG_Lua_class_register_instance(lua_State *L,swig_lua_class *clss)
{
  const int SWIGUNUSED begin = lua_gettop(L);
  int i;
  /* if name already there (class is already registered) then do nothing */
  SWIG_Lua_get_class_registry(L);  /* get the registry */
  lua_pushstring(L,clss->fqname);  /* get the name */
  lua_rawget(L,-2);
  if(!lua_isnil(L,-1)) {
    lua_pop(L,2);
    assert(lua_gettop(L)==begin);
    return;
  }
  lua_pop(L,2); /* tidy stack */
  /* Recursively initialize all bases */
  for(i=0;clss->bases[i];i++)
  {
    SWIG_Lua_class_register_instance(L,clss->bases[i]);
  }
  /* Again, get registry and push name */
  SWIG_Lua_get_class_registry(L);  /* get the registry */
  lua_pushstring(L,clss->fqname);  /* get the name */
  lua_newtable(L);    /* create the metatable */
#if defined(SWIG_LUA_SQUASH_BASES) && (SWIG_LUA_TARGET == SWIG_LUA_FLAVOR_LUA)
  /* If squashing is requested, then merges all bases metatable into this one.
   * It would get us all special methods: __getitem, __add etc.
   * This would set .fn, .type, and other .xxx incorrectly, but we will overwrite it right away
   */
  {
    int new_metatable_index = lua_absindex(L,-1);
    for(i=0;clss->bases[i];i++)
    {
      int base_metatable;
      SWIG_Lua_get_class_metatable(L,clss->bases[i]->fqname);
      base_metatable = lua_absindex(L,-1);
      SWIG_Lua_merge_tables_by_index(L,new_metatable_index, base_metatable);
      lua_pop(L,1);
    }
  }
  /* And now we will overwrite all incorrectly set data */
#endif
  /* add string of class name called ".type" */
  lua_pushstring(L,".type");
  lua_pushstring(L,clss->fqname);
  lua_rawset(L,-3);
  /* add a table called bases */
  lua_pushstring(L,".bases");
  lua_newtable(L);
  lua_rawset(L,-3);
  /* add a table called ".get" */
  lua_pushstring(L,".get");
  lua_newtable(L);
  lua_rawset(L,-3);
  /* add a table called ".set" */
  lua_pushstring(L,".set");
  lua_newtable(L);
  lua_rawset(L,-3);
  /* add a table called ".fn" */
  lua_pushstring(L,".fn");
  lua_newtable(L);
  /* add manual disown method */
  SWIG_Lua_add_function(L,"__disown",SWIG_Lua_class_disown);
  lua_rawset(L,-3);
  /* add accessor fns for using the .get,.set&.fn */
  SWIG_Lua_add_function(L,"__index",SWIG_Lua_class_get);
  SWIG_Lua_add_function(L,"__newindex",SWIG_Lua_class_set);
  SWIG_Lua_add_function(L,"__gc",SWIG_Lua_class_destruct);
  /* add it */
  lua_rawset(L,-3);  /* metatable into registry */
  lua_pop(L,1);      /* tidy stack (remove registry) */
  assert(lua_gettop(L) == begin);

#if defined(SWIG_LUA_SQUASH_BASES) && (SWIG_LUA_TARGET == SWIG_LUA_FLAVOR_LUA)
  /* Now merge all symbols from .fn, .set, .get etc from bases to our tables */
  SWIG_Lua_class_squash_bases(L,clss);
#endif
  SWIG_Lua_get_class_metatable(L,clss->fqname);
  SWIG_Lua_add_class_instance_details(L,clss);  /* recursive adding of details (atts & ops) */
  lua_pop(L,1);      /* tidy stack (remove class metatable) */
  assert( lua_gettop(L) == begin );
}

SWIGINTERN void  SWIG_Lua_class_register(lua_State *L,swig_lua_class *clss)
{
  int SWIGUNUSED begin;
  assert(lua_istable(L,-1)); /* This is a table (module or namespace) where classes will be added */
  SWIG_Lua_class_register_instance(L,clss);
  SWIG_Lua_class_register_static(L,clss);

  /* Add links from static part to instance part and vice versa */
  /* [SWIG registry]                                   [Module]
   *    "MyClass" ----> [MyClass metatable] <=====     "MyClass" -+> [static part]
   *                     ".get" ----> ...        |                |     getmetatable()----|
   *                     ".set" ----> ...        |                |                       |
   *                     ".static" --------------)----------------/           [static part metatable]
   *                                             |                                ".get" --> ...
   *                                             |                                ".set" --> ....
   *                                             |=============================== ".instance"
   */
  begin = lua_gettop(L);
  lua_pushstring(L,clss->cls_static->name);
  lua_rawget(L,-2); /* get class static table */
  assert(lua_istable(L,-1));
  lua_getmetatable(L,-1);
  assert(lua_istable(L,-1)); /* get class static metatable */
  lua_pushstring(L,".instance"); /* prepare key */

  SWIG_Lua_get_class_metatable(L,clss->fqname); /* get class metatable */
  assert(lua_istable(L,-1));
  lua_pushstring(L,".static"); /* prepare key */
  lua_pushvalue(L, -4); /* push static class TABLE */
  assert(lua_istable(L,-1));
  lua_rawset(L,-3); /* assign static class table(!NOT metatable) as ".static" member of class metatable */
  lua_rawset(L,-3); /* assign class metatable as ".instance" member of class static METATABLE */
  lua_pop(L,2);
  assert(lua_gettop(L) == begin);
}
#endif /* SWIG_LUA_TARGET == SWIG_LUA_FLAVOR_LUA */

#if (SWIG_LUA_TARGET == SWIG_LUA_FLAVOR_ELUA) || (SWIG_LUA_TARGET == SWIG_LUA_FLAVOR_ELUAC)
SWIGINTERN void SWIG_Lua_elua_class_register_instance(lua_State *L, swig_lua_class *clss)
{
  const int SWIGUNUSED begin = lua_gettop(L);
  int i;
  /* if name already there (class is already registered) then do nothing */
  SWIG_Lua_get_class_registry(L);  /* get the registry */
  lua_pushstring(L,clss->fqname);  /* get the name */
  lua_rawget(L,-2);
  if(!lua_isnil(L,-1)) {
    lua_pop(L,2);
    assert(lua_gettop(L)==begin);
    return;
  }
  lua_pop(L,2); /* tidy stack */
  /* Recursively initialize all bases */
  for(i=0;clss->bases[i];i++)
  {
    SWIG_Lua_elua_class_register_instance(L,clss->bases[i]);
  }
  /* Again, get registry and push name */
  SWIG_Lua_get_class_registry(L);  /* get the registry */
  lua_pushstring(L,clss->fqname);  /* get the name */
  assert(clss->metatable);
  lua_pushrotable(L, (void*)(clss->metatable));    /* create the metatable */
  lua_rawset(L,-3);
  lua_pop(L,1);
  assert(lua_gettop(L) == begin);
}
#endif /* elua && eluac */

/* -----------------------------------------------------------------------------
 * Class/structure conversion fns
 * ----------------------------------------------------------------------------- */

/* helper to add metatable to new lua object */
SWIGINTERN void SWIG_Lua_AddMetatable(lua_State *L,swig_type_info *type)
{
  if (type->clientdata)  /* there is clientdata: so add the metatable */
  {
    SWIG_Lua_get_class_metatable(L,((swig_lua_class*)(type->clientdata))->fqname);
    if (lua_istable(L,-1))
    {
      lua_setmetatable(L,-2);
    }
    else
    {
      lua_pop(L,1);
    }
  }
}

/* pushes a new object into the lua stack */
SWIGRUNTIME void SWIG_Lua_NewPointerObj(lua_State *L,void *ptr,swig_type_info *type, int own)
{
  swig_lua_userdata *usr;
  if (!ptr){
    lua_pushnil(L);
    return;
  }
  usr=(swig_lua_userdata*)lua_newuserdata(L,sizeof(swig_lua_userdata));  /* get data */
  usr->ptr=ptr;  /* set the ptr */
  usr->type=type;
  usr->own=own;
#if (SWIG_LUA_TARGET != SWIG_LUA_FLAVOR_ELUAC)
  SWIG_Lua_AddMetatable(L,type); /* add metatable */
#endif
}

/* takes a object from the lua stack & converts it into an object of the correct type
 (if possible) */
SWIGRUNTIME int  SWIG_Lua_ConvertPtr(lua_State *L,int index,void **ptr,swig_type_info *type,int flags)
{
  swig_lua_userdata *usr;
  swig_cast_info *cast;
  /* special case: lua nil => NULL pointer */
  if (lua_isnil(L,index))
  {
    *ptr=0;
    return (flags & SWIG_POINTER_NO_NULL) ? SWIG_NullReferenceError : SWIG_OK;
  }
  if (lua_islightuserdata(L,index))
  {
    *ptr=lua_touserdata(L,index);
    return (flags & SWIG_POINTER_NO_NULL) ? SWIG_NullReferenceError : SWIG_OK;
  }
  usr=(swig_lua_userdata*)lua_touserdata(L,index);  /* get data */
  if (usr)
  {
    if (flags & SWIG_POINTER_DISOWN) /* must disown the object */
    {
        usr->own=0;
    }
    if (!type)            /* special cast void*, no casting fn */
    {
      *ptr=usr->ptr;
      return SWIG_OK; /* ok */
    }
    cast=SWIG_TypeCheckStruct(usr->type,type); /* performs normal type checking */
    if (cast)
    {
      int newmemory = 0;
      *ptr=SWIG_TypeCast(cast,usr->ptr,&newmemory);
      assert(!newmemory); /* newmemory handling not yet implemented */
      return SWIG_OK;  /* ok */
    }
  }
  return SWIG_ERROR;  /* error */
}

SWIGRUNTIME void* SWIG_Lua_MustGetPtr(lua_State *L,int index,swig_type_info *type,int flags,
       int argnum,const char *func_name){
  void *result;
  if (!SWIG_IsOK(SWIG_ConvertPtr(L,index,&result,type,flags))){
    luaL_error (L,"Error in %s, expected a %s at argument number %d\n",
		func_name,(type && type->str)?type->str:"void*",argnum);
  }
  return result;
}

/* pushes a packed userdata. user for member fn pointers only */
SWIGRUNTIME void SWIG_Lua_NewPackedObj(lua_State *L,void *ptr,size_t size,swig_type_info *type)
{
  swig_lua_rawdata *raw;
  assert(ptr); /* not acceptable to pass in a NULL value */
  raw=(swig_lua_rawdata*)lua_newuserdata(L,sizeof(swig_lua_rawdata)-1+size);  /* alloc data */
  raw->type=type;
  raw->own=0;
  memcpy(raw->data,ptr,size); /* copy the data */
  SWIG_Lua_AddMetatable(L,type); /* add metatable */
}

/* converts a packed userdata. user for member fn pointers only */
SWIGRUNTIME int  SWIG_Lua_ConvertPacked(lua_State *L,int index,void *ptr,size_t size,swig_type_info *type)
{
  swig_lua_rawdata *raw;
  raw=(swig_lua_rawdata*)lua_touserdata(L,index);  /* get data */
  if (!raw) return SWIG_ERROR;  /* error */
  if (type==0 || type==raw->type) /* void* or identical type */
  {
    memcpy(ptr,raw->data,size); /* copy it */
    return SWIG_OK; /* ok */
  }
  return SWIG_ERROR;  /* error */
}

/* a function to get the typestring of a piece of data */
SWIGRUNTIME const char *SWIG_Lua_typename(lua_State *L, int tp)
{
  swig_lua_userdata *usr;
  if (lua_isuserdata(L,tp))
  {
    usr=(swig_lua_userdata*)lua_touserdata(L,tp);  /* get data */
    if (usr && usr->type && usr->type->str)
      return usr->type->str;
    return "userdata (unknown type)";
  }
  return lua_typename(L,lua_type(L,tp));
}

/* lua callable function to get the userdata's type */
SWIGRUNTIME int SWIG_Lua_type(lua_State *L)
{
  lua_pushstring(L,SWIG_Lua_typename(L,1));
  return 1;
}

/* -----------------------------------------------------------------------------
 * global variable support code: class/struct typemap functions
 * ----------------------------------------------------------------------------- */

#if ((SWIG_LUA_TARGET != SWIG_LUA_FLAVOR_ELUA) && (SWIG_LUA_TARGET != SWIG_LUA_FLAVOR_ELUAC))
/* Install Constants */
SWIGINTERN void
SWIG_Lua_InstallConstants(lua_State *L, swig_lua_const_info constants[]) {
  int i;
  for (i = 0; constants[i].type; i++) {
    switch(constants[i].type) {
    case SWIG_LUA_INT:
      lua_pushstring(L,constants[i].name);
      lua_pushinteger(L,(lua_Integer)constants[i].lvalue);
      lua_rawset(L,-3);
      break;
    case SWIG_LUA_FLOAT:
      lua_pushstring(L,constants[i].name);
      lua_pushnumber(L,(lua_Number)constants[i].dvalue);
      lua_rawset(L,-3);
      break;
    case SWIG_LUA_CHAR:
      lua_pushstring(L,constants[i].name);
      {
        char c = (char)constants[i].lvalue;
        lua_pushlstring(L,&c,1);
      }
      lua_rawset(L,-3);
      break;
    case SWIG_LUA_STRING:
      lua_pushstring(L,constants[i].name);
      lua_pushstring(L,(char *) constants[i].pvalue);
      lua_rawset(L,-3);
      break;
    case SWIG_LUA_POINTER:
      lua_pushstring(L,constants[i].name);
      SWIG_NewPointerObj(L,constants[i].pvalue, *(constants[i]).ptype,0);
      lua_rawset(L,-3);
      break;
    case SWIG_LUA_BINARY:
      lua_pushstring(L,constants[i].name);
      SWIG_NewMemberObj(L,constants[i].pvalue,constants[i].lvalue,*(constants[i]).ptype);
      lua_rawset(L,-3);
      break;
    default:
      break;
    }
  }
}
#endif

/* -----------------------------------------------------------------------------
 * executing lua code from within the wrapper
 * ----------------------------------------------------------------------------- */

#ifndef SWIG_DOSTRING_FAIL /* Allows redefining of error function */
#define SWIG_DOSTRING_FAIL(S) fprintf(stderr,"%s\n",S)
#endif
/* Executes a C string in Lua which is a really simple way of calling lua from C
Unfortunately lua keeps changing its APIs, so we need a conditional compile
In lua 5.0.X it's lua_dostring()
In lua 5.1.X it's luaL_dostring()
*/
SWIGINTERN int
SWIG_Lua_dostring(lua_State *L, const char *str) {
  int ok,top;
  if (str==0 || str[0]==0) return 0; /* nothing to do */
  top=lua_gettop(L); /* save stack */
#if (defined(LUA_VERSION_NUM) && (LUA_VERSION_NUM>=501))
  ok=luaL_dostring(L,str);	/* looks like this is lua 5.1.X or later, good */
#else
  ok=lua_dostring(L,str);	/* might be lua 5.0.x, using lua_dostring */
#endif
  if (ok!=0) {
    SWIG_DOSTRING_FAIL(lua_tostring(L,-1));
  }
  lua_settop(L,top); /* restore the stack */
  return ok;
}

#ifdef __cplusplus
}
#endif

/* ------------------------------ end luarun.swg  ------------------------------ */
