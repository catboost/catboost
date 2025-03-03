/* -----------------------------------------------------------------------------
 * lua_fnptr.i
 *
 * SWIG Library file containing the main typemap code to support Lua modules.
 * ----------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 *                          Basic function pointer support
 * ----------------------------------------------------------------------------- */
/*
The structure: SWIGLUA_FN provides a simple (local only) wrapping for a function.

For example if you wanted to have a C/C++ function take a lua function as a parameter.
You could declare it as:
  int my_func(int a, int b, SWIGLUA_FN fn);
note: it should be passed by value, not byref or as a pointer.

The SWIGLUA_FN holds a pointer to the lua_State, and the stack index where the function is held.
The macro SWIGLUA_FN_GET() will put a copy of the lua function at the top of the stack.
After that its fairly simple to write the rest of the code (assuming know how to use lua),
just push the parameters, call the function and return the result.

  int my_func(int a, int b, SWIGLUA_FN fn)
  {
    SWIGLUA_FN_GET(fn);
    lua_pushnumber(fn.L,a);
    lua_pushnumber(fn.L,b);
    lua_call(fn.L,2,1);    // 2 in, 1 out
    return luaL_checknumber(fn.L,-1);
  }

SWIG will automatically performs the wrapping of the arguments in and out.

However: if you wish to store the function between calls, look to the SWIGLUA_REF below.

*/
// this is for the C code only, we don't want SWIG to wrapper it for us.
%{
typedef struct{
  lua_State* L; /* the state */
  int idx;      /* the index on the stack */
}SWIGLUA_FN;

#define SWIGLUA_FN_GET(fn) {lua_pushvalue(fn.L,fn.idx);}
%}

// the actual typemap
%typemap(in,checkfn="lua_isfunction") SWIGLUA_FN
%{  $1.L=L; $1.idx=$input; %}

/* -----------------------------------------------------------------------------
 *                          Storing lua object support
 * ----------------------------------------------------------------------------- */
/*
The structure: SWIGLUA_REF provides a mechanism to store object (usually functions)
between calls to the interpreter.

For example if you wanted to have a C/C++ function take a lua function as a parameter.
Then call it later, You could declare it as:
  SWIGLUA_REF myref;
  void set_func(SWIGLUA_REF ref);
  SWIGLUA_REF get_func();
  void call_func(int val);
note: it should be passed by value, not byref or as a pointer.

The SWIGLUA_REF holds a pointer to the lua_State, and an integer reference to the object.
Because it holds a permanent ref to an object, the SWIGLUA_REF must be handled with a bit more care.
It should be initialised to {0,0}. The function swiglua_ref_set() should be used to set it.
swiglua_ref_clear() should be used to clear it when not in use, and swiglua_ref_get() to get the
data back.

Note: the typemap does not check that the object is in fact a function,
if you need that you must add it yourself.


  int my_func(int a, int b, SWIGLUA_FN fn)
  {
    SWIGLUA_FN_GET(fn);
    lua_pushnumber(fn.L,a);
    lua_pushnumber(fn.L,b);
    lua_call(fn.L,2,1);    // 2 in, 1 out
    return luaL_checknumber(fn.L,-1);
  }

SWIG will automatically performs the wrapping of the arguments in and out.

However: if you wish to store the function between calls, look to the SWIGLUA_REF below.

*/

%{
typedef struct{
  lua_State* L; /* the state */
  int ref;      /* a ref in the lua global index */
}SWIGLUA_REF;


void swiglua_ref_clear(SWIGLUA_REF* pref){
 	if (pref->L!=0 && pref->ref!=LUA_NOREF && pref->ref!=LUA_REFNIL){
		luaL_unref(pref->L,LUA_REGISTRYINDEX,pref->ref);
	}
	pref->L=0; pref->ref=0;
}

void swiglua_ref_set(SWIGLUA_REF* pref,lua_State* L,int idx){
	pref->L=L;
	lua_pushvalue(L,idx);                 /* copy obj to top */
	pref->ref=luaL_ref(L,LUA_REGISTRYINDEX); /* remove obj from top & put into registry */
}

void swiglua_ref_get(SWIGLUA_REF* pref){
	if (pref->L!=0)
		lua_rawgeti(pref->L,LUA_REGISTRYINDEX,pref->ref);
}

%}

%typemap(in) SWIGLUA_REF
%{  swiglua_ref_set(&$1,L,$input); %}

%typemap(out) SWIGLUA_REF
%{  if ($1.L!=0)  {swiglua_ref_get(&$1);} else {lua_pushnil(L);}
  SWIG_arg++; %}

