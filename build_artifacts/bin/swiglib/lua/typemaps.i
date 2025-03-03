/* -----------------------------------------------------------------------------
 * typemaps.swg
 *
 * SWIG Library file containing the main typemap code to support Lua modules.
 * ----------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 *                          Basic inout typemaps
 * ----------------------------------------------------------------------------- */
/*
These provide the basic ability for passing in & out of standard numeric data types
(int,long,float,double, etc)

The basic code looks like this:

%typemap(in,checkfn="lua_isnumber") int *INPUT(int temp), int &INPUT(int temp)
%{ temp = (int)lua_tonumber(L,$input);
   $1 = &temp; %}

%typemap(in, numinputs=0) int *OUTPUT (int temp)
%{ $1 = &temp; %}

%typemap(argout) int *OUTPUT
%{  lua_pushnumber(L, (double) *$1); SWIG_arg++;%}

%typemap(in) int *INOUT = int *INPUT;
%typemap(argout) int *INOUT = int *OUTPUT;

However the code below is a mixture of #defines & such, so nowhere as easy to read

To make you code work correctly its not just a matter of %including this file
You also have to give SWIG the hints on which to use where

eg
extern int add_pointer(int* a1,int* a2); // a1 & a2 are pointer values to be added
extern void swap(int* s1, int* s2);	// does the swap

You will need to either change the argument names
extern int add_pointer(int* INPUT,int* INPUT);

or provide a %apply statement

%apply int* INOUT{ int *s1, int *s2 };
	// if SWIG sees int* s1, int* s2, assume they are inout params
*/


%define SWIG_NUMBER_TYPEMAP(TYPE)
%typemap(in,checkfn="lua_isnumber")	TYPE *INPUT($*ltype temp), TYPE &INPUT($*ltype temp)
%{ temp = ($*ltype)lua_tonumber(L,$input);
   $1 = &temp; %}
%typemap(in, numinputs=0) TYPE *OUTPUT ($*ltype temp)
%{ $1 = &temp; %}
%typemap(argout) TYPE *OUTPUT
%{  lua_pushnumber(L, (lua_Number) *$1); SWIG_arg++;%}
%typemap(in) TYPE *INOUT = TYPE *INPUT;
%typemap(argout) TYPE *INOUT = TYPE *OUTPUT;
%typemap(in) TYPE &OUTPUT = TYPE *OUTPUT;
%typemap(argout) TYPE &OUTPUT = TYPE *OUTPUT;
%typemap(in) TYPE &INOUT = TYPE *INPUT;
%typemap(argout) TYPE &INOUT = TYPE *OUTPUT;
// const version (the $*ltype is the basic number without ptr or const's)
%typemap(in,checkfn="lua_isnumber")	const TYPE *INPUT($*ltype temp)
%{ temp = ($*ltype)lua_tonumber(L,$input);
   $1 = &temp; %}
%enddef

// now the code
SWIG_NUMBER_TYPEMAP(unsigned char); SWIG_NUMBER_TYPEMAP(signed char);

SWIG_NUMBER_TYPEMAP(short); SWIG_NUMBER_TYPEMAP(unsigned short); SWIG_NUMBER_TYPEMAP(signed short);
SWIG_NUMBER_TYPEMAP(int); SWIG_NUMBER_TYPEMAP(unsigned int); SWIG_NUMBER_TYPEMAP(signed int);
SWIG_NUMBER_TYPEMAP(long); SWIG_NUMBER_TYPEMAP(unsigned long); SWIG_NUMBER_TYPEMAP(signed long);
SWIG_NUMBER_TYPEMAP(float);
SWIG_NUMBER_TYPEMAP(double);
SWIG_NUMBER_TYPEMAP(enum SWIGTYPE);
// also for long longs's
SWIG_NUMBER_TYPEMAP(long long); SWIG_NUMBER_TYPEMAP(unsigned long long); SWIG_NUMBER_TYPEMAP(signed long long);

// note we dont do char, as a char* is probably a string not a ptr to a single char

// similar for booleans
%typemap(in,checkfn="lua_isboolean") bool *INPUT(bool temp), bool &INPUT(bool temp)
%{ temp = (lua_toboolean(L,$input)!=0);
   $1 = &temp; %}

%typemap(in, numinputs=0) bool *OUTPUT (bool temp),bool &OUTPUT (bool temp)
%{ $1 = &temp; %}

%typemap(argout) bool *OUTPUT,bool &OUTPUT
%{  lua_pushboolean(L, (int)((*$1)!=0)); SWIG_arg++;%}

%typemap(in) bool *INOUT = bool *INPUT;
%typemap(argout) bool *INOUT = bool *OUTPUT;
%typemap(in) bool &INOUT = bool &INPUT;
%typemap(argout) bool &INOUT = bool &OUTPUT;

/* -----------------------------------------------------------------------------
 *                          Basic Array typemaps
 * ----------------------------------------------------------------------------- */
/*
I have no idea why this kind of code does not exist in SWIG as standard,
but here is it.
This code will convert to/from 1D numeric arrays.
In order to reduce code bloat, there are a few macros
and quite a few functions defined
(unfortunately this makes it a lot less clear)

assuming we have functions
void process_array(int arr[3]);	// nice fixed size array
void process_var_array(float arr[],int len);	// variable sized array
void process_var_array_inout(double* arr,int len);	// variable sized array
			// data passed in & out
void process_enum_inout_array_var(enum Days *arrinout, int len);	// using enums
void return_array_5(int arrout[5]);	// out array only

in order to wrap them correctly requires a typemap

// inform SWIG of the correct typemap
// For fixed length, you must specify it as <type> INPUT[ANY]
%apply (int INPUT[ANY]) {(int arr[3])};
// variable length arrays are just the same
%apply (float INPUT[],int) {(float arr[],int len)};
// it is also ok, to map the TYPE* instead of a TYPE[]
%apply (double *INOUT,int) {(double arr*,int len)};
// for the enum's you must use enum SWIGTYPE
%apply (enum SWIGTYPE *INOUT,int) {(enum Days *arrinout, int len)};
// fixed length out if also fine
%apply (int OUTPUT[ANY]) {(int arrout[5])};

Generally, you could use %typemap(...)=...
but the %apply is neater & easier

a few things of note:
* all Lua tables are indexed from 1, all C/C++ arrays are indexed from 0
	therefore t={6,5,3} -- t[1]==6, t[2]==5, t[3]==3
	when passed to process_array(int arr[3]) becomes
	arr[0]==6, arr[1]==5, arr[2]==3
* for OUTPUT arrays, no array need be passed in, the fn will return a Lua table
	so for the above mentioned return_array_5() would look like
	arr=return_array_5() -- no parameters passed in
* for INOUT arrays, a table must be passed in, and a new table will be returned
	(this is consistent with the way that numbers are processed)
	if you want just use
	arr={...}
	arr=process_var_array_inout(arr)	-- arr is replaced by the new version

The following are not yet supported:
* variable length output only array (inout works ok)
* multidimensional arrays
* arrays of objects/structs
* arrays of pointers

*/

/*
The internals of the array management stuff
helper fns/macros
SWIG_ALLOC_ARRAY(TYPE,LEN)	// returns a typed array TYPE[LEN]
SWIG_FREE_ARRAY(PTR)		// delete the ptr (if not zero)

// counts the specified table & gets the size
// integer version
int SWIG_itable_size(lua_State* L, int index);
// other version
int SWIG_table_size(lua_State* L, int index);

SWIG_DECLARE_TYPEMAP_ARR_FN(NAME,TYPE)
// this fn declares up 4 functions for helping to read/write tables
// these can then be called by the macros ...
// all assume the table is an integer indexes from 1
// but the C array is a indexed from 0
	// created a fixed size array, reads the specified table
	// and then fills the array with numbers
	// returns ptr to the array if ok, or 0 for error
	// (also pushes a error message to the stack)
TYPE* SWIG_get_NAME_num_array_fixed(lua_State* L, int index, int size);
	// as per SWIG_get_NAME_num_array_fixed()
	// but reads the entire table & creates an array of the correct size
	// (if the table is empty, it returns an error rather than a zero length array)
TYPE* SWIG_get_NAME_num_array_var(lua_State* L, int index, int* size);
	// writes a table to Lua with all the specified numbers
void SWIG_write_NAME_num_array(lua_State* L,TYPE *array,int size);
	// read the specified table, and fills the array with numbers
	// returns 1 of ok (only fails if it doesn't find numbers)
	// helper fn (called by SWIG_get_NAME_num_array_*() fns)
int SWIG_read_NAME_num_array(lua_State* L,int index,TYPE *array,int size);

*/

/* Reported that you don't need to check for NULL for delete & free
There probably is some compiler that its not true for, so the code is left here just in case.
#ifdef __cplusplus	
#define SWIG_ALLOC_ARRAY(TYPE,LEN) 	new TYPE[LEN]
#define SWIG_FREE_ARRAY(PTR)		if(PTR){delete[] PTR;}
#else
#define SWIG_ALLOC_ARRAY(TYPE,LEN) 	(TYPE *)malloc(LEN*sizeof(TYPE))
#define SWIG_FREE_ARRAY(PTR)		if(PTR){free(PTR);}
#endif
*/
%{
#ifdef __cplusplus	/* generic alloc/dealloc fns*/
#define SWIG_ALLOC_ARRAY(TYPE,LEN) 	new TYPE[LEN]
#define SWIG_FREE_ARRAY(PTR)		delete[] PTR
#else
#define SWIG_ALLOC_ARRAY(TYPE,LEN) 	(TYPE *)malloc(LEN*sizeof(TYPE))
#define SWIG_FREE_ARRAY(PTR)		free(PTR)
#endif
/* counting the size of arrays:*/
SWIGINTERN int SWIG_itable_size(lua_State* L, int index)
{
	int n=0;
	while(1){
		lua_rawgeti(L,index,n+1);
		if (lua_isnil(L,-1))break;
		++n;
		lua_pop(L,1);
	}
	lua_pop(L,1);
	return n;
}

SWIGINTERN int SWIG_table_size(lua_State* L, int index)
{
	int n=0;
	lua_pushnil(L);  /* first key*/
	while (lua_next(L, index) != 0) {
		++n;
		lua_pop(L, 1);  /* removes `value'; keeps `key' for next iteration*/
	}
	return n;
}

/* super macro to declare array typemap helper fns */
#define SWIG_DECLARE_TYPEMAP_ARR_FN(NAME,TYPE)\
	SWIGINTERN int SWIG_read_##NAME##_num_array(lua_State* L,int index,TYPE *array,int size){\
		int i;\
		for (i = 0; i < size; i++) {\
			lua_rawgeti(L,index,i+1);\
			if (lua_isnumber(L,-1)){\
				array[i] = (TYPE)lua_tonumber(L,-1);\
			} else {\
				lua_pop(L,1);\
				return 0;\
			}\
			lua_pop(L,1);\
		}\
		return 1;\
	}\
	SWIGINTERN TYPE* SWIG_get_##NAME##_num_array_fixed(lua_State* L, int index, int size){\
		TYPE *array;\
		if (!lua_istable(L,index) || SWIG_itable_size(L,index) != size) {\
			SWIG_Lua_pushferrstring(L,"expected a table of size %d",size);\
			return 0;\
		}\
		array=SWIG_ALLOC_ARRAY(TYPE,size);\
		if (!SWIG_read_##NAME##_num_array(L,index,array,size)){\
			SWIG_Lua_pusherrstring(L,"table must contain numbers");\
			SWIG_FREE_ARRAY(array);\
			return 0;\
		}\
		return array;\
	}\
	SWIGINTERN TYPE* SWIG_get_##NAME##_num_array_var(lua_State* L, int index, int* size)\
	{\
		TYPE *array;\
		if (!lua_istable(L,index)) {\
			SWIG_Lua_pusherrstring(L,"expected a table");\
			return 0;\
		}\
		*size=SWIG_itable_size(L,index);\
		if (*size<1){\
			SWIG_Lua_pusherrstring(L,"table appears to be empty");\
			return 0;\
		}\
		array=SWIG_ALLOC_ARRAY(TYPE,*size);\
		if (!SWIG_read_##NAME##_num_array(L,index,array,*size)){\
			SWIG_Lua_pusherrstring(L,"table must contain numbers");\
			SWIG_FREE_ARRAY(array);\
			return 0;\
		}\
		return array;\
	}\
	SWIGINTERN void SWIG_write_##NAME##_num_array(lua_State* L,TYPE *array,int size){\
		int i;\
		lua_newtable(L);\
		for (i = 0; i < size; i++){\
			lua_pushnumber(L,(lua_Number)array[i]);\
			lua_rawseti(L,-2,i+1);/* -1 is the number, -2 is the table*/ \
		}\
	}
%}

/*
This is one giant macro to define the typemaps & the helpers
for array handling
*/
%define SWIG_TYPEMAP_NUM_ARR(NAME,TYPE)
%{SWIG_DECLARE_TYPEMAP_ARR_FN(NAME,TYPE)%}

// fixed size array's
%typemap(in) TYPE INPUT[ANY]
%{	$1 = SWIG_get_##NAME##_num_array_fixed(L,$input,$1_dim0);
	if (!$1) SWIG_fail;%}

%typemap(freearg) TYPE INPUT[ANY]
%{	SWIG_FREE_ARRAY($1);%}

// variable size array's
%typemap(in) (TYPE *INPUT,int)
%{	$1 = SWIG_get_##NAME##_num_array_var(L,$input,&$2);
	if (!$1) SWIG_fail;%}

%typemap(freearg) (TYPE *INPUT,int)
%{	SWIG_FREE_ARRAY($1);%}

// out fixed arrays
%typemap(in,numinputs=0) TYPE OUTPUT[ANY]
%{  $1 = SWIG_ALLOC_ARRAY(TYPE,$1_dim0); %}

%typemap(argout) TYPE OUTPUT[ANY]
%{	SWIG_write_##NAME##_num_array(L,$1,$1_dim0); SWIG_arg++; %}

%typemap(freearg) TYPE OUTPUT[ANY]
%{	SWIG_FREE_ARRAY($1); %}

// inout fixed arrays
%typemap(in) TYPE INOUT[ANY]=TYPE INPUT[ANY];
%typemap(argout) TYPE INOUT[ANY]=TYPE OUTPUT[ANY];
%typemap(freearg) TYPE INOUT[ANY]=TYPE INPUT[ANY];
// inout variable arrays
%typemap(in) (TYPE *INOUT,int)=(TYPE *INPUT,int);
%typemap(argout) (TYPE *INOUT,int)
%{	SWIG_write_##NAME##_num_array(L,$1,$2); SWIG_arg++; %}
%typemap(freearg) (TYPE *INOUT,int)=(TYPE *INPUT,int);

// TODO out variable arrays (is there a standard form for such things?)

// referencing so that (int *INPUT,int) and (int INPUT[],int) are the same
%typemap(in) (TYPE INPUT[],int)=(TYPE *INPUT,int);
%typemap(freearg) (TYPE INPUT[],int)=(TYPE *INPUT,int);

%enddef

// the following line of code
// declares the C helper fns for the array typemaps
// as well as defining typemaps for
// fixed len arrays in & out, & variable length arrays in

SWIG_TYPEMAP_NUM_ARR(schar,signed char);
SWIG_TYPEMAP_NUM_ARR(uchar,unsigned char);
SWIG_TYPEMAP_NUM_ARR(int,int);
SWIG_TYPEMAP_NUM_ARR(uint,unsigned int);
SWIG_TYPEMAP_NUM_ARR(short,short);
SWIG_TYPEMAP_NUM_ARR(ushort,unsigned short);
SWIG_TYPEMAP_NUM_ARR(long,long);
SWIG_TYPEMAP_NUM_ARR(ulong,unsigned long);
SWIG_TYPEMAP_NUM_ARR(float,float);
SWIG_TYPEMAP_NUM_ARR(double,double);

// again enums are a problem so they need their own type
// we use the int conversion routine & recast it
%typemap(in) enum SWIGTYPE INPUT[ANY]
%{	$1 = ($ltype)SWIG_get_int_num_array_fixed(L,$input,$1_dim0);
	if (!$1) SWIG_fail;%}

%typemap(freearg) enum SWIGTYPE INPUT[ANY]
%{	SWIG_FREE_ARRAY($1);%}

// variable size arrays
%typemap(in) (enum SWIGTYPE *INPUT,int)
%{	$1 = ($ltype)SWIG_get_int_num_array_var(L,$input,&$2);
	if (!$1) SWIG_fail;%}

%typemap(freearg) (enum SWIGTYPE *INPUT,int)
%{	SWIG_FREE_ARRAY($1);%}

// out fixed arrays
%typemap(in,numinputs=0) enum SWIGTYPE OUTPUT[ANY]
%{  $1 = SWIG_ALLOC_ARRAY(enum SWIGTYPE,$1_dim0); %}

%typemap(argout) enum SWIGTYPE OUTPUT[ANY]
%{	SWIG_write_int_num_array(L,(int*)$1,$1_dim0); SWIG_arg++; %}

%typemap(freearg) enum SWIGTYPE OUTPUT[ANY]
%{	SWIG_FREE_ARRAY($1); %}

// inout fixed arrays
%typemap(in) enum SWIGTYPE INOUT[ANY]=enum SWIGTYPE INPUT[ANY];
%typemap(argout) enum SWIGTYPE INOUT[ANY]=enum SWIGTYPE OUTPUT[ANY];
%typemap(freearg) enum SWIGTYPE INOUT[ANY]=enum SWIGTYPE INPUT[ANY];
// inout variable arrays
%typemap(in) (enum SWIGTYPE *INOUT,int)=(enum SWIGTYPE *INPUT,int);
%typemap(argout) (enum SWIGTYPE *INOUT,int)
%{	SWIG_write_int_num_array(L,(int*)$1,$2); SWIG_arg++; %}
%typemap(freearg) (enum SWIGTYPE *INOUT,int)=(enum SWIGTYPE *INPUT,int);


/* Surprisingly pointer arrays are easier:
this is because all ptr arrays become void**
so only a few fns are needed & a few casts

The function defined are
	// created a fixed size array, reads the specified table
	// and then fills the array with pointers (checking the type)
	// returns ptr to the array if ok, or 0 for error
	// (also pushes a error message to the stack)
void** SWIG_get_ptr_array_fixed(lua_State* L, int index, int size,swig_type_info *type);
	// as per SWIG_get_ptr_array_fixed()
	// but reads the entire table & creates an array of the correct size
	// (if the table is empty, it returns an error rather than a zero length array)
void** SWIG_get_ptr_array_var(lua_State* L, int index, int* size,swig_type_info *type);
	// writes a table to Lua with all the specified pointers
	// all pointers have the ownership value 'own' (normally 0)
void SWIG_write_ptr_array(lua_State* L,void **array,int size,int own);
	// read the specified table, and fills the array with ptrs
	// returns 1 of ok (only fails if it doesn't find correct type of ptrs)
	// helper fn (called by SWIG_get_ptr_array_*() fns)
int SWIG_read_ptr_array(lua_State* L,int index,void **array,int size,swig_type_info *type);

The key thing to remember is that it is assumed that there is no
modification of pointers ownership in the arrays

eg A fn:
void pointers_in(TYPE* arr[],int len);
will make copies of the pointer into a temp array and then pass it into the fn
Lua does not remember that this fn held the pointers, so it is not safe to keep
these pointers until later

eg A fn:
void pointers_out(TYPE* arr[3]);
will return a table containing three pointers
however these pointers are NOT owned by Lua, merely borrowed
so if the C/C++ frees then Lua is not aware

*/

%{
SWIGINTERN int SWIG_read_ptr_array(lua_State* L,int index,void **array,int size,swig_type_info *type){
	int i;
	for (i = 0; i < size; i++) {
		lua_rawgeti(L,index,i+1);
		if (!lua_isuserdata(L,-1) || SWIG_ConvertPtr(L,-1,&array[i],type,0)==-1){
			lua_pop(L,1);
			return 0;
		}
		lua_pop(L,1);
	}
	return 1;
}
SWIGINTERN void** SWIG_get_ptr_array_fixed(lua_State* L, int index, int size,swig_type_info *type){
	void **array;
	if (!lua_istable(L,index) || SWIG_itable_size(L,index) != size) {
		SWIG_Lua_pushferrstring(L,"expected a table of size %d",size);
		return 0;
	}
	array=SWIG_ALLOC_ARRAY(void*,size);
	if (!SWIG_read_ptr_array(L,index,array,size,type)){
		SWIG_Lua_pushferrstring(L,"table must contain pointers of type %s",type->name);
		SWIG_FREE_ARRAY(array);
		return 0;
	}
	return array;
}
SWIGINTERN void** SWIG_get_ptr_array_var(lua_State* L, int index, int* size,swig_type_info *type){
	void **array;
	if (!lua_istable(L,index)) {
		SWIG_Lua_pusherrstring(L,"expected a table");
		return 0;
	}
	*size=SWIG_itable_size(L,index);
	if (*size<1){
		SWIG_Lua_pusherrstring(L,"table appears to be empty");
		return 0;
	}
	array=SWIG_ALLOC_ARRAY(void*,*size);
	if (!SWIG_read_ptr_array(L,index,array,*size,type)){
		SWIG_Lua_pushferrstring(L,"table must contain pointers of type %s",type->name);
		SWIG_FREE_ARRAY(array);
		return 0;
	}
	return array;
}
SWIGINTERN void SWIG_write_ptr_array(lua_State* L,void **array,int size,swig_type_info *type,int own){
	int i;
	lua_newtable(L);
	for (i = 0; i < size; i++){
		SWIG_NewPointerObj(L,array[i],type,own);
		lua_rawseti(L,-2,i+1);/* -1 is the number, -2 is the table*/
	}
}
%}

// fixed size array's
%typemap(in) SWIGTYPE* INPUT[ANY]
%{	$1 = ($ltype)SWIG_get_ptr_array_fixed(L,$input,$1_dim0,$*1_descriptor);
	if (!$1) SWIG_fail;%}

%typemap(freearg) SWIGTYPE* INPUT[ANY]
%{	SWIG_FREE_ARRAY($1);%}

// variable size array's
%typemap(in) (SWIGTYPE **INPUT,int)
%{	$1 = ($ltype)SWIG_get_ptr_array_var(L,$input,&$2,$*1_descriptor);
	if (!$1) SWIG_fail;%}

%typemap(freearg) (SWIGTYPE **INPUT,int)
%{	SWIG_FREE_ARRAY($1);%}

// out fixed arrays
%typemap(in,numinputs=0) SWIGTYPE* OUTPUT[ANY]
%{  $1 = SWIG_ALLOC_ARRAY($*1_type,$1_dim0); %}

%typemap(argout) SWIGTYPE* OUTPUT[ANY]
%{	SWIG_write_ptr_array(L,(void**)$1,$1_dim0,$*1_descriptor,0); SWIG_arg++; %}

%typemap(freearg) SWIGTYPE* OUTPUT[ANY]
%{	SWIG_FREE_ARRAY($1); %}

// inout fixed arrays
%typemap(in) SWIGTYPE* INOUT[ANY]=SWIGTYPE* INPUT[ANY];
%typemap(argout) SWIGTYPE* INOUT[ANY]=SWIGTYPE* OUTPUT[ANY];
%typemap(freearg) SWIGTYPE* INOUT[ANY]=SWIGTYPE* INPUT[ANY];
// inout variable arrays
%typemap(in) (SWIGTYPE** INOUT,int)=(SWIGTYPE** INPUT,int);
%typemap(argout) (SWIGTYPE** INOUT,int)
%{	SWIG_write_ptr_array(L,(void**)$1,$2,$*1_descriptor,0); SWIG_arg++; %}
%typemap(freearg) (SWIGTYPE**INOUT,int)=(SWIGTYPE**INPUT,int);

/* -----------------------------------------------------------------------------
 *                          Pointer-Pointer typemaps
 * ----------------------------------------------------------------------------- */
/*
This code is to deal with the issue for pointer-pointer's
In particular for factory methods.

for example take the following code segment:

struct iMath;    // some structure
int Create_Math(iMath** pptr); // its factory (assume it mallocs)

to use it you might have the following C code:

iMath* ptr;
int ok;
ok=Create_Math(&ptr);
// do things with ptr
//...
free(ptr);

With the following SWIG code
%apply SWIGTYPE** OUTPUT{iMath **pptr };

You can get natural wrapping in Lua as follows:
ok,ptr=Create_Math() -- ptr is a iMath* which is returned with the int
ptr=nil -- the iMath* will be GC'ed as normal
*/

%typemap(in,numinputs=0) SWIGTYPE** OUTPUT ($*ltype temp)
%{ temp = ($*ltype)0;
   $1 = &temp; %}
%typemap(argout) SWIGTYPE** OUTPUT
%{SWIG_NewPointerObj(L,*$1,$*descriptor,1); SWIG_arg++; %}

