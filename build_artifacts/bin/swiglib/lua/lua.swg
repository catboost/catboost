/* -----------------------------------------------------------------------------
 * lua.swg
 *
 * SWIG Configuration File for Lua.
 * This file is parsed by SWIG before reading any other interface file.
 * ----------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 *                          includes
 * ----------------------------------------------------------------------------- */

%include <luatypemaps.swg>         /* The typemaps */
%include <luaruntime.swg>          /* The runtime stuff */
%include <luakw.swg>               /* Warnings for Lua keywords */

//%include <typemaps/swigmacros.swg>
/* -----------------------------------------------------------------------------
 *                          constants typemaps
 * ----------------------------------------------------------------------------- */
// this basically adds to a table of constants
%typemap(consttab) int, unsigned int, short, unsigned short, long, unsigned long, unsigned char, signed char, bool, enum SWIGTYPE
       {SWIG_LUA_CONSTTAB_INT("$symname", $value)}

%typemap(consttab) float, double
       {SWIG_LUA_CONSTTAB_FLOAT("$symname", $value)}

%typemap(consttab) long long, unsigned long long, signed long long
       {SWIG_LUA_CONSTTAB_FLOAT("$symname", $value)}

%typemap(consttab) const long long&, const unsigned long long&, const signed long long&
       {SWIG_LUA_CONSTTAB_FLOAT("$symname", *$value)}

%typemap(consttab) char *, const char *, char [], const char []
       {SWIG_LUA_CONSTTAB_STRING("$symname", $value)}

// note: char is treated as a separate special type
// signed char & unsigned char are numbers
%typemap(consttab) char
       {SWIG_LUA_CONSTTAB_CHAR("$symname", $value)}

%typemap(consttab) long long, unsigned long long
       {SWIG_LUA_CONSTTAB_STRING("$symname", "$value")}

%typemap(consttab) SWIGTYPE *, SWIGTYPE *const, SWIGTYPE &, SWIGTYPE &&, SWIGTYPE []
       { SWIG_LUA_CONSTTAB_POINTER("$symname",$value, $1_descriptor) }

%typemap(consttab) SWIGTYPE
       { SWIG_LUA_CONSTTAB_POINTER("$symname",&$value, $&1_descriptor) }

// member function pointers
%typemap(consttab) SWIGTYPE (CLASS::*)
       { SWIG_LUA_CONSTTAB_BINARY("$symname", sizeof($type),&$value, $1_descriptor) }


/* -----------------------------------------------------------------------------
 *                          Overloaded operator support
 * ----------------------------------------------------------------------------- */
// lua calls the + operator '__add'
// python likes to call it '__add__'
// Assuming most SWIGers will probably use the __add__ if they extend their classes
// we have two sets of renames
// one to rename the operator+() to __add()
//	(this lets SWIG rename the operator overloads)
// another is to rename __add__() to __add()
//	(this means that people who wrote SWIG code to do that add will also work)

#ifdef __cplusplus
// this is extra renaming for lua
// not all operators are supported, so only those that are, are listed
%rename(__add)			*::operator+;
%rename(__sub)			*::operator-;
%rename(__mul)			*::operator*;
%rename(__div)			*::operator/;
%rename(__unm)      *::operator-();
%rename(__unm)      *::operator-() const;

%rename(__eq)			*::operator==;	
%ignore *::operator!=;      // note: Lua does not have a notequal operator
						// it just uses 'not (a==b)'
%rename(__lt)			*::operator<;
%ignore *::operator>;   	// ditto less than vs greater than
%rename(__le)			*::operator<=;	
%ignore *::operator>=;  // ditto less than vs greater than
%ignore *::operator!;  // does not support not

%rename(__call)			*::operator();	// the fn call operator

// lua does not support overloading of:
// 	logical/bitwise operators
// 	assign operator
// 	+=,-=,*=, etc
// therefore ignoring them for now
// it also doesn't support non class operators
// eg friends or XX operator+(XX,XX)
// also ignoring
// note: some of these might be better to rename, but not doing that for now
%ignore *::operator&&;	%ignore operator&&;
%ignore *::operator||;	%ignore operator||;
%ignore *::operator+=;
%ignore *::operator-=;
%ignore *::operator*=;
%ignore *::operator/=;
%ignore *::operator%=;
%ignore *::operator++;	%ignore *::operator--;

%ignore *::operator=;	// note: this might be better to rename to assign() or similar

%ignore operator+;
%ignore operator-;
%ignore operator*;
%ignore operator/;
%ignore operator%;
%ignore operator[];
%ignore operator>;	%ignore operator>=;	
%ignore operator<;	%ignore operator<=;
%ignore operator==;	%ignore operator!=;


// renaming the python operators to be compatible with lua
// this means that if a developer has written a fn __add__()
// it will be used for the lua +
%rename(__add)			*::__add__;
%rename(__sub)			*::__sub__;
%rename(__mul)			*::__mul__;
%rename(__div)			*::__div__;
%rename(__unm)			*::__neg__;		// lua calls unary minus,'unm' not 'neg'
%rename(__tostring)		*::__str__;		// both map to __tostring
%rename(__tostring)		*::__repr__;	// both map to __tostring


%rename(__pow)			*::__pow__;		// lua power '^' operator
%rename(__concat)		*::__concat__;  // lua concat '..' operator
%rename(__eq)			*::__eq__;
%rename(__lt)			*::__lt__;
%rename(__le)			*::__le__;
%rename(__call)			*::__call__;	// the fn call operator()

// the [] operator has two parts, the get & the set
%rename(__getitem)			*::__getitem__;	// the v=X[i] (get operator)
%rename(__setitem)			*::__setitem__;	// the X[i]=v (set operator)


#endif


/* ------------------------------------------------------------
 *                              Exceptions
 * ------------------------------------------------------------ */
/* Confession: I don't really like C++ exceptions
The python/lua ones are great, but C++ ones I don't like
(mainly because I cannot get the stack trace out of it)
Therefore I have not bothered to try doing much in this

Therefore currently its just enough to get a few test cases running ok

note: if you wish to throw anything related to std::exception
use %include <std_except.i> instead
*/

// number as number+error
%typemap(throws) int,unsigned int,signed int,
				long,unsigned long,signed long,
				short,unsigned short,signed short,
				float,double,
				long long,unsigned long long,
				unsigned char, signed char,
                int&,unsigned int&,signed int&,
				long&,unsigned long&,signed long&,
				short&,unsigned short&,signed short&,
				float&,double&,
				long long&,unsigned long long&,
				unsigned char&, signed char&
%{lua_pushnumber(L,(lua_Number)$1);SWIG_fail; %}

%typemap(throws) bool,bool& 
%{lua_pushboolean(L,(int)($1==true));SWIG_fail; %}

// enum as number+error
%typemap(throws) enum SWIGTYPE
%{lua_pushnumber(L,(lua_Number)(int)$1);SWIG_fail; %}

// strings are just sent as errors
%typemap(throws) char *, const char *
%{lua_pushstring(L,$1);SWIG_fail;%}

// char is changed to a string
%typemap(throws) char
%{lua_pushlstring(L,&$1,1);SWIG_fail;%}

/*
Throwing object is a serious problem:
Assuming some code throws a 'FooBar'
There are a few options:
- return a pointer to it: but its unclear how long this will last for.
- return a copy of it: but not all objects are copyable
	(see exception_partial_info in the test suite for a case where you cannot do this)
- convert to a string & throw that
	it's not so useful, but it works (this is more lua like).
The third option (though not nice) is used
For a more useful solution: see std_except for more details
*/

// basic typemap for structs, classes, pointers & references
// convert to string and error
%typemap(throws) SWIGTYPE
%{(void)$1; /* ignore it */
lua_pushfstring(L,"object exception:%s",SWIG_TypePrettyName($1_descriptor));
SWIG_fail;%}

// code to make a copy of the object and return this
// if you have a function which throws a FooBar & you want SWIG to return a copy of the object as its error
// then use one of the below
//	%apply SWIGTYPE EXCEPTION_BY_VAL {FooBar};
//	%apply SWIGTYPE& EXCEPTION_BY_VAL {FooBar&}; // note: need & twice
%typemap(throws) SWIGTYPE EXCEPTION_BY_VAL
%{SWIG_NewPointerObj(L,(void *)new $1_ltype(($1_ltype &) $1),$&1_descriptor,1);
SWIG_fail;%}

// similar for object reference
// note: swig typemaps seem a little confused around here, therefore we use $basetype
%typemap(throws) SWIGTYPE& EXCEPTION_BY_VAL
%{SWIG_NewPointerObj(L,(void *)new $basetype($1),$1_descriptor,1);
SWIG_fail;%}


// note: no support for object pointers
// its not clear how long the pointer is valid for, therefore not supporting it

/* -----------------------------------------------------------------------------
 *                          extras
 * ----------------------------------------------------------------------------- */
// this %define is to allow insertion of lua source code into the wrapper file
#define %luacode  %insert("luacode")


/* ------------------------------ end lua.swg  ------------------------------ */
