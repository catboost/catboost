/* -----------------------------------------------------------------------------
 * std_vector.i
 *
 * std::vector typemaps for LUA
 * ----------------------------------------------------------------------------- */

%{
#include <vector>
%}
%include <std_except.i> // the general exceptions
/*
A really cut down version of the vector class.

Note: this does not match the true std::vector class
but instead is an approximate, so that SWIG knows how to wrapper it.
(Eg, all access is by value, not ref, as SWIG turns refs to pointers)

And no support for iterators & insert/erase

It would be useful to have a vector<->Lua table conversion routine

*/
namespace std {

	template<class T>
    class vector {
      public:
        typedef size_t size_type;
        typedef ptrdiff_t difference_type;
        typedef T value_type;
        typedef value_type* pointer;
        typedef const value_type* const_pointer;
        typedef value_type& reference;
        typedef const value_type& const_reference;

        vector();
        vector(unsigned int);
        vector(const vector& other);
        vector(unsigned int,T);

        unsigned int size() const;
        unsigned int max_size() const;
        bool empty() const;
        void clear();
        void push_back(T val);
        void pop_back();
        T front()const; // only read front & back
        T back()const;  // not write to them
        // operator [] given later:

		%extend // this is a extra bit of SWIG code
		{
			// [] is replaced by __getitem__ & __setitem__
			// simply throws a string, which causes a lua error
			T __getitem__(unsigned int idx) throw (std::out_of_range)
			{
				if (idx>=self->size())
					throw std::out_of_range("in vector::__getitem__()");
				return (*self)[idx];
			}
			void __setitem__(unsigned int idx,T val) throw (std::out_of_range)
			{
				if (idx>=self->size())
					throw std::out_of_range("in vector::__setitem__()");
				(*self)[idx]=val;
			}
		};
    };

}

/*
Vector<->LuaTable fns
These look a bit like the array<->LuaTable fns
but are templated, not %defined
(you must have template support for STL)

*/
/*
%{
// reads a table into a vector of numbers
// lua numbers will be cast into the type required (rounding may occur)
// return 0 if non numbers found in the table
// returns new'ed ptr if ok
template<class T>
std::vector<T>* SWIG_read_number_vector(lua_State* L,int index)
{
	int i=0;
	std::vector<T>* vec=new std::vector<T>();
	while(1)
	{
		lua_rawgeti(L,index,i+1);
		if (!lua_isnil(L,-1))
		{
			lua_pop(L,1);
			break;	// finished
		}
		if (!lua_isnumber(L,-1))
		{
			lua_pop(L,1);
			delete vec;
			return 0;	// error
		}
		vec->push_back((T)lua_tonumber(L,-1));
		lua_pop(L,1);
		++i;
	}
	return vec;	// ok
}
// writes a vector of numbers out as a lua table
template<class T>
int SWIG_write_number_vector(lua_State* L,std::vector<T> *vec)
{
	lua_newtable(L);
	for(int i=0;i<vec->size();++i)
	{
		lua_pushnumber(L,(double)((*vec)[i]));
		lua_rawseti(L,-2,i+1);// -1 is the number, -2 is the table
	}
}
%}

// then the typemaps

%define SWIG_TYPEMAP_NUM_VECTOR(T)

// in
%typemap(in) std::vector<T> *INPUT
%{	$1 = SWIG_read_number_vector<T>(L,$input);
	if (!$1) SWIG_fail;%}

%typemap(freearg) std::vector<T> *INPUT
%{	delete $1;%}

// out
%typemap(argout) std::vector<T> *OUTPUT
%{	SWIG_write_number_vector(L,$1); SWIG_arg++; %}

%enddef
*/
