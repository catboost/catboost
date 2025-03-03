/* -----------------------------------------------------------------------------
 * std_vector.i
 *
 * SWIG typemaps for std::vector types
 * ----------------------------------------------------------------------------- */

%include <std_common.i>

// ------------------------------------------------------------------------
// std::vector
// 
// The aim of all that follows would be to integrate std::vector with 
// Python as much as possible, namely, to allow the user to pass and 
// be returned Python tuples or lists.
// const declarations are used to guess the intent of the function being
// exported; therefore, the following rationale is applied:
// 
//   -- f(std::vector<T>), f(const std::vector<T>&), f(const std::vector<T>*):
//      the parameter being read-only, either a Python sequence or a
//      previously wrapped std::vector<T> can be passed.
//   -- f(std::vector<T>&), f(std::vector<T>*):
//      the parameter must be modified; therefore, only a wrapped std::vector
//      can be passed.
//   -- std::vector<T> f():
//      the vector is returned by copy; therefore, a Python sequence of T:s 
//      is returned which is most easily used in other Python functions
//   -- std::vector<T>& f(), std::vector<T>* f(), const std::vector<T>& f(),
//      const std::vector<T>* f():
//      the vector is returned by reference; therefore, a wrapped std::vector
//      is returned
// ------------------------------------------------------------------------

%{
#include <vector>
#include <algorithm>
#include <stdexcept>
%}

// exported class

namespace std {
    template <class T> class vector {
    public:
        typedef size_t size_type;
        typedef ptrdiff_t difference_type;
        typedef T value_type;
        typedef value_type* pointer;
        typedef const value_type* const_pointer;
        typedef value_type& reference;
        typedef const value_type& const_reference;

        vector(unsigned int size = 0);
        vector(unsigned int size, const T& value);
        vector(const vector& other);

        unsigned int size() const;
        bool empty() const;
        void clear();
        void push_back(const T& x);
	T operator [] ( int f );
	vector <T> &operator = ( vector <T> &other );
	%extend {
	    void set( int i, const T &x ) {
		self->resize(i+1);
		(*self)[i] = x;
	    }
	};
	%extend {
	    T *to_array() {
		T *array = new T[self->size() + 1];
		for( int i = 0; i < self->size(); i++ ) 
		    array[i] = (*self)[i];
		return array;
	    }
	};
    };
};

%insert(ml) %{
  
  let array_to_vector v argcons array = 
    for i = 0 to (Array.length array) - 1 do
	ignore ((invoke v) "set" (C_list [ C_int i ; (argcons array.(i)) ]))
    done ;
    v
    
  let vector_to_array v argcons array =
    for i = 0; to (get_int ((invoke v) "size" C_void)) - 1 do
	array.(i) <- argcons ((invoke v) "[]" (C_int i))
    done ; 
    v
      
%}

%insert(mli) %{
    val array_to_vector : c_obj -> ('a -> c_obj) -> 'a array -> c_obj
    val vector_to_array : c_obj -> (c_obj -> 'a) -> 'a array -> c_obj
%}
