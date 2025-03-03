/* -----------------------------------------------------------------------------
 * std_vector.i
 *
 * SWIG typemaps for std::vector
 * ----------------------------------------------------------------------------- */

%include <std_common.i>

// ------------------------------------------------------------------------
// std::vector
// 
// The aim of all that follows would be to integrate std::vector with 
// Guile as much as possible, namely, to allow the user to pass and 
// be returned Guile vectors or lists.
// const declarations are used to guess the intent of the function being
// exported; therefore, the following rationale is applied:
// 
//   -- f(std::vector<T>), f(const std::vector<T>&), f(const std::vector<T>*):
//      the parameter being read-only, either a Guile sequence or a
//      previously wrapped std::vector<T> can be passed.
//   -- f(std::vector<T>&), f(std::vector<T>*):
//      the parameter must be modified; therefore, only a wrapped std::vector
//      can be passed.
//   -- std::vector<T> f():
//      the vector is returned by copy; therefore, a Guile vector of T:s 
//      is returned which is most easily used in other Guile functions
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
    
    template<class T> class vector {
        %typemap(in) vector<T> {
            if (scm_is_vector($input)) {
                unsigned long size = scm_c_vector_length($input);
                $1 = std::vector< T >(size);
                for (unsigned long i=0; i<size; i++) {
                    SCM o = scm_vector_ref($input,scm_from_ulong(i));
                    (($1_type &)$1)[i] =
                        *((T*) SWIG_MustGetPtr(o,$descriptor(T *),$argnum, 0));
                }
            } else if (scm_is_null($input)) {
                $1 = std::vector< T >();
            } else if (scm_is_pair($input)) {
                SCM head, tail;
                $1 = std::vector< T >();
                tail = $input;
                while (!scm_is_null(tail)) {
                    head = SCM_CAR(tail);
                    tail = SCM_CDR(tail);
                    $1.push_back(*((T*)SWIG_MustGetPtr(head,
                                                       $descriptor(T *),
                                                       $argnum, 0)));
                }
            } else {
                $1 = *(($&1_type)
                       SWIG_MustGetPtr($input,$&1_descriptor,$argnum, 0));
            }
        }
        %typemap(in) const vector<T>& (std::vector<T> temp),
                     const vector<T>* (std::vector<T> temp) {
            if (scm_is_vector($input)) {
                unsigned long size = scm_c_vector_length($input);
                temp = std::vector< T >(size);
                $1 = &temp;
                for (unsigned long i=0; i<size; i++) {
                    SCM o = scm_vector_ref($input,scm_from_ulong(i));
                    temp[i] = *((T*) SWIG_MustGetPtr(o,
                                                     $descriptor(T *),
                                                     $argnum, 0));
                }
            } else if (scm_is_null($input)) {
                temp = std::vector< T >();
                $1 = &temp;
            } else if (scm_is_pair($input)) {
                temp = std::vector< T >();
                $1 = &temp;
                SCM head, tail;
                tail = $input;
                while (!scm_is_null(tail)) {
                    head = SCM_CAR(tail);
                    tail = SCM_CDR(tail);
                    temp.push_back(*((T*) SWIG_MustGetPtr(head,
                                                          $descriptor(T *),
                                                          $argnum, 0)));
                }
            } else {
                $1 = ($1_ltype) SWIG_MustGetPtr($input,$1_descriptor,$argnum, 0);
            }
        }
        %typemap(out) vector<T> {
            $result = scm_make_vector(scm_from_long($1.size()),SCM_UNSPECIFIED);
            for (unsigned int i=0; i<$1.size(); i++) {
                T* x = new T((($1_type &)$1)[i]);
                scm_vector_set_x($result,scm_from_long(i),
                                SWIG_NewPointerObj(x, $descriptor(T *), 1));
            }
        }
        %typecheck(SWIG_TYPECHECK_VECTOR) vector<T> {
            /* native sequence? */
            if (scm_is_vector($input)) {
                unsigned int size = scm_c_vector_length($input);
                if (size == 0) {
                    /* an empty sequence can be of any type */
                    $1 = 1;
                } else {
                    /* check the first element only */
                    SCM o = scm_vector_ref($input,scm_from_ulong(0));
                    T* x;
                    if (SWIG_ConvertPtr(o,(void**) &x,
                                          $descriptor(T *), 0) != -1)
                        $1 = 1;
                    else
                        $1 = 0;
                }
            } else if (scm_is_null($input)) {
                /* again, an empty sequence can be of any type */
                $1 = 1;
            } else if (scm_is_pair($input)) {
                /* check the first element only */
                T* x;
                SCM head = SCM_CAR($input);
                if (SWIG_ConvertPtr(head,(void**) &x,
                                      $descriptor(T *), 0) != -1)
                    $1 = 1;
                else
                    $1 = 0;
            } else {
                /* wrapped vector? */
                std::vector< T >* v;
                if (SWIG_ConvertPtr($input,(void **) &v, 
                                      $&1_descriptor, 0) != -1)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
        %typecheck(SWIG_TYPECHECK_VECTOR) const vector<T>&,
                                          const vector<T>* {
            /* native sequence? */
            if (scm_is_vector($input)) {
                unsigned int size = scm_c_vector_length($input);
                if (size == 0) {
                    /* an empty sequence can be of any type */
                    $1 = 1;
                } else {
                    /* check the first element only */
                    T* x;
                    SCM o = scm_vector_ref($input,scm_from_ulong(0));
                    if (SWIG_ConvertPtr(o,(void**) &x,
                                          $descriptor(T *), 0) != -1)
                        $1 = 1;
                    else
                        $1 = 0;
                }
            } else if (scm_is_null($input)) {
                /* again, an empty sequence can be of any type */
                $1 = 1;
            } else if (scm_is_pair($input)) {
                /* check the first element only */
                T* x;
                SCM head = SCM_CAR($input);
                if (SWIG_ConvertPtr(head,(void**) &x,
                                      $descriptor(T *), 0) != -1)
                    $1 = 1;
                else
                    $1 = 0;
            } else {
                /* wrapped vector? */
                std::vector< T >* v;
                if (SWIG_ConvertPtr($input,(void **) &v, 
                                      $1_descriptor, 0) != -1)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
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

        %rename(length) size;
        unsigned int size() const;
        %rename("empty?") empty;
        bool empty() const;
        %rename("clear!") clear;
        void clear();
        %rename("set!") set;
        %rename("pop!") pop;
        %rename("push!") push_back;
        void push_back(const T& x);
        %extend {
            T pop() throw (std::out_of_range) {
                if (self->size() == 0)
                    throw std::out_of_range("pop from empty vector");
                T x = self->back();
                self->pop_back();
                return x;
            }
            const T& ref(int i) throw (std::out_of_range) {
                int size = int(self->size());
                if (i>=0 && i<size)
                    return (*self)[i];
                else
                    throw std::out_of_range("vector index out of range");
            }
            void set(int i, const T& x) throw (std::out_of_range) {
                int size = int(self->size());
                if (i>=0 && i<size)
                    (*self)[i] = x;
                else
                    throw std::out_of_range("vector index out of range");
            }
        }
    };


    // specializations for built-ins
    %define specialize_stl_vector(T,CHECK,CONVERT_FROM,CONVERT_TO)
    template<> class vector<T> {
        %typemap(in) vector<T> {
            if (scm_is_vector($input)) {
                unsigned long size = scm_c_vector_length($input);
                $1 = std::vector< T >(size);
                for (unsigned long i=0; i<size; i++) {
                    SCM o = scm_vector_ref($input,scm_from_ulong(i));
                    if (CHECK(o))
                        (($1_type &)$1)[i] = (T)(CONVERT_FROM(o));
                    else
                        scm_wrong_type_arg(FUNC_NAME, $argnum, $input);
                }
            } else if (scm_is_null($input)) {
                $1 = std::vector<T >();
            } else if (scm_is_pair($input)) {
                SCM v = scm_vector($input);
                unsigned long size = scm_c_vector_length(v);
                $1 = std::vector< T >(size);
                for (unsigned long i=0; i<size; i++) {
                    SCM o = scm_vector_ref(v,scm_from_ulong(i));
                    if (CHECK(o))
                        (($1_type &)$1)[i] = (T)(CONVERT_FROM(o));
                    else
                        scm_wrong_type_arg(FUNC_NAME, $argnum, $input);
                }
            } else {
                $1 = *(($&1_type)
                       SWIG_MustGetPtr($input,$&1_descriptor,$argnum, 0));
            }
        }
        %typemap(in) const vector<T>& (std::vector<T> temp),
                     const vector<T>* (std::vector<T> temp) {
            if (scm_is_vector($input)) {
                unsigned long size = scm_c_vector_length($input);
                temp = std::vector< T >(size);
                $1 = &temp;
                for (unsigned long i=0; i<size; i++) {
                    SCM o = scm_vector_ref($input,scm_from_ulong(i));
                    if (CHECK(o))
                        temp[i] = (T)(CONVERT_FROM(o));
                    else
                        scm_wrong_type_arg(FUNC_NAME, $argnum, $input);
                }
            } else if (scm_is_null($input)) {
                temp = std::vector< T >();
                $1 = &temp;
            } else if (scm_is_pair($input)) {
                SCM v = scm_vector($input);
                unsigned long size = scm_c_vector_length(v);
                temp = std::vector< T >(size);
                $1 = &temp;
                for (unsigned long i=0; i<size; i++) {
                    SCM o = scm_vector_ref(v,scm_from_ulong(i));
                    if (CHECK(o))
                        temp[i] = (T)(CONVERT_FROM(o));
                    else
                        scm_wrong_type_arg(FUNC_NAME, $argnum, $input);
                }
            } else {
                $1 = ($1_ltype) SWIG_MustGetPtr($input,$1_descriptor,$argnum, 0);
            }
        }
        %typemap(out) vector<T> {
            $result = scm_make_vector(scm_from_long($1.size()),SCM_UNSPECIFIED);
            for (unsigned int i=0; i<$1.size(); i++) {
                SCM x = CONVERT_TO((($1_type &)$1)[i]);
                scm_vector_set_x($result,scm_from_long(i),x);
            }
        }
        %typecheck(SWIG_TYPECHECK_VECTOR) vector<T> {
            /* native sequence? */
            if (scm_is_vector($input)) {
                unsigned int size = scm_c_vector_length($input);
                if (size == 0) {
                    /* an empty sequence can be of any type */
                    $1 = 1;
                } else {
                    /* check the first element only */
                    SCM o = scm_vector_ref($input,scm_from_ulong(0));
                    $1 = CHECK(o) ? 1 : 0;
                }
            } else if (scm_is_null($input)) {
                /* again, an empty sequence can be of any type */
                $1 = 1;
            } else if (scm_is_pair($input)) {
                /* check the first element only */
                SCM head = SCM_CAR($input);
                $1 = CHECK(head) ? 1 : 0;
            } else {
                /* wrapped vector? */
                std::vector< T >* v;
                $1 = (SWIG_ConvertPtr($input,(void **) &v, 
                                        $&1_descriptor, 0) != -1) ? 1 : 0;
            }
        }
        %typecheck(SWIG_TYPECHECK_VECTOR) const vector<T>&,
                                          const vector<T>* {
            /* native sequence? */
            if (scm_is_vector($input)) {
                unsigned int size = scm_c_vector_length($input);
                if (size == 0) {
                    /* an empty sequence can be of any type */
                    $1 = 1;
                } else {
                    /* check the first element only */
                    SCM o = scm_vector_ref($input,scm_from_ulong(0));
                    $1 = CHECK(o) ? 1 : 0;
                }
            } else if (scm_is_null($input)) {
                /* again, an empty sequence can be of any type */
                $1 = 1;
            } else if (scm_is_pair($input)) {
                /* check the first element only */
                SCM head = SCM_CAR($input);
                $1 = CHECK(head) ? 1 : 0;
            } else {
                /* wrapped vector? */
                std::vector< T >* v;
                $1 = (SWIG_ConvertPtr($input,(void **) &v, 
                                        $1_descriptor, 0) != -1) ? 1 : 0;
            }
        }
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

        %rename(length) size;
        unsigned int size() const;
        %rename("empty?") empty;
        bool empty() const;
        %rename("clear!") clear;
        void clear();
        %rename("set!") set;
        %rename("pop!") pop;
        %rename("push!") push_back;
        void push_back(T x);
        %extend {
            T pop() throw (std::out_of_range) {
                if (self->size() == 0)
                    throw std::out_of_range("pop from empty vector");
                T x = self->back();
                self->pop_back();
                return x;
            }
            T ref(int i) throw (std::out_of_range) {
                int size = int(self->size());
                if (i>=0 && i<size)
                    return (*self)[i];
                else
                    throw std::out_of_range("vector index out of range");
            }
            void set(int i, T x) throw (std::out_of_range) {
                int size = int(self->size());
                if (i>=0 && i<size)
                    (*self)[i] = x;
                else
                    throw std::out_of_range("vector index out of range");
            }
        }
    };
    %enddef

    specialize_stl_vector(bool,scm_is_bool,scm_is_true,SWIG_bool2scm);
    specialize_stl_vector(char,scm_is_number,scm_to_long,scm_from_long);
    specialize_stl_vector(int,scm_is_number,scm_to_long,scm_from_long);
    specialize_stl_vector(long,scm_is_number,scm_to_long,scm_from_long);
    specialize_stl_vector(short,scm_is_number,scm_to_long,scm_from_long);
    specialize_stl_vector(unsigned char,scm_is_number,scm_to_ulong,scm_from_ulong);
    specialize_stl_vector(unsigned int,scm_is_number,scm_to_ulong,scm_from_ulong);
    specialize_stl_vector(unsigned long,scm_is_number,scm_to_ulong,scm_from_ulong);
    specialize_stl_vector(unsigned short,scm_is_number,scm_to_ulong,scm_from_ulong);
    specialize_stl_vector(float,scm_is_number,scm_to_double,scm_from_double);
    specialize_stl_vector(double,scm_is_number,scm_to_double,scm_from_double);
    specialize_stl_vector(std::string,scm_is_string,SWIG_scm2string,SWIG_string2scm);
}

