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
// MzScheme as much as possible, namely, to allow the user to pass and 
// be returned MzScheme vectors or lists.
// const declarations are used to guess the intent of the function being
// exported; therefore, the following rationale is applied:
// 
//   -- f(std::vector<T>), f(const std::vector<T>&), f(const std::vector<T>*):
//      the parameter being read-only, either a MzScheme sequence or a
//      previously wrapped std::vector<T> can be passed.
//   -- f(std::vector<T>&), f(std::vector<T>*):
//      the parameter must be modified; therefore, only a wrapped std::vector
//      can be passed.
//   -- std::vector<T> f():
//      the vector is returned by copy; therefore, a MzScheme vector of T:s 
//      is returned which is most easily used in other MzScheme functions
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
            if (SCHEME_VECTORP($input)) {
                unsigned int size = SCHEME_VEC_SIZE($input);
                $1 = std::vector<T >(size);
                Scheme_Object** items = SCHEME_VEC_ELS($input);
                for (unsigned int i=0; i<size; i++) {
                    (($1_type &)$1)[i] =
                        *((T*) SWIG_MustGetPtr(items[i],
                                               $descriptor(T *),
                                               $argnum, 0));
                }
            } else if (SCHEME_NULLP($input)) {
                $1 = std::vector<T >();
            } else if (SCHEME_PAIRP($input)) {
                Scheme_Object *head, *tail;
                $1 = std::vector<T >();
                tail = $input;
                while (!SCHEME_NULLP(tail)) {
                    head = scheme_car(tail);
                    tail = scheme_cdr(tail);
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
            if (SCHEME_VECTORP($input)) {
                unsigned int size = SCHEME_VEC_SIZE($input);
                temp = std::vector<T >(size);
                $1 = &temp;
                Scheme_Object** items = SCHEME_VEC_ELS($input);
                for (unsigned int i=0; i<size; i++) {
                    temp[i] = *((T*) SWIG_MustGetPtr(items[i],
                                                     $descriptor(T *),
                                                     $argnum, 0));
                }
            } else if (SCHEME_NULLP($input)) {
                temp = std::vector<T >();
                $1 = &temp;
            } else if (SCHEME_PAIRP($input)) {
                temp = std::vector<T >();
                $1 = &temp;
                Scheme_Object *head, *tail;
                tail = $input;
                while (!SCHEME_NULLP(tail)) {
                    head = scheme_car(tail);
                    tail = scheme_cdr(tail);
                    temp.push_back(*((T*) SWIG_MustGetPtr(head,
                                                          $descriptor(T *),
                                                          $argnum, 0)));
                }
            } else {
                $1 = ($1_ltype) SWIG_MustGetPtr($input,$1_descriptor,$argnum, 0);
            }
        }
        %typemap(out) vector<T> {
            $result = scheme_make_vector($1.size(),scheme_undefined);
            Scheme_Object** els = SCHEME_VEC_ELS($result);
            for (unsigned int i=0; i<$1.size(); i++) {
                T* x = new T((($1_type &)$1)[i]);
                els[i] = SWIG_NewPointerObj(x,$descriptor(T *), 1);
            }
        }
        %typecheck(SWIG_TYPECHECK_VECTOR) vector<T> {
            /* native sequence? */
            if (SCHEME_VECTORP($input)) {
                unsigned int size = SCHEME_VEC_SIZE($input);
                if (size == 0) {
                    /* an empty sequence can be of any type */
                    $1 = 1;
                } else {
                    /* check the first element only */
                    T* x;
                    Scheme_Object** items = SCHEME_VEC_ELS($input);
                    if (SWIG_ConvertPtr(items[0],(void**) &x,
                                    $descriptor(T *), 0) != -1)
                        $1 = 1;
                    else
                        $1 = 0;
                }
            } else if (SCHEME_NULLP($input)) {
                /* again, an empty sequence can be of any type */
                $1 = 1;
            } else if (SCHEME_PAIRP($input)) {
                /* check the first element only */
                T* x;
                Scheme_Object *head = scheme_car($input);
                if (SWIG_ConvertPtr(head,(void**) &x,
                                $descriptor(T *), 0) != -1)
                    $1 = 1;
                else
                    $1 = 0;
            } else {
                /* wrapped vector? */
                std::vector<T >* v;
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
            if (SCHEME_VECTORP($input)) {
                unsigned int size = SCHEME_VEC_SIZE($input);
                if (size == 0) {
                    /* an empty sequence can be of any type */
                    $1 = 1;
                } else {
                    /* check the first element only */
                    T* x;
                    Scheme_Object** items = SCHEME_VEC_ELS($input);
                    if (SWIG_ConvertPtr(items[0],(void**) &x,
                                    $descriptor(T *), 0) != -1)
                        $1 = 1;
                    else
                        $1 = 0;
                }
            } else if (SCHEME_NULLP($input)) {
                /* again, an empty sequence can be of any type */
                $1 = 1;
            } else if (SCHEME_PAIRP($input)) {
                /* check the first element only */
                T* x;
                Scheme_Object *head = scheme_car($input);
                if (SWIG_ConvertPtr(head,(void**) &x,
                                $descriptor(T *), 0) != -1)
                    $1 = 1;
                else
                    $1 = 0;
            } else {
                /* wrapped vector? */
                std::vector<T >* v;
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
            T& ref(int i) throw (std::out_of_range) {
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

    %define specialize_std_vector(T,CHECK,CONVERT_FROM,CONVERT_TO)
    template<> class vector<T> {
        %typemap(in) vector<T> {
            if (SCHEME_VECTORP($input)) {
                unsigned int size = SCHEME_VEC_SIZE($input);
                $1 = std::vector<T >(size);
                Scheme_Object** items = SCHEME_VEC_ELS($input);
                for (unsigned int i=0; i<size; i++) {
                    Scheme_Object* o = items[i];
                    if (CHECK(o))
                        (($1_type &)$1)[i] = (T)(CONVERT_FROM(o));
                    else
                        scheme_wrong_type(FUNC_NAME, "vector<" #T ">", 
                                          $argnum - 1, argc, argv);
                }
            } else if (SCHEME_NULLP($input)) {
                $1 = std::vector<T >();
            } else if (SCHEME_PAIRP($input)) {
                Scheme_Object *head, *tail;
                $1 = std::vector<T >();
                tail = $input;
                while (!SCHEME_NULLP(tail)) {
                    head = scheme_car(tail);
                    tail = scheme_cdr(tail);
                    if (CHECK(head))
                        $1.push_back((T)(CONVERT_FROM(head)));
                    else
                        scheme_wrong_type(FUNC_NAME, "vector<" #T ">", 
                                          $argnum - 1, argc, argv);
                }
            } else {
                $1 = *(($&1_type)
                       SWIG_MustGetPtr($input,$&1_descriptor,$argnum, 0));
            }
        }
        %typemap(in) const vector<T>& (std::vector<T> temp),
                     const vector<T>* (std::vector<T> temp) {
            if (SCHEME_VECTORP($input)) {
                unsigned int size = SCHEME_VEC_SIZE($input);
                temp = std::vector<T >(size);
                $1 = &temp;
                Scheme_Object** items = SCHEME_VEC_ELS($input);
                for (unsigned int i=0; i<size; i++) {
                    Scheme_Object* o = items[i];
                    if (CHECK(o))
                        temp[i] = (T)(CONVERT_FROM(o));
                    else
                        scheme_wrong_type(FUNC_NAME, "vector<" #T ">", 
                                          $argnum - 1, argc, argv);
                }
            } else if (SCHEME_NULLP($input)) {
                temp = std::vector<T >();
                $1 = &temp;
            } else if (SCHEME_PAIRP($input)) {
                temp = std::vector<T >();
                $1 = &temp;
                Scheme_Object *head, *tail;
                tail = $input;
                while (!SCHEME_NULLP(tail)) {
                    head = scheme_car(tail);
                    tail = scheme_cdr(tail);
                    if (CHECK(head))
                        temp.push_back((T)(CONVERT_FROM(head)));
                    else
                        scheme_wrong_type(FUNC_NAME, "vector<" #T ">", 
                                          $argnum - 1, argc, argv);
                }
            } else {
                $1 = ($1_ltype) SWIG_MustGetPtr($input,$1_descriptor,$argnum - 1, 0);
            }
        }
        %typemap(out) vector<T> {
            $result = scheme_make_vector($1.size(),scheme_undefined);
            Scheme_Object** els = SCHEME_VEC_ELS($result);
            for (unsigned int i=0; i<$1.size(); i++)
                els[i] = CONVERT_TO((($1_type &)$1)[i]);
        }
        %typecheck(SWIG_TYPECHECK_VECTOR) vector<T> {
            /* native sequence? */
            if (SCHEME_VECTORP($input)) {
                unsigned int size = SCHEME_VEC_SIZE($input);
                if (size == 0) {
                    /* an empty sequence can be of any type */
                    $1 = 1;
                } else {
                    /* check the first element only */
                    T* x;
                    Scheme_Object** items = SCHEME_VEC_ELS($input);
                    $1 = CHECK(items[0]) ? 1 : 0;
                }
            } else if (SCHEME_NULLP($input)) {
                /* again, an empty sequence can be of any type */
                $1 = 1;
            } else if (SCHEME_PAIRP($input)) {
                /* check the first element only */
                T* x;
                Scheme_Object *head = scheme_car($input);
                $1 = CHECK(head) ? 1 : 0;
            } else {
                /* wrapped vector? */
                std::vector<T >* v;
                $1 = (SWIG_ConvertPtr($input,(void **) &v, 
                                  $&1_descriptor, 0) != -1) ? 1 : 0;
            }
        }
        %typecheck(SWIG_TYPECHECK_VECTOR) const vector<T>&,
                                          const vector<T>* {
            /* native sequence? */
            if (SCHEME_VECTORP($input)) {
                unsigned int size = SCHEME_VEC_SIZE($input);
                if (size == 0) {
                    /* an empty sequence can be of any type */
                    $1 = 1;
                } else {
                    /* check the first element only */
                    T* x;
                    Scheme_Object** items = SCHEME_VEC_ELS($input);
                    $1 = CHECK(items[0]) ? 1 : 0;
                }
            } else if (SCHEME_NULLP($input)) {
                /* again, an empty sequence can be of any type */
                $1 = 1;
            } else if (SCHEME_PAIRP($input)) {
                /* check the first element only */
                T* x;
                Scheme_Object *head = scheme_car($input);
                $1 = CHECK(head) ? 1 : 0;
            } else {
                /* wrapped vector? */
                std::vector<T >* v;
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

    specialize_std_vector(bool,SCHEME_BOOLP,SCHEME_TRUEP,\
                          swig_make_boolean);
    specialize_std_vector(char,SCHEME_INTP,SCHEME_INT_VAL,\
                          scheme_make_integer_value);
    specialize_std_vector(int,SCHEME_INTP,SCHEME_INT_VAL,\
                          scheme_make_integer_value);
    specialize_std_vector(short,SCHEME_INTP,SCHEME_INT_VAL,\
                          scheme_make_integer_value);
    specialize_std_vector(long,SCHEME_INTP,SCHEME_INT_VAL,\
                          scheme_make_integer_value);
    specialize_std_vector(unsigned char,SCHEME_INTP,SCHEME_INT_VAL,\
                          scheme_make_integer_value);
    specialize_std_vector(unsigned int,SCHEME_INTP,SCHEME_INT_VAL,\
                          scheme_make_integer_value);
    specialize_std_vector(unsigned short,SCHEME_INTP,SCHEME_INT_VAL,\
                          scheme_make_integer_value);
    specialize_std_vector(unsigned long,SCHEME_INTP,SCHEME_INT_VAL,\
                          scheme_make_integer_value);
    specialize_std_vector(float,SCHEME_REALP,scheme_real_to_double,\
                          scheme_make_double);
    specialize_std_vector(double,SCHEME_REALP,scheme_real_to_double,\
                          scheme_make_double);
    specialize_std_vector(std::string,SCHEME_STRINGP,swig_scm_to_string,\
                          swig_make_string);

}

