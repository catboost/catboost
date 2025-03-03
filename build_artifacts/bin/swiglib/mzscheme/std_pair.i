/* -----------------------------------------------------------------------------
 * std_pair.i
 *
 * SWIG typemaps for std::pair
 * ----------------------------------------------------------------------------- */

%include <std_common.i>
%include <exception.i>


// ------------------------------------------------------------------------
// std::pair
//
// See std_vector.i for the rationale of typemap application
// ------------------------------------------------------------------------

%{
#include <utility>
%}

// exported class

namespace std {

    template<class T, class U> struct pair {
        %typemap(in) pair<T,U> (std::pair<T,U>* m) {
            if (SCHEME_PAIRP($input)) {
                T* x;
                U* y;
                Scheme_Object *first, *second;
                first = scheme_car($input);
                second = scheme_cdr($input);
                x = (T*) SWIG_MustGetPtr(first,$descriptor(T *),$argnum, 0);
                y = (U*) SWIG_MustGetPtr(second,$descriptor(U *),$argnum, 0);
                $1 = std::make_pair(*x,*y);
            } else {
                $1 = *(($&1_type)
                       SWIG_MustGetPtr($input,$&1_descriptor,$argnum, 0));
            }
        }
        %typemap(in) const pair<T,U>& (std::pair<T,U> temp,
                                       std::pair<T,U>* m),
                     const pair<T,U>* (std::pair<T,U> temp,
                                       std::pair<T,U>* m) {
            if (SCHEME_PAIRP($input)) {
                T* x;
                U* y;
                Scheme_Object *first, *second;
                first = scheme_car($input);
                second = scheme_cdr($input);
                x = (T*) SWIG_MustGetPtr(first,$descriptor(T *),$argnum, 0);
                y = (U*) SWIG_MustGetPtr(second,$descriptor(U *),$argnum, 0);
                temp = std::make_pair(*x,*y);
                $1 = &temp;
            } else {
                $1 = ($1_ltype)
                    SWIG_MustGetPtr($input,$1_descriptor,$argnum, 0);
            }
        }
        %typemap(out) pair<T,U> {
            T* x = new T($1.first);
            U* y = new U($1.second);
            Scheme_Object* first = SWIG_NewPointerObj(x,$descriptor(T *), 1);
            Scheme_Object* second = SWIG_NewPointerObj(y,$descriptor(U *), 1);
            $result = scheme_make_pair(first,second);
        }
        %typecheck(SWIG_TYPECHECK_PAIR) pair<T,U> {
            /* native pair? */
            if (SCHEME_PAIRP($input)) {
                T* x;
                U* y;
                Scheme_Object* first = scheme_car($input);
                Scheme_Object* second = scheme_cdr($input);
                if (SWIG_ConvertPtr(first,(void**) &x,
                                    $descriptor(T *), 0) != -1 &&
                    SWIG_ConvertPtr(second,(void**) &y,
                                    $descriptor(U *), 0) != -1) {
                        $1 = 1;
                } else {
                    $1 = 0;
                }
            } else {
                /* wrapped pair? */
                std::pair<T,U >* p;
                if (SWIG_ConvertPtr($input,(void **) &p,
                                    $&1_descriptor, 0) != -1)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
        %typecheck(SWIG_TYPECHECK_PAIR) const pair<T,U>&,
                                        const pair<T,U>* {
            /* native pair? */
            if (SCHEME_PAIRP($input)) {
                T* x;
                U* y;
                Scheme_Object* first = scheme_car($input);
                Scheme_Object* second = scheme_cdr($input);
                if (SWIG_ConvertPtr(first,(void**) &x,
                                    $descriptor(T *), 0) != -1 &&
                    SWIG_ConvertPtr(second,(void**) &y,
                                    $descriptor(U *), 0) != -1) {
                        $1 = 1;
                } else {
                    $1 = 0;
                }
            } else {
                /* wrapped pair? */
                std::pair<T,U >* p;
                if (SWIG_ConvertPtr($input,(void **) &p,
                                    $1_descriptor, 0) != -1)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }

        typedef T first_type;
        typedef U second_type;

        pair();
        pair(T first, U second);
        pair(const pair& other);

        template <class U1, class U2> pair(const pair<U1, U2> &other);

        T first;
        U second;
    };

    // specializations for built-ins

    %define specialize_std_pair_on_first(T,CHECK,CONVERT_FROM,CONVERT_TO)
    template<class U> struct pair<T,U> {
        %typemap(in) pair<T,U> (std::pair<T,U>* m) {
            if (SCHEME_PAIRP($input)) {
                U* y;
                Scheme_Object *first, *second;
                first = scheme_car($input);
                second = scheme_cdr($input);
                if (!CHECK(first))
                    SWIG_exception(SWIG_TypeError,
                                   "pair<" #T "," #U "> expected");
                y = (U*) SWIG_MustGetPtr(second,$descriptor(U *),$argnum, 0);
                $1 = std::make_pair(CONVERT_FROM(first),*y);
            } else {
                $1 = *(($&1_type)
                       SWIG_MustGetPtr($input,$&1_descriptor,$argnum, 0));
            }
        }
        %typemap(in) const pair<T,U>& (std::pair<T,U> temp,
                                       std::pair<T,U>* m),
                     const pair<T,U>* (std::pair<T,U> temp,
                                       std::pair<T,U>* m) {
            if (SCHEME_PAIRP($input)) {
                U* y;
                Scheme_Object *first, *second;
                first = scheme_car($input);
                second = scheme_cdr($input);
                if (!CHECK(first))
                    SWIG_exception(SWIG_TypeError,
                                   "pair<" #T "," #U "> expected");
                y = (U*) SWIG_MustGetPtr(second,$descriptor(U *),$argnum, 0);
                temp = std::make_pair(CONVERT_FROM(first),*y);
                $1 = &temp;
            } else {
                $1 = ($1_ltype)
                    SWIG_MustGetPtr($input,$1_descriptor,$argnum, 0);
            }
        }
        %typemap(out) pair<T,U> {
            U* y = new U($1.second);
            Scheme_Object* second = SWIG_NewPointerObj(y,$descriptor(U *), 1);
            $result = scheme_make_pair(CONVERT_TO($1.first),second);
        }
        %typecheck(SWIG_TYPECHECK_PAIR) pair<T,U> {
            /* native pair? */
            if (SCHEME_PAIRP($input)) {
                U* y;
                Scheme_Object* first = scheme_car($input);
                Scheme_Object* second = scheme_cdr($input);
                if (CHECK(first) &&
                    SWIG_ConvertPtr(second,(void**) &y,
                                    $descriptor(U *), 0) != -1) {
                        $1 = 1;
                } else {
                    $1 = 0;
                }
            } else {
                /* wrapped pair? */
                std::pair<T,U >* p;
                if (SWIG_ConvertPtr($input,(void **) &p,
                                    $&1_descriptor, 0) != -1)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
        %typecheck(SWIG_TYPECHECK_PAIR) const pair<T,U>&,
                                        const pair<T,U>* {
            /* native pair? */
            if (SCHEME_PAIRP($input)) {
                U* y;
                Scheme_Object* first = scheme_car($input);
                Scheme_Object* second = scheme_cdr($input);
                if (CHECK(first) &&
                    SWIG_ConvertPtr(second,(void**) &y,
                                    $descriptor(U *), 0) != -1) {
                        $1 = 1;
                } else {
                    $1 = 0;
                }
            } else {
                /* wrapped pair? */
                std::pair<T,U >* p;
                if (SWIG_ConvertPtr($input,(void **) &p,
                                    $1_descriptor, 0) != -1)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
        pair();
        pair(T first, U second);
        pair(const pair& other);

        template <class U1, class U2> pair(const pair<U1, U2> &other);

        T first;
        U second;
    };
    %enddef

    %define specialize_std_pair_on_second(U,CHECK,CONVERT_FROM,CONVERT_TO)
    template<class T> struct pair<T,U> {
        %typemap(in) pair<T,U> (std::pair<T,U>* m) {
            if (SCHEME_PAIRP($input)) {
                T* x;
                Scheme_Object *first, *second;
                first = scheme_car($input);
                second = scheme_cdr($input);
                x = (T*) SWIG_MustGetPtr(first,$descriptor(T *),$argnum, 0);
                if (!CHECK(second))
                    SWIG_exception(SWIG_TypeError,
                                   "pair<" #T "," #U "> expected");
                $1 = std::make_pair(*x,CONVERT_FROM(second));
            } else {
                $1 = *(($&1_type)
                       SWIG_MustGetPtr($input,$&1_descriptor,$argnum, 0));
            }
        }
        %typemap(in) const pair<T,U>& (std::pair<T,U> temp,
                                       std::pair<T,U>* m),
                     const pair<T,U>* (std::pair<T,U> temp,
                                       std::pair<T,U>* m) {
            if (SCHEME_PAIRP($input)) {
                T* x;
                Scheme_Object *first, *second;
                first = scheme_car($input);
                second = scheme_cdr($input);
                x = (T*) SWIG_MustGetPtr(first,$descriptor(T *),$argnum, 0);
                if (!CHECK(second))
                    SWIG_exception(SWIG_TypeError,
                                   "pair<" #T "," #U "> expected");
                temp = std::make_pair(*x,CONVERT_FROM(second));
                $1 = &temp;
            } else {
                $1 = ($1_ltype)
                    SWIG_MustGetPtr($input,$1_descriptor,$argnum, 0);
            }
        }
        %typemap(out) pair<T,U> {
            T* x = new T($1.first);
            Scheme_Object* first = SWIG_NewPointerObj(x,$descriptor(T *), 1);
            $result = scheme_make_pair(first,CONVERT_TO($1.second));
        }
        %typecheck(SWIG_TYPECHECK_PAIR) pair<T,U> {
            /* native pair? */
            if (SCHEME_PAIRP($input)) {
                T* x;
                Scheme_Object* first = scheme_car($input);
                Scheme_Object* second = scheme_cdr($input);
                if (SWIG_ConvertPtr(first,(void**) &x,
                                    $descriptor(T *), 0) != -1 &&
                    CHECK(second)) {
                        $1 = 1;
                } else {
                    $1 = 0;
                }
            } else {
                /* wrapped pair? */
                std::pair<T,U >* p;
                if (SWIG_ConvertPtr($input,(void **) &p,
                                    $&1_descriptor, 0) != -1)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
        %typecheck(SWIG_TYPECHECK_PAIR) const pair<T,U>&,
                                        const pair<T,U>* {
            /* native pair? */
            if (SCHEME_PAIRP($input)) {
                T* x;
                Scheme_Object* first = scheme_car($input);
                Scheme_Object* second = scheme_cdr($input);
                if (SWIG_ConvertPtr(first,(void**) &x,
                                    $descriptor(T *), 0) != -1 &&
                    CHECK(second)) {
                        $1 = 1;
                } else {
                    $1 = 0;
                }
            } else {
                /* wrapped pair? */
                std::pair<T,U >* p;
                if (SWIG_ConvertPtr($input,(void **) &p,
                                    $1_descriptor, 0) != -1)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
        pair();
        pair(T first, U second);
        pair(const pair& other);

        template <class U1, class U2> pair(const pair<U1, U2> &other);

        T first;
        U second;
    };
    %enddef

    %define specialize_std_pair_on_both(T,CHECK_T,CONVERT_T_FROM,CONVERT_T_TO,
                                        U,CHECK_U,CONVERT_U_FROM,CONVERT_U_TO)
    template<> struct pair<T,U> {
        %typemap(in) pair<T,U> (std::pair<T,U>* m) {
            if (SCHEME_PAIRP($input)) {
                Scheme_Object *first, *second;
                first = scheme_car($input);
                second = scheme_cdr($input);
                if (!CHECK_T(first) || !CHECK_U(second))
                    SWIG_exception(SWIG_TypeError,
                                   "pair<" #T "," #U "> expected");
                $1 = make_pair(CONVERT_T_FROM(first),
                               CONVERT_U_FROM(second));
            } else {
                $1 = *(($&1_type)
                       SWIG_MustGetPtr($input,$&1_descriptor,$argnum, 0));
            }
        }
        %typemap(in) const pair<T,U>& (std::pair<T,U> temp,
                                       std::pair<T,U>* m),
                     const pair<T,U>* (std::pair<T,U> temp,
                                       std::pair<T,U>* m) {
            if (SCHEME_PAIRP($input)) {
                Scheme_Object *first, *second;
            T *x;
                first = scheme_car($input);
                second = scheme_cdr($input);
                x = (T*) SWIG_MustGetPtr(first,$descriptor(T *),$argnum, 0);
                if (!CHECK_T(first) || !CHECK_U(second))
                    SWIG_exception(SWIG_TypeError,
                                   "pair<" #T "," #U "> expected");
                temp = make_pair(CONVERT_T_FROM(first),
                               CONVERT_U_FROM(second));
                $1 = &temp;
            } else {
                $1 = ($1_ltype)
                    SWIG_MustGetPtr($input,$1_descriptor,$argnum, 0);
            }
        }
        %typemap(out) pair<T,U> {
            $result = scheme_make_pair(CONVERT_T_TO($1.first),
                                       CONVERT_U_TO($1.second));
        }
        %typecheck(SWIG_TYPECHECK_PAIR) pair<T,U> {
            /* native pair? */
            if (SCHEME_PAIRP($input)) {
                Scheme_Object* first = scheme_car($input);
                Scheme_Object* second = scheme_cdr($input);
                if (CHECK_T(first) && CHECK_U(second)) {
                        $1 = 1;
                } else {
                    $1 = 0;
                }
            } else {
                /* wrapped pair? */
                std::pair<T,U >* p;
                if (SWIG_ConvertPtr($input,(void **) &p,
                                    $&1_descriptor, 0) != -1)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
        %typecheck(SWIG_TYPECHECK_PAIR) const pair<T,U>&,
                                        const pair<T,U>* {
            /* native pair? */
            if (SCHEME_PAIRP($input)) {
                Scheme_Object* first = scheme_car($input);
                Scheme_Object* second = scheme_cdr($input);
                if (CHECK_T(first) && CHECK_U(second)) {
                        $1 = 1;
                } else {
                    $1 = 0;
                }
            } else {
                /* wrapped pair? */
                std::pair<T,U >* p;
                if (SWIG_ConvertPtr($input,(void **) &p,
                                    $1_descriptor, 0) != -1)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
        pair();
        pair(T first, U second);
        pair(const pair& other);

        template <class U1, class U2> pair(const pair<U1, U2> &other);

        T first;
        U second;
    };
    %enddef


    specialize_std_pair_on_first(bool,SCHEME_BOOLP,
                              SCHEME_TRUEP,swig_make_boolean);
    specialize_std_pair_on_first(int,SCHEME_INTP,
                              SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_first(short,SCHEME_INTP,
                              SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_first(long,SCHEME_INTP,
                              SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_first(unsigned int,SCHEME_INTP,
                              SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_first(unsigned short,SCHEME_INTP,
                              SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_first(unsigned long,SCHEME_INTP,
                              SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_first(double,SCHEME_REALP,
                              scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_first(float,SCHEME_REALP,
                              scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_first(std::string,SCHEME_STRINGP,
                              swig_scm_to_string,swig_make_string);

    specialize_std_pair_on_second(bool,SCHEME_BOOLP,
                                SCHEME_TRUEP,swig_make_boolean);
    specialize_std_pair_on_second(int,SCHEME_INTP,
                                SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_second(short,SCHEME_INTP,
                                SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_second(long,SCHEME_INTP,
                                SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_second(unsigned int,SCHEME_INTP,
                                SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_second(unsigned short,SCHEME_INTP,
                                SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_second(unsigned long,SCHEME_INTP,
                                SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_second(double,SCHEME_REALP,
                                scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_second(float,SCHEME_REALP,
                                scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_second(std::string,SCHEME_STRINGP,
                                swig_scm_to_string,swig_make_string);

    specialize_std_pair_on_both(bool,SCHEME_BOOLP,
                               SCHEME_TRUEP,swig_make_boolean,
                               bool,SCHEME_BOOLP,
                               SCHEME_TRUEP,swig_make_boolean);
    specialize_std_pair_on_both(bool,SCHEME_BOOLP,
                               SCHEME_TRUEP,swig_make_boolean,
                               int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(bool,SCHEME_BOOLP,
                               SCHEME_TRUEP,swig_make_boolean,
                               short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(bool,SCHEME_BOOLP,
                               SCHEME_TRUEP,swig_make_boolean,
                               long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(bool,SCHEME_BOOLP,
                               SCHEME_TRUEP,swig_make_boolean,
                               unsigned int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(bool,SCHEME_BOOLP,
                               SCHEME_TRUEP,swig_make_boolean,
                               unsigned short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(bool,SCHEME_BOOLP,
                               SCHEME_TRUEP,swig_make_boolean,
                               unsigned long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(bool,SCHEME_BOOLP,
                               SCHEME_TRUEP,swig_make_boolean,
                               double,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_both(bool,SCHEME_BOOLP,
                               SCHEME_TRUEP,swig_make_boolean,
                               float,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_both(bool,SCHEME_BOOLP,
                               SCHEME_TRUEP,swig_make_boolean,
                               std::string,SCHEME_STRINGP,
                               swig_scm_to_string,swig_make_string);
    specialize_std_pair_on_both(int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               bool,SCHEME_BOOLP,
                               SCHEME_TRUEP,swig_make_boolean);
    specialize_std_pair_on_both(int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               unsigned int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               unsigned short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               unsigned long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               double,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_both(int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               float,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_both(int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               std::string,SCHEME_STRINGP,
                               swig_scm_to_string,swig_make_string);
    specialize_std_pair_on_both(short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               bool,SCHEME_BOOLP,
                               SCHEME_TRUEP,swig_make_boolean);
    specialize_std_pair_on_both(short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               unsigned int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               unsigned short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               unsigned long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               double,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_both(short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               float,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_both(short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               std::string,SCHEME_STRINGP,
                               swig_scm_to_string,swig_make_string);
    specialize_std_pair_on_both(long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               bool,SCHEME_BOOLP,
                               SCHEME_TRUEP,swig_make_boolean);
    specialize_std_pair_on_both(long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               unsigned int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               unsigned short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               unsigned long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               double,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_both(long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               float,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_both(long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               std::string,SCHEME_STRINGP,
                               swig_scm_to_string,swig_make_string);
    specialize_std_pair_on_both(unsigned int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               bool,SCHEME_BOOLP,
                               SCHEME_TRUEP,swig_make_boolean);
    specialize_std_pair_on_both(unsigned int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(unsigned int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(unsigned int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(unsigned int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               unsigned int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(unsigned int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               unsigned short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(unsigned int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               unsigned long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(unsigned int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               double,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_both(unsigned int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               float,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_both(unsigned int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               std::string,SCHEME_STRINGP,
                               swig_scm_to_string,swig_make_string);
    specialize_std_pair_on_both(unsigned short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               bool,SCHEME_BOOLP,
                               SCHEME_TRUEP,swig_make_boolean);
    specialize_std_pair_on_both(unsigned short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(unsigned short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(unsigned short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(unsigned short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               unsigned int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(unsigned short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               unsigned short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(unsigned short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               unsigned long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(unsigned short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               double,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_both(unsigned short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               float,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_both(unsigned short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               std::string,SCHEME_STRINGP,
                               swig_scm_to_string,swig_make_string);
    specialize_std_pair_on_both(unsigned long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               bool,SCHEME_BOOLP,
                               SCHEME_TRUEP,swig_make_boolean);
    specialize_std_pair_on_both(unsigned long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(unsigned long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(unsigned long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(unsigned long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               unsigned int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(unsigned long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               unsigned short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(unsigned long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               unsigned long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(unsigned long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               double,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_both(unsigned long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               float,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_both(unsigned long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value,
                               std::string,SCHEME_STRINGP,
                               swig_scm_to_string,swig_make_string);
    specialize_std_pair_on_both(double,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double,
                               bool,SCHEME_BOOLP,
                               SCHEME_TRUEP,swig_make_boolean);
    specialize_std_pair_on_both(double,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double,
                               int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(double,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double,
                               short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(double,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double,
                               long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(double,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double,
                               unsigned int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(double,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double,
                               unsigned short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(double,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double,
                               unsigned long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(double,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double,
                               double,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_both(double,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double,
                               float,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_both(double,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double,
                               std::string,SCHEME_STRINGP,
                               swig_scm_to_string,swig_make_string);
    specialize_std_pair_on_both(float,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double,
                               bool,SCHEME_BOOLP,
                               SCHEME_TRUEP,swig_make_boolean);
    specialize_std_pair_on_both(float,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double,
                               int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(float,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double,
                               short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(float,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double,
                               long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(float,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double,
                               unsigned int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(float,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double,
                               unsigned short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(float,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double,
                               unsigned long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(float,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double,
                               double,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_both(float,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double,
                               float,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_both(float,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double,
                               std::string,SCHEME_STRINGP,
                               swig_scm_to_string,swig_make_string);
    specialize_std_pair_on_both(std::string,SCHEME_STRINGP,
                               swig_scm_to_string,swig_make_string,
                               bool,SCHEME_BOOLP,
                               SCHEME_TRUEP,swig_make_boolean);
    specialize_std_pair_on_both(std::string,SCHEME_STRINGP,
                               swig_scm_to_string,swig_make_string,
                               int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(std::string,SCHEME_STRINGP,
                               swig_scm_to_string,swig_make_string,
                               short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(std::string,SCHEME_STRINGP,
                               swig_scm_to_string,swig_make_string,
                               long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(std::string,SCHEME_STRINGP,
                               swig_scm_to_string,swig_make_string,
                               unsigned int,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(std::string,SCHEME_STRINGP,
                               swig_scm_to_string,swig_make_string,
                               unsigned short,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(std::string,SCHEME_STRINGP,
                               swig_scm_to_string,swig_make_string,
                               unsigned long,SCHEME_INTP,
                               SCHEME_INT_VAL,scheme_make_integer_value);
    specialize_std_pair_on_both(std::string,SCHEME_STRINGP,
                               swig_scm_to_string,swig_make_string,
                               double,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_both(std::string,SCHEME_STRINGP,
                               swig_scm_to_string,swig_make_string,
                               float,SCHEME_REALP,
                               scheme_real_to_double,scheme_make_double);
    specialize_std_pair_on_both(std::string,SCHEME_STRINGP,
                               swig_scm_to_string,swig_make_string,
                               std::string,SCHEME_STRINGP,
                               swig_scm_to_string,swig_make_string);
}
