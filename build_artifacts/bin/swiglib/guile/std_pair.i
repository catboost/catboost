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
        %typemap(in) pair<T,U> %{
            if (scm_is_pair($input)) {
                T* x;
                U* y;
                SCM first, second;
                first = SCM_CAR($input);
                second = SCM_CDR($input);
                x = (T*) SWIG_MustGetPtr(first,$descriptor(T *),$argnum, 0);
                y = (U*) SWIG_MustGetPtr(second,$descriptor(U *),$argnum, 0);
                $1 = std::make_pair(*x,*y);
            } else {
                $1 = *(($&1_type)
                       SWIG_MustGetPtr($input,$&1_descriptor,$argnum, 0));
            }
        %}
        %typemap(in) const pair<T,U>& (std::pair<T,U> *temp = 0),
                     const pair<T,U>* (std::pair<T,U> *temp = 0) %{
            if (scm_is_pair($input)) {
                T* x;
                U* y;
                SCM first, second;
                first = SCM_CAR($input);
                second = SCM_CDR($input);
                x = (T*) SWIG_MustGetPtr(first,$descriptor(T *),$argnum, 0);
                y = (U*) SWIG_MustGetPtr(second,$descriptor(U *),$argnum, 0);
                temp = new std::pair< T, U >(*x,*y);
                $1 = temp;
            } else {
                $1 = ($1_ltype)
                    SWIG_MustGetPtr($input,$1_descriptor,$argnum, 0);
            }
        %}
        %typemap(freearg) const pair<T,U>&, const pair<T,U>* %{ delete temp$argnum; %}
        %typemap(out) pair<T,U> {
            T* x = new T($1.first);
            U* y = new U($1.second);
            SCM first = SWIG_NewPointerObj(x,$descriptor(T *), 1);
            SCM second = SWIG_NewPointerObj(y,$descriptor(U *), 1);
            $result = scm_cons(first,second);
        }
        %typecheck(SWIG_TYPECHECK_PAIR) pair<T,U> {
            /* native pair? */
            if (scm_is_pair($input)) {
                T* x;
                U* y;
                SCM first = SCM_CAR($input);
                SCM second = SCM_CDR($input);
                if (SWIG_ConvertPtr(first,(void**) &x,
                                    $descriptor(T *), 0) == 0 &&
                    SWIG_ConvertPtr(second,(void**) &y,
                                    $descriptor(U *), 0) == 0) {
                    $1 = 1;
                } else {
                    $1 = 0;
                }
            } else {
                /* wrapped pair? */
                std::pair< T, U >* m;
                if (SWIG_ConvertPtr($input,(void **) &m,
                                    $&1_descriptor, 0) == 0)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
        %typecheck(SWIG_TYPECHECK_PAIR) const pair<T,U>&,
                                        const pair<T,U>* {
            /* native pair? */
            if (scm_is_pair($input)) {
                T* x;
                U* y;
                SCM first = SCM_CAR($input);
                SCM second = SCM_CDR($input);
                if (SWIG_ConvertPtr(first,(void**) &x,
                                    $descriptor(T *), 0) == 0 &&
                    SWIG_ConvertPtr(second,(void**) &y,
                                    $descriptor(U *), 0) == 0) {
                    $1 = 1;
                } else {
                    $1 = 0;
                }
            } else {
                /* wrapped pair? */
                std::pair< T, U >* m;
                if (SWIG_ConvertPtr($input,(void **) &m,
                                    $1_descriptor, 0) == 0)
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
        %typemap(in) pair<T,U> %{
            if (scm_is_pair($input)) {
                U* y;
                SCM first, second;
                first = SCM_CAR($input);
                second = SCM_CDR($input);
                if (!CHECK(first))
                    SWIG_exception(SWIG_TypeError,
                                   "pair<" #T "," #U "> expected");
                y = (U*) SWIG_MustGetPtr(second,$descriptor(U *),$argnum, 0);
                $1 = std::make_pair(CONVERT_FROM(first),*y);
            } else {
                $1 = *(($&1_type)
                       SWIG_MustGetPtr($input,$&1_descriptor,$argnum, 0));
            }
        %}
        %typemap(in) const pair<T,U>& (std::pair<T,U> *temp = 0),
                     const pair<T,U>* (std::pair<T,U> *temp = 0) %{
            if (scm_is_pair($input)) {
                U* y;
                SCM first, second;
                first = SCM_CAR($input);
                second = SCM_CDR($input);
                if (!CHECK(first))
                    SWIG_exception(SWIG_TypeError,
                                   "pair<" #T "," #U "> expected");
                y = (U*) SWIG_MustGetPtr(second,$descriptor(U *),$argnum, 0);
                temp = new std::pair< T, U >(CONVERT_FROM(first),*y);
                $1 = temp;
            } else {
                $1 = ($1_ltype)
                    SWIG_MustGetPtr($input,$1_descriptor,$argnum, 0);
            }
        %}
        %typemap(freearg) const pair<T,U>&, const pair<T,U>* %{ delete temp$argnum; %}
        %typemap(out) pair<T,U> {
            U* y = new U($1.second);
            SCM second = SWIG_NewPointerObj(y,$descriptor(U *), 1);
            $result = scm_cons(CONVERT_TO($1.first),second);
        }
        %typecheck(SWIG_TYPECHECK_PAIR) pair<T,U> {
            /* native pair? */
            if (scm_is_pair($input)) {
                U* y;
                SCM first = SCM_CAR($input);
                SCM second = SCM_CDR($input);
                if (CHECK(first) &&
                    SWIG_ConvertPtr(second,(void**) &y,
                                    $descriptor(U *), 0) == 0) {
                    $1 = 1;
                } else {
                    $1 = 0;
                }
            } else {
                /* wrapped pair? */
                std::pair< T, U >* m;
                if (SWIG_ConvertPtr($input,(void **) &m,
                                    $&1_descriptor, 0) == 0)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
        %typecheck(SWIG_TYPECHECK_PAIR) const pair<T,U>&,
                                        const pair<T,U>* {
            /* native pair? */
            if (scm_is_pair($input)) {
                U* y;
                SCM first = SCM_CAR($input);
                SCM second = SCM_CDR($input);
                if (CHECK(first) &&
                    SWIG_ConvertPtr(second,(void**) &y,
                                    $descriptor(U *), 0) == 0) {
                    $1 = 1;
                } else {
                    $1 = 0;
                }
            } else {
                /* wrapped pair? */
                std::pair< T, U >* m;
                if (SWIG_ConvertPtr($input,(void **) &m,
                                    $1_descriptor, 0) == 0)
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
        %typemap(in) pair<T,U> %{
            if (scm_is_pair($input)) {
                T* x;
                SCM first, second;
                first = SCM_CAR($input);
                second = SCM_CDR($input);
                x = (T*) SWIG_MustGetPtr(first,$descriptor(T *),$argnum, 0);
                if (!CHECK(second))
                    SWIG_exception(SWIG_TypeError,
                                   "pair<" #T "," #U "> expected");
                $1 = std::make_pair(*x,CONVERT_FROM(second));
            } else {
                $1 = *(($&1_type)
                       SWIG_MustGetPtr($input,$&1_descriptor,$argnum, 0));
            }
        %}
        %typemap(in) const pair<T,U>& (std::pair<T,U> *temp = 0),
                     const pair<T,U>* (std::pair<T,U> *temp = 0) %{
            if (scm_is_pair($input)) {
                T* x;
                SCM first, second;
                first = SCM_CAR($input);
                second = SCM_CDR($input);
                x = (T*) SWIG_MustGetPtr(first,$descriptor(T *),$argnum, 0);
                if (!CHECK(second))
                    SWIG_exception(SWIG_TypeError,
                                   "pair<" #T "," #U "> expected");
                temp = new std::pair< T, U >(*x,CONVERT_FROM(second));
                $1 = temp;
            } else {
                $1 = ($1_ltype)
                    SWIG_MustGetPtr($input,$1_descriptor,$argnum, 0);
            }
        %}
        %typemap(freearg) const pair<T,U>&, const pair<T,U>* %{ delete temp$argnum; %}
        %typemap(out) pair<T,U> {
            T* x = new T($1.first);
            SCM first = SWIG_NewPointerObj(x,$descriptor(T *), 1);
            $result = scm_cons(first,CONVERT_TO($1.second));
        }
        %typecheck(SWIG_TYPECHECK_PAIR) pair<T,U> {
            /* native pair? */
            if (scm_is_pair($input)) {
                T* x;
                SCM first = SCM_CAR($input);
                SCM second = SCM_CDR($input);
                if (SWIG_ConvertPtr(first,(void**) &x,
                                    $descriptor(T *), 0) == 0 &&
                    CHECK(second)) {
                    $1 = 1;
                } else {
                    $1 = 0;
                }
            } else {
                /* wrapped pair? */
                std::pair< T, U >* m;
                if (SWIG_ConvertPtr($input,(void **) &m,
                                    $&1_descriptor, 0) == 0)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
        %typecheck(SWIG_TYPECHECK_PAIR) const pair<T,U>&,
                                        const pair<T,U>* {
            /* native pair? */
            if (scm_is_pair($input)) {
                T* x;
                SCM first = SCM_CAR($input);
                SCM second = SCM_CDR($input);
                if (SWIG_ConvertPtr(first,(void**) &x,
                                    $descriptor(T *), 0) == 0 &&
                    CHECK(second)) {
                    $1 = 1;
                } else {
                    $1 = 0;
                }
            } else {
                /* wrapped pair? */
                std::pair< T, U >* m;
                if (SWIG_ConvertPtr($input,(void **) &m,
                                    $1_descriptor, 0) == 0)
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
        %typemap(in) pair<T,U> %{
            if (scm_is_pair($input)) {
                SCM first, second;
                first = SCM_CAR($input);
                second = SCM_CDR($input);
                if (!CHECK_T(first) || !CHECK_U(second))
                    SWIG_exception(SWIG_TypeError,
                                   "pair<" #T "," #U "> expected");
                $1 = std::make_pair(CONVERT_T_FROM(first),
                                    CONVERT_U_FROM(second));
            } else {
                $1 = *(($&1_type)
                       SWIG_MustGetPtr($input,$&1_descriptor,$argnum, 0));
            }
        %}
        %typemap(in) const pair<T,U>& (std::pair<T,U> *temp = 0),
                     const pair<T,U>* (std::pair<T,U> *temp = 0) %{
            if (scm_is_pair($input)) {
                SCM first, second;
                first = SCM_CAR($input);
                second = SCM_CDR($input);
                if (!CHECK_T(first) || !CHECK_U(second))
                    SWIG_exception(SWIG_TypeError,
                                   "pair<" #T "," #U "> expected");
                temp = new std::pair< T, U >(CONVERT_T_FROM(first), CONVERT_U_FROM(second));
                $1 = temp;
            } else {
                $1 = ($1_ltype)
                    SWIG_MustGetPtr($input,$1_descriptor,$argnum, 0);
            }
        %}
        %typemap(freearg) const pair<T,U>&, const pair<T,U>* %{ delete temp$argnum; %}
        %typemap(out) pair<T,U> {
            $result = scm_cons(CONVERT_T_TO($1.first),
                              CONVERT_U_TO($1.second));
        }
        %typecheck(SWIG_TYPECHECK_PAIR) pair<T,U> {
            /* native pair? */
            if (scm_is_pair($input)) {
                SCM first = SCM_CAR($input);
                SCM second = SCM_CDR($input);
                if (CHECK_T(first) && CHECK_U(second)) {
                    $1 = 1;
                } else {
                    $1 = 0;
                }
            } else {
                /* wrapped pair? */
                std::pair< T, U >* m;
                if (SWIG_ConvertPtr($input,(void **) &m,
                                    $&1_descriptor, 0) == 0)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
        %typecheck(SWIG_TYPECHECK_PAIR) const pair<T,U>&,
                                        const pair<T,U>* {
            /* native pair? */
            if (scm_is_pair($input)) {
                SCM first = SCM_CAR($input);
                SCM second = SCM_CDR($input);
                if (CHECK_T(first) && CHECK_U(second)) {
                    $1 = 1;
                } else {
                    $1 = 0;
                }
            } else {
                /* wrapped pair? */
                std::pair< T, U >* m;
                if (SWIG_ConvertPtr($input,(void **) &m,
                                    $1_descriptor, 0) == 0)
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


    specialize_std_pair_on_first(bool,scm_is_bool,
                              scm_is_true,SWIG_bool2scm);
    specialize_std_pair_on_first(int,scm_is_number,
                              scm_to_long,scm_from_long);
    specialize_std_pair_on_first(short,scm_is_number,
                              scm_to_long,scm_from_long);
    specialize_std_pair_on_first(long,scm_is_number,
                              scm_to_long,scm_from_long);
    specialize_std_pair_on_first(unsigned int,scm_is_number,
                              scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_first(unsigned short,scm_is_number,
                              scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_first(unsigned long,scm_is_number,
                              scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_first(double,scm_is_number,
                              scm_to_double,scm_from_double);
    specialize_std_pair_on_first(float,scm_is_number,
                              scm_to_double,scm_from_double);
    specialize_std_pair_on_first(std::string,scm_is_string,
                              SWIG_scm2string,SWIG_string2scm);

    specialize_std_pair_on_second(bool,scm_is_bool,
                                scm_is_true,SWIG_bool2scm);
    specialize_std_pair_on_second(int,scm_is_number,
                                scm_to_long,scm_from_long);
    specialize_std_pair_on_second(short,scm_is_number,
                                scm_to_long,scm_from_long);
    specialize_std_pair_on_second(long,scm_is_number,
                                scm_to_long,scm_from_long);
    specialize_std_pair_on_second(unsigned int,scm_is_number,
                                scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_second(unsigned short,scm_is_number,
                                scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_second(unsigned long,scm_is_number,
                                scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_second(double,scm_is_number,
                                scm_to_double,scm_from_double);
    specialize_std_pair_on_second(float,scm_is_number,
                                scm_to_double,scm_from_double);
    specialize_std_pair_on_second(std::string,scm_is_string,
                                SWIG_scm2string,SWIG_string2scm);

    specialize_std_pair_on_both(bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm,
                               bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm);
    specialize_std_pair_on_both(bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm,
                               int,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm,
                               short,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm,
                               long,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm,
                               unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm,
                               unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm,
                               unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm,
                               double,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_pair_on_both(bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm,
                               float,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_pair_on_both(bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm,
                               std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm);
    specialize_std_pair_on_both(int,scm_is_number,
                               scm_to_long,scm_from_long,
                               bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm);
    specialize_std_pair_on_both(int,scm_is_number,
                               scm_to_long,scm_from_long,
                               int,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(int,scm_is_number,
                               scm_to_long,scm_from_long,
                               short,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(int,scm_is_number,
                               scm_to_long,scm_from_long,
                               long,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(int,scm_is_number,
                               scm_to_long,scm_from_long,
                               unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(int,scm_is_number,
                               scm_to_long,scm_from_long,
                               unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(int,scm_is_number,
                               scm_to_long,scm_from_long,
                               unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(int,scm_is_number,
                               scm_to_long,scm_from_long,
                               double,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_pair_on_both(int,scm_is_number,
                               scm_to_long,scm_from_long,
                               float,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_pair_on_both(int,scm_is_number,
                               scm_to_long,scm_from_long,
                               std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm);
    specialize_std_pair_on_both(short,scm_is_number,
                               scm_to_long,scm_from_long,
                               bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm);
    specialize_std_pair_on_both(short,scm_is_number,
                               scm_to_long,scm_from_long,
                               int,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(short,scm_is_number,
                               scm_to_long,scm_from_long,
                               short,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(short,scm_is_number,
                               scm_to_long,scm_from_long,
                               long,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(short,scm_is_number,
                               scm_to_long,scm_from_long,
                               unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(short,scm_is_number,
                               scm_to_long,scm_from_long,
                               unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(short,scm_is_number,
                               scm_to_long,scm_from_long,
                               unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(short,scm_is_number,
                               scm_to_long,scm_from_long,
                               double,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_pair_on_both(short,scm_is_number,
                               scm_to_long,scm_from_long,
                               float,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_pair_on_both(short,scm_is_number,
                               scm_to_long,scm_from_long,
                               std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm);
    specialize_std_pair_on_both(long,scm_is_number,
                               scm_to_long,scm_from_long,
                               bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm);
    specialize_std_pair_on_both(long,scm_is_number,
                               scm_to_long,scm_from_long,
                               int,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(long,scm_is_number,
                               scm_to_long,scm_from_long,
                               short,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(long,scm_is_number,
                               scm_to_long,scm_from_long,
                               long,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(long,scm_is_number,
                               scm_to_long,scm_from_long,
                               unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(long,scm_is_number,
                               scm_to_long,scm_from_long,
                               unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(long,scm_is_number,
                               scm_to_long,scm_from_long,
                               unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(long,scm_is_number,
                               scm_to_long,scm_from_long,
                               double,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_pair_on_both(long,scm_is_number,
                               scm_to_long,scm_from_long,
                               float,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_pair_on_both(long,scm_is_number,
                               scm_to_long,scm_from_long,
                               std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm);
    specialize_std_pair_on_both(unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm);
    specialize_std_pair_on_both(unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               int,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               short,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               long,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               double,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_pair_on_both(unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               float,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_pair_on_both(unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm);
    specialize_std_pair_on_both(unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm);
    specialize_std_pair_on_both(unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               int,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               short,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               long,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               double,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_pair_on_both(unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               float,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_pair_on_both(unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm);
    specialize_std_pair_on_both(unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm);
    specialize_std_pair_on_both(unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               int,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               short,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               long,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               double,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_pair_on_both(unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               float,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_pair_on_both(unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm);
    specialize_std_pair_on_both(double,scm_is_number,
                               scm_to_double,scm_from_double,
                               bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm);
    specialize_std_pair_on_both(double,scm_is_number,
                               scm_to_double,scm_from_double,
                               int,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(double,scm_is_number,
                               scm_to_double,scm_from_double,
                               short,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(double,scm_is_number,
                               scm_to_double,scm_from_double,
                               long,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(double,scm_is_number,
                               scm_to_double,scm_from_double,
                               unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(double,scm_is_number,
                               scm_to_double,scm_from_double,
                               unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(double,scm_is_number,
                               scm_to_double,scm_from_double,
                               unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(double,scm_is_number,
                               scm_to_double,scm_from_double,
                               double,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_pair_on_both(double,scm_is_number,
                               scm_to_double,scm_from_double,
                               float,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_pair_on_both(double,scm_is_number,
                               scm_to_double,scm_from_double,
                               std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm);
    specialize_std_pair_on_both(float,scm_is_number,
                               scm_to_double,scm_from_double,
                               bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm);
    specialize_std_pair_on_both(float,scm_is_number,
                               scm_to_double,scm_from_double,
                               int,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(float,scm_is_number,
                               scm_to_double,scm_from_double,
                               short,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(float,scm_is_number,
                               scm_to_double,scm_from_double,
                               long,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(float,scm_is_number,
                               scm_to_double,scm_from_double,
                               unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(float,scm_is_number,
                               scm_to_double,scm_from_double,
                               unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(float,scm_is_number,
                               scm_to_double,scm_from_double,
                               unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(float,scm_is_number,
                               scm_to_double,scm_from_double,
                               double,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_pair_on_both(float,scm_is_number,
                               scm_to_double,scm_from_double,
                               float,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_pair_on_both(float,scm_is_number,
                               scm_to_double,scm_from_double,
                               std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm);
    specialize_std_pair_on_both(std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm,
                               bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm);
    specialize_std_pair_on_both(std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm,
                               int,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm,
                               short,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm,
                               long,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_pair_on_both(std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm,
                               unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm,
                               unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm,
                               unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_pair_on_both(std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm,
                               double,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_pair_on_both(std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm,
                               float,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_pair_on_both(std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm,
                               std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm);
}
