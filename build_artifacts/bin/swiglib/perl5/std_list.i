/* -----------------------------------------------------------------------------
 * std_list.i
 *
 * SWIG typemaps for std::list types
 * ----------------------------------------------------------------------------- */

%include <std_common.i>
%include <exception.i>

// containers


// ------------------------------------------------------------------------
// std::list
// 
// The aim of all that follows would be to integrate std::list with 
// Perl as much as possible, namely, to allow the user to pass and 
// be returned Perl arrays.
// const declarations are used to guess the intent of the function being
// exported; therefore, the following rationale is applied:
// 
//   -- f(std::list<T>), f(const std::list<T>&), f(const std::list<T>*):
//      the parameter being read-only, either a Perl sequence or a
//      previously wrapped std::list<T> can be passed.
//   -- f(std::list<T>&), f(std::list<T>*):
//      the parameter must be modified; therefore, only a wrapped std::list
//      can be passed.
//   -- std::list<T> f():
//      the list is returned by copy; therefore, a Perl sequence of T:s 
//      is returned which is most easily used in other Perl functions
//   -- std::list<T>& f(), std::list<T>* f(), const std::list<T>& f(),
//      const std::list<T>* f():
//      the list is returned by reference; therefore, a wrapped std::list
//      is returned
// ------------------------------------------------------------------------

%{
#include <list>
%}
%fragment("<algorithm>");
%fragment("<stdexcept>");

// exported class

namespace std {
    
    template<class T> class list {
        %typemap(in) list<T> (std::list<T>* v) {
            if (SWIG_ConvertPtr($input,(void **) &v, 
                                $&1_descriptor,1) != -1) {
                $1 = *v;
            } else if (SvROK($input)) {
                AV *av = (AV *)SvRV($input);
                if (SvTYPE(av) != SVt_PVAV)
                    SWIG_croak("Type error in argument $argnum of $symname. "
                               "Expected an array of " #T);
                SV **tv;
                I32 len = av_len(av) + 1;
                T* obj;
                for (int i=0; i<len; i++) {
                    tv = av_fetch(av, i, 0);
                    if (SWIG_ConvertPtr(*tv, (void **)&obj, 
                                        $descriptor(T *),0) != -1) {
                        $1.push_back(*obj);
                    } else {
                        SWIG_croak("Type error in argument $argnum of "
                                   "$symname. "
                                   "Expected an array of " #T);
                    }
                }
            } else {
                SWIG_croak("Type error in argument $argnum of $symname. "
                           "Expected an array of " #T);
            }
        }
        %typemap(in) const list<T>& (std::list<T> temp,
                                       std::list<T>* v),
                     const list<T>* (std::list<T> temp,
                                       std::list<T>* v) {
            if (SWIG_ConvertPtr($input,(void **) &v, 
                                $1_descriptor,1) != -1) {
                $1 = v;
            } else if (SvROK($input)) {
                AV *av = (AV *)SvRV($input);
                if (SvTYPE(av) != SVt_PVAV)
                    SWIG_croak("Type error in argument $argnum of $symname. "
                               "Expected an array of " #T);
                SV **tv;
                I32 len = av_len(av) + 1;
                T* obj;
                for (int i=0; i<len; i++) {
                    tv = av_fetch(av, i, 0);
                    if (SWIG_ConvertPtr(*tv, (void **)&obj, 
                                        $descriptor(T *),0) != -1) {
                        temp.push_back(*obj);
                    } else {
                        SWIG_croak("Type error in argument $argnum of "
                                   "$symname. "
                                   "Expected an array of " #T);
                    }
                }
                $1 = &temp;
            } else {
                SWIG_croak("Type error in argument $argnum of $symname. "
                           "Expected an array of " #T);
            }
        }
        %typemap(out) list<T> {
	    std::list< T >::const_iterator i;
            unsigned int j;
            int len = $1.size();
            SV **svs = new SV*[len];
            for (i=$1.begin(), j=0; i!=$1.end(); i++, j++) {
                T* ptr = new T(*i);
                svs[j] = sv_newmortal();
                SWIG_MakePtr(svs[j], (void*) ptr, 
                             $descriptor(T *), $shadow|$owner);
            }
            AV *myav = av_make(len, svs);
            delete[] svs;
            $result = newRV_noinc((SV*) myav);
            sv_2mortal($result);
            argvi++;
        }
        %typecheck(SWIG_TYPECHECK_LIST) list<T> {
            {
                /* wrapped list? */
                std::list< T >* v;
                if (SWIG_ConvertPtr($input,(void **) &v, 
                                    $1_&descriptor,0) != -1) {
                    $1 = 1;
                } else if (SvROK($input)) {
                    /* native sequence? */
                    AV *av = (AV *)SvRV($input);
                    if (SvTYPE(av) == SVt_PVAV) {
                        SV **tv;
                        I32 len = av_len(av) + 1;
                        if (len == 0) {
                            /* an empty sequence can be of any type */
                            $1 = 1;
                        } else {
                            /* check the first element only */
                            T* obj;
                            tv = av_fetch(av, 0, 0);
                            if (SWIG_ConvertPtr(*tv, (void **)&obj, 
                                                $descriptor(T *),0) != -1)
                                $1 = 1;
                            else
                                $1 = 0;
                        }
                    }
                } else {
                    $1 = 0;
                }
            }
        }
        %typecheck(SWIG_TYPECHECK_LIST) const list<T>&,
                                          const list<T>* {
            {
                /* wrapped list? */
                std::list< T >* v;
                if (SWIG_ConvertPtr($input,(void **) &v, 
                                    $1_descriptor,0) != -1) {
                    $1 = 1;
                } else if (SvROK($input)) {
                    /* native sequence? */
                    AV *av = (AV *)SvRV($input);
                    if (SvTYPE(av) == SVt_PVAV) {
                        SV **tv;
                        I32 len = av_len(av) + 1;
                        if (len == 0) {
                            /* an empty sequence can be of any type */
                            $1 = 1;
                        } else {
                            /* check the first element only */
                            T* obj;
                            tv = av_fetch(av, 0, 0);
                            if (SWIG_ConvertPtr(*tv, (void **)&obj, 
                                                $descriptor(T *),0) != -1)
                                $1 = 1;
                            else
                                $1 = 0;
                        }
                    }
                } else {
                    $1 = 0;
                }
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

        list();
        list(const list& other);

        unsigned int size() const;
        bool empty() const;
        void clear();
        %rename(push) push_back;
        void push_back(const T& x);
    };


    // specializations for built-ins

    %define specialize_std_list(T,CHECK_T,TO_T,FROM_T)
    template<> class list<T> {
        %typemap(in) list<T> (std::list<T>* v) {
            if (SWIG_ConvertPtr($input,(void **) &v, 
                                $&1_descriptor,1) != -1){
                $1 = *v;
            } else if (SvROK($input)) {
                AV *av = (AV *)SvRV($input);
                if (SvTYPE(av) != SVt_PVAV)
                    SWIG_croak("Type error in argument $argnum of $symname. "
                               "Expected an array of " #T);
                SV **tv;
                I32 len = av_len(av) + 1;
                for (int i=0; i<len; i++) {
                    tv = av_fetch(av, i, 0);
                    if (CHECK_T(*tv)) {
                        $1.push_back(TO_T(*tv));
                    } else {
                        SWIG_croak("Type error in argument $argnum of "
                                   "$symname. "
                                   "Expected an array of " #T);
                    }
                }
            } else {
                SWIG_croak("Type error in argument $argnum of $symname. "
                           "Expected an array of " #T);
            }
        }
        %typemap(in) const list<T>& (std::list<T> temp,
                                       std::list<T>* v),
                     const list<T>* (std::list<T> temp,
                                       std::list<T>* v) {
            if (SWIG_ConvertPtr($input,(void **) &v, 
                                $1_descriptor,1) != -1) {
                $1 = v;
            } else if (SvROK($input)) {
                AV *av = (AV *)SvRV($input);
                if (SvTYPE(av) != SVt_PVAV)
                    SWIG_croak("Type error in argument $argnum of $symname. "
                               "Expected an array of " #T);
                SV **tv;
                I32 len = av_len(av) + 1;
                T* obj;
                for (int i=0; i<len; i++) {
                    tv = av_fetch(av, i, 0);
                    if (CHECK_T(*tv)) {
                        temp.push_back(TO_T(*tv));
                    } else {
                        SWIG_croak("Type error in argument $argnum of "
                                   "$symname. "
                                   "Expected an array of " #T);
                    }
                }
                $1 = &temp;
            } else {
                SWIG_croak("Type error in argument $argnum of $symname. "
                           "Expected an array of " #T);
            }
        }
        %typemap(out) list<T> {
	    std::list< T >::const_iterator i;
            unsigned int j;
            int len = $1.size();
            SV **svs = new SV*[len];
            for (i=$1.begin(), j=0; i!=$1.end(); i++, j++) {
                svs[j] = sv_newmortal();
                FROM_T(svs[j], *i);
            }
            AV *myav = av_make(len, svs);
            delete[] svs;
            $result = newRV_noinc((SV*) myav);
            sv_2mortal($result);
            argvi++;
        }
        %typecheck(SWIG_TYPECHECK_LIST) list<T> {
            {
                /* wrapped list? */
                std::list< T >* v;
                if (SWIG_ConvertPtr($input,(void **) &v, 
                                    $1_&descriptor,0) != -1) {
                    $1 = 1;
                } else if (SvROK($input)) {
                    /* native sequence? */
                    AV *av = (AV *)SvRV($input);
                    if (SvTYPE(av) == SVt_PVAV) {
                        SV **tv;
                        I32 len = av_len(av) + 1;
                        if (len == 0) {
                            /* an empty sequence can be of any type */
                            $1 = 1;
                        } else {
                            /* check the first element only */
                            tv = av_fetch(av, 0, 0);
                            if (CHECK_T(*tv))
                                $1 = 1;
                            else
                                $1 = 0;
                        }
                    }
                } else {
                    $1 = 0;
                }
            }
        }
        %typecheck(SWIG_TYPECHECK_LIST) const list<T>&,
                                          const list<T>* {
            {
                /* wrapped list? */
                std::list< T >* v;
                if (SWIG_ConvertPtr($input,(void **) &v, 
                                    $1_descriptor,0) != -1) {
                    $1 = 1;
                } else if (SvROK($input)) {
                    /* native sequence? */
                    AV *av = (AV *)SvRV($input);
                    if (SvTYPE(av) == SVt_PVAV) {
                        SV **tv;
                        I32 len = av_len(av) + 1;
                        if (len == 0) {
                            /* an empty sequence can be of any type */
                            $1 = 1;
                        } else {
                            /* check the first element only */
                            tv = av_fetch(av, 0, 0);
                            if (CHECK_T(*tv))
                                $1 = 1;
                            else
                                $1 = 0;
                        }
                    }
                } else {
                    $1 = 0;
                }
            }
        }
      public:
        typedef size_t size_type;
        typedef T value_type;
        typedef const value_type& const_reference;

        list();
        list(const list& other);

        unsigned int size() const;
        bool empty() const;
        void clear();
        %rename(push) push_back;
        void push_back(T x);
    };
    %enddef

    specialize_std_list(bool,SvIOK,SvIVX,sv_setiv);
    specialize_std_list(char,SvIOK,SvIVX,sv_setiv);
    specialize_std_list(int,SvIOK,SvIVX,sv_setiv);
    specialize_std_list(short,SvIOK,SvIVX,sv_setiv);
    specialize_std_list(long,SvIOK,SvIVX,sv_setiv);
    specialize_std_list(unsigned char,SvIOK,SvIVX,sv_setiv);
    specialize_std_list(unsigned int,SvIOK,SvIVX,sv_setiv);
    specialize_std_list(unsigned short,SvIOK,SvIVX,sv_setiv);
    specialize_std_list(unsigned long,SvIOK,SvIVX,sv_setiv);
    specialize_std_list(float,SvNIOK,SwigSvToNumber,sv_setnv);
    specialize_std_list(double,SvNIOK,SwigSvToNumber,sv_setnv);
    specialize_std_list(std::string,SvPOK,SvPVX,SwigSvFromString);

}

