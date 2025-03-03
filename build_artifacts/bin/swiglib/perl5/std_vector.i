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
// Perl as much as possible, namely, to allow the user to pass and 
// be returned Perl arrays.
// const declarations are used to guess the intent of the function being
// exported; therefore, the following rationale is applied:
// 
//   -- f(std::vector<T>), f(const std::vector<T>&), f(const std::vector<T>*):
//      the parameter being read-only, either a Perl sequence or a
//      previously wrapped std::vector<T> can be passed.
//   -- f(std::vector<T>&), f(std::vector<T>*):
//      the parameter must be modified; therefore, only a wrapped std::vector
//      can be passed.
//   -- std::vector<T> f():
//      the vector is returned by copy; therefore, a Perl sequence of T:s 
//      is returned which is most easily used in other Perl functions
//   -- std::vector<T>& f(), std::vector<T>* f(), const std::vector<T>& f(),
//      const std::vector<T>* f():
//      the vector is returned by reference; therefore, a wrapped std::vector
//      is returned
// ------------------------------------------------------------------------

%{
#include <vector>
%}
%fragment("<algorithm>");
%fragment("<stdexcept>");

// exported class

namespace std {
    
    template<class T> class vector {
        %typemap(in) vector<T> (std::vector<T>* v) {
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
        %typemap(in) const vector<T>& (std::vector<T> temp,
                                       std::vector<T>* v),
                     const vector<T>* (std::vector<T> temp,
                                       std::vector<T>* v) {
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
        %typemap(out) vector<T> {
            size_t len = $1.size();
            SV **svs = new SV*[len];
            for (size_t i=0; i<len; i++) {
                T* ptr = new T($1[i]);
                svs[i] = sv_newmortal();
                SWIG_MakePtr(svs[i], (void*) ptr, 
                             $descriptor(T *), $shadow|$owner);
            }
            AV *myav = av_make(len, svs);
            delete[] svs;
            $result = newRV_noinc((SV*) myav);
            sv_2mortal($result);
            argvi++;
        }
        %typecheck(SWIG_TYPECHECK_VECTOR) vector<T> {
            {
                /* wrapped vector? */
                std::vector< T >* v;
                if (SWIG_ConvertPtr($input,(void **) &v, 
                                    $&1_descriptor,0) != -1) {
                    $1 = 1;
                } else if (SvROK($input)) {
                    /* native sequence? */
                    AV *av = (AV *)SvRV($input);
                    if (SvTYPE(av) == SVt_PVAV) {
                        I32 len = av_len(av) + 1;
                        if (len == 0) {
                            /* an empty sequence can be of any type */
                            $1 = 1;
                        } else {
                            /* check the first element only */
                            T* obj;
                            SV **tv = av_fetch(av, 0, 0);
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
        %typecheck(SWIG_TYPECHECK_VECTOR) const vector<T>&,
                                          const vector<T>* {
            {
                /* wrapped vector? */
                std::vector< T >* v;
                if (SWIG_ConvertPtr($input,(void **) &v, 
                                    $1_descriptor,0) != -1) {
                    $1 = 1;
                } else if (SvROK($input)) {
                    /* native sequence? */
                    AV *av = (AV *)SvRV($input);
                    if (SvTYPE(av) == SVt_PVAV) {
                        I32 len = av_len(av) + 1;
                        if (len == 0) {
                            /* an empty sequence can be of any type */
                            $1 = 1;
                        } else {
                            /* check the first element only */
                            T* obj;
                            SV **tv = av_fetch(av, 0, 0);
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

        vector(unsigned int size = 0);
        vector(unsigned int size, const T& value);
        vector(const vector& other);

        unsigned int size() const;
        bool empty() const;
        void clear();
        %rename(push) push_back;
        void push_back(const T& x);
        %extend {
            T pop() throw (std::out_of_range) {
                if (self->size() == 0)
                    throw std::out_of_range("pop from empty vector");
                T x = self->back();
                self->pop_back();
                return x;
            }
            T& get(int i) throw (std::out_of_range) {
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

    // specializations for pointers
    template<class T> class vector<T*> {
        %typemap(in) vector<T*> (std::vector<T*>* v) {
	    int res = SWIG_ConvertPtr($input,(void **) &v, $&1_descriptor,0);
            if (SWIG_IsOK(res)){
                $1 = *v;
            } else if (SvROK($input)) {
                AV *av = (AV *)SvRV($input);
                if (SvTYPE(av) != SVt_PVAV)
                    SWIG_croak("Type error in argument $argnum of $symname. "
                               "Expected an array of " #T);
                I32 len = av_len(av) + 1;
                for (int i=0; i<len; i++) {
		    void *v;
		    SV **tv = av_fetch(av, i, 0);
		    int res = SWIG_ConvertPtr(*tv, &v, $descriptor(T *),0);
                    if (SWIG_IsOK(res)) {
                        $1.push_back(%static_cast(v, T *));
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
        %typemap(in) const vector<T *>& (std::vector<T *> temp,std::vector<T *>* v),
                     const vector<T *>* (std::vector<T *> temp,std::vector<T *>* v) {
	    int res = SWIG_ConvertPtr($input,(void **) &v, $1_descriptor,0);
            if (SWIG_IsOK(res)) {
                $1 = v;
            } else if (SvROK($input)) {
                AV *av = (AV *)SvRV($input);
                if (SvTYPE(av) != SVt_PVAV)
                    SWIG_croak("Type error in argument $argnum of $symname. "
                               "Expected an array of " #T);
                I32 len = av_len(av) + 1;
                for (int i=0; i<len; i++) {
		    void *v;
		    SV **tv = av_fetch(av, i, 0);
		    int res = SWIG_ConvertPtr(*tv, &v, $descriptor(T *),0);
                    if (SWIG_IsOK(res)) {
                        temp.push_back(%static_cast(v, T *));
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
        %typemap(out) vector<T *> {
            size_t len = $1.size();
            SV **svs = new SV*[len];
            for (size_t i=0; i<len; i++) {
                T *x = (($1_type &)$1)[i];
                svs[i] = sv_newmortal();
		sv_setsv(svs[i], SWIG_NewPointerObj(x, $descriptor(T *), 0));
            }
            AV *myav = av_make(len, svs);
            delete[] svs;
            $result = newRV_noinc((SV*) myav);
            sv_2mortal($result);
            argvi++;
        }
        %typecheck(SWIG_TYPECHECK_VECTOR) vector<T *> {
            {
                /* wrapped vector? */
                std::vector< T *>* v;
		int res = SWIG_ConvertPtr($input,(void **) &v, $&1_descriptor,0);
                if (SWIG_IsOK(res)) {
                    $1 = 1;
                } else if (SvROK($input)) {
                    /* native sequence? */
                    AV *av = (AV *)SvRV($input);
                    if (SvTYPE(av) == SVt_PVAV) {
                        I32 len = av_len(av) + 1;
                        if (len == 0) {
                            /* an empty sequence can be of any type */
                            $1 = 1;
                        } else {
                            /* check the first element only */
			    void *v;
			    SV **tv = av_fetch(av, 0, 0);
			    int res = SWIG_ConvertPtr(*tv, &v, $descriptor(T *),0);
                            if (SWIG_IsOK(res))
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
        %typecheck(SWIG_TYPECHECK_VECTOR) const vector<T *>&,const vector<T *>* {
            {
                /* wrapped vector? */
                std::vector< T *> *v;
		int res = SWIG_ConvertPtr($input,%as_voidptrptr(&v), $1_descriptor,0);
                if (SWIG_IsOK(res)) {
                    $1 = 1;
                } else if (SvROK($input)) {
                    /* native sequence? */
                    AV *av = (AV *)SvRV($input);
                    if (SvTYPE(av) == SVt_PVAV) {
                        I32 len = av_len(av) + 1;
                        if (len == 0) {
                            /* an empty sequence can be of any type */
                            $1 = 1;
                        } else {
                            /* check the first element only */
			    void *v;
			    SV **tv = av_fetch(av, 0, 0);
			    int res = SWIG_ConvertPtr(*tv, &v, $descriptor(T *),0);
                            if (SWIG_IsOK(res))
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
        typedef T* value_type;
        typedef value_type* pointer;
        typedef const value_type* const_pointer;
        typedef value_type& reference;
        typedef const value_type& const_reference;

        vector(unsigned int size = 0);
        vector(unsigned int size, T *value);
        vector(const vector& other);

        unsigned int size() const;
        bool empty() const;
        void clear();
        %rename(push) push_back;
        void push_back(T *x);
        %extend {
            T *pop() throw (std::out_of_range) {
                if (self->size() == 0)
                    throw std::out_of_range("pop from empty vector");
                T *x = self->back();
                self->pop_back();
                return x;
            }
            T *get(int i) throw (std::out_of_range) {
                int size = int(self->size());
                if (i>=0 && i<size)
                    return (*self)[i];
                else
                    throw std::out_of_range("vector index out of range");
            }
            void set(int i, T *x) throw (std::out_of_range) {
                int size = int(self->size());
                if (i>=0 && i<size)
                    (*self)[i] = x;
                else
                    throw std::out_of_range("vector index out of range");
            }
        }
    };


    // specializations for built-ins

    %define specialize_std_vector(T,CHECK_T,TO_T,FROM_T)
    template<> class vector<T> {
        %typemap(in) vector<T> (std::vector<T>* v) {
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
                        $1.push_back((T)TO_T(*tv));
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
        %typemap(in) const vector<T>& (std::vector<T> temp,
                                       std::vector<T>* v),
                     const vector<T>* (std::vector<T> temp,
                                       std::vector<T>* v) {
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
                for (int i=0; i<len; i++) {
                    tv = av_fetch(av, i, 0);
                    if (CHECK_T(*tv)) {
                        temp.push_back((T)TO_T(*tv));
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
        %typemap(out) vector<T> {
            size_t len = $1.size();
            SV **svs = new SV*[len];
            for (size_t i=0; i<len; i++) {
                svs[i] = sv_newmortal();
                FROM_T(svs[i], $1[i]);
            }
            AV *myav = av_make(len, svs);
            delete[] svs;
            $result = newRV_noinc((SV*) myav);
            sv_2mortal($result);
            argvi++;
        }
        %typecheck(SWIG_TYPECHECK_VECTOR) vector<T> {
            {
                /* wrapped vector? */
                std::vector< T >* v;
                if (SWIG_ConvertPtr($input,(void **) &v, 
                                    $&1_descriptor,0) != -1) {
                    $1 = 1;
                } else if (SvROK($input)) {
                    /* native sequence? */
                    AV *av = (AV *)SvRV($input);
                    if (SvTYPE(av) == SVt_PVAV) {
                        I32 len = av_len(av) + 1;
                        if (len == 0) {
                            /* an empty sequence can be of any type */
                            $1 = 1;
                        } else {
                            /* check the first element only */
                            SV **tv = av_fetch(av, 0, 0);
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
        %typecheck(SWIG_TYPECHECK_VECTOR) const vector<T>&,
                                          const vector<T>* {
            {
                /* wrapped vector? */
                std::vector< T >* v;
                if (SWIG_ConvertPtr($input,(void **) &v, 
                                    $1_descriptor,0) != -1) {
                    $1 = 1;
                } else if (SvROK($input)) {
                    /* native sequence? */
                    AV *av = (AV *)SvRV($input);
                    if (SvTYPE(av) == SVt_PVAV) {
                        I32 len = av_len(av) + 1;
                        if (len == 0) {
                            /* an empty sequence can be of any type */
                            $1 = 1;
                        } else {
                            /* check the first element only */
                            SV **tv = av_fetch(av, 0, 0);
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
        typedef ptrdiff_t difference_type;
        typedef T value_type;
        typedef value_type* pointer;
        typedef const value_type* const_pointer;
        typedef value_type& reference;
        typedef const value_type& const_reference;

        vector(unsigned int size = 0);
        vector(unsigned int size, T value);
        vector(const vector& other);

        unsigned int size() const;
        bool empty() const;
        void clear();
        %rename(push) push_back;
        void push_back(T x);
        %extend {
            T pop() throw (std::out_of_range) {
                if (self->size() == 0)
                    throw std::out_of_range("pop from empty vector");
                T x = self->back();
                self->pop_back();
                return x;
            }
            T get(int i) throw (std::out_of_range) {
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

    specialize_std_vector(bool,SvIOK,SvIVX,sv_setiv);
    specialize_std_vector(char,SvIOK,SvIVX,sv_setiv);
    specialize_std_vector(int,SvIOK,SvIVX,sv_setiv);
    specialize_std_vector(short,SvIOK,SvIVX,sv_setiv);
    specialize_std_vector(long,SvIOK,SvIVX,sv_setiv);
    specialize_std_vector(unsigned char,SvIOK,SvIVX,sv_setiv);
    specialize_std_vector(unsigned int,SvIOK,SvIVX,sv_setiv);
    specialize_std_vector(unsigned short,SvIOK,SvIVX,sv_setiv);
    specialize_std_vector(unsigned long,SvIOK,SvIVX,sv_setiv);
    specialize_std_vector(float,SvNIOK,SwigSvToNumber,sv_setnv);
    specialize_std_vector(double,SvNIOK,SwigSvToNumber,sv_setnv);
    specialize_std_vector(std::string,SvPOK,SwigSvToString,SwigSvFromString);
}

