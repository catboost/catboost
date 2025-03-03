/* -----------------------------------------------------------------------------
 * std_vector.i
 * ----------------------------------------------------------------------------- */

%include <std_common.i>

// ------------------------------------------------------------------------
// std::vector
// 
// The aim of all that follows would be to integrate std::vector with 
// Tcl as much as possible, namely, to allow the user to pass and 
// be returned Tcl lists.
// const declarations are used to guess the intent of the function being
// exported; therefore, the following rationale is applied:
// 
//   -- f(std::vector< T >), f(const std::vector< T >&), f(const std::vector< T >*):
//      the parameter being read-only, either a Tcl list or a
//      previously wrapped std::vector< T > can be passed.
//   -- f(std::vector< T >&), f(std::vector< T >*):
//      the parameter must be modified; therefore, only a wrapped std::vector
//      can be passed.
//   -- std::vector< T > f():
//      the vector is returned by copy; therefore, a Tcl list of T:s 
//      is returned which is most easily used in other Tcl functions procs
//   -- std::vector< T >& f(), std::vector< T >* f(), const std::vector< T >& f(),
//      const std::vector< T >* f():
//      the vector is returned by reference; therefore, a wrapped std::vector
//      is returned
// ------------------------------------------------------------------------

%fragment("<string>");
%fragment("<stdexcept>");
%fragment("<algorithm>");
%{
#include <vector>

Tcl_Obj* SwigString_FromString(const std::string &s) {
    return Tcl_NewStringObj(s.data(), (int)s.length());
}

int Tcl_GetBoolFromObj(Tcl_Interp *interp, Tcl_Obj *o, bool *val) {
  int v;
  int res = Tcl_GetBooleanFromObj(interp, o, &v);
  if (res == TCL_OK) {
    *val = v ? true : false;
  }
  return res;  
}
 
int SwigString_AsString(Tcl_Interp *interp, Tcl_Obj *o, std::string *val) {
    int len;
    const char* temp = Tcl_GetStringFromObj(o, &len);
    if (temp == NULL)
        return TCL_ERROR;
    val->assign(temp, len);
    return TCL_OK;
}

// behaviour of this is such as the real Tcl_GetIntFromObj
template <typename Type>
int SwigInt_As(Tcl_Interp *interp, Tcl_Obj *o, Type *val) {
    int temp_val, return_val;
    return_val = Tcl_GetIntFromObj(interp, o, &temp_val);
    *val = (Type) temp_val;
    return return_val;
}

// behaviour of this is such as the real Tcl_GetDoubleFromObj
template <typename Type>
int SwigDouble_As(Tcl_Interp *interp, Tcl_Obj *o, Type *val) {
    int return_val;
    double temp_val;
    return_val = Tcl_GetDoubleFromObj(interp, o, &temp_val);
    *val = (Type) temp_val;
    return return_val;
}

%}

// exported class

namespace std {
    
    template<class T> class vector {
        %typemap(in) vector< T > (std::vector< T > *v) {
            Tcl_Obj **listobjv;
            int       nitems;
            int       i;
            T*        temp;

            if (SWIG_ConvertPtr($input, (void **) &v, \
                                $&1_descriptor, 0) == 0){
                $1 = *v;
            } else {
                // It isn't a vector< T > so it should be a list of T's
                if(Tcl_ListObjGetElements(interp, $input, \
                                          &nitems, &listobjv) == TCL_ERROR)
                    return TCL_ERROR;
                $1 = std::vector< T >();
                for (i = 0; i < nitems; i++) {
                    if ((SWIG_ConvertPtr(listobjv[i],(void **) &temp,
                                         $descriptor(T *),0)) != 0) {
                        char message[] = 
                            "list of " #T " expected";
                        Tcl_SetResult(interp, message, TCL_VOLATILE);
                        return TCL_ERROR;
                    }
                    $1.push_back(*temp);
                } 
            }
        }

        %typemap(in) const vector< T >* (std::vector< T > *v, std::vector< T > w),
                     const vector< T >& (std::vector< T > *v, std::vector< T > w) {
            Tcl_Obj **listobjv;
            int       nitems;
            int       i;
            T*        temp;

            if(SWIG_ConvertPtr($input, (void **) &v, \
                               $&1_descriptor, 0) == 0) {
                $1 = v;
            } else {
                // It isn't a vector< T > so it should be a list of T's
                if(Tcl_ListObjGetElements(interp, $input, 
                                          &nitems, &listobjv) == TCL_ERROR)
                    return TCL_ERROR;
                w = std::vector< T >();
                for (i = 0; i < nitems; i++) {
                    if ((SWIG_ConvertPtr(listobjv[i],(void **) &temp,
                                         $descriptor(T *),0)) != 0) {
                        char message[] = 
                            "list of " #T " expected";
                        Tcl_SetResult(interp, message, TCL_VOLATILE);
                        return TCL_ERROR;
                    }
                    w.push_back(*temp);
                } 
                $1 = &w;
            }
        }

        %typemap(out) vector< T > {
            for (unsigned int i=0; i<$1.size(); i++) {
                T* ptr = new T((($1_type &)$1)[i]);
                Tcl_ListObjAppendElement(interp, $result, \
                                         SWIG_NewInstanceObj(ptr, 
                                                             $descriptor(T *), 
                                                             0));
            }
        }

        %typecheck(SWIG_TYPECHECK_VECTOR) vector< T > {
            Tcl_Obj **listobjv;
            int       nitems;
            T*        temp;
            std::vector< T > *v;
            
            if(SWIG_ConvertPtr($input, (void **) &v, \
                               $&1_descriptor, 0) == 0) {
                /* wrapped vector */
                $1 = 1;
            } else {
                // It isn't a vector< T > so it should be a list of T's
                if(Tcl_ListObjGetElements(interp, $input, 
                                          &nitems, &listobjv) == TCL_ERROR)
                    $1 = 0;
                else
                    if (nitems == 0)
                        $1 = 1;
                //check the first value to see if it is of correct type
                    else if ((SWIG_ConvertPtr(listobjv[0],
                                              (void **) &temp, 
                                              $descriptor(T *),0)) != 0)
                        $1 = 0;
                    else
                        $1 = 1;
            }
        }
        
        %typecheck(SWIG_TYPECHECK_VECTOR) const vector< T >&,
                                          const vector< T >* {
            Tcl_Obj **listobjv;
            int       nitems;
            T*         temp;
            std::vector< T > *v;

            if(SWIG_ConvertPtr($input, (void **) &v, \
                               $1_descriptor, 0) == 0){
                /* wrapped vector */
                $1 = 1;
            } else {
                // It isn't a vector< T > so it should be a list of T's
                if(Tcl_ListObjGetElements(interp, $input, 
                                          &nitems, &listobjv) == TCL_ERROR)
                    $1 = 0;
                else
                    if (nitems == 0)
                        $1 = 1;
                //check the first value to see if it is of correct type
                    else if ((SWIG_ConvertPtr(listobjv[0],
                                              (void **) &temp,
                                              $descriptor(T *),0)) != 0)
                        $1 = 0;
                    else
                        $1 = 1;
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
                if (i<0) i += size;
                if (i>=0 && i<size)
                    return (*self)[i];
                else
                    throw std::out_of_range("vector index out of range");
            }
            void set(int i, const T& x) throw (std::out_of_range) {
                int size = int(self->size());
                if (i<0) i+= size;
                if (i>=0 && i<size)
                    (*self)[i] = x;
                else
                    throw std::out_of_range("vector index out of range");
            }
        }
    };


    // specializations for built-ins

    %define specialize_std_vector(T, CONVERT_FROM, CONVERT_TO)
    template<> class vector< T > {

        %typemap(in) vector< T > (std::vector< T > *v){
            Tcl_Obj **listobjv;
            int       nitems;
            int       i;
            T         temp;

            if(SWIG_ConvertPtr($input, (void **) &v, \
                               $&1_descriptor, 0) == 0) {
                $1 = *v;
            } else {
                // It isn't a vector< T > so it should be a list of T's
                if(Tcl_ListObjGetElements(interp, $input, 
                                          &nitems, &listobjv) == TCL_ERROR)
                    return TCL_ERROR;					      
                $1 = std::vector< T >();
                for (i = 0; i < nitems; i++) {
                    if (CONVERT_FROM(interp, listobjv[i], &temp) == TCL_ERROR)
                        return TCL_ERROR;
                    $1.push_back(temp);
                } 
            }
        }
      
        %typemap(in) const vector< T >& (std::vector< T > *v,std::vector< T > w),
                     const vector< T >* (std::vector< T > *v,std::vector< T > w) {
            Tcl_Obj **listobjv;
            int       nitems;
            int       i;
            T         temp;

            if(SWIG_ConvertPtr($input, (void **) &v, \
                               $1_descriptor, 0) == 0) {
                $1 = v;
            } else {
                // It isn't a vector< T > so it should be a list of T's
                if(Tcl_ListObjGetElements(interp, $input, 
                                          &nitems, &listobjv) == TCL_ERROR)
                    return TCL_ERROR;
                w = std::vector< T >();
                for (i = 0; i < nitems; i++) {
                    if (CONVERT_FROM(interp, listobjv[i], &temp) == TCL_ERROR)
                        return TCL_ERROR;
                    w.push_back(temp);
                } 
                $1 = &w;
            }
        }

        %typemap(out) vector< T > {
            for (unsigned int i=0; i<$1.size(); i++) {
                Tcl_ListObjAppendElement(interp, $result, \
                                         CONVERT_TO((($1_type &)$1)[i]));
            }
        }
       
        %typecheck(SWIG_TYPECHECK_VECTOR) vector< T > {
            Tcl_Obj **listobjv;
            int       nitems;
            T         temp;
            std::vector< T > *v;

            if(SWIG_ConvertPtr($input, (void **) &v, \
                               $&1_descriptor, 0) == 0){
                /* wrapped vector */
                $1 = 1;
            } else {
                // It isn't a vector< T > so it should be a list of T's
                if(Tcl_ListObjGetElements(interp, $input, 
                                          &nitems, &listobjv) == TCL_ERROR)
                    $1 = 0;
                else
                    if (nitems == 0)
                        $1 = 1;
                //check the first value to see if it is of correct type
                if (CONVERT_FROM(interp, listobjv[0], &temp) == TCL_ERROR)
                    $1 = 0;
                else
                    $1 = 1;
            }
        }      

        %typecheck(SWIG_TYPECHECK_VECTOR) const vector< T >&,
	                                      const vector< T >*{
            Tcl_Obj **listobjv;
            int       nitems;
            T         temp;
            std::vector< T > *v;

            if(SWIG_ConvertPtr($input, (void **) &v, \
                               $1_descriptor, 0) == 0){
                /* wrapped vector */
                $1 = 1;
            } else {
                // It isn't a vector< T > so it should be a list of T's
                if(Tcl_ListObjGetElements(interp, $input, 
                                          &nitems, &listobjv) == TCL_ERROR)
                    $1 = 0;
                else
                    if (nitems == 0)
                        $1 = 1;
                //check the first value to see if it is of correct type
                if (CONVERT_FROM(interp, listobjv[0], &temp) == TCL_ERROR)
                    $1 = 0;
                else
                    $1 = 1;
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
                if (i<0) i += size;
                if (i>=0 && i<size)
                    return (*self)[i];
                else
                    throw std::out_of_range("vector index out of range");
            }
            void set(int i, T x) throw (std::out_of_range) {
                int size = int(self->size());
                if (i<0) i+= size;
                if (i>=0 && i<size)
                    (*self)[i] = x;
                else
                    throw std::out_of_range("vector index out of range");
            }
        }
    };
    %enddef

    specialize_std_vector(bool, Tcl_GetBoolFromObj, Tcl_NewBooleanObj);
    specialize_std_vector(char, SwigInt_As<char>,Tcl_NewIntObj);
    specialize_std_vector(int, Tcl_GetIntFromObj,Tcl_NewIntObj);
    specialize_std_vector(short, SwigInt_As<short>, Tcl_NewIntObj);
    specialize_std_vector(long, SwigInt_As<long>, Tcl_NewIntObj);
    specialize_std_vector(unsigned char, 
                          SwigInt_As<unsigned char>, Tcl_NewIntObj);
    specialize_std_vector(unsigned int, 
                          SwigInt_As<unsigned int>, Tcl_NewIntObj);
    specialize_std_vector(unsigned short, 
                          SwigInt_As<unsigned short>, Tcl_NewIntObj);
    specialize_std_vector(unsigned long, 
                          SwigInt_As<unsigned long>, Tcl_NewIntObj);
    specialize_std_vector(double, Tcl_GetDoubleFromObj, Tcl_NewDoubleObj);
    specialize_std_vector(float, SwigDouble_As<float>, Tcl_NewDoubleObj);
    specialize_std_vector(std::string, 
                          SwigString_AsString, SwigString_FromString);

}


