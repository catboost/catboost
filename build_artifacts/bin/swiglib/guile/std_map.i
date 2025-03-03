/* -----------------------------------------------------------------------------
 * std_map.i
 *
 * SWIG typemaps for std::map
 * ----------------------------------------------------------------------------- */

%include <std_common.i>

// ------------------------------------------------------------------------
// std::map
//
// The aim of all that follows would be to integrate std::map with
// Guile as much as possible, namely, to allow the user to pass and
// be returned Scheme association lists.
// const declarations are used to guess the intent of the function being
// exported; therefore, the following rationale is applied:
//
//   -- f(std::map<T>), f(const std::map<T>&), f(const std::map<T>*):
//      the parameter being read-only, either a Scheme alist or a
//      previously wrapped std::map<T> can be passed.
//   -- f(std::map<T>&), f(std::map<T>*):
//      the parameter must be modified; therefore, only a wrapped std::map
//      can be passed.
//   -- std::map<T> f():
//      the map is returned by copy; therefore, a Scheme alist
//      is returned which is most easily used in other Scheme functions
//   -- std::map<T>& f(), std::map<T>* f(), const std::map<T>& f(),
//      const std::map<T>* f():
//      the map is returned by reference; therefore, a wrapped std::map
//      is returned
// ------------------------------------------------------------------------

%{
#include <map>
#include <algorithm>
#include <stdexcept>
%}

// exported class

namespace std {

    template<class K, class T, class C = std::less<K> > class map {
        %typemap(in) map< K, T, C > {
            if (scm_is_null($input)) {
                $1 = std::map< K, T, C >();
            } else if (scm_is_pair($input)) {
                $1 = std::map< K, T, C >();
                SCM alist = $input;
                while (!scm_is_null(alist)) {
                    K* k;
                    T* x;
                    SCM entry, key, val;
                    entry = SCM_CAR(alist);
                    if (!scm_is_pair(entry))
                        SWIG_exception(SWIG_TypeError,"alist expected");
                    key = SCM_CAR(entry);
                    val = SCM_CDR(entry);
                    k = (K*) SWIG_MustGetPtr(key,$descriptor(K *),$argnum, 0);
                    if (SWIG_ConvertPtr(val,(void**) &x,
                                    $descriptor(T *), 0) != 0) {
                        if (!scm_is_pair(val))
                            SWIG_exception(SWIG_TypeError,"alist expected");
                        val = SCM_CAR(val);
                        x = (T*) SWIG_MustGetPtr(val,$descriptor(T *),$argnum, 0);
                    }
                    (($1_type &)$1)[*k] = *x;
                    alist = SCM_CDR(alist);
                }
            } else {
                $1 = *(($&1_type)
                       SWIG_MustGetPtr($input,$&1_descriptor,$argnum, 0));
            }
        }
        %typemap(in) const map< K, T, C >& (std::map< K, T, C > temp),
                     const map< K, T, C >* (std::map< K, T, C > temp) {
            if (scm_is_null($input)) {
                temp = std::map< K, T, C >();
                $1 = &temp;
            } else if (scm_is_pair($input)) {
                temp = std::map< K, T, C >();
                $1 = &temp;
                SCM alist = $input;
                while (!scm_is_null(alist)) {
                    K* k;
                    T* x;
                    SCM entry, key, val;
                    entry = SCM_CAR(alist);
                    if (!scm_is_pair(entry))
                        SWIG_exception(SWIG_TypeError,"alist expected");
                    key = SCM_CAR(entry);
                    val = SCM_CDR(entry);
                    k = (K*) SWIG_MustGetPtr(key,$descriptor(K *),$argnum, 0);
                    if (SWIG_ConvertPtr(val,(void**) &x,
                                    $descriptor(T *), 0) != 0) {
                        if (!scm_is_pair(val))
                            SWIG_exception(SWIG_TypeError,"alist expected");
                        val = SCM_CAR(val);
                        x = (T*) SWIG_MustGetPtr(val,$descriptor(T *),$argnum, 0);
                    }
                    temp[*k] = *x;
                    alist = SCM_CDR(alist);
                }
            } else {
                $1 = ($1_ltype) SWIG_MustGetPtr($input,$1_descriptor,$argnum, 0);
            }
        }
        %typemap(out) map< K, T, C > {
            SCM alist = SCM_EOL;
            for (std::map< K, T, C >::reverse_iterator i=$1.rbegin(); i!=$1.rend(); ++i) {
                K* key = new K(i->first);
                T* val = new T(i->second);
                SCM k = SWIG_NewPointerObj(key,$descriptor(K *), 1);
                SCM x = SWIG_NewPointerObj(val,$descriptor(T *), 1);
                SCM entry = scm_cons(k,x);
                alist = scm_cons(entry,alist);
            }
            $result = alist;
        }
        %typecheck(SWIG_TYPECHECK_MAP) map< K, T, C > {
            /* native sequence? */
            if (scm_is_null($input)) {
                /* an empty sequence can be of any type */
                $1 = 1;
            } else if (scm_is_pair($input)) {
                /* check the first element only */
                K* k;
                T* x;
                SCM head = SCM_CAR($input);
                if (scm_is_pair(head)) {
                    SCM key = SCM_CAR(head);
                    SCM val = SCM_CDR(head);
                    if (SWIG_ConvertPtr(key,(void**) &k,
                                    $descriptor(K *), 0) != 0) {
                        $1 = 0;
                    } else {
                        if (SWIG_ConvertPtr(val,(void**) &x,
                                        $descriptor(T *), 0) == 0) {
                            $1 = 1;
                        } else if (scm_is_pair(val)) {
                            val = SCM_CAR(val);
                            if (SWIG_ConvertPtr(val,(void**) &x,
                                            $descriptor(T *), 0) == 0)
                                $1 = 1;
                            else
                                $1 = 0;
                        } else {
                            $1 = 0;
                        }
                    }
                } else {
                    $1 = 0;
                }
            } else {
                /* wrapped map? */
                std::map< K, T, C >* m;
                if (SWIG_ConvertPtr($input,(void **) &m,
                                $&1_descriptor, 0) == 0)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
        %typecheck(SWIG_TYPECHECK_MAP) const map< K, T, C >&,
                                       const map< K, T, C >* {
            /* native sequence? */
            if (scm_is_null($input)) {
                /* an empty sequence can be of any type */
                $1 = 1;
            } else if (scm_is_pair($input)) {
                /* check the first element only */
                K* k;
                T* x;
                SCM head = SCM_CAR($input);
                if (scm_is_pair(head)) {
                    SCM key = SCM_CAR(head);
                    SCM val = SCM_CDR(head);
                    if (SWIG_ConvertPtr(key,(void**) &k,
                                    $descriptor(K *), 0) != 0) {
                        $1 = 0;
                    } else {
                        if (SWIG_ConvertPtr(val,(void**) &x,
                                        $descriptor(T *), 0) == 0) {
                            $1 = 1;
                        } else if (scm_is_pair(val)) {
                            val = SCM_CAR(val);
                            if (SWIG_ConvertPtr(val,(void**) &x,
                                            $descriptor(T *), 0) == 0)
                                $1 = 1;
                            else
                                $1 = 0;
                        } else {
                            $1 = 0;
                        }
                    }
                } else {
                    $1 = 0;
                }
            } else {
                /* wrapped map? */
                std::map< K, T, C >* m;
                if (SWIG_ConvertPtr($input,(void **) &m,
                                $1_descriptor, 0) == 0)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
        %rename("length") size;
        %rename("null?") empty;
        %rename("clear!") clear;
        %rename("ref") __getitem__;
        %rename("set!") __setitem__;
        %rename("delete!") __delitem__;
        %rename("has-key?") has_key;
      public:
        typedef size_t size_type;
        typedef ptrdiff_t difference_type;
        typedef K key_type;
        typedef T mapped_type;
        typedef std::pair< const K, T > value_type;
        typedef value_type* pointer;
        typedef const value_type* const_pointer;
        typedef value_type& reference;
        typedef const value_type& const_reference;

        map();
        map(const map& other);
        
        unsigned int size() const;
        bool empty() const;
        void clear();
        %extend {
            const T& __getitem__(const K& key) throw (std::out_of_range) {
                std::map< K, T, C >::iterator i = self->find(key);
                if (i != self->end())
                    return i->second;
                else
                    throw std::out_of_range("key not found");
            }
            void __setitem__(const K& key, const T& x) {
                (*self)[key] = x;
            }
            void __delitem__(const K& key) throw (std::out_of_range) {
                std::map< K, T, C >::iterator i = self->find(key);
                if (i != self->end())
                    self->erase(i);
                else
                    throw std::out_of_range("key not found");
            }
            bool has_key(const K& key) {
                std::map< K, T, C >::iterator i = self->find(key);
                return i != self->end();
            }
            SCM keys() {
                SCM result = SCM_EOL;
                for (std::map< K, T, C >::reverse_iterator i=self->rbegin(); i!=self->rend(); ++i) {
                    K* key = new K(i->first);
                    SCM k = SWIG_NewPointerObj(key,$descriptor(K *), 1);
                    result = scm_cons(k,result);
                }
                return result;
            }
        }
    };


    // specializations for built-ins

    %define specialize_std_map_on_key(K,CHECK,CONVERT_FROM,CONVERT_TO)

    template<class T> class map< K, T, C > {
        %typemap(in) map< K, T, C > {
            if (scm_is_null($input)) {
                $1 = std::map< K, T, C >();
            } else if (scm_is_pair($input)) {
                $1 = std::map< K, T, C >();
                SCM alist = $input;
                while (!scm_is_null(alist)) {
                    T* x;
                    SCM entry, key, val;
                    entry = SCM_CAR(alist);
                    if (!scm_is_pair(entry))
                        SWIG_exception(SWIG_TypeError,"alist expected");
                    key = SCM_CAR(entry);
                    val = SCM_CDR(entry);
                    if (!CHECK(key))
                        SWIG_exception(SWIG_TypeError,
                                       "map<" #K "," #T "," #C "> expected");
                    if (SWIG_ConvertPtr(val,(void**) &x,
                                    $descriptor(T *), 0) != 0) {
                        if (!scm_is_pair(val))
                            SWIG_exception(SWIG_TypeError,"alist expected");
                        val = SCM_CAR(val);
                        x = (T*) SWIG_MustGetPtr(val,$descriptor(T *),$argnum, 0);
                    }
                    (($1_type &)$1)[CONVERT_FROM(key)] = *x;
                    alist = SCM_CDR(alist);
                }
            } else {
                $1 = *(($&1_type)
                       SWIG_MustGetPtr($input,$&1_descriptor,$argnum, 0));
            }
        }
        %typemap(in) const map< K, T, C >& (std::map< K, T, C > temp),
                     const map< K, T, C >* (std::map< K, T, C > temp) {
            if (scm_is_null($input)) {
                temp = std::map< K, T, C >();
                $1 = &temp;
            } else if (scm_is_pair($input)) {
                temp = std::map< K, T, C >();
                $1 = &temp;
                SCM alist = $input;
                while (!scm_is_null(alist)) {
                    T* x;
                    SCM entry, key, val;
                    entry = SCM_CAR(alist);
                    if (!scm_is_pair(entry))
                        SWIG_exception(SWIG_TypeError,"alist expected");
                    key = SCM_CAR(entry);
                    val = SCM_CDR(entry);
                    if (!CHECK(key))
                        SWIG_exception(SWIG_TypeError,
                                       "map<" #K "," #T "," #C "> expected");
                    if (SWIG_ConvertPtr(val,(void**) &x,
                                    $descriptor(T *), 0) != 0) {
                        if (!scm_is_pair(val))
                            SWIG_exception(SWIG_TypeError,"alist expected");
                        val = SCM_CAR(val);
                        x = (T*) SWIG_MustGetPtr(val,$descriptor(T *),$argnum, 0);
                    }
                    temp[CONVERT_FROM(key)] = *x;
                    alist = SCM_CDR(alist);
                }
            } else {
                $1 = ($1_ltype) SWIG_MustGetPtr($input,$1_descriptor,$argnum, 0);
            }
        }
        %typemap(out) map< K, T, C > {
            SCM alist = SCM_EOL;
            for (std::map< K, T, C >::reverse_iterator i=$1.rbegin(); i!=$1.rend(); ++i) {
                T* val = new T(i->second);
                SCM k = CONVERT_TO(i->first);
                SCM x = SWIG_NewPointerObj(val,$descriptor(T *), 1);
                SCM entry = scm_cons(k,x);
                alist = scm_cons(entry,alist);
            }
            $result = alist;
        }
        %typecheck(SWIG_TYPECHECK_MAP) map< K, T, C > {
            // native sequence?
            if (scm_is_null($input)) {
                /* an empty sequence can be of any type */
                $1 = 1;
            } else if (scm_is_pair($input)) {
                // check the first element only
                T* x;
                SCM head = SCM_CAR($input);
                if (scm_is_pair(head)) {
                    SCM key = SCM_CAR(head);
                    SCM val = SCM_CDR(head);
                    if (!CHECK(key)) {
                        $1 = 0;
                    } else {
                        if (SWIG_ConvertPtr(val,(void**) &x,
                                        $descriptor(T *), 0) == 0) {
                            $1 = 1;
                        } else if (scm_is_pair(val)) {
                            val = SCM_CAR(val);
                            if (SWIG_ConvertPtr(val,(void**) &x,
                                            $descriptor(T *), 0) == 0)
                                $1 = 1;
                            else
                                $1 = 0;
                        } else {
                            $1 = 0;
                        }
                    }
                } else {
                    $1 = 0;
                }
            } else {
                // wrapped map?
                std::map< K, T, C >* m;
                if (SWIG_ConvertPtr($input,(void **) &m,
                                $&1_descriptor, 0) == 0)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
        %typecheck(SWIG_TYPECHECK_MAP) const map< K, T, C >&,
                                       const map< K, T, C >* {
            // native sequence?
            if (scm_is_null($input)) {
                /* an empty sequence can be of any type */
                $1 = 1;
            } else if (scm_is_pair($input)) {
                // check the first element only
                T* x;
                SCM head = SCM_CAR($input);
                if (scm_is_pair(head)) {
                    SCM key = SCM_CAR(head);
                    SCM val = SCM_CDR(head);
                    if (!CHECK(key)) {
                        $1 = 0;
                    } else {
                        if (SWIG_ConvertPtr(val,(void**) &x,
                                        $descriptor(T *), 0) == 0) {
                            $1 = 1;
                        } else if (scm_is_pair(val)) {
                            val = SCM_CAR(val);
                            if (SWIG_ConvertPtr(val,(void**) &x,
                                            $descriptor(T *), 0) == 0)
                                $1 = 1;
                            else
                                $1 = 0;
                        } else {
                            $1 = 0;
                        }
                    }
                } else {
                    $1 = 0;
                }
            } else {
                // wrapped map?
                std::map< K, T, C >* m;
                if (SWIG_ConvertPtr($input,(void **) &m,
                                $1_descriptor, 0) == 0)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
        %rename("length") size;
        %rename("null?") empty;
        %rename("clear!") clear;
        %rename("ref") __getitem__;
        %rename("set!") __setitem__;
        %rename("delete!") __delitem__;
        %rename("has-key?") has_key;
      public:
        typedef size_t size_type;
        typedef ptrdiff_t difference_type;
        typedef K key_type;
        typedef T mapped_type;
        typedef std::pair< const K, T > value_type;
        typedef value_type* pointer;
        typedef const value_type* const_pointer;
        typedef value_type& reference;
        typedef const value_type& const_reference;

        map();
        map(const map& other);
        
        unsigned int size() const;
        bool empty() const;
        void clear();
        %extend {
            T& __getitem__(K key) throw (std::out_of_range) {
                std::map< K, T, C >::iterator i = self->find(key);
                if (i != self->end())
                    return i->second;
                else
                    throw std::out_of_range("key not found");
            }
            void __setitem__(K key, const T& x) {
                (*self)[key] = x;
            }
            void __delitem__(K key) throw (std::out_of_range) {
                std::map< K, T, C >::iterator i = self->find(key);
                if (i != self->end())
                    self->erase(i);
                else
                    throw std::out_of_range("key not found");
            }
            bool has_key(K key) {
                std::map< K, T, C >::iterator i = self->find(key);
                return i != self->end();
            }
            SCM keys() {
                SCM result = SCM_EOL;
                for (std::map< K, T, C >::reverse_iterator i=self->rbegin(); i!=self->rend(); ++i) {
                    SCM k = CONVERT_TO(i->first);
                    result = scm_cons(k,result);
                }
                return result;
            }
        }
    };
    %enddef

    %define specialize_std_map_on_value(T,CHECK,CONVERT_FROM,CONVERT_TO)
    template<class K> class map< K, T, C > {
        %typemap(in) map< K, T, C > {
            if (scm_is_null($input)) {
                $1 = std::map< K, T, C >();
            } else if (scm_is_pair($input)) {
                $1 = std::map< K, T, C >();
                SCM alist = $input;
                while (!scm_is_null(alist)) {
                    K* k;
                    SCM entry, key, val;
                    entry = SCM_CAR(alist);
                    if (!scm_is_pair(entry))
                        SWIG_exception(SWIG_TypeError,"alist expected");
                    key = SCM_CAR(entry);
                    val = SCM_CDR(entry);
                    k = (K*) SWIG_MustGetPtr(key,$descriptor(K *),$argnum, 0);
                    if (!CHECK(val)) {
                        if (!scm_is_pair(val))
                            SWIG_exception(SWIG_TypeError,"alist expected");
                        val = SCM_CAR(val);
                        if (!CHECK(val))
                            SWIG_exception(SWIG_TypeError,
                                           "map<" #K "," #T "," #C "> expected");
                    }
                    (($1_type &)$1)[*k] = CONVERT_FROM(val);
                    alist = SCM_CDR(alist);
                }
            } else {
                $1 = *(($&1_type)
                       SWIG_MustGetPtr($input,$&1_descriptor,$argnum, 0));
            }
        }
        %typemap(in) const map< K, T, C >& (std::map< K, T, C > temp),
                     const map< K, T, C >* (std::map< K, T, C > temp) {
            if (scm_is_null($input)) {
                temp = std::map< K, T, C >();
                $1 = &temp;
            } else if (scm_is_pair($input)) {
                temp = std::map< K, T, C >();
                $1 = &temp;
                SCM alist = $input;
                while (!scm_is_null(alist)) {
                    K* k;
                    SCM entry, key, val;
                    entry = SCM_CAR(alist);
                    if (!scm_is_pair(entry))
                        SWIG_exception(SWIG_TypeError,"alist expected");
                    key = SCM_CAR(entry);
                    val = SCM_CDR(entry);
                    k = (K*) SWIG_MustGetPtr(key,$descriptor(K *),$argnum, 0);
                    if (!CHECK(val)) {
                        if (!scm_is_pair(val))
                            SWIG_exception(SWIG_TypeError,"alist expected");
                        val = SCM_CAR(val);
                        if (!CHECK(val))
                            SWIG_exception(SWIG_TypeError,
                                           "map<" #K "," #T "," #C "> expected");
                    }
                    temp[*k] = CONVERT_FROM(val);
                    alist = SCM_CDR(alist);
                }
            } else {
                $1 = ($1_ltype) SWIG_MustGetPtr($input,$1_descriptor,$argnum, 0);
            }
        }
        %typemap(out) map< K, T, C > {
            SCM alist = SCM_EOL;
            for (std::map< K, T, C >::reverse_iterator i=$1.rbegin(); i!=$1.rend(); ++i) {
                K* key = new K(i->first);
                SCM k = SWIG_NewPointerObj(key,$descriptor(K *), 1);
                SCM x = CONVERT_TO(i->second);
                SCM entry = scm_cons(k,x);
                alist = scm_cons(entry,alist);
            }
            $result = alist;
        }
        %typecheck(SWIG_TYPECHECK_MAP) map< K, T, C > {
            // native sequence?
            if (scm_is_null($input)) {
                /* an empty sequence can be of any type */
                $1 = 1;
            } else if (scm_is_pair($input)) {
                // check the first element only
                K* k;
                SCM head = SCM_CAR($input);
                if (scm_is_pair(head)) {
                    SCM val = SCM_CDR(head);
                    if (SWIG_ConvertPtr(val,(void **) &k,
                                    $descriptor(K *), 0) != 0) {
                        $1 = 0;
                    } else {
                        if (CHECK(val)) {
                            $1 = 1;
                        } else if (scm_is_pair(val)) {
                            val = SCM_CAR(val);
                            if (CHECK(val))
                                $1 = 1;
                            else
                                $1 = 0;
                        } else {
                            $1 = 0;
                        }
                    }
                } else {
                    $1 = 0;
                }
            } else {
                // wrapped map?
                std::map< K, T, C >* m;
                if (SWIG_ConvertPtr($input,(void **) &m,
                                $&1_descriptor, 0) == 0)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
        %typecheck(SWIG_TYPECHECK_MAP) const map< K, T, C >&,
                                       const map< K, T, C >* {
            // native sequence?
            if (scm_is_null($input)) {
                /* an empty sequence can be of any type */
                $1 = 1;
            } else if (scm_is_pair($input)) {
                // check the first element only
                K* k;
                SCM head = SCM_CAR($input);
                if (scm_is_pair(head)) {
                    SCM val = SCM_CDR(head);
                    if (SWIG_ConvertPtr(val,(void **) &k,
                                    $descriptor(K *), 0) != 0) {
                        $1 = 0;
                    } else {
                        if (CHECK(val)) {
                            $1 = 1;
                        } else if (scm_is_pair(val)) {
                            val = SCM_CAR(val);
                            if (CHECK(val))
                                $1 = 1;
                            else
                                $1 = 0;
                        } else {
                            $1 = 0;
                        }
                    }
                } else {
                    $1 = 0;
                }
            } else {
                // wrapped map?
                std::map< K, T, C >* m;
                if (SWIG_ConvertPtr($input,(void **) &m,
                                $1_descriptor, 0) == 0)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
        %rename("length") size;
        %rename("null?") empty;
        %rename("clear!") clear;
        %rename("ref") __getitem__;
        %rename("set!") __setitem__;
        %rename("delete!") __delitem__;
        %rename("has-key?") has_key;
      public:
        typedef size_t size_type;
        typedef ptrdiff_t difference_type;
        typedef K key_type;
        typedef T mapped_type;
        typedef std::pair< const K, T > value_type;
        typedef value_type* pointer;
        typedef const value_type* const_pointer;
        typedef value_type& reference;
        typedef const value_type& const_reference;

        map();
        map(const map& other);
        
        unsigned int size() const;
        bool empty() const;
        void clear();
        %extend {
            T __getitem__(const K& key) throw (std::out_of_range) {
                std::map< K, T, C >::iterator i = self->find(key);
                if (i != self->end())
                    return i->second;
                else
                    throw std::out_of_range("key not found");
            }
            void __setitem__(const K& key, T x) {
                (*self)[key] = x;
            }
            void __delitem__(const K& key) throw (std::out_of_range) {
                std::map< K, T, C >::iterator i = self->find(key);
                if (i != self->end())
                    self->erase(i);
                else
                    throw std::out_of_range("key not found");
            }
            bool has_key(const K& key) {
                std::map< K, T, C >::iterator i = self->find(key);
                return i != self->end();
            }
            SCM keys() {
                SCM result = SCM_EOL;
                for (std::map< K, T, C >::reverse_iterator i=self->rbegin(); i!=self->rend(); ++i) {
                    K* key = new K(i->first);
                    SCM k = SWIG_NewPointerObj(key,$descriptor(K *), 1);
                    result = scm_cons(k,result);
                }
                return result;
            }
        }
    };
    %enddef

    %define specialize_std_map_on_both(K,CHECK_K,CONVERT_K_FROM,CONVERT_K_TO,
                                       T,CHECK_T,CONVERT_T_FROM,CONVERT_T_TO)
    template<> class map< K, T, C > {
        %typemap(in) map< K, T, C > {
            if (scm_is_null($input)) {
                $1 = std::map< K, T, C >();
            } else if (scm_is_pair($input)) {
                $1 = std::map< K, T, C >();
                SCM alist = $input;
                while (!scm_is_null(alist)) {
                    SCM entry, key, val;
                    entry = SCM_CAR(alist);
                    if (!scm_is_pair(entry))
                        SWIG_exception(SWIG_TypeError,"alist expected");
                    key = SCM_CAR(entry);
                    val = SCM_CDR(entry);
                    if (!CHECK_K(key))
                        SWIG_exception(SWIG_TypeError,
                                           "map<" #K "," #T "," #C "> expected");
                    if (!CHECK_T(val)) {
                        if (!scm_is_pair(val))
                            SWIG_exception(SWIG_TypeError,"alist expected");
                        val = SCM_CAR(val);
                        if (!CHECK_T(val))
                            SWIG_exception(SWIG_TypeError,
                                           "map<" #K "," #T "," #C "> expected");
                    }
                    (($1_type &)$1)[CONVERT_K_FROM(key)] = 
                                               CONVERT_T_FROM(val);
                    alist = SCM_CDR(alist);
                }
            } else {
                $1 = *(($&1_type)
                       SWIG_MustGetPtr($input,$&1_descriptor,$argnum, 0));
            }
        }
        %typemap(in) const map< K, T, C >& (std::map< K, T, C > temp),
                     const map< K, T, C >* (std::map< K, T, C > temp) {
            if (scm_is_null($input)) {
                temp = std::map< K, T, C >();
                $1 = &temp;
            } else if (scm_is_pair($input)) {
                temp = std::map< K, T, C >();
                $1 = &temp;
                SCM alist = $input;
                while (!scm_is_null(alist)) {
                    SCM entry, key, val;
                    entry = SCM_CAR(alist);
                    if (!scm_is_pair(entry))
                        SWIG_exception(SWIG_TypeError,"alist expected");
                    key = SCM_CAR(entry);
                    val = SCM_CDR(entry);
                    if (!CHECK_K(key))
                        SWIG_exception(SWIG_TypeError,
                                           "map<" #K "," #T "," #C "> expected");
                    if (!CHECK_T(val)) {
                        if (!scm_is_pair(val))
                            SWIG_exception(SWIG_TypeError,"alist expected");
                        val = SCM_CAR(val);
                        if (!CHECK_T(val))
                            SWIG_exception(SWIG_TypeError,
                                           "map<" #K "," #T "," #C "> expected");
                    }
                    temp[CONVERT_K_FROM(key)] = CONVERT_T_FROM(val);
                    alist = SCM_CDR(alist);
                }
            } else {
                $1 = ($1_ltype) SWIG_MustGetPtr($input,$1_descriptor,$argnum, 0);
            }
        }
        %typemap(out) map< K, T, C > {
            SCM alist = SCM_EOL;
            for (std::map< K, T, C >::reverse_iterator i=$1.rbegin(); i!=$1.rend(); ++i) {
                SCM k = CONVERT_K_TO(i->first);
                SCM x = CONVERT_T_TO(i->second);
                SCM entry = scm_cons(k,x);
                alist = scm_cons(entry,alist);
            }
            $result = alist;
        }
        %typecheck(SWIG_TYPECHECK_MAP) map< K, T, C > {
            // native sequence?
            if (scm_is_null($input)) {
                /* an empty sequence can be of any type */
                $1 = 1;
            } else if (scm_is_pair($input)) {
                // check the first element only
                SCM head = SCM_CAR($input);
                if (scm_is_pair(head)) {
                    SCM key = SCM_CAR(head);
                    SCM val = SCM_CDR(head);
                    if (!CHECK_K(key)) {
                        $1 = 0;
                    } else {
                        if (CHECK_T(val)) {
                            $1 = 1;
                        } else if (scm_is_pair(val)) {
                            val = SCM_CAR(val);
                            if (CHECK_T(val))
                                $1 = 1;
                            else
                                $1 = 0;
                        } else {
                            $1 = 0;
                        }
                    }
                } else {
                    $1 = 0;
                }
            } else {
                // wrapped map?
                std::map< K, T, C >* m;
                if (SWIG_ConvertPtr($input,(void **) &m,
                                $&1_descriptor, 0) == 0)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
        %typecheck(SWIG_TYPECHECK_MAP) const map< K, T, C >&,
                                       const map< K, T, C >* {
            // native sequence?
            if (scm_is_null($input)) {
                /* an empty sequence can be of any type */
                $1 = 1;
            } else if (scm_is_pair($input)) {
                // check the first element only
                SCM head = SCM_CAR($input);
                if (scm_is_pair(head)) {
                    SCM key = SCM_CAR(head);
                    SCM val = SCM_CDR(head);
                    if (!CHECK_K(key)) {
                        $1 = 0;
                    } else {
                        if (CHECK_T(val)) {
                            $1 = 1;
                        } else if (scm_is_pair(val)) {
                            val = SCM_CAR(val);
                            if (CHECK_T(val))
                                $1 = 1;
                            else
                                $1 = 0;
                        } else {
                            $1 = 0;
                        }
                    }
                } else {
                    $1 = 0;
                }
            } else {
                // wrapped map?
                std::map< K, T, C >* m;
                if (SWIG_ConvertPtr($input,(void **) &m,
                                $1_descriptor, 0) == 0)
                    $1 = 1;
                else
                    $1 = 0;
            }
        }
        %rename("length") size;
        %rename("null?") empty;
        %rename("clear!") clear;
        %rename("ref") __getitem__;
        %rename("set!") __setitem__;
        %rename("delete!") __delitem__;
        %rename("has-key?") has_key;
      public:
        typedef size_t size_type;
        typedef ptrdiff_t difference_type;
        typedef K key_type;
        typedef T mapped_type;
        typedef std::pair< const K, T > value_type;
        typedef value_type* pointer;
        typedef const value_type* const_pointer;
        typedef value_type& reference;
        typedef const value_type& const_reference;

        map();
        map(const map& other);
        
        unsigned int size() const;
        bool empty() const;
        void clear();
        %extend {
            T __getitem__(K key) throw (std::out_of_range) {
                std::map< K, T, C >::iterator i = self->find(key);
                if (i != self->end())
                    return i->second;
                else
                    throw std::out_of_range("key not found");
            }
            void __setitem__(K key, T x) {
                (*self)[key] = x;
            }
            void __delitem__(K key) throw (std::out_of_range) {
                std::map< K, T, C >::iterator i = self->find(key);
                if (i != self->end())
                    self->erase(i);
                else
                    throw std::out_of_range("key not found");
            }
            bool has_key(K key) {
                std::map< K, T, C >::iterator i = self->find(key);
                return i != self->end();
            }
            SCM keys() {
                SCM result = SCM_EOL;
                for (std::map< K, T, C >::reverse_iterator i=self->rbegin(); i!=self->rend(); ++i) {
                    SCM k = CONVERT_K_TO(i->first);
                    result = scm_cons(k,result);
                }
                return result;
            }
        }
    };
    %enddef


    specialize_std_map_on_key(bool,scm_is_bool,
                              scm_is_true,SWIG_bool2scm);
    specialize_std_map_on_key(int,scm_is_number,
                              scm_to_long,scm_from_long);
    specialize_std_map_on_key(short,scm_is_number,
                              scm_to_long,scm_from_long);
    specialize_std_map_on_key(long,scm_is_number,
                              scm_to_long,scm_from_long);
    specialize_std_map_on_key(unsigned int,scm_is_number,
                              scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_key(unsigned short,scm_is_number,
                              scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_key(unsigned long,scm_is_number,
                              scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_key(double,scm_is_number,
                              scm_to_double,scm_from_double);
    specialize_std_map_on_key(float,scm_is_number,
                              scm_to_double,scm_from_double);
    specialize_std_map_on_key(std::string,scm_is_string,
                              SWIG_scm2string,SWIG_string2scm);

    specialize_std_map_on_value(bool,scm_is_bool,
                                scm_is_true,SWIG_bool2scm);
    specialize_std_map_on_value(int,scm_is_number,
                                scm_to_long,scm_from_long);
    specialize_std_map_on_value(short,scm_is_number,
                                scm_to_long,scm_from_long);
    specialize_std_map_on_value(long,scm_is_number,
                                scm_to_long,scm_from_long);
    specialize_std_map_on_value(unsigned int,scm_is_number,
                                scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_value(unsigned short,scm_is_number,
                                scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_value(unsigned long,scm_is_number,
                                scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_value(double,scm_is_number,
                                scm_to_double,scm_from_double);
    specialize_std_map_on_value(float,scm_is_number,
                                scm_to_double,scm_from_double);
    specialize_std_map_on_value(std::string,scm_is_string,
                                SWIG_scm2string,SWIG_string2scm);

    specialize_std_map_on_both(bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm,
                               bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm);
    specialize_std_map_on_both(bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm,
                               int,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm,
                               short,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm,
                               long,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm,
                               unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm,
                               unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm,
                               unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm,
                               double,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_map_on_both(bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm,
                               float,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_map_on_both(bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm,
                               std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm);
    specialize_std_map_on_both(int,scm_is_number,
                               scm_to_long,scm_from_long,
                               bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm);
    specialize_std_map_on_both(int,scm_is_number,
                               scm_to_long,scm_from_long,
                               int,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(int,scm_is_number,
                               scm_to_long,scm_from_long,
                               short,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(int,scm_is_number,
                               scm_to_long,scm_from_long,
                               long,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(int,scm_is_number,
                               scm_to_long,scm_from_long,
                               unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(int,scm_is_number,
                               scm_to_long,scm_from_long,
                               unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(int,scm_is_number,
                               scm_to_long,scm_from_long,
                               unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(int,scm_is_number,
                               scm_to_long,scm_from_long,
                               double,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_map_on_both(int,scm_is_number,
                               scm_to_long,scm_from_long,
                               float,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_map_on_both(int,scm_is_number,
                               scm_to_long,scm_from_long,
                               std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm);
    specialize_std_map_on_both(short,scm_is_number,
                               scm_to_long,scm_from_long,
                               bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm);
    specialize_std_map_on_both(short,scm_is_number,
                               scm_to_long,scm_from_long,
                               int,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(short,scm_is_number,
                               scm_to_long,scm_from_long,
                               short,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(short,scm_is_number,
                               scm_to_long,scm_from_long,
                               long,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(short,scm_is_number,
                               scm_to_long,scm_from_long,
                               unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(short,scm_is_number,
                               scm_to_long,scm_from_long,
                               unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(short,scm_is_number,
                               scm_to_long,scm_from_long,
                               unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(short,scm_is_number,
                               scm_to_long,scm_from_long,
                               double,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_map_on_both(short,scm_is_number,
                               scm_to_long,scm_from_long,
                               float,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_map_on_both(short,scm_is_number,
                               scm_to_long,scm_from_long,
                               std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm);
    specialize_std_map_on_both(long,scm_is_number,
                               scm_to_long,scm_from_long,
                               bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm);
    specialize_std_map_on_both(long,scm_is_number,
                               scm_to_long,scm_from_long,
                               int,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(long,scm_is_number,
                               scm_to_long,scm_from_long,
                               short,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(long,scm_is_number,
                               scm_to_long,scm_from_long,
                               long,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(long,scm_is_number,
                               scm_to_long,scm_from_long,
                               unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(long,scm_is_number,
                               scm_to_long,scm_from_long,
                               unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(long,scm_is_number,
                               scm_to_long,scm_from_long,
                               unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(long,scm_is_number,
                               scm_to_long,scm_from_long,
                               double,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_map_on_both(long,scm_is_number,
                               scm_to_long,scm_from_long,
                               float,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_map_on_both(long,scm_is_number,
                               scm_to_long,scm_from_long,
                               std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm);
    specialize_std_map_on_both(unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm);
    specialize_std_map_on_both(unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               int,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               short,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               long,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               double,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_map_on_both(unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               float,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_map_on_both(unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm);
    specialize_std_map_on_both(unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm);
    specialize_std_map_on_both(unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               int,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               short,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               long,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               double,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_map_on_both(unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               float,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_map_on_both(unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm);
    specialize_std_map_on_both(unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm);
    specialize_std_map_on_both(unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               int,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               short,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               long,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               double,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_map_on_both(unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               float,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_map_on_both(unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong,
                               std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm);
    specialize_std_map_on_both(double,scm_is_number,
                               scm_to_double,scm_from_double,
                               bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm);
    specialize_std_map_on_both(double,scm_is_number,
                               scm_to_double,scm_from_double,
                               int,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(double,scm_is_number,
                               scm_to_double,scm_from_double,
                               short,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(double,scm_is_number,
                               scm_to_double,scm_from_double,
                               long,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(double,scm_is_number,
                               scm_to_double,scm_from_double,
                               unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(double,scm_is_number,
                               scm_to_double,scm_from_double,
                               unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(double,scm_is_number,
                               scm_to_double,scm_from_double,
                               unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(double,scm_is_number,
                               scm_to_double,scm_from_double,
                               double,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_map_on_both(double,scm_is_number,
                               scm_to_double,scm_from_double,
                               float,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_map_on_both(double,scm_is_number,
                               scm_to_double,scm_from_double,
                               std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm);
    specialize_std_map_on_both(float,scm_is_number,
                               scm_to_double,scm_from_double,
                               bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm);
    specialize_std_map_on_both(float,scm_is_number,
                               scm_to_double,scm_from_double,
                               int,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(float,scm_is_number,
                               scm_to_double,scm_from_double,
                               short,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(float,scm_is_number,
                               scm_to_double,scm_from_double,
                               long,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(float,scm_is_number,
                               scm_to_double,scm_from_double,
                               unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(float,scm_is_number,
                               scm_to_double,scm_from_double,
                               unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(float,scm_is_number,
                               scm_to_double,scm_from_double,
                               unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(float,scm_is_number,
                               scm_to_double,scm_from_double,
                               double,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_map_on_both(float,scm_is_number,
                               scm_to_double,scm_from_double,
                               float,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_map_on_both(float,scm_is_number,
                               scm_to_double,scm_from_double,
                               std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm);
    specialize_std_map_on_both(std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm,
                               bool,scm_is_bool,
                               scm_is_true,SWIG_bool2scm);
    specialize_std_map_on_both(std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm,
                               int,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm,
                               short,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm,
                               long,scm_is_number,
                               scm_to_long,scm_from_long);
    specialize_std_map_on_both(std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm,
                               unsigned int,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm,
                               unsigned short,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm,
                               unsigned long,scm_is_number,
                               scm_to_ulong,scm_from_ulong);
    specialize_std_map_on_both(std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm,
                               double,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_map_on_both(std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm,
                               float,scm_is_number,
                               scm_to_double,scm_from_double);
    specialize_std_map_on_both(std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm,
                               std::string,scm_is_string,
                               SWIG_scm2string,SWIG_string2scm);
}
