/* -----------------------------------------------------------------------------
 * std_map.i
 *
 * SWIG typemaps for std::map
 * ----------------------------------------------------------------------------- */

%include <std_common.i>

// ------------------------------------------------------------------------
// std::map
// ------------------------------------------------------------------------

%{
#include <map>
%}
%fragment("<algorithm>");
%fragment("<stdexcept>");

// exported class

namespace std {

    template<class K, class T, class C = std::less<K> > class map {
        // add typemaps here
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
            const T& get(const K& key) throw (std::out_of_range) {
                std::map< K, T, C >::iterator i = self->find(key);
                if (i != self->end())
                    return i->second;
                else
                    throw std::out_of_range("key not found");
            }
            void set(const K& key, const T& x) {
                (*self)[key] = x;
            }
            void del(const K& key) throw (std::out_of_range) {
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
        }
    };

// Legacy macros (deprecated)
%define specialize_std_map_on_key(K,CHECK,CONVERT_FROM,CONVERT_TO)
#warning "specialize_std_map_on_key ignored - macro is deprecated and no longer necessary"
%enddef

%define specialize_std_map_on_value(T,CHECK,CONVERT_FROM,CONVERT_TO)
#warning "specialize_std_map_on_value ignored - macro is deprecated and no longer necessary"
%enddef

%define specialize_std_map_on_both(K,CHECK_K,CONVERT_K_FROM,CONVERT_K_TO, T,CHECK_T,CONVERT_T_FROM,CONVERT_T_TO)
#warning "specialize_std_map_on_both ignored - macro is deprecated and no longer necessary"
%enddef

}
