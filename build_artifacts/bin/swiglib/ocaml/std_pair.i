/* -----------------------------------------------------------------------------
 * std_pair.i
 *
 * SWIG typemaps for std::pair
 * ----------------------------------------------------------------------------- */

%include <std_common.i>
%include <exception.i>

// ------------------------------------------------------------------------
// std::pair
// ------------------------------------------------------------------------

%{
#include <utility>
%}

namespace std {

  template<class T, class U> struct pair {
    typedef T first_type;
    typedef U second_type;

    pair();
    pair(T first, U second);
    pair(const pair& other);

    template <class U1, class U2> pair(const pair<U1, U2> &other);

    T first;
    U second;
  };

  // add specializations here

}
