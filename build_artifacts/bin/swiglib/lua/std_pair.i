/* -----------------------------------------------------------------------------
 * std_pair.i
 *
 * std::pair typemaps for LUA
 * ----------------------------------------------------------------------------- */

%{
#include <utility>
%}

namespace std {
  template <class T, class U > struct pair {
    typedef T first_type;
    typedef U second_type;

    pair();
    pair(T first, U second);
    pair(const pair& other);

    T first;
    U second;
  };

  template <class T, class U >
  pair<T,U> make_pair(const T& first, const U& second);
}
