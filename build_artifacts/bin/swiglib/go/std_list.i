/* -----------------------------------------------------------------------------
 * std_list.i
 * ----------------------------------------------------------------------------- */

%{
#include <list>
#include <stdexcept>
%}

namespace std {

  template<class T>
  class list {
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

    size_type size() const;
    bool empty() const;
    %rename(isEmpty) empty;
    void clear();
    void push_front(const value_type& x);
    void pop_front();
    void push_back(const value_type& x);
    void pop_back();
    void remove(value_type x);
    void reverse();
    void unique();
    void sort();
    void merge(list& x);
  };

}
