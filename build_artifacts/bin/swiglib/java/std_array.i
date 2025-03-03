/* -----------------------------------------------------------------------------
 * std_array.i
 * ----------------------------------------------------------------------------- */

%include <std_common.i>

namespace std {

  template<class T, size_t N> class array {
  public:
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef T value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;

    array();
    array(const array& other);

    size_type size() const;
    %rename(isEmpty) empty;
    bool empty() const;
    void fill(const T& u);
    %extend {
      const_reference get(int i) throw (std::out_of_range) {
        int size = int(self->size());
        if (i>=0 && i<size)
          return (*self)[i];
        else
          throw std::out_of_range("array index out of range");
      }
      void set(int i, const value_type& val) throw (std::out_of_range) {
        int size = int(self->size());
        if (i>=0 && i<size)
          (*self)[i] = val;
        else
          throw std::out_of_range("array index out of range");
      }
    }
  };
}

