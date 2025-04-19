// kept old implementation

#ifdef _LIBCPP_ENABLE_REMOVED_ALLOCATOR_CONST
template <class _Tp>
class _LIBCPP_TEMPLATE_VIS allocator<const _Tp>
    : private __non_trivial_if<!is_void<_Tp>::value, allocator<const _Tp> > {
  static_assert(!is_volatile<_Tp>::value, "std::allocator does not support volatile types");

public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef const _Tp value_type;
  typedef true_type propagate_on_container_move_assignment;
#  if _LIBCPP_STD_VER <= 23 || defined(_LIBCPP_ENABLE_CXX26_REMOVED_ALLOCATOR_MEMBERS)
  _LIBCPP_DEPRECATED_IN_CXX23 typedef true_type is_always_equal;
#  endif

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 allocator() _NOEXCEPT = default;

  template <class _Up>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 allocator(const allocator<_Up>&) _NOEXCEPT {}

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 const _Tp* allocate(size_t __n) {
    if (__n > allocator_traits<allocator>::max_size(*this))
      __throw_bad_array_new_length();
    if (__libcpp_is_constant_evaluated()) {
      return static_cast<const _Tp*>(::operator new(__n * sizeof(_Tp)));
    } else {
      return static_cast<const _Tp*>(std::__libcpp_allocate(__n * sizeof(_Tp), _LIBCPP_ALIGNOF(_Tp)));
    }
  }

#  if _LIBCPP_STD_VER >= 23
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr allocation_result<const _Tp*> allocate_at_least(size_t __n) {
    return {allocate(__n), __n};
  }
#  endif

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void deallocate(const _Tp* __p, size_t __n) {
    if (__libcpp_is_constant_evaluated()) {
      ::operator delete(const_cast<_Tp*>(__p));
    } else {
      std::__libcpp_deallocate((void*)const_cast<_Tp*>(__p), __n * sizeof(_Tp), _LIBCPP_ALIGNOF(_Tp));
    }
  }

  // C++20 Removed members
#  if _LIBCPP_STD_VER <= 17 || defined(_LIBCPP_ENABLE_CXX20_REMOVED_ALLOCATOR_MEMBERS)
  _LIBCPP_DEPRECATED_IN_CXX17 typedef const _Tp* pointer;
  _LIBCPP_DEPRECATED_IN_CXX17 typedef const _Tp* const_pointer;
  _LIBCPP_DEPRECATED_IN_CXX17 typedef const _Tp& reference;
  _LIBCPP_DEPRECATED_IN_CXX17 typedef const _Tp& const_reference;

  template <class _Up>
  struct _LIBCPP_DEPRECATED_IN_CXX17 rebind {
    typedef allocator<_Up> other;
  };

  _LIBCPP_DEPRECATED_IN_CXX17 _LIBCPP_HIDE_FROM_ABI const_pointer address(const_reference __x) const _NOEXCEPT {
    return std::addressof(__x);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI _LIBCPP_DEPRECATED_IN_CXX17 const _Tp* allocate(size_t __n, const void*) {
    return allocate(__n);
  }

  _LIBCPP_DEPRECATED_IN_CXX17 _LIBCPP_HIDE_FROM_ABI size_type max_size() const _NOEXCEPT {
    return size_type(~0) / sizeof(_Tp);
  }

  template <class _Up, class... _Args>
  _LIBCPP_DEPRECATED_IN_CXX17 _LIBCPP_HIDE_FROM_ABI void construct(_Up* __p, _Args&&... __args) {
    ::new ((void*)__p) _Up(std::forward<_Args>(__args)...);
  }

  _LIBCPP_DEPRECATED_IN_CXX17 _LIBCPP_HIDE_FROM_ABI void destroy(pointer __p) { __p->~_Tp(); }
#  endif
};
#endif // _LIBCPP_ENABLE_REMOVED_ALLOCATOR_CONST