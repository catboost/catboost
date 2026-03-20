// MSVCRT already has the correct prototype in <stdlib.h> if __cplusplus is defined
#      if !defined(_LIBCPP_MSVCRT)
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI long abs(long __x) _NOEXCEPT { return __builtin_labs(__x); }
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI long long abs(long long __x) _NOEXCEPT { return __builtin_llabs(__x); }
#      endif // !defined(_LIBCPP_MSVCRT)

[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI float abs(float __lcpp_x) _NOEXCEPT {
  return __builtin_fabsf(__lcpp_x); // Use builtins to prevent needing math.h
}

[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI double abs(double __lcpp_x) _NOEXCEPT {
  return __builtin_fabs(__lcpp_x);
}

[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI long double abs(long double __lcpp_x) _NOEXCEPT {
  return __builtin_fabsl(__lcpp_x);
}