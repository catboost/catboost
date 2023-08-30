#ifndef PYTHONIC_NUMPY_ROLL_HPP
#define PYTHONIC_NUMPY_ROLL_HPP

#include "pythonic/include/numpy/roll.hpp"

#include "pythonic/utils/functor.hpp"
#include "pythonic/utils/numpy_conversion.hpp"
#include "pythonic/types/ndarray.hpp"

PYTHONIC_NS_BEGIN

namespace numpy
{
  template <class T, class pS>
  types::ndarray<T, pS> roll(types::ndarray<T, pS> const &expr, long shift)
  {
    if (expr.flat_size() == 0)
      return expr.copy();
    if (shift < 0)
      shift += expr.flat_size();
    shift %= expr.flat_size();
    types::ndarray<T, pS> out(expr._shape, builtins::None);
    std::copy(expr.fbegin(), expr.fend() - shift,
              std::copy(expr.fend() - shift, expr.fend(), out.fbegin()));
    return out;
  }

  namespace
  {
    template <class To, class From, size_t N>
    To _roll(To to, From from, long, long, types::array<long, N> const &,
             utils::int_<N>)
    {
      *to = *from;
      return to + 1;
    }

    template <class To, class From, size_t N, size_t M>
    To _roll(To to, From from, long shift, long axis,
             types::array<long, N> const &shape, utils::int_<M>)
    {
      long dim = shape[M];
      long offset = std::accumulate(shape.begin() + M + 1, shape.end(), 1L,
                                    std::multiplies<long>());
      if (axis == M) {
        const From split = from + (dim - shift) * offset;
        for (From iter = split, end = from + dim * offset; iter != end;
             iter += offset)
          to = _roll(to, iter, shift, axis, shape, utils::int_<M + 1>());
        for (From iter = from, end = split; iter != end; iter += offset)
          to = _roll(to, iter, shift, axis, shape, utils::int_<M + 1>());
      } else {
        for (From iter = from, end = from + dim * offset; iter != end;
             iter += offset)
          to = _roll(to, iter, shift, axis, shape, utils::int_<M + 1>());
      }
      return to;
    }
  }

  template <class T, class pS>
  types::ndarray<T, pS> roll(types::ndarray<T, pS> const &expr, long shift,
                             long axis)
  {
    auto expr_shape = sutils::array(expr._shape);
    if (expr_shape[axis] == 0)
      return expr.copy();
    if (shift < 0)
      shift += expr_shape[axis];
    types::ndarray<T, pS> out(expr._shape, builtins::None);
    _roll(out.fbegin(), expr.fbegin(), shift, axis, expr_shape,
          utils::int_<0>());
    return out;
  }

  NUMPY_EXPR_TO_NDARRAY0_IMPL(roll);
}
PYTHONIC_NS_END

#endif
