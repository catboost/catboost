/* _unur_hypot
 *
 * Computs the value of sqrt(x^2 + y^2) in a way that avoids overflow.
 *
 * Replacement for missing C99 function hypot()
 *
 * Copied and renamed from NumPy's npy_hypot into _unur_hypot
 * by Josef Leydold, Tue Nov  1 11:17:37 CET 2011
 */

#include <unur_source.h>

/* Copied from NumPy: https://github.com/numpy/numpy/blob/43c113dd7aa36ef833315035858ea98a7b4732c5/numpy/core/src/common/npy_config.h#L73 */
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(_WIN64)
#ifdef HAVE_DECL_HYPOT
#undef HAVE_DECL_HYPOT
#endif
#endif

/* Copied from NumPy's npy_hypot: https://github.com/numpy/numpy/blob/663094532de48e131793faaa0ba06eb7c051ab47/numpy/core/src/npymath/npy_math_internal.h.src#L191-L219 */
#if !HAVE_DECL_HYPOT
double _unur_hypot(double x, double y)
{
    double yx;

    x = fabs(x);
    y = fabs(y);
    if (x < y) {
        double temp = x;
        x = y;
        y = temp;
    }
    if (x == 0.) {
        return 0.;
    }
    else {
        yx = y/x;
        return x*sqrt(1.+yx*yx);
    }
} /* end of _unur_hypot() */
#endif
