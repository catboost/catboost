#[cfg(feature = "approx")]
mod approx_methods
{
    use crate::imp_prelude::*;

    impl<A, S, D> ArrayBase<S, D>
    where
        S: Data<Elem = A>,
        D: Dimension,
    {
        /// A test for equality that uses the elementwise absolute difference to compute the
        /// approximate equality of two arrays.
        ///
        /// **Requires crate feature `"approx"`**
        pub fn abs_diff_eq<S2>(&self, other: &ArrayBase<S2, D>, epsilon: A::Epsilon) -> bool
        where
            A: ::approx::AbsDiffEq<S2::Elem>,
            A::Epsilon: Clone,
            S2: Data,
        {
            <Self as ::approx::AbsDiffEq<_>>::abs_diff_eq(self, other, epsilon)
        }

        /// A test for equality that uses an elementwise relative comparison if the values are far
        /// apart; and the absolute difference otherwise.
        ///
        /// **Requires crate feature `"approx"`**
        pub fn relative_eq<S2>(&self, other: &ArrayBase<S2, D>, epsilon: A::Epsilon, max_relative: A::Epsilon) -> bool
        where
            A: ::approx::RelativeEq<S2::Elem>,
            A::Epsilon: Clone,
            S2: Data,
        {
            <Self as ::approx::RelativeEq<_>>::relative_eq(self, other, epsilon, max_relative)
        }
    }
}

macro_rules! impl_approx_traits {
    ($approx:ident, $doc:expr) => {
        mod $approx {
            use crate::imp_prelude::*;
            use crate::Zip;
            use $approx::{AbsDiffEq, RelativeEq, UlpsEq};

            #[doc = $doc]
            impl<A, B, S, S2, D> AbsDiffEq<ArrayBase<S2, D>> for ArrayBase<S, D>
            where
                A: AbsDiffEq<B>,
                A::Epsilon: Clone,
                S: Data<Elem = A>,
                S2: Data<Elem = B>,
                D: Dimension,
            {
                type Epsilon = A::Epsilon;

                fn default_epsilon() -> A::Epsilon {
                    A::default_epsilon()
                }

                fn abs_diff_eq(&self, other: &ArrayBase<S2, D>, epsilon: A::Epsilon) -> bool {
                    if self.shape() != other.shape() {
                        return false;
                    }

                    Zip::from(self)
                        .and(other)
                        .all(move |a, b| A::abs_diff_eq(a, b, epsilon.clone()))
                }
            }

            #[doc = $doc]
            impl<A, B, S, S2, D> RelativeEq<ArrayBase<S2, D>> for ArrayBase<S, D>
            where
                A: RelativeEq<B>,
                A::Epsilon: Clone,
                S: Data<Elem = A>,
                S2: Data<Elem = B>,
                D: Dimension,
            {
                fn default_max_relative() -> A::Epsilon {
                    A::default_max_relative()
                }

                fn relative_eq(
                    &self,
                    other: &ArrayBase<S2, D>,
                    epsilon: A::Epsilon,
                    max_relative: A::Epsilon,
                ) -> bool {
                    if self.shape() != other.shape() {
                        return false;
                    }

                    Zip::from(self).and(other).all(move |a, b| {
                        A::relative_eq(a, b, epsilon.clone(), max_relative.clone())
                    })
                }
            }

            #[doc = $doc]
            impl<A, B, S, S2, D> UlpsEq<ArrayBase<S2, D>> for ArrayBase<S, D>
            where
                A: UlpsEq<B>,
                A::Epsilon: Clone,
                S: Data<Elem = A>,
                S2: Data<Elem = B>,
                D: Dimension,
            {
                fn default_max_ulps() -> u32 {
                    A::default_max_ulps()
                }

                fn ulps_eq(
                    &self,
                    other: &ArrayBase<S2, D>,
                    epsilon: A::Epsilon,
                    max_ulps: u32,
                ) -> bool {
                    if self.shape() != other.shape() {
                        return false;
                    }

                    Zip::from(self)
                        .and(other)
                        .all(move |a, b| A::ulps_eq(a, b, epsilon.clone(), max_ulps))
                }
            }

            #[cfg(test)]
            mod tests {
                use crate::prelude::*;
                use alloc::vec;
                use $approx::{
                    assert_abs_diff_eq, assert_abs_diff_ne, assert_relative_eq, assert_relative_ne,
                    assert_ulps_eq, assert_ulps_ne,
                };

                #[test]
                fn abs_diff_eq() {
                    let a: Array2<f32> = array![[0., 2.], [-0.000010001, 100000000.]];
                    let mut b: Array2<f32> = array![[0., 1.], [-0.000010002, 100000001.]];
                    assert_abs_diff_ne!(a, b);
                    b[(0, 1)] = 2.;
                    assert_abs_diff_eq!(a, b);

                    // Check epsilon.
                    assert_abs_diff_eq!(array![0.0f32], array![1e-40f32], epsilon = 1e-40f32);
                    assert_abs_diff_ne!(array![0.0f32], array![1e-40f32], epsilon = 1e-41f32);

                    // Make sure we can compare different shapes without failure.
                    let c = array![[1., 2.]];
                    assert_abs_diff_ne!(a, c);
                }

                #[test]
                fn relative_eq() {
                    let a: Array2<f32> = array![[1., 2.], [-0.000010001, 100000000.]];
                    let mut b: Array2<f32> = array![[1., 1.], [-0.000010002, 100000001.]];
                    assert_relative_ne!(a, b);
                    b[(0, 1)] = 2.;
                    assert_relative_eq!(a, b);

                    // Check epsilon.
                    assert_relative_eq!(array![0.0f32], array![1e-40f32], epsilon = 1e-40f32);
                    assert_relative_ne!(array![0.0f32], array![1e-40f32], epsilon = 1e-41f32);

                    // Make sure we can compare different shapes without failure.
                    let c = array![[1., 2.]];
                    assert_relative_ne!(a, c);
                }

                #[test]
                fn ulps_eq() {
                    let a: Array2<f32> = array![[1., 2.], [-0.000010001, 100000000.]];
                    let mut b: Array2<f32> = array![[1., 1.], [-0.000010002, 100000001.]];
                    assert_ulps_ne!(a, b);
                    b[(0, 1)] = 2.;
                    assert_ulps_eq!(a, b);

                    // Check epsilon.
                    assert_ulps_eq!(array![0.0f32], array![1e-40f32], epsilon = 1e-40f32);
                    assert_ulps_ne!(array![0.0f32], array![1e-40f32], epsilon = 1e-41f32);

                    // Make sure we can compare different shapes without failure.
                    let c = array![[1., 2.]];
                    assert_ulps_ne!(a, c);
                }
            }
        }
    };
}

#[cfg(feature = "approx")]
impl_approx_traits!(approx, "**Requires crate feature `\"approx\"`.**");
