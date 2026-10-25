//! Example of rotation with Euler angles.
//!
//! This is an example of some coordinate transformations (using Euler angles)
//! for illustrative purposes. Note that other crates such as
//! [`cgmath`](https://crates.io/crates/cgmath) or
//! [`nalgebra`](https://crates.io/crates/nalgebra) may be better-suited if
//! most of your work is coordinate transformations, since they have built-in
//! geometry primitives.
//!
//! This is the original Python program:
//!
//! ```python
//! # Euler angles (rows) for four coordinate systems (columns).
//! nelems = 4
//! bunge = np.ones((3, nelems))
//!
//! # Precompute sines and cosines
//! s1 = np.sin(bunge[0, :])
//! c1 = np.cos(bunge[0, :])
//! s2 = np.sin(bunge[1, :])
//! c2 = np.cos(bunge[1, :])
//! s3 = np.sin(bunge[2, :])
//! c3 = np.cos(bunge[2, :])
//!
//! # Rotation matrices.
//! rmat = np.zeros((3, 3, nelems), order='F')
//! for i in range(nelems):
//!     rmat[0, 0, i] = c1[i] * c3[i] - s1[i] * s3[i] * c2[i]
//!     rmat[0, 1, i] = -c1[i] * s3[i] - s1[i] * c2[i] * c3[i]
//!     rmat[0, 2, i] = s1[i] * s2[i]
//!
//!     rmat[1, 0, i] = s1[i] * c3[i] + c1[i] * c2[i] * s3[i]
//!     rmat[1, 1, i] = -s1[i] * s3[i] + c1[i] * c2[i] * c3[i]
//!     rmat[1, 2, i] = -c1[i] * s2[i]
//!
//!     rmat[2, 0, i] = s2[i] * s3[i]
//!     rmat[2, 1, i] = s2[i] * c3[i]
//!     rmat[2, 2, i] = c2[i]
//!
//! # Unit vectors of coordinate systems to rotate.
//! eye2d = np.eye(3)
//!
//! # Unit vectors after rotation.
//! rotated = np.zeros((3, 3, nelems), order='F')
//! for i in range(nelems):
//!     rotated[:,:,i] = rmat[:,:,i].dot(eye2d)
//! ```
//!
//! This is a direct translation to `ndarray`:
//!
//! ```
//! use ndarray::prelude::*;
//!
//! let nelems = 4;
//! let bunge = Array::ones((3, nelems));
//!
//! let s1 = bunge.slice(s![0, ..]).mapv(f64::sin);
//! let c1 = bunge.slice(s![0, ..]).mapv(f64::cos);
//! let s2 = bunge.slice(s![1, ..]).mapv(f64::sin);
//! let c2 = bunge.slice(s![1, ..]).mapv(f64::cos);
//! let s3 = bunge.slice(s![2, ..]).mapv(f64::sin);
//! let c3 = bunge.slice(s![2, ..]).mapv(f64::cos);
//!
//! let mut rmat = Array::zeros((3, 3, nelems).f());
//! for i in 0..nelems {
//!     rmat[[0, 0, i]] = c1[i] * c3[i] - s1[i] * s3[i] * c2[i];
//!     rmat[[0, 1, i]] = -c1[i] * s3[i] - s1[i] * c2[i] * c3[i];
//!     rmat[[0, 2, i]] = s1[i] * s2[i];
//!
//!     rmat[[1, 0, i]] = s1[i] * c3[i] + c1[i] * c2[i] * s3[i];
//!     rmat[[1, 1, i]] = -s1[i] * s3[i] + c1[i] * c2[i] * c3[i];
//!     rmat[[1, 2, i]] = -c1[i] * s2[i];
//!
//!     rmat[[2, 0, i]] = s2[i] * s3[i];
//!     rmat[[2, 1, i]] = s2[i] * c3[i];
//!     rmat[[2, 2, i]] = c2[i];
//! }
//!
//! let eye2d = Array::eye(3);
//!
//! let mut rotated = Array::zeros((3, 3, nelems).f());
//! for i in 0..nelems {
//!     rotated
//!         .slice_mut(s![.., .., i])
//!         .assign(&rmat.slice(s![.., .., i]).dot(&eye2d));
//! }
//! ```
//!
//! Instead of looping over indices, a cleaner (and usually faster) option is
//! to zip arrays together. It's also possible to avoid some of the temporary
//! memory allocations in the original program. The improved version looks like
//! this:
//!
//! ```
//! use ndarray::prelude::*;
//!
//! let nelems = 4;
//! let bunge = Array2::<f64>::ones((3, nelems));
//!
//! let mut rmat = Array::zeros((3, 3, nelems).f());
//! azip!((mut rmat in rmat.axis_iter_mut(Axis(2)), bunge in bunge.axis_iter(Axis(1))) {
//!     let s1 = bunge[0].sin();
//!     let c1 = bunge[0].cos();
//!     let s2 = bunge[1].sin();
//!     let c2 = bunge[1].cos();
//!     let s3 = bunge[2].sin();
//!     let c3 = bunge[2].cos();
//!
//!     rmat[[0, 0]] = c1 * c3 - s1 * s3 * c2;
//!     rmat[[0, 1]] = -c1 * s3 - s1 * c2 * c3;
//!     rmat[[0, 2]] = s1 * s2;
//!
//!     rmat[[1, 0]] = s1 * c3 + c1 * c2 * s3;
//!     rmat[[1, 1]] = -s1 * s3 + c1 * c2 * c3;
//!     rmat[[1, 2]] = -c1 * s2;
//!
//!     rmat[[2, 0]] = s2 * s3;
//!     rmat[[2, 1]] = s2 * c3;
//!     rmat[[2, 2]] = c2;
//! });
//!
//! let eye2d = Array2::<f64>::eye(3);
//!
//! let mut rotated = Array3::<f64>::zeros((3, 3, nelems).f());
//! azip!((mut rotated in rotated.axis_iter_mut(Axis(2)), rmat in rmat.axis_iter(Axis(2))) {
//!     rotated.assign(&rmat.dot(&eye2d));
//! });
//! ```
