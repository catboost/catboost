// Copyright 2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Linear algebra.

pub use self::impl_linalg::general_mat_mul;
pub use self::impl_linalg::general_mat_vec_mul;
pub use self::impl_linalg::kron;
pub use self::impl_linalg::Dot;

mod impl_linalg;
