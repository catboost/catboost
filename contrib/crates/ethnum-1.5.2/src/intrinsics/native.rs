//! This module contains native implementations for intrinsics. These are used
//! when generated IR intrinsics are disabled.

mod add;
mod ctz;
mod divmod;
mod mul;
mod rot;
mod shl;
mod shr;
mod sub;

pub use self::{add::*, ctz::*, divmod::*, mul::*, rot::*, shl::*, shr::*, sub::*};
