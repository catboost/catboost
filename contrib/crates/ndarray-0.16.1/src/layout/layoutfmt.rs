// Copyright 2017 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::Layout;

const LAYOUT_NAMES: &[&str] = &["C", "F", "c", "f"];

use std::fmt;

impl fmt::Debug for Layout
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    {
        if self.0 == 0 {
            write!(f, "Custom")?
        } else {
            (0..32).filter(|&i| self.is(1 << i)).try_fold((), |_, i| {
                if let Some(name) = LAYOUT_NAMES.get(i) {
                    write!(f, "{}", name)
                } else {
                    write!(f, "{:#x}", i)
                }
            })?;
        };
        write!(f, " ({:#x})", self.0)
    }
}
