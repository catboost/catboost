// Copyright 2021 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Guard value that will abort if it is dropped.
/// To defuse, this value must be forgotten before the end of the scope.
///
/// The string value is added to the message printed if aborting.
#[must_use]
pub(crate) struct AbortIfPanic(pub(crate) &'static &'static str);

impl AbortIfPanic
{
    /// Defuse the AbortIfPanic guard. This *must* be done when finished.
    #[inline]
    pub(crate) fn defuse(self)
    {
        std::mem::forget(self);
    }
}

impl Drop for AbortIfPanic
{
    // The compiler should be able to remove this, if it can see through that there
    // is no panic in the code section.
    fn drop(&mut self)
    {
        #[cfg(feature = "std")]
        {
            eprintln!("ndarray: panic in no-panic section, aborting: {}", self.0);
            std::process::abort()
        }
        #[cfg(not(feature = "std"))]
        {
            // no-std uses panic-in-panic (should abort)
            panic!("ndarray: panic in no-panic section, bailing out: {}", self.0);
        }
    }
}
