// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use core::{
    fmt::{self, Write},
    iter::FusedIterator,
};
use tinyvec::ArrayVec;

/// External iterator for replacements for a string's characters.
#[derive(Clone)]
pub struct Replacements<I> {
    iter: I,
    // At this time, the longest replacement sequence has length 2, so we just
    // need buffer space for 1 codepoint.
    buffer: Option<char>,
}

impl<I: Iterator<Item = char>> Replacements<I> {
    /// Create a new iterator that replaces [CJK Compatibility Ideograph] codepoints with normal forms using [Standardized Variation Sequences].
    ///
    /// Note that this iterator can also be obtained by directly calling [`.cjk_compat_variants()`] on the iterator.
    ///
    /// [CJK Compatibility Ideograph]: https://www.unicode.org/glossary/#compatibility_ideograph
    /// [Standardized Variation Sequences]: https://www.unicode.org/glossary/#standardized_variation_sequence
    /// [`.cjk_compat_variants()`]: crate::UnicodeNormalization::cjk_compat_variants
    #[inline]
    pub fn new_cjk_compat_variants(iter: I) -> Replacements<I> {
        Replacements { iter, buffer: None }
    }
}

impl<I: Iterator<Item = char>> Iterator for Replacements<I> {
    type Item = char;

    #[inline]
    fn next(&mut self) -> Option<char> {
        if let Some(c) = self.buffer.take() {
            return Some(c);
        }

        match self.iter.next() {
            Some(ch) => {
                // At this time, the longest replacement sequence has length 2.
                let mut buffer = ArrayVec::<[char; 2]>::new();
                super::char::decompose_cjk_compat_variants(ch, |d| buffer.push(d));
                self.buffer = buffer.get(1).copied();
                Some(buffer[0])
            }
            None => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, _) = self.iter.size_hint();
        (lower, None)
    }
}

impl<I: Iterator<Item = char> + FusedIterator> FusedIterator for Replacements<I> {}

impl<I: Iterator<Item = char> + Clone> fmt::Display for Replacements<I> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for c in self.clone() {
            f.write_char(c)?;
        }
        Ok(())
    }
}
