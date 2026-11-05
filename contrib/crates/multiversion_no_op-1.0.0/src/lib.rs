// Copyright Mozilla Foundation
//
// Licensed under the Apache License (Version 2.0), or the MIT license,
// (the "Licenses") at your option. You may not use this file except in
// compliance with one of the Licenses. You may obtain copies of the
// Licenses at:
//
//    https://www.apache.org/licenses/LICENSE-2.0
//    https://opensource.org/licenses/MIT
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the Licenses is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the Licenses for the specific language governing permissions and
// limitations under the Licenses.

#![no_std]

//! # `multiversion_no_op`
//!
//! This crate provides a pass-through function attribute called `multiversion`.
//!
//! The purpose of this crate is to optimize build times on targets that do not actually
//! need multiversioning. That is, targets that need multiversioning can use the real
//! [`multiversion`](https://crates.io/crates/multiversion) crate, and targets that do
//! not need multiversioning can use this crate instead.
//!
//! This crate works in the `no_std` context.
//!
//! ## License
//!
//! Apache-2.0 OR MIT; the file named `COPYRIGHT`.
//!
//! ## Usage
//!
//! See [`encoding_rs`](https://github.com/hsivonen/encoding_rs/) crate for a usage example.

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn multiversion(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}
