// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! # SQL Parser for Rust
//!
//! This crate provides an ANSI:SQL 2011 lexer and parser that can parse SQL
//! into an Abstract Syntax Tree ([`AST`]). See the [sqlparser crates.io page]
//! for more information.
//!
//! For more information:
//! 1. [`Parser::parse_sql`] and [`Parser::new`] for the Parsing API
//! 2. [`ast`] for the AST structure
//! 3. [`Dialect`] for supported SQL dialects
//! 4. [`Spanned`] for source text locations (see "Source Spans" below for details)
//!
//! [`Spanned`]: ast::Spanned
//!
//! # Example parsing SQL text
//!
//! ```
//! use sqlparser::dialect::GenericDialect;
//! use sqlparser::parser::Parser;
//!
//! let dialect = GenericDialect {}; // or AnsiDialect
//!
//! let sql = "SELECT a, b, 123, myfunc(b) \
//!            FROM table_1 \
//!            WHERE a > b AND b < 100 \
//!            ORDER BY a DESC, b";
//!
//! let ast = Parser::parse_sql(&dialect, sql).unwrap();
//!
//! println!("AST: {:?}", ast);
//! ```
//!
//! # Creating SQL text from AST
//!
//! This crate allows users to recover the original SQL text (with comments
//! removed, normalized whitespace and identifier capitalization), which is
//! useful for tools that analyze and manipulate SQL.
//!
//! ```
//! # use sqlparser::dialect::GenericDialect;
//! # use sqlparser::parser::Parser;
//! let sql = "SELECT a FROM table_1";
//!
//! // parse to a Vec<Statement>
//! let ast = Parser::parse_sql(&GenericDialect, sql).unwrap();
//!
//! // The original SQL text can be generated from the AST
//! assert_eq!(ast[0].to_string(), sql);
//! ```
//! [sqlparser crates.io page]: https://crates.io/crates/sqlparser
//! [`Parser::parse_sql`]: crate::parser::Parser::parse_sql
//! [`Parser::new`]: crate::parser::Parser::new
//! [`AST`]: crate::ast
//! [`ast`]: crate::ast
//! [`Dialect`]: crate::dialect::Dialect
//!
//! # Source Spans
//!
//! Starting with version  `0.53.0` sqlparser introduced source spans to the
//! AST. This feature provides source information for syntax errors, enabling
//! better error messages. See [issue #1548] for more information and the
//! [`Spanned`] trait to access the spans.
//!
//! [issue #1548]: https://github.com/apache/datafusion-sqlparser-rs/issues/1548
//! [`Spanned`]: ast::Spanned
//!
//! ## Migration Guide
//!
//! For the next few releases, we will be incrementally adding source spans to the
//! AST nodes, trying to minimize the impact on existing users. Some breaking
//! changes are inevitable, and the following is a summary of the changes:
//!
//! #### New fields for spans (must be added to any existing pattern matches)
//!
//! The primary change is that new fields will be added to AST nodes to store the source `Span` or `TokenWithLocation`.
//!
//! This will require
//! 1. Adding new fields to existing pattern matches.
//! 2. Filling in the proper span information when constructing AST nodes.
//!
//! For example, since `Ident` now stores a `Span`, to construct an `Ident` you
//! must provide now provide one:
//!
//! Previously:
//! ```text
//! # use sqlparser::ast::Ident;
//! Ident {
//!     value: "name".into(),
//!     quote_style: None,
//! }
//! ```
//! Now
//! ```rust
//! # use sqlparser::ast::Ident;
//! # use sqlparser::tokenizer::Span;
//! Ident {
//!     value: "name".into(),
//!     quote_style: None,
//!     span: Span::empty(),
//! };
//! ```
//!
//! Similarly, when pattern matching on `Ident`, you must now account for the
//! `span` field.
//!
//! #### Misc.
//! - [`TokenWithLocation`] stores a full `Span`, rather than just a source location.
//!   Users relying on `token.location` should use `token.location.start` instead.
//!
//![`TokenWithLocation`]: tokenizer::TokenWithLocation

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::upper_case_acronyms)]

// Allow proc-macros to find this crate
extern crate self as sqlparser;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[macro_use]
#[cfg(test)]
extern crate pretty_assertions;

pub mod ast;
#[macro_use]
pub mod dialect;
pub mod keywords;
pub mod parser;
pub mod tokenizer;

#[doc(hidden)]
// This is required to make utilities accessible by both the crate-internal
// unit-tests and by the integration tests <https://stackoverflow.com/a/44541071/1026>
// External users are not supposed to rely on this module.
pub mod test_utils;
