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

use core::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use core::fmt::{self, Debug, Formatter};
use core::hash::{Hash, Hasher};

use crate::tokenizer::TokenWithSpan;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "visitor")]
use sqlparser_derive::{Visit, VisitMut};

/// A wrapper over [`TokenWithSpan`]s that ignores the token and source
/// location in comparisons and hashing.
///
/// This type is used when the token and location is not relevant for semantics,
/// but is still needed for accurate source location tracking, for example, in
/// the nodes in the [ast](crate::ast) module.
///
/// Note: **All** `AttachedTokens` are equal.
///
/// # Examples
///
/// Same token, different location are equal
/// ```
/// # use sqlparser::ast::helpers::attached_token::AttachedToken;
/// # use sqlparser::tokenizer::{Location, Span, Token, TokenWithLocation};
/// // commas @ line 1, column 10
/// let tok1 = TokenWithLocation::new(
///   Token::Comma,
///   Span::new(Location::new(1, 10), Location::new(1, 11)),
/// );
/// // commas @ line 2, column 20
/// let tok2 = TokenWithLocation::new(
///   Token::Comma,
///   Span::new(Location::new(2, 20), Location::new(2, 21)),
/// );
///
/// assert_ne!(tok1, tok2); // token with locations are *not* equal
/// assert_eq!(AttachedToken(tok1), AttachedToken(tok2)); // attached tokens are
/// ```
///
/// Different token, different location are equal 🤯
///
/// ```
/// # use sqlparser::ast::helpers::attached_token::AttachedToken;
/// # use sqlparser::tokenizer::{Location, Span, Token, TokenWithLocation};
/// // commas @ line 1, column 10
/// let tok1 = TokenWithLocation::new(
///   Token::Comma,
///   Span::new(Location::new(1, 10), Location::new(1, 11)),
/// );
/// // period @ line 2, column 20
/// let tok2 = TokenWithLocation::new(
///  Token::Period,
///   Span::new(Location::new(2, 10), Location::new(2, 21)),
/// );
///
/// assert_ne!(tok1, tok2); // token with locations are *not* equal
/// assert_eq!(AttachedToken(tok1), AttachedToken(tok2)); // attached tokens are
/// ```
/// // period @ line 2, column 20
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct AttachedToken(pub TokenWithSpan);

impl AttachedToken {
    /// Return a new Empty AttachedToken
    pub fn empty() -> Self {
        AttachedToken(TokenWithSpan::new_eof())
    }
}

// Conditional Implementations
impl Debug for AttachedToken {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

// Blanket Implementations
impl PartialEq for AttachedToken {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for AttachedToken {}

impl PartialOrd for AttachedToken {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AttachedToken {
    fn cmp(&self, _: &Self) -> Ordering {
        Ordering::Equal
    }
}

impl Hash for AttachedToken {
    fn hash<H: Hasher>(&self, _state: &mut H) {
        // Do nothing
    }
}

impl From<TokenWithSpan> for AttachedToken {
    fn from(value: TokenWithSpan) -> Self {
        AttachedToken(value)
    }
}

impl From<AttachedToken> for TokenWithSpan {
    fn from(value: AttachedToken) -> Self {
        value.0
    }
}
