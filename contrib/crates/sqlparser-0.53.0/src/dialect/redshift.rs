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

use crate::dialect::Dialect;
use core::iter::Peekable;
use core::str::Chars;

use super::PostgreSqlDialect;

/// A [`Dialect`] for [RedShift](https://aws.amazon.com/redshift/)
#[derive(Debug)]
pub struct RedshiftSqlDialect {}

// In most cases the redshift dialect is identical to [`PostgresSqlDialect`].
//
// Notable differences:
// 1. Redshift treats brackets `[` and `]` differently. For example, `SQL SELECT a[1][2] FROM b`
// in the Postgres dialect, the query will be parsed as an array, while in the Redshift dialect it will
// be a json path
impl Dialect for RedshiftSqlDialect {
    fn is_delimited_identifier_start(&self, ch: char) -> bool {
        ch == '"' || ch == '['
    }

    /// Determine if quoted characters are proper for identifier
    /// It's needed to distinguish treating square brackets as quotes from
    /// treating them as json path. If there is identifier then we assume
    /// there is no json path.
    fn is_proper_identifier_inside_quotes(&self, mut chars: Peekable<Chars<'_>>) -> bool {
        chars.next();
        let mut not_white_chars = chars.skip_while(|ch| ch.is_whitespace()).peekable();
        if let Some(&ch) = not_white_chars.peek() {
            return self.is_identifier_start(ch);
        }
        false
    }

    fn is_identifier_start(&self, ch: char) -> bool {
        // Extends Postgres dialect with sharp
        PostgreSqlDialect {}.is_identifier_start(ch) || ch == '#'
    }

    fn is_identifier_part(&self, ch: char) -> bool {
        // Extends Postgres dialect with sharp
        PostgreSqlDialect {}.is_identifier_part(ch) || ch == '#'
    }

    /// redshift has `CONVERT(type, value)` instead of `CONVERT(value, type)`
    /// <https://docs.aws.amazon.com/redshift/latest/dg/r_CONVERT_function.html>
    fn convert_type_before_value(&self) -> bool {
        true
    }

    fn supports_connect_by(&self) -> bool {
        true
    }

    /// Redshift expects the `TOP` option before the `ALL/DISTINCT` option:
    /// <https://docs.aws.amazon.com/redshift/latest/dg/r_SELECT_list.html#r_SELECT_list-parameters>
    fn supports_top_before_distinct(&self) -> bool {
        true
    }

    /// Redshift supports PartiQL: <https://docs.aws.amazon.com/redshift/latest/dg/super-overview.html>
    fn supports_partiql(&self) -> bool {
        true
    }
}
