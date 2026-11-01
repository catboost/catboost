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

/// A [`Dialect`] for [Microsoft SQL Server](https://www.microsoft.com/en-us/sql-server/)
#[derive(Debug)]
pub struct MsSqlDialect {}

impl Dialect for MsSqlDialect {
    fn is_delimited_identifier_start(&self, ch: char) -> bool {
        ch == '"' || ch == '['
    }

    fn is_identifier_start(&self, ch: char) -> bool {
        // See https://docs.microsoft.com/en-us/sql/relational-databases/databases/database-identifiers?view=sql-server-2017#rules-for-regular-identifiers
        ch.is_alphabetic() || ch == '_' || ch == '#' || ch == '@'
    }

    fn is_identifier_part(&self, ch: char) -> bool {
        ch.is_alphabetic()
            || ch.is_ascii_digit()
            || ch == '@'
            || ch == '$'
            || ch == '#'
            || ch == '_'
    }

    /// SQL Server has `CONVERT(type, value)` instead of `CONVERT(value, type)`
    /// <https://learn.microsoft.com/en-us/sql/t-sql/functions/cast-and-convert-transact-sql?view=sql-server-ver16>
    fn convert_type_before_value(&self) -> bool {
        true
    }

    fn supports_connect_by(&self) -> bool {
        true
    }

    fn supports_eq_alias_assignment(&self) -> bool {
        true
    }

    fn supports_try_convert(&self) -> bool {
        true
    }

    /// In MSSQL, there is no boolean type, and `true` and `false` are valid column names
    fn supports_boolean_literals(&self) -> bool {
        false
    }

    fn supports_methods(&self) -> bool {
        true
    }

    fn supports_named_fn_args_with_colon_operator(&self) -> bool {
        true
    }

    fn supports_named_fn_args_with_expr_name(&self) -> bool {
        true
    }

    fn supports_named_fn_args_with_rarrow_operator(&self) -> bool {
        false
    }
}
