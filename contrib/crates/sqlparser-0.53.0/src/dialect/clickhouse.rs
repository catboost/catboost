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

// A [`Dialect`] for [ClickHouse](https://clickhouse.com/).
#[derive(Debug)]
pub struct ClickHouseDialect {}

impl Dialect for ClickHouseDialect {
    fn is_identifier_start(&self, ch: char) -> bool {
        // See https://clickhouse.com/docs/en/sql-reference/syntax/#syntax-identifiers
        ch.is_ascii_lowercase() || ch.is_ascii_uppercase() || ch == '_'
    }

    fn is_identifier_part(&self, ch: char) -> bool {
        self.is_identifier_start(ch) || ch.is_ascii_digit()
    }

    fn supports_string_literal_backslash_escape(&self) -> bool {
        true
    }

    fn supports_select_wildcard_except(&self) -> bool {
        true
    }

    fn describe_requires_table_keyword(&self) -> bool {
        true
    }

    fn require_interval_qualifier(&self) -> bool {
        true
    }

    fn supports_limit_comma(&self) -> bool {
        true
    }
}
