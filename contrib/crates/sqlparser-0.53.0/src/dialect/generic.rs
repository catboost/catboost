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

/// A permissive, general purpose [`Dialect`], which parses a wide variety of SQL
/// statements, from many different dialects.
#[derive(Debug, Default)]
pub struct GenericDialect;

impl Dialect for GenericDialect {
    fn is_delimited_identifier_start(&self, ch: char) -> bool {
        ch == '"' || ch == '`'
    }

    fn is_identifier_start(&self, ch: char) -> bool {
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

    fn supports_unicode_string_literal(&self) -> bool {
        true
    }

    fn supports_group_by_expr(&self) -> bool {
        true
    }

    fn supports_connect_by(&self) -> bool {
        true
    }

    fn supports_match_recognize(&self) -> bool {
        true
    }

    fn supports_start_transaction_modifier(&self) -> bool {
        true
    }

    fn supports_window_function_null_treatment_arg(&self) -> bool {
        true
    }

    fn supports_dictionary_syntax(&self) -> bool {
        true
    }

    fn supports_window_clause_named_window_reference(&self) -> bool {
        true
    }

    fn supports_parenthesized_set_variables(&self) -> bool {
        true
    }

    fn supports_select_wildcard_except(&self) -> bool {
        true
    }

    fn support_map_literal_syntax(&self) -> bool {
        true
    }

    fn allow_extract_custom(&self) -> bool {
        true
    }

    fn allow_extract_single_quotes(&self) -> bool {
        true
    }

    fn supports_create_index_with_clause(&self) -> bool {
        true
    }

    fn supports_explain_with_utility_options(&self) -> bool {
        true
    }

    fn supports_limit_comma(&self) -> bool {
        true
    }

    fn supports_asc_desc_in_column_definition(&self) -> bool {
        true
    }

    fn supports_try_convert(&self) -> bool {
        true
    }

    fn supports_comment_on(&self) -> bool {
        true
    }

    fn supports_load_extension(&self) -> bool {
        true
    }

    fn supports_named_fn_args_with_assignment_operator(&self) -> bool {
        true
    }

    fn supports_struct_literal(&self) -> bool {
        true
    }
}
