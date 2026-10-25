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

use core::fmt;

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "visitor")]
use sqlparser_derive::{Visit, VisitMut};

use super::display_separated;

/// Unary operators
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum UnaryOperator {
    /// Plus, e.g. `+9`
    Plus,
    /// Minus, e.g. `-9`
    Minus,
    /// Not, e.g. `NOT(true)`
    Not,
    /// Bitwise Not, e.g. `~9` (PostgreSQL-specific)
    PGBitwiseNot,
    /// Square root, e.g. `|/9` (PostgreSQL-specific)
    PGSquareRoot,
    /// Cube root, e.g. `||/27` (PostgreSQL-specific)
    PGCubeRoot,
    /// Factorial, e.g. `9!` (PostgreSQL-specific)
    PGPostfixFactorial,
    /// Factorial, e.g. `!!9` (PostgreSQL-specific)
    PGPrefixFactorial,
    /// Absolute value, e.g. `@ -9` (PostgreSQL-specific)
    PGAbs,
    /// Unary logical not operator: e.g. `! false` (Hive-specific)
    BangNot,
}

impl fmt::Display for UnaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            UnaryOperator::Plus => "+",
            UnaryOperator::Minus => "-",
            UnaryOperator::Not => "NOT",
            UnaryOperator::PGBitwiseNot => "~",
            UnaryOperator::PGSquareRoot => "|/",
            UnaryOperator::PGCubeRoot => "||/",
            UnaryOperator::PGPostfixFactorial => "!",
            UnaryOperator::PGPrefixFactorial => "!!",
            UnaryOperator::PGAbs => "@",
            UnaryOperator::BangNot => "!",
        })
    }
}

/// Binary operators
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum BinaryOperator {
    /// Plus, e.g. `a + b`
    Plus,
    /// Minus, e.g. `a - b`
    Minus,
    /// Multiply, e.g. `a * b`
    Multiply,
    /// Divide, e.g. `a / b`
    Divide,
    /// Modulo, e.g. `a % b`
    Modulo,
    /// String/Array Concat operator, e.g. `a || b`
    StringConcat,
    /// Greater than, e.g. `a > b`
    Gt,
    /// Less than, e.g. `a < b`
    Lt,
    /// Greater equal, e.g. `a >= b`
    GtEq,
    /// Less equal, e.g. `a <= b`
    LtEq,
    /// Spaceship, e.g. `a <=> b`
    Spaceship,
    /// Equal, e.g. `a = b`
    Eq,
    /// Not equal, e.g. `a <> b`
    NotEq,
    /// And, e.g. `a AND b`
    And,
    /// Or, e.g. `a OR b`
    Or,
    /// XOR, e.g. `a XOR b`
    Xor,
    /// Bitwise or, e.g. `a | b`
    BitwiseOr,
    /// Bitwise and, e.g. `a & b`
    BitwiseAnd,
    /// Bitwise XOR, e.g. `a ^ b`
    BitwiseXor,
    /// Integer division operator `//` in DuckDB
    DuckIntegerDivide,
    /// MySQL [`DIV`](https://dev.mysql.com/doc/refman/8.0/en/arithmetic-functions.html) integer division
    MyIntegerDivide,
    /// Support for custom operators (such as Postgres custom operators)
    Custom(String),
    /// Bitwise XOR, e.g. `a # b` (PostgreSQL-specific)
    PGBitwiseXor,
    /// Bitwise shift left, e.g. `a << b` (PostgreSQL-specific)
    PGBitwiseShiftLeft,
    /// Bitwise shift right, e.g. `a >> b` (PostgreSQL-specific)
    PGBitwiseShiftRight,
    /// Exponent, e.g. `a ^ b` (PostgreSQL-specific)
    PGExp,
    /// Overlap operator, e.g. `a && b` (PostgreSQL-specific)
    PGOverlap,
    /// String matches regular expression (case sensitively), e.g. `a ~ b` (PostgreSQL-specific)
    PGRegexMatch,
    /// String matches regular expression (case insensitively), e.g. `a ~* b` (PostgreSQL-specific)
    PGRegexIMatch,
    /// String does not match regular expression (case sensitively), e.g. `a !~ b` (PostgreSQL-specific)
    PGRegexNotMatch,
    /// String does not match regular expression (case insensitively), e.g. `a !~* b` (PostgreSQL-specific)
    PGRegexNotIMatch,
    /// String matches pattern (case sensitively), e.g. `a ~~ b` (PostgreSQL-specific)
    PGLikeMatch,
    /// String matches pattern (case insensitively), e.g. `a ~~* b` (PostgreSQL-specific)
    PGILikeMatch,
    /// String does not match pattern (case sensitively), e.g. `a !~~ b` (PostgreSQL-specific)
    PGNotLikeMatch,
    /// String does not match pattern (case insensitively), e.g. `a !~~* b` (PostgreSQL-specific)
    PGNotILikeMatch,
    /// String "starts with", eg: `a ^@ b` (PostgreSQL-specific)
    PGStartsWith,
    /// The `->` operator.
    ///
    /// On PostgreSQL, this operator extracts a JSON object field or array
    /// element, for example `'{"a":"b"}'::json -> 'a'` or `[1, 2, 3]'::json
    /// -> 2`.
    ///
    /// See <https://www.postgresql.org/docs/current/functions-json.html>.
    Arrow,
    /// The `->>` operator.
    ///
    /// On PostgreSQL, this operator extracts a JSON object field or JSON
    /// array element and converts it to text, for example `'{"a":"b"}'::json
    /// ->> 'a'` or `[1, 2, 3]'::json ->> 2`.
    ///
    /// See <https://www.postgresql.org/docs/current/functions-json.html>.
    LongArrow,
    /// The `#>` operator.
    ///
    /// On PostgreSQL, this operator extracts a JSON sub-object at the specified
    /// path, for example:
    ///
    /// ```notrust
    ///'{"a": {"b": ["foo","bar"]}}'::json #> '{a,b,1}'
    /// ```
    ///
    /// See <https://www.postgresql.org/docs/current/functions-json.html>.
    HashArrow,
    /// The `#>>` operator.
    ///
    /// A PostgreSQL-specific operator that extracts JSON sub-object at the
    /// specified path, for example
    ///
    /// ```notrust
    ///'{"a": {"b": ["foo","bar"]}}'::json #>> '{a,b,1}'
    /// ```
    ///
    /// See <https://www.postgresql.org/docs/current/functions-json.html>.
    HashLongArrow,
    /// The `@@` operator.
    ///
    /// On PostgreSQL, this is used for JSON and text searches.
    ///
    /// See <https://www.postgresql.org/docs/current/functions-json.html>.
    /// See <https://www.postgresql.org/docs/current/functions-textsearch.html>.
    AtAt,
    /// The `@>` operator.
    ///
    /// On PostgreSQL, this is used for JSON and text searches.
    ///
    /// See <https://www.postgresql.org/docs/current/functions-json.html>.
    /// See <https://www.postgresql.org/docs/current/functions-textsearch.html>.
    AtArrow,
    /// The `<@` operator.
    ///
    /// On PostgreSQL, this is used for JSON and text searches.
    ///
    /// See <https://www.postgresql.org/docs/current/functions-json.html>.
    /// See <https://www.postgresql.org/docs/current/functions-textsearch.html>.
    ArrowAt,
    /// The `#-` operator.
    ///
    /// On PostgreSQL, this operator is used to delete a field or array element
    /// at a specified path.
    ///
    /// See <https://www.postgresql.org/docs/current/functions-json.html>.
    HashMinus,
    /// The `@?` operator.
    ///
    /// On PostgreSQL, this operator is used to check the given JSON path
    /// returns an item for the JSON value.
    ///
    /// See <https://www.postgresql.org/docs/current/functions-json.html>.
    AtQuestion,
    /// The `?` operator.
    ///
    /// On PostgreSQL, this operator is used to check whether a string exists as a top-level key
    /// within the JSON value
    ///
    /// See <https://www.postgresql.org/docs/current/functions-json.html>.
    Question,
    /// The `?&` operator.
    ///
    /// On PostgreSQL, this operator is used to check whether all of the the indicated array
    /// members exist as top-level keys.
    ///
    /// See <https://www.postgresql.org/docs/current/functions-json.html>.
    QuestionAnd,
    /// The `?|` operator.
    ///
    /// On PostgreSQL, this operator is used to check whether any of the the indicated array
    /// members exist as top-level keys.
    ///
    /// See <https://www.postgresql.org/docs/current/functions-json.html>.
    QuestionPipe,
    /// PostgreSQL-specific custom operator.
    ///
    /// See [CREATE OPERATOR](https://www.postgresql.org/docs/current/sql-createoperator.html)
    /// for more information.
    PGCustomBinaryOperator(Vec<String>),
}

impl fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BinaryOperator::Plus => f.write_str("+"),
            BinaryOperator::Minus => f.write_str("-"),
            BinaryOperator::Multiply => f.write_str("*"),
            BinaryOperator::Divide => f.write_str("/"),
            BinaryOperator::Modulo => f.write_str("%"),
            BinaryOperator::StringConcat => f.write_str("||"),
            BinaryOperator::Gt => f.write_str(">"),
            BinaryOperator::Lt => f.write_str("<"),
            BinaryOperator::GtEq => f.write_str(">="),
            BinaryOperator::LtEq => f.write_str("<="),
            BinaryOperator::Spaceship => f.write_str("<=>"),
            BinaryOperator::Eq => f.write_str("="),
            BinaryOperator::NotEq => f.write_str("<>"),
            BinaryOperator::And => f.write_str("AND"),
            BinaryOperator::Or => f.write_str("OR"),
            BinaryOperator::Xor => f.write_str("XOR"),
            BinaryOperator::BitwiseOr => f.write_str("|"),
            BinaryOperator::BitwiseAnd => f.write_str("&"),
            BinaryOperator::BitwiseXor => f.write_str("^"),
            BinaryOperator::DuckIntegerDivide => f.write_str("//"),
            BinaryOperator::MyIntegerDivide => f.write_str("DIV"),
            BinaryOperator::Custom(s) => f.write_str(s),
            BinaryOperator::PGBitwiseXor => f.write_str("#"),
            BinaryOperator::PGBitwiseShiftLeft => f.write_str("<<"),
            BinaryOperator::PGBitwiseShiftRight => f.write_str(">>"),
            BinaryOperator::PGExp => f.write_str("^"),
            BinaryOperator::PGOverlap => f.write_str("&&"),
            BinaryOperator::PGRegexMatch => f.write_str("~"),
            BinaryOperator::PGRegexIMatch => f.write_str("~*"),
            BinaryOperator::PGRegexNotMatch => f.write_str("!~"),
            BinaryOperator::PGRegexNotIMatch => f.write_str("!~*"),
            BinaryOperator::PGLikeMatch => f.write_str("~~"),
            BinaryOperator::PGILikeMatch => f.write_str("~~*"),
            BinaryOperator::PGNotLikeMatch => f.write_str("!~~"),
            BinaryOperator::PGNotILikeMatch => f.write_str("!~~*"),
            BinaryOperator::PGStartsWith => f.write_str("^@"),
            BinaryOperator::Arrow => f.write_str("->"),
            BinaryOperator::LongArrow => f.write_str("->>"),
            BinaryOperator::HashArrow => f.write_str("#>"),
            BinaryOperator::HashLongArrow => f.write_str("#>>"),
            BinaryOperator::AtAt => f.write_str("@@"),
            BinaryOperator::AtArrow => f.write_str("@>"),
            BinaryOperator::ArrowAt => f.write_str("<@"),
            BinaryOperator::HashMinus => f.write_str("#-"),
            BinaryOperator::AtQuestion => f.write_str("@?"),
            BinaryOperator::Question => f.write_str("?"),
            BinaryOperator::QuestionAnd => f.write_str("?&"),
            BinaryOperator::QuestionPipe => f.write_str("?|"),
            BinaryOperator::PGCustomBinaryOperator(idents) => {
                write!(f, "OPERATOR({})", display_separated(idents, "."))
            }
        }
    }
}
