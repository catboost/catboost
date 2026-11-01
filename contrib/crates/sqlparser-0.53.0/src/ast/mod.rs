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

//! SQL Abstract Syntax Tree (AST) types
#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec::Vec,
};
use helpers::attached_token::AttachedToken;

use core::ops::Deref;
use core::{
    fmt::{self, Display},
    hash,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "visitor")]
use sqlparser_derive::{Visit, VisitMut};

use crate::tokenizer::Span;

pub use self::data_type::{
    ArrayElemTypeDef, CharLengthUnits, CharacterLength, DataType, EnumMember, ExactNumberInfo,
    StructBracketKind, TimezoneInfo,
};
pub use self::dcl::{
    AlterRoleOperation, ResetConfig, RoleOption, SecondaryRoles, SetConfigValue, Use,
};
pub use self::ddl::{
    AlterColumnOperation, AlterIndexOperation, AlterPolicyOperation, AlterTableOperation,
    ClusteredBy, ColumnDef, ColumnOption, ColumnOptionDef, ColumnPolicy, ColumnPolicyProperty,
    ConstraintCharacteristics, CreateFunction, Deduplicate, DeferrableInitial, GeneratedAs,
    GeneratedExpressionMode, IdentityParameters, IdentityProperty, IdentityPropertyFormatKind,
    IdentityPropertyKind, IdentityPropertyOrder, IndexOption, IndexType, KeyOrIndexDisplay,
    NullsDistinctOption, Owner, Partition, ProcedureParam, ReferentialAction, TableConstraint,
    TagsColumnOption, UserDefinedTypeCompositeAttributeDef, UserDefinedTypeRepresentation,
    ViewColumnDef,
};
pub use self::dml::{CreateIndex, CreateTable, Delete, Insert};
pub use self::operator::{BinaryOperator, UnaryOperator};
pub use self::query::{
    AfterMatchSkip, ConnectBy, Cte, CteAsMaterialized, Distinct, EmptyMatchesMode,
    ExceptSelectItem, ExcludeSelectItem, ExprWithAlias, Fetch, ForClause, ForJson, ForXml,
    FormatClause, GroupByExpr, GroupByWithModifier, IdentWithAlias, IlikeSelectItem, Interpolate,
    InterpolateExpr, Join, JoinConstraint, JoinOperator, JsonTableColumn,
    JsonTableColumnErrorHandling, JsonTableNamedColumn, JsonTableNestedColumn, LateralView,
    LockClause, LockType, MatchRecognizePattern, MatchRecognizeSymbol, Measure,
    NamedWindowDefinition, NamedWindowExpr, NonBlock, Offset, OffsetRows, OpenJsonTableColumn,
    OrderBy, OrderByExpr, PivotValueSource, ProjectionSelect, Query, RenameSelectItem,
    RepetitionQuantifier, ReplaceSelectElement, ReplaceSelectItem, RowsPerMatch, Select,
    SelectInto, SelectItem, SetExpr, SetOperator, SetQuantifier, Setting, SymbolDefinition, Table,
    TableAlias, TableAliasColumnDef, TableFactor, TableFunctionArgs, TableVersion, TableWithJoins,
    Top, TopQuantity, ValueTableMode, Values, WildcardAdditionalOptions, With, WithFill,
};

pub use self::trigger::{
    TriggerEvent, TriggerExecBody, TriggerExecBodyType, TriggerObject, TriggerPeriod,
    TriggerReferencing, TriggerReferencingType,
};

pub use self::value::{
    escape_double_quote_string, escape_quoted_string, DateTimeField, DollarQuotedString,
    TrimWhereField, Value,
};

use crate::ast::helpers::stmt_data_loading::{
    DataLoadingOptions, StageLoadSelectItem, StageParamsObject,
};
#[cfg(feature = "visitor")]
pub use visitor::*;

mod data_type;
mod dcl;
mod ddl;
mod dml;
pub mod helpers;
mod operator;
mod query;
mod spans;
pub use spans::Spanned;

mod trigger;
mod value;

#[cfg(feature = "visitor")]
mod visitor;

pub struct DisplaySeparated<'a, T>
where
    T: fmt::Display,
{
    slice: &'a [T],
    sep: &'static str,
}

impl<T> fmt::Display for DisplaySeparated<'_, T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut delim = "";
        for t in self.slice {
            write!(f, "{delim}")?;
            delim = self.sep;
            write!(f, "{t}")?;
        }
        Ok(())
    }
}

pub fn display_separated<'a, T>(slice: &'a [T], sep: &'static str) -> DisplaySeparated<'a, T>
where
    T: fmt::Display,
{
    DisplaySeparated { slice, sep }
}

pub fn display_comma_separated<T>(slice: &[T]) -> DisplaySeparated<'_, T>
where
    T: fmt::Display,
{
    DisplaySeparated { slice, sep: ", " }
}

/// An identifier, decomposed into its value or character data and the quote style.
#[derive(Debug, Clone, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct Ident {
    /// The value of the identifier without quotes.
    pub value: String,
    /// The starting quote if any. Valid quote characters are the single quote,
    /// double quote, backtick, and opening square bracket.
    pub quote_style: Option<char>,
    /// The span of the identifier in the original SQL string.
    pub span: Span,
}

impl PartialEq for Ident {
    fn eq(&self, other: &Self) -> bool {
        let Ident {
            value,
            quote_style,
            // exhaustiveness check; we ignore spans in comparisons
            span: _,
        } = self;

        value == &other.value && quote_style == &other.quote_style
    }
}

impl core::hash::Hash for Ident {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        let Ident {
            value,
            quote_style,
            // exhaustiveness check; we ignore spans in hashes
            span: _,
        } = self;

        value.hash(state);
        quote_style.hash(state);
    }
}

impl Eq for Ident {}

impl Ident {
    /// Create a new identifier with the given value and no quotes and an empty span.
    pub fn new<S>(value: S) -> Self
    where
        S: Into<String>,
    {
        Ident {
            value: value.into(),
            quote_style: None,
            span: Span::empty(),
        }
    }

    /// Create a new quoted identifier with the given quote and value. This function
    /// panics if the given quote is not a valid quote character.
    pub fn with_quote<S>(quote: char, value: S) -> Self
    where
        S: Into<String>,
    {
        assert!(quote == '\'' || quote == '"' || quote == '`' || quote == '[');
        Ident {
            value: value.into(),
            quote_style: Some(quote),
            span: Span::empty(),
        }
    }

    pub fn with_span<S>(span: Span, value: S) -> Self
    where
        S: Into<String>,
    {
        Ident {
            value: value.into(),
            quote_style: None,
            span,
        }
    }

    pub fn with_quote_and_span<S>(quote: char, span: Span, value: S) -> Self
    where
        S: Into<String>,
    {
        assert!(quote == '\'' || quote == '"' || quote == '`' || quote == '[');
        Ident {
            value: value.into(),
            quote_style: Some(quote),
            span,
        }
    }
}

impl From<&str> for Ident {
    fn from(value: &str) -> Self {
        Ident {
            value: value.to_string(),
            quote_style: None,
            span: Span::empty(),
        }
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.quote_style {
            Some(q) if q == '"' || q == '\'' || q == '`' => {
                let escaped = value::escape_quoted_string(&self.value, q);
                write!(f, "{q}{escaped}{q}")
            }
            Some('[') => write!(f, "[{}]", self.value),
            None => f.write_str(&self.value),
            _ => panic!("unexpected quote style"),
        }
    }
}

/// A name of a table, view, custom type, etc., possibly multi-part, i.e. db.schema.obj
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct ObjectName(pub Vec<Ident>);

impl fmt::Display for ObjectName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", display_separated(&self.0, "."))
    }
}

/// Represents an Array Expression, either
/// `ARRAY[..]`, or `[..]`
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct Array {
    /// The list of expressions between brackets
    pub elem: Vec<Expr>,

    /// `true` for  `ARRAY[..]`, `false` for `[..]`
    pub named: bool,
}

impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}[{}]",
            if self.named { "ARRAY" } else { "" },
            display_comma_separated(&self.elem)
        )
    }
}

/// Represents an INTERVAL expression, roughly in the following format:
/// `INTERVAL '<value>' [ <leading_field> [ (<leading_precision>) ] ]
/// [ TO <last_field> [ (<fractional_seconds_precision>) ] ]`,
/// e.g. `INTERVAL '123:45.67' MINUTE(3) TO SECOND(2)`.
///
/// The parser does not validate the `<value>`, nor does it ensure
/// that the `<leading_field>` units >= the units in `<last_field>`,
/// so the user will have to reject intervals like `HOUR TO YEAR`.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct Interval {
    pub value: Box<Expr>,
    pub leading_field: Option<DateTimeField>,
    pub leading_precision: Option<u64>,
    pub last_field: Option<DateTimeField>,
    /// The seconds precision can be specified in SQL source as
    /// `INTERVAL '__' SECOND(_, x)` (in which case the `leading_field`
    /// will be `Second` and the `last_field` will be `None`),
    /// or as `__ TO SECOND(x)`.
    pub fractional_seconds_precision: Option<u64>,
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let value = self.value.as_ref();
        match (
            &self.leading_field,
            self.leading_precision,
            self.fractional_seconds_precision,
        ) {
            (
                Some(DateTimeField::Second),
                Some(leading_precision),
                Some(fractional_seconds_precision),
            ) => {
                // When the leading field is SECOND, the parser guarantees that
                // the last field is None.
                assert!(self.last_field.is_none());
                write!(
                    f,
                    "INTERVAL {value} SECOND ({leading_precision}, {fractional_seconds_precision})"
                )
            }
            _ => {
                write!(f, "INTERVAL {value}")?;
                if let Some(leading_field) = &self.leading_field {
                    write!(f, " {leading_field}")?;
                }
                if let Some(leading_precision) = self.leading_precision {
                    write!(f, " ({leading_precision})")?;
                }
                if let Some(last_field) = &self.last_field {
                    write!(f, " TO {last_field}")?;
                }
                if let Some(fractional_seconds_precision) = self.fractional_seconds_precision {
                    write!(f, " ({fractional_seconds_precision})")?;
                }
                Ok(())
            }
        }
    }
}

/// A field definition within a struct
///
/// [bigquery]: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#struct_type
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct StructField {
    pub field_name: Option<Ident>,
    pub field_type: DataType,
}

impl fmt::Display for StructField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.field_name {
            write!(f, "{name} {}", self.field_type)
        } else {
            write!(f, "{}", self.field_type)
        }
    }
}

/// A field definition within a union
///
/// [duckdb]: https://duckdb.org/docs/sql/data_types/union.html
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct UnionField {
    pub field_name: Ident,
    pub field_type: DataType,
}

impl fmt::Display for UnionField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.field_name, self.field_type)
    }
}

/// A dictionary field within a dictionary.
///
/// [duckdb]: https://duckdb.org/docs/sql/data_types/struct#creating-structs
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct DictionaryField {
    pub key: Ident,
    pub value: Box<Expr>,
}

impl fmt::Display for DictionaryField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.key, self.value)
    }
}

/// Represents a Map expression.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct Map {
    pub entries: Vec<MapEntry>,
}

impl Display for Map {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MAP {{{}}}", display_comma_separated(&self.entries))
    }
}

/// A map field within a map.
///
/// [duckdb]: https://duckdb.org/docs/sql/data_types/map.html#creating-maps
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct MapEntry {
    pub key: Box<Expr>,
    pub value: Box<Expr>,
}

impl fmt::Display for MapEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.key, self.value)
    }
}

/// Options for `CAST` / `TRY_CAST`
/// BigQuery: <https://cloud.google.com/bigquery/docs/reference/standard-sql/format-elements#formatting_syntax>
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum CastFormat {
    Value(Value),
    ValueAtTimeZone(Value, Value),
}

/// Represents the syntax/style used in a map access.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum MapAccessSyntax {
    /// Access using bracket notation. `mymap[mykey]`
    Bracket,
    /// Access using period notation. `mymap.mykey`
    Period,
}

/// Expression used to access a value in a nested structure.
///
/// Example: `SAFE_OFFSET(0)` in
/// ```sql
/// SELECT mymap[SAFE_OFFSET(0)];
/// ```
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct MapAccessKey {
    pub key: Expr,
    pub syntax: MapAccessSyntax,
}

impl fmt::Display for MapAccessKey {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.syntax {
            MapAccessSyntax::Bracket => write!(f, "[{}]", self.key),
            MapAccessSyntax::Period => write!(f, ".{}", self.key),
        }
    }
}

/// An element of a JSON path.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum JsonPathElem {
    /// Accesses an object field using dot notation, e.g. `obj:foo.bar.baz`.
    ///
    /// See <https://docs.snowflake.com/en/user-guide/querying-semistructured#dot-notation>.
    Dot { key: String, quoted: bool },
    /// Accesses an object field or array element using bracket notation,
    /// e.g. `obj['foo']`.
    ///
    /// See <https://docs.snowflake.com/en/user-guide/querying-semistructured#bracket-notation>.
    Bracket { key: Expr },
}

/// A JSON path.
///
/// See <https://docs.snowflake.com/en/user-guide/querying-semistructured>.
/// See <https://docs.databricks.com/en/sql/language-manual/sql-ref-json-path-expression.html>.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct JsonPath {
    pub path: Vec<JsonPathElem>,
}

impl fmt::Display for JsonPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, elem) in self.path.iter().enumerate() {
            match elem {
                JsonPathElem::Dot { key, quoted } => {
                    if i == 0 {
                        write!(f, ":")?;
                    } else {
                        write!(f, ".")?;
                    }

                    if *quoted {
                        write!(f, "\"{}\"", escape_double_quote_string(key))?;
                    } else {
                        write!(f, "{key}")?;
                    }
                }
                JsonPathElem::Bracket { key } => {
                    write!(f, "[{key}]")?;
                }
            }
        }
        Ok(())
    }
}

/// The syntax used for in a cast expression.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum CastKind {
    /// The standard SQL cast syntax, e.g. `CAST(<expr> as <datatype>)`
    Cast,
    /// A cast that returns `NULL` on failure, e.g. `TRY_CAST(<expr> as <datatype>)`.
    ///
    /// See <https://docs.snowflake.com/en/sql-reference/functions/try_cast>.
    /// See <https://learn.microsoft.com/en-us/sql/t-sql/functions/try-cast-transact-sql>.
    TryCast,
    /// A cast that returns `NULL` on failure, bigQuery-specific ,  e.g. `SAFE_CAST(<expr> as <datatype>)`.
    ///
    /// See <https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#safe_casting>.
    SafeCast,
    /// `<expr> :: <datatype>`
    DoubleColon,
}

/// `EXTRACT` syntax variants.
///
/// In Snowflake dialect, the `EXTRACT` expression can support either the `from` syntax
/// or the comma syntax.
///
/// See <https://docs.snowflake.com/en/sql-reference/functions/extract>
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum ExtractSyntax {
    /// `EXTRACT( <date_or_time_part> FROM <date_or_time_expr> )`
    From,
    /// `EXTRACT( <date_or_time_part> , <date_or_timestamp_expr> )`
    Comma,
}

/// The syntax used in a CEIL or FLOOR expression.
///
/// The `CEIL/FLOOR(<datetime value expression> TO <time unit>)` is an Amazon Kinesis Data Analytics extension.
/// See <https://docs.aws.amazon.com/kinesisanalytics/latest/sqlref/sql-reference-ceil.html> for
/// details.
///
/// Other dialects either support `CEIL/FLOOR( <expr> [, <scale>])` format or just
/// `CEIL/FLOOR(<expr>)`.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum CeilFloorKind {
    /// `CEIL( <expr> TO <DateTimeField>)`
    DateTimeField(DateTimeField),
    /// `CEIL( <expr> [, <scale>])`
    Scale(Value),
}

/// An SQL expression of any type.
///
/// # Semantics / Type Checking
///
/// The parser does not distinguish between expressions of different types
/// (e.g. boolean vs string). The caller is responsible for detecting and
/// validating types as necessary (for example  `WHERE 1` vs `SELECT 1=1`)
/// See the [README.md] for more details.
///
/// [README.md]: https://github.com/apache/datafusion-sqlparser-rs/blob/main/README.md#syntax-vs-semantics
///
/// # Equality and Hashing Does not Include Source Locations
///
/// The `Expr` type implements `PartialEq` and `Eq` based on the semantic value
/// of the expression (not bitwise comparison). This means that `Expr` instances
/// that are semantically equivalent but have different spans (locations in the
/// source tree) will compare as equal.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "visitor",
    derive(Visit, VisitMut),
    visit(with = "visit_expr")
)]
pub enum Expr {
    /// Identifier e.g. table name or column name
    Identifier(Ident),
    /// Multi-part identifier, e.g. `table_alias.column` or `schema.table.col`
    CompoundIdentifier(Vec<Ident>),
    /// Access data nested in a value containing semi-structured data, such as
    /// the `VARIANT` type on Snowflake. for example `src:customer[0].name`.
    ///
    /// See <https://docs.snowflake.com/en/user-guide/querying-semistructured>.
    /// See <https://docs.databricks.com/en/sql/language-manual/functions/colonsign.html>.
    JsonAccess {
        /// The value being queried.
        value: Box<Expr>,
        /// The path to the data to extract.
        path: JsonPath,
    },
    /// CompositeAccess (postgres) eg: SELECT (information_schema._pg_expandarray(array['i','i'])).n
    CompositeAccess {
        expr: Box<Expr>,
        key: Ident,
    },
    /// `IS FALSE` operator
    IsFalse(Box<Expr>),
    /// `IS NOT FALSE` operator
    IsNotFalse(Box<Expr>),
    /// `IS TRUE` operator
    IsTrue(Box<Expr>),
    /// `IS NOT TRUE` operator
    IsNotTrue(Box<Expr>),
    /// `IS NULL` operator
    IsNull(Box<Expr>),
    /// `IS NOT NULL` operator
    IsNotNull(Box<Expr>),
    /// `IS UNKNOWN` operator
    IsUnknown(Box<Expr>),
    /// `IS NOT UNKNOWN` operator
    IsNotUnknown(Box<Expr>),
    /// `IS DISTINCT FROM` operator
    IsDistinctFrom(Box<Expr>, Box<Expr>),
    /// `IS NOT DISTINCT FROM` operator
    IsNotDistinctFrom(Box<Expr>, Box<Expr>),
    /// `[ NOT ] IN (val1, val2, ...)`
    InList {
        expr: Box<Expr>,
        list: Vec<Expr>,
        negated: bool,
    },
    /// `[ NOT ] IN (SELECT ...)`
    InSubquery {
        expr: Box<Expr>,
        subquery: Box<Query>,
        negated: bool,
    },
    /// `[ NOT ] IN UNNEST(array_expression)`
    InUnnest {
        expr: Box<Expr>,
        array_expr: Box<Expr>,
        negated: bool,
    },
    /// `<expr> [ NOT ] BETWEEN <low> AND <high>`
    Between {
        expr: Box<Expr>,
        negated: bool,
        low: Box<Expr>,
        high: Box<Expr>,
    },
    /// Binary operation e.g. `1 + 1` or `foo > bar`
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOperator,
        right: Box<Expr>,
    },
    /// `[NOT] LIKE <pattern> [ESCAPE <escape_character>]`
    Like {
        negated: bool,
        // Snowflake supports the ANY keyword to match against a list of patterns
        // https://docs.snowflake.com/en/sql-reference/functions/like_any
        any: bool,
        expr: Box<Expr>,
        pattern: Box<Expr>,
        escape_char: Option<String>,
    },
    /// `ILIKE` (case-insensitive `LIKE`)
    ILike {
        negated: bool,
        // Snowflake supports the ANY keyword to match against a list of patterns
        // https://docs.snowflake.com/en/sql-reference/functions/like_any
        any: bool,
        expr: Box<Expr>,
        pattern: Box<Expr>,
        escape_char: Option<String>,
    },
    /// SIMILAR TO regex
    SimilarTo {
        negated: bool,
        expr: Box<Expr>,
        pattern: Box<Expr>,
        escape_char: Option<String>,
    },
    /// MySQL: RLIKE regex or REGEXP regex
    RLike {
        negated: bool,
        expr: Box<Expr>,
        pattern: Box<Expr>,
        // true for REGEXP, false for RLIKE (no difference in semantics)
        regexp: bool,
    },
    /// `ANY` operation e.g. `foo > ANY(bar)`, comparison operator is one of `[=, >, <, =>, =<, !=]`
    /// <https://docs.snowflake.com/en/sql-reference/operators-subquery#all-any>
    AnyOp {
        left: Box<Expr>,
        compare_op: BinaryOperator,
        right: Box<Expr>,
        // ANY and SOME are synonymous: https://docs.cloudera.com/cdw-runtime/cloud/using-hiveql/topics/hive_comparison_predicates.html
        is_some: bool,
    },
    /// `ALL` operation e.g. `foo > ALL(bar)`, comparison operator is one of `[=, >, <, =>, =<, !=]`
    /// <https://docs.snowflake.com/en/sql-reference/operators-subquery#all-any>
    AllOp {
        left: Box<Expr>,
        compare_op: BinaryOperator,
        right: Box<Expr>,
    },
    /// Unary operation e.g. `NOT foo`
    UnaryOp {
        op: UnaryOperator,
        expr: Box<Expr>,
    },
    /// CONVERT a value to a different data type or character encoding. e.g. `CONVERT(foo USING utf8mb4)`
    Convert {
        /// CONVERT (false) or TRY_CONVERT (true)
        /// <https://learn.microsoft.com/en-us/sql/t-sql/functions/try-convert-transact-sql?view=sql-server-ver16>
        is_try: bool,
        /// The expression to convert
        expr: Box<Expr>,
        /// The target data type
        data_type: Option<DataType>,
        /// The target character encoding
        charset: Option<ObjectName>,
        /// whether the target comes before the expr (MSSQL syntax)
        target_before_value: bool,
        /// How to translate the expression.
        ///
        /// [MSSQL]: https://learn.microsoft.com/en-us/sql/t-sql/functions/cast-and-convert-transact-sql?view=sql-server-ver16#style
        styles: Vec<Expr>,
    },
    /// `CAST` an expression to a different data type e.g. `CAST(foo AS VARCHAR(123))`
    Cast {
        kind: CastKind,
        expr: Box<Expr>,
        data_type: DataType,
        // Optional CAST(string_expression AS type FORMAT format_string_expression) as used by BigQuery
        // https://cloud.google.com/bigquery/docs/reference/standard-sql/format-elements#formatting_syntax
        format: Option<CastFormat>,
    },
    /// AT a timestamp to a different timezone e.g. `FROM_UNIXTIME(0) AT TIME ZONE 'UTC-06:00'`
    AtTimeZone {
        timestamp: Box<Expr>,
        time_zone: Box<Expr>,
    },
    /// Extract a field from a timestamp e.g. `EXTRACT(MONTH FROM foo)`
    /// Or `EXTRACT(MONTH, foo)`
    ///
    /// Syntax:
    /// ```sql
    /// EXTRACT(DateTimeField FROM <expr>) | EXTRACT(DateTimeField, <expr>)
    /// ```
    Extract {
        field: DateTimeField,
        syntax: ExtractSyntax,
        expr: Box<Expr>,
    },
    /// ```sql
    /// CEIL(<expr> [TO DateTimeField])
    /// ```
    /// ```sql
    /// CEIL( <input_expr> [, <scale_expr> ] )
    /// ```
    Ceil {
        expr: Box<Expr>,
        field: CeilFloorKind,
    },
    /// ```sql
    /// FLOOR(<expr> [TO DateTimeField])
    /// ```
    /// ```sql
    /// FLOOR( <input_expr> [, <scale_expr> ] )
    ///
    Floor {
        expr: Box<Expr>,
        field: CeilFloorKind,
    },
    /// ```sql
    /// POSITION(<expr> in <expr>)
    /// ```
    Position {
        expr: Box<Expr>,
        r#in: Box<Expr>,
    },
    /// ```sql
    /// SUBSTRING(<expr> [FROM <expr>] [FOR <expr>])
    /// ```
    /// or
    /// ```sql
    /// SUBSTRING(<expr>, <expr>, <expr>)
    /// ```
    Substring {
        expr: Box<Expr>,
        substring_from: Option<Box<Expr>>,
        substring_for: Option<Box<Expr>>,

        /// false if the expression is represented using the `SUBSTRING(expr [FROM start] [FOR len])` syntax
        /// true if the expression is represented using the `SUBSTRING(expr, start, len)` syntax
        /// This flag is used for formatting.
        special: bool,
    },
    /// ```sql
    /// TRIM([BOTH | LEADING | TRAILING] [<expr> FROM] <expr>)
    /// TRIM(<expr>)
    /// TRIM(<expr>, [, characters]) -- only Snowflake or Bigquery
    /// ```
    Trim {
        expr: Box<Expr>,
        // ([BOTH | LEADING | TRAILING]
        trim_where: Option<TrimWhereField>,
        trim_what: Option<Box<Expr>>,
        trim_characters: Option<Vec<Expr>>,
    },
    /// ```sql
    /// OVERLAY(<expr> PLACING <expr> FROM <expr>[ FOR <expr> ]
    /// ```
    Overlay {
        expr: Box<Expr>,
        overlay_what: Box<Expr>,
        overlay_from: Box<Expr>,
        overlay_for: Option<Box<Expr>>,
    },
    /// `expr COLLATE collation`
    Collate {
        expr: Box<Expr>,
        collation: ObjectName,
    },
    /// Nested expression e.g. `(foo > bar)` or `(1)`
    Nested(Box<Expr>),
    /// A literal value, such as string, number, date or NULL
    Value(Value),
    /// <https://dev.mysql.com/doc/refman/8.0/en/charset-introducer.html>
    IntroducedString {
        introducer: String,
        value: Value,
    },
    /// A constant of form `<data_type> 'value'`.
    /// This can represent ANSI SQL `DATE`, `TIME`, and `TIMESTAMP` literals (such as `DATE '2020-01-01'`),
    /// as well as constants of other types (a non-standard PostgreSQL extension).
    TypedString {
        data_type: DataType,
        value: String,
    },
    /// Access a map-like object by field (e.g. `column['field']` or `column[4]`
    /// Note that depending on the dialect, struct like accesses may be
    /// parsed as [`Subscript`](Self::Subscript) or [`MapAccess`](Self::MapAccess)
    /// <https://clickhouse.com/docs/en/sql-reference/data-types/map/>
    MapAccess {
        column: Box<Expr>,
        keys: Vec<MapAccessKey>,
    },
    /// Scalar function call e.g. `LEFT(foo, 5)`
    Function(Function),
    /// Arbitrary expr method call
    ///
    /// Syntax:
    ///
    /// `<arbitrary-expr>.<function-call>.<function-call-expr>...`
    ///
    /// > `arbitrary-expr` can be any expression including a function call.
    ///
    /// Example:
    ///
    /// ```sql
    /// SELECT (SELECT ',' + name FROM sys.objects  FOR XML PATH(''), TYPE).value('.','NVARCHAR(MAX)')
    /// SELECT CONVERT(XML,'<Book>abc</Book>').value('.','NVARCHAR(MAX)').value('.','NVARCHAR(MAX)')
    /// ```
    ///
    /// (mssql): <https://learn.microsoft.com/en-us/sql/t-sql/xml/xml-data-type-methods?view=sql-server-ver16>
    Method(Method),
    /// `CASE [<operand>] WHEN <condition> THEN <result> ... [ELSE <result>] END`
    ///
    /// Note we only recognize a complete single expression as `<condition>`,
    /// not `< 0` nor `1, 2, 3` as allowed in a `<simple when clause>` per
    /// <https://jakewheat.github.io/sql-overview/sql-2011-foundation-grammar.html#simple-when-clause>
    Case {
        operand: Option<Box<Expr>>,
        conditions: Vec<Expr>,
        results: Vec<Expr>,
        else_result: Option<Box<Expr>>,
    },
    /// An exists expression `[ NOT ] EXISTS(SELECT ...)`, used in expressions like
    /// `WHERE [ NOT ] EXISTS (SELECT ...)`.
    Exists {
        subquery: Box<Query>,
        negated: bool,
    },
    /// A parenthesized subquery `(SELECT ...)`, used in expression like
    /// `SELECT (subquery) AS x` or `WHERE (subquery) = x`
    Subquery(Box<Query>),
    /// The `GROUPING SETS` expr.
    GroupingSets(Vec<Vec<Expr>>),
    /// The `CUBE` expr.
    Cube(Vec<Vec<Expr>>),
    /// The `ROLLUP` expr.
    Rollup(Vec<Vec<Expr>>),
    /// ROW / TUPLE a single value, such as `SELECT (1, 2)`
    Tuple(Vec<Expr>),
    /// `Struct` literal expression
    /// Syntax:
    /// ```sql
    /// STRUCT<[field_name] field_type, ...>( expr1 [, ... ])
    ///
    /// [BigQuery](https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#struct_type)
    /// [Databricks](https://docs.databricks.com/en/sql/language-manual/functions/struct.html)
    /// ```
    Struct {
        /// Struct values.
        values: Vec<Expr>,
        /// Struct field definitions.
        fields: Vec<StructField>,
    },
    /// `BigQuery` specific: An named expression in a typeless struct [1]
    ///
    /// Syntax
    /// ```sql
    /// 1 AS A
    /// ```
    /// [1]: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#struct_type
    Named {
        expr: Box<Expr>,
        name: Ident,
    },
    /// `DuckDB` specific `Struct` literal expression [1]
    ///
    /// Syntax:
    /// ```sql
    /// syntax: {'field_name': expr1[, ... ]}
    /// ```
    /// [1]: https://duckdb.org/docs/sql/data_types/struct#creating-structs
    Dictionary(Vec<DictionaryField>),
    /// `DuckDB` specific `Map` literal expression [1]
    ///
    /// Syntax:
    /// ```sql
    /// syntax: Map {key1: value1[, ... ]}
    /// ```
    /// [1]: https://duckdb.org/docs/sql/data_types/map#creating-maps
    Map(Map),
    /// An access of nested data using subscript syntax, for example `array[2]`.
    Subscript {
        expr: Box<Expr>,
        subscript: Box<Subscript>,
    },
    /// An array expression e.g. `ARRAY[1, 2]`
    Array(Array),
    /// An interval expression e.g. `INTERVAL '1' YEAR`
    Interval(Interval),
    /// `MySQL` specific text search function [(1)].
    ///
    /// Syntax:
    /// ```sql
    /// MATCH (<col>, <col>, ...) AGAINST (<expr> [<search modifier>])
    ///
    /// <col> = CompoundIdentifier
    /// <expr> = String literal
    /// ```
    /// [(1)]: https://dev.mysql.com/doc/refman/8.0/en/fulltext-search.html#function_match
    MatchAgainst {
        /// `(<col>, <col>, ...)`.
        columns: Vec<Ident>,
        /// `<expr>`.
        match_value: Value,
        /// `<search modifier>`
        opt_search_modifier: Option<SearchModifier>,
    },
    Wildcard(AttachedToken),
    /// Qualified wildcard, e.g. `alias.*` or `schema.table.*`.
    /// (Same caveats apply to `QualifiedWildcard` as to `Wildcard`.)
    QualifiedWildcard(ObjectName, AttachedToken),
    /// Some dialects support an older syntax for outer joins where columns are
    /// marked with the `(+)` operator in the WHERE clause, for example:
    ///
    /// ```sql
    /// SELECT t1.c1, t2.c2 FROM t1, t2 WHERE t1.c1 = t2.c2 (+)
    /// ```
    ///
    /// which is equivalent to
    ///
    /// ```sql
    /// SELECT t1.c1, t2.c2 FROM t1 LEFT OUTER JOIN t2 ON t1.c1 = t2.c2
    /// ```
    ///
    /// See <https://docs.snowflake.com/en/sql-reference/constructs/where#joins-in-the-where-clause>.
    OuterJoin(Box<Expr>),
    /// A reference to the prior level in a CONNECT BY clause.
    Prior(Box<Expr>),
    /// A lambda function.
    ///
    /// Syntax:
    /// ```plaintext
    /// param -> expr | (param1, ...) -> expr
    /// ```
    ///
    /// See <https://docs.databricks.com/en/sql/language-manual/sql-ref-lambda-functions.html>.
    Lambda(LambdaFunction),
}

/// The contents inside the `[` and `]` in a subscript expression.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum Subscript {
    /// Accesses the element of the array at the given index.
    Index { index: Expr },

    /// Accesses a slice of an array on PostgreSQL, e.g.
    ///
    /// ```plaintext
    /// => select (array[1,2,3,4,5,6])[2:5];
    /// -----------
    /// {2,3,4,5}
    /// ```
    ///
    /// The lower and/or upper bound can be omitted to slice from the start or
    /// end of the array respectively.
    ///
    /// See <https://www.postgresql.org/docs/current/arrays.html#ARRAYS-ACCESSING>.
    ///
    /// Also supports an optional "stride" as the last element (this is not
    /// supported by postgres), e.g.
    ///
    /// ```plaintext
    /// => select (array[1,2,3,4,5,6])[1:6:2];
    /// -----------
    /// {1,3,5}
    /// ```
    Slice {
        lower_bound: Option<Expr>,
        upper_bound: Option<Expr>,
        stride: Option<Expr>,
    },
}

impl fmt::Display for Subscript {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Subscript::Index { index } => write!(f, "{index}"),
            Subscript::Slice {
                lower_bound,
                upper_bound,
                stride,
            } => {
                if let Some(lower) = lower_bound {
                    write!(f, "{lower}")?;
                }
                write!(f, ":")?;
                if let Some(upper) = upper_bound {
                    write!(f, "{upper}")?;
                }
                if let Some(stride) = stride {
                    write!(f, ":")?;
                    write!(f, "{stride}")?;
                }
                Ok(())
            }
        }
    }
}

/// A lambda function.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct LambdaFunction {
    /// The parameters to the lambda function.
    pub params: OneOrManyWithParens<Ident>,
    /// The body of the lambda function.
    pub body: Box<Expr>,
}

impl fmt::Display for LambdaFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> {}", self.params, self.body)
    }
}

/// Encapsulates the common pattern in SQL where either one unparenthesized item
/// such as an identifier or expression is permitted, or multiple of the same
/// item in a parenthesized list. For accessing items regardless of the form,
/// `OneOrManyWithParens` implements `Deref<Target = [T]>` and `IntoIterator`,
/// so you can call slice methods on it and iterate over items
/// # Examples
/// Acessing as a slice:
/// ```
/// # use sqlparser::ast::OneOrManyWithParens;
/// let one = OneOrManyWithParens::One("a");
///
/// assert_eq!(one[0], "a");
/// assert_eq!(one.len(), 1);
/// ```
/// Iterating:
/// ```
/// # use sqlparser::ast::OneOrManyWithParens;
/// let one = OneOrManyWithParens::One("a");
/// let many = OneOrManyWithParens::Many(vec!["a", "b"]);
///
/// assert_eq!(one.into_iter().chain(many).collect::<Vec<_>>(), vec!["a", "a", "b"] );
/// ```
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum OneOrManyWithParens<T> {
    /// A single `T`, unparenthesized.
    One(T),
    /// One or more `T`s, parenthesized.
    Many(Vec<T>),
}

impl<T> Deref for OneOrManyWithParens<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            OneOrManyWithParens::One(one) => core::slice::from_ref(one),
            OneOrManyWithParens::Many(many) => many,
        }
    }
}

impl<T> AsRef<[T]> for OneOrManyWithParens<T> {
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<'a, T> IntoIterator for &'a OneOrManyWithParens<T> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Owned iterator implementation of `OneOrManyWithParens`
#[derive(Debug, Clone)]
pub struct OneOrManyWithParensIntoIter<T> {
    inner: OneOrManyWithParensIntoIterInner<T>,
}

#[derive(Debug, Clone)]
enum OneOrManyWithParensIntoIterInner<T> {
    One(core::iter::Once<T>),
    Many(<Vec<T> as IntoIterator>::IntoIter),
}

impl<T> core::iter::FusedIterator for OneOrManyWithParensIntoIter<T>
where
    core::iter::Once<T>: core::iter::FusedIterator,
    <Vec<T> as IntoIterator>::IntoIter: core::iter::FusedIterator,
{
}

impl<T> core::iter::ExactSizeIterator for OneOrManyWithParensIntoIter<T>
where
    core::iter::Once<T>: core::iter::ExactSizeIterator,
    <Vec<T> as IntoIterator>::IntoIter: core::iter::ExactSizeIterator,
{
}

impl<T> core::iter::Iterator for OneOrManyWithParensIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            OneOrManyWithParensIntoIterInner::One(one) => one.next(),
            OneOrManyWithParensIntoIterInner::Many(many) => many.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            OneOrManyWithParensIntoIterInner::One(one) => one.size_hint(),
            OneOrManyWithParensIntoIterInner::Many(many) => many.size_hint(),
        }
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        match self.inner {
            OneOrManyWithParensIntoIterInner::One(one) => one.count(),
            OneOrManyWithParensIntoIterInner::Many(many) => many.count(),
        }
    }

    fn fold<B, F>(mut self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        match &mut self.inner {
            OneOrManyWithParensIntoIterInner::One(one) => one.fold(init, f),
            OneOrManyWithParensIntoIterInner::Many(many) => many.fold(init, f),
        }
    }
}

impl<T> core::iter::DoubleEndedIterator for OneOrManyWithParensIntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            OneOrManyWithParensIntoIterInner::One(one) => one.next_back(),
            OneOrManyWithParensIntoIterInner::Many(many) => many.next_back(),
        }
    }
}

impl<T> IntoIterator for OneOrManyWithParens<T> {
    type Item = T;

    type IntoIter = OneOrManyWithParensIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        let inner = match self {
            OneOrManyWithParens::One(one) => {
                OneOrManyWithParensIntoIterInner::One(core::iter::once(one))
            }
            OneOrManyWithParens::Many(many) => {
                OneOrManyWithParensIntoIterInner::Many(many.into_iter())
            }
        };

        OneOrManyWithParensIntoIter { inner }
    }
}

impl<T> fmt::Display for OneOrManyWithParens<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OneOrManyWithParens::One(value) => write!(f, "{value}"),
            OneOrManyWithParens::Many(values) => {
                write!(f, "({})", display_comma_separated(values))
            }
        }
    }
}

impl fmt::Display for CastFormat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CastFormat::Value(v) => write!(f, "{v}"),
            CastFormat::ValueAtTimeZone(v, tz) => write!(f, "{v} AT TIME ZONE {tz}"),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expr::Identifier(s) => write!(f, "{s}"),
            Expr::MapAccess { column, keys } => {
                write!(f, "{column}{}", display_separated(keys, ""))
            }
            Expr::Wildcard(_) => f.write_str("*"),
            Expr::QualifiedWildcard(prefix, _) => write!(f, "{}.*", prefix),
            Expr::CompoundIdentifier(s) => write!(f, "{}", display_separated(s, ".")),
            Expr::IsTrue(ast) => write!(f, "{ast} IS TRUE"),
            Expr::IsNotTrue(ast) => write!(f, "{ast} IS NOT TRUE"),
            Expr::IsFalse(ast) => write!(f, "{ast} IS FALSE"),
            Expr::IsNotFalse(ast) => write!(f, "{ast} IS NOT FALSE"),
            Expr::IsNull(ast) => write!(f, "{ast} IS NULL"),
            Expr::IsNotNull(ast) => write!(f, "{ast} IS NOT NULL"),
            Expr::IsUnknown(ast) => write!(f, "{ast} IS UNKNOWN"),
            Expr::IsNotUnknown(ast) => write!(f, "{ast} IS NOT UNKNOWN"),
            Expr::InList {
                expr,
                list,
                negated,
            } => write!(
                f,
                "{} {}IN ({})",
                expr,
                if *negated { "NOT " } else { "" },
                display_comma_separated(list)
            ),
            Expr::InSubquery {
                expr,
                subquery,
                negated,
            } => write!(
                f,
                "{} {}IN ({})",
                expr,
                if *negated { "NOT " } else { "" },
                subquery
            ),
            Expr::InUnnest {
                expr,
                array_expr,
                negated,
            } => write!(
                f,
                "{} {}IN UNNEST({})",
                expr,
                if *negated { "NOT " } else { "" },
                array_expr
            ),
            Expr::Between {
                expr,
                negated,
                low,
                high,
            } => write!(
                f,
                "{} {}BETWEEN {} AND {}",
                expr,
                if *negated { "NOT " } else { "" },
                low,
                high
            ),
            Expr::BinaryOp { left, op, right } => write!(f, "{left} {op} {right}"),
            Expr::Like {
                negated,
                expr,
                pattern,
                escape_char,
                any,
            } => match escape_char {
                Some(ch) => write!(
                    f,
                    "{} {}LIKE {}{} ESCAPE '{}'",
                    expr,
                    if *negated { "NOT " } else { "" },
                    if *any { "ANY " } else { "" },
                    pattern,
                    ch
                ),
                _ => write!(
                    f,
                    "{} {}LIKE {}{}",
                    expr,
                    if *negated { "NOT " } else { "" },
                    if *any { "ANY " } else { "" },
                    pattern
                ),
            },
            Expr::ILike {
                negated,
                expr,
                pattern,
                escape_char,
                any,
            } => match escape_char {
                Some(ch) => write!(
                    f,
                    "{} {}ILIKE {}{} ESCAPE '{}'",
                    expr,
                    if *negated { "NOT " } else { "" },
                    if *any { "ANY" } else { "" },
                    pattern,
                    ch
                ),
                _ => write!(
                    f,
                    "{} {}ILIKE {}{}",
                    expr,
                    if *negated { "NOT " } else { "" },
                    if *any { "ANY " } else { "" },
                    pattern
                ),
            },
            Expr::RLike {
                negated,
                expr,
                pattern,
                regexp,
            } => write!(
                f,
                "{} {}{} {}",
                expr,
                if *negated { "NOT " } else { "" },
                if *regexp { "REGEXP" } else { "RLIKE" },
                pattern
            ),
            Expr::SimilarTo {
                negated,
                expr,
                pattern,
                escape_char,
            } => match escape_char {
                Some(ch) => write!(
                    f,
                    "{} {}SIMILAR TO {} ESCAPE '{}'",
                    expr,
                    if *negated { "NOT " } else { "" },
                    pattern,
                    ch
                ),
                _ => write!(
                    f,
                    "{} {}SIMILAR TO {}",
                    expr,
                    if *negated { "NOT " } else { "" },
                    pattern
                ),
            },
            Expr::AnyOp {
                left,
                compare_op,
                right,
                is_some,
            } => {
                let add_parens = !matches!(right.as_ref(), Expr::Subquery(_));
                write!(
                    f,
                    "{left} {compare_op} {}{}{right}{}",
                    if *is_some { "SOME" } else { "ANY" },
                    if add_parens { "(" } else { "" },
                    if add_parens { ")" } else { "" },
                )
            }
            Expr::AllOp {
                left,
                compare_op,
                right,
            } => {
                let add_parens = !matches!(right.as_ref(), Expr::Subquery(_));
                write!(
                    f,
                    "{left} {compare_op} ALL{}{right}{}",
                    if add_parens { "(" } else { "" },
                    if add_parens { ")" } else { "" },
                )
            }
            Expr::UnaryOp { op, expr } => {
                if op == &UnaryOperator::PGPostfixFactorial {
                    write!(f, "{expr}{op}")
                } else if op == &UnaryOperator::Not {
                    write!(f, "{op} {expr}")
                } else {
                    write!(f, "{op}{expr}")
                }
            }
            Expr::Convert {
                is_try,
                expr,
                target_before_value,
                data_type,
                charset,
                styles,
            } => {
                write!(f, "{}CONVERT(", if *is_try { "TRY_" } else { "" })?;
                if let Some(data_type) = data_type {
                    if let Some(charset) = charset {
                        write!(f, "{expr}, {data_type} CHARACTER SET {charset}")
                    } else if *target_before_value {
                        write!(f, "{data_type}, {expr}")
                    } else {
                        write!(f, "{expr}, {data_type}")
                    }
                } else if let Some(charset) = charset {
                    write!(f, "{expr} USING {charset}")
                } else {
                    write!(f, "{expr}") // This should never happen
                }?;
                if !styles.is_empty() {
                    write!(f, ", {}", display_comma_separated(styles))?;
                }
                write!(f, ")")
            }
            Expr::Cast {
                kind,
                expr,
                data_type,
                format,
            } => match kind {
                CastKind::Cast => {
                    if let Some(format) = format {
                        write!(f, "CAST({expr} AS {data_type} FORMAT {format})")
                    } else {
                        write!(f, "CAST({expr} AS {data_type})")
                    }
                }
                CastKind::TryCast => {
                    if let Some(format) = format {
                        write!(f, "TRY_CAST({expr} AS {data_type} FORMAT {format})")
                    } else {
                        write!(f, "TRY_CAST({expr} AS {data_type})")
                    }
                }
                CastKind::SafeCast => {
                    if let Some(format) = format {
                        write!(f, "SAFE_CAST({expr} AS {data_type} FORMAT {format})")
                    } else {
                        write!(f, "SAFE_CAST({expr} AS {data_type})")
                    }
                }
                CastKind::DoubleColon => {
                    write!(f, "{expr}::{data_type}")
                }
            },
            Expr::Extract {
                field,
                syntax,
                expr,
            } => match syntax {
                ExtractSyntax::From => write!(f, "EXTRACT({field} FROM {expr})"),
                ExtractSyntax::Comma => write!(f, "EXTRACT({field}, {expr})"),
            },
            Expr::Ceil { expr, field } => match field {
                CeilFloorKind::DateTimeField(DateTimeField::NoDateTime) => {
                    write!(f, "CEIL({expr})")
                }
                CeilFloorKind::DateTimeField(dt_field) => write!(f, "CEIL({expr} TO {dt_field})"),
                CeilFloorKind::Scale(s) => write!(f, "CEIL({expr}, {s})"),
            },
            Expr::Floor { expr, field } => match field {
                CeilFloorKind::DateTimeField(DateTimeField::NoDateTime) => {
                    write!(f, "FLOOR({expr})")
                }
                CeilFloorKind::DateTimeField(dt_field) => write!(f, "FLOOR({expr} TO {dt_field})"),
                CeilFloorKind::Scale(s) => write!(f, "FLOOR({expr}, {s})"),
            },
            Expr::Position { expr, r#in } => write!(f, "POSITION({expr} IN {in})"),
            Expr::Collate { expr, collation } => write!(f, "{expr} COLLATE {collation}"),
            Expr::Nested(ast) => write!(f, "({ast})"),
            Expr::Value(v) => write!(f, "{v}"),
            Expr::IntroducedString { introducer, value } => write!(f, "{introducer} {value}"),
            Expr::TypedString { data_type, value } => {
                write!(f, "{data_type}")?;
                write!(f, " '{}'", &value::escape_single_quote_string(value))
            }
            Expr::Function(fun) => write!(f, "{fun}"),
            Expr::Method(method) => write!(f, "{method}"),
            Expr::Case {
                operand,
                conditions,
                results,
                else_result,
            } => {
                write!(f, "CASE")?;
                if let Some(operand) = operand {
                    write!(f, " {operand}")?;
                }
                for (c, r) in conditions.iter().zip(results) {
                    write!(f, " WHEN {c} THEN {r}")?;
                }

                if let Some(else_result) = else_result {
                    write!(f, " ELSE {else_result}")?;
                }
                write!(f, " END")
            }
            Expr::Exists { subquery, negated } => write!(
                f,
                "{}EXISTS ({})",
                if *negated { "NOT " } else { "" },
                subquery
            ),
            Expr::Subquery(s) => write!(f, "({s})"),
            Expr::GroupingSets(sets) => {
                write!(f, "GROUPING SETS (")?;
                let mut sep = "";
                for set in sets {
                    write!(f, "{sep}")?;
                    sep = ", ";
                    write!(f, "({})", display_comma_separated(set))?;
                }
                write!(f, ")")
            }
            Expr::Cube(sets) => {
                write!(f, "CUBE (")?;
                let mut sep = "";
                for set in sets {
                    write!(f, "{sep}")?;
                    sep = ", ";
                    if set.len() == 1 {
                        write!(f, "{}", set[0])?;
                    } else {
                        write!(f, "({})", display_comma_separated(set))?;
                    }
                }
                write!(f, ")")
            }
            Expr::Rollup(sets) => {
                write!(f, "ROLLUP (")?;
                let mut sep = "";
                for set in sets {
                    write!(f, "{sep}")?;
                    sep = ", ";
                    if set.len() == 1 {
                        write!(f, "{}", set[0])?;
                    } else {
                        write!(f, "({})", display_comma_separated(set))?;
                    }
                }
                write!(f, ")")
            }
            Expr::Substring {
                expr,
                substring_from,
                substring_for,
                special,
            } => {
                write!(f, "SUBSTRING({expr}")?;
                if let Some(from_part) = substring_from {
                    if *special {
                        write!(f, ", {from_part}")?;
                    } else {
                        write!(f, " FROM {from_part}")?;
                    }
                }
                if let Some(for_part) = substring_for {
                    if *special {
                        write!(f, ", {for_part}")?;
                    } else {
                        write!(f, " FOR {for_part}")?;
                    }
                }

                write!(f, ")")
            }
            Expr::Overlay {
                expr,
                overlay_what,
                overlay_from,
                overlay_for,
            } => {
                write!(
                    f,
                    "OVERLAY({expr} PLACING {overlay_what} FROM {overlay_from}"
                )?;
                if let Some(for_part) = overlay_for {
                    write!(f, " FOR {for_part}")?;
                }

                write!(f, ")")
            }
            Expr::IsDistinctFrom(a, b) => write!(f, "{a} IS DISTINCT FROM {b}"),
            Expr::IsNotDistinctFrom(a, b) => write!(f, "{a} IS NOT DISTINCT FROM {b}"),
            Expr::Trim {
                expr,
                trim_where,
                trim_what,
                trim_characters,
            } => {
                write!(f, "TRIM(")?;
                if let Some(ident) = trim_where {
                    write!(f, "{ident} ")?;
                }
                if let Some(trim_char) = trim_what {
                    write!(f, "{trim_char} FROM {expr}")?;
                } else {
                    write!(f, "{expr}")?;
                }
                if let Some(characters) = trim_characters {
                    write!(f, ", {}", display_comma_separated(characters))?;
                }

                write!(f, ")")
            }
            Expr::Tuple(exprs) => {
                write!(f, "({})", display_comma_separated(exprs))
            }
            Expr::Struct { values, fields } => {
                if !fields.is_empty() {
                    write!(
                        f,
                        "STRUCT<{}>({})",
                        display_comma_separated(fields),
                        display_comma_separated(values)
                    )
                } else {
                    write!(f, "STRUCT({})", display_comma_separated(values))
                }
            }
            Expr::Named { expr, name } => {
                write!(f, "{} AS {}", expr, name)
            }
            Expr::Dictionary(fields) => {
                write!(f, "{{{}}}", display_comma_separated(fields))
            }
            Expr::Map(map) => {
                write!(f, "{map}")
            }
            Expr::Subscript {
                expr,
                subscript: key,
            } => {
                write!(f, "{expr}[{key}]")
            }
            Expr::Array(set) => {
                write!(f, "{set}")
            }
            Expr::JsonAccess { value, path } => {
                write!(f, "{value}{path}")
            }
            Expr::CompositeAccess { expr, key } => {
                write!(f, "{expr}.{key}")
            }
            Expr::AtTimeZone {
                timestamp,
                time_zone,
            } => {
                write!(f, "{timestamp} AT TIME ZONE {time_zone}")
            }
            Expr::Interval(interval) => {
                write!(f, "{interval}")
            }
            Expr::MatchAgainst {
                columns,
                match_value: match_expr,
                opt_search_modifier,
            } => {
                write!(f, "MATCH ({}) AGAINST ", display_comma_separated(columns),)?;

                if let Some(search_modifier) = opt_search_modifier {
                    write!(f, "({match_expr} {search_modifier})")?;
                } else {
                    write!(f, "({match_expr})")?;
                }

                Ok(())
            }
            Expr::OuterJoin(expr) => {
                write!(f, "{expr} (+)")
            }
            Expr::Prior(expr) => write!(f, "PRIOR {expr}"),
            Expr::Lambda(lambda) => write!(f, "{lambda}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum WindowType {
    WindowSpec(WindowSpec),
    NamedWindow(Ident),
}

impl Display for WindowType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WindowType::WindowSpec(spec) => write!(f, "({})", spec),
            WindowType::NamedWindow(name) => write!(f, "{}", name),
        }
    }
}

/// A window specification (i.e. `OVER ([window_name] PARTITION BY .. ORDER BY .. etc.)`)
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct WindowSpec {
    /// Optional window name.
    ///
    /// You can find it at least in [MySQL][1], [BigQuery][2], [PostgreSQL][3]
    ///
    /// [1]: https://dev.mysql.com/doc/refman/8.0/en/window-functions-named-windows.html
    /// [2]: https://cloud.google.com/bigquery/docs/reference/standard-sql/window-function-calls
    /// [3]: https://www.postgresql.org/docs/current/sql-expressions.html#SYNTAX-WINDOW-FUNCTIONS
    pub window_name: Option<Ident>,
    /// `OVER (PARTITION BY ...)`
    pub partition_by: Vec<Expr>,
    /// `OVER (ORDER BY ...)`
    pub order_by: Vec<OrderByExpr>,
    /// `OVER (window frame)`
    pub window_frame: Option<WindowFrame>,
}

impl fmt::Display for WindowSpec {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut delim = "";
        if let Some(window_name) = &self.window_name {
            delim = " ";
            write!(f, "{window_name}")?;
        }
        if !self.partition_by.is_empty() {
            f.write_str(delim)?;
            delim = " ";
            write!(
                f,
                "PARTITION BY {}",
                display_comma_separated(&self.partition_by)
            )?;
        }
        if !self.order_by.is_empty() {
            f.write_str(delim)?;
            delim = " ";
            write!(f, "ORDER BY {}", display_comma_separated(&self.order_by))?;
        }
        if let Some(window_frame) = &self.window_frame {
            f.write_str(delim)?;
            if let Some(end_bound) = &window_frame.end_bound {
                write!(
                    f,
                    "{} BETWEEN {} AND {}",
                    window_frame.units, window_frame.start_bound, end_bound
                )?;
            } else {
                write!(f, "{} {}", window_frame.units, window_frame.start_bound)?;
            }
        }
        Ok(())
    }
}

/// Specifies the data processed by a window function, e.g.
/// `RANGE UNBOUNDED PRECEDING` or `ROWS BETWEEN 5 PRECEDING AND CURRENT ROW`.
///
/// Note: The parser does not validate the specified bounds; the caller should
/// reject invalid bounds like `ROWS UNBOUNDED FOLLOWING` before execution.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct WindowFrame {
    pub units: WindowFrameUnits,
    pub start_bound: WindowFrameBound,
    /// The right bound of the `BETWEEN .. AND` clause. The end bound of `None`
    /// indicates the shorthand form (e.g. `ROWS 1 PRECEDING`), which must
    /// behave the same as `end_bound = WindowFrameBound::CurrentRow`.
    pub end_bound: Option<WindowFrameBound>,
    // TBD: EXCLUDE
}

impl Default for WindowFrame {
    /// Returns default value for window frame
    ///
    /// See [this page](https://www.sqlite.org/windowfunctions.html#frame_specifications) for more details.
    fn default() -> Self {
        Self {
            units: WindowFrameUnits::Range,
            start_bound: WindowFrameBound::Preceding(None),
            end_bound: None,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum WindowFrameUnits {
    Rows,
    Range,
    Groups,
}

impl fmt::Display for WindowFrameUnits {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            WindowFrameUnits::Rows => "ROWS",
            WindowFrameUnits::Range => "RANGE",
            WindowFrameUnits::Groups => "GROUPS",
        })
    }
}

/// Specifies Ignore / Respect NULL within window functions.
/// For example
/// `FIRST_VALUE(column2) IGNORE NULLS OVER (PARTITION BY column1)`
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum NullTreatment {
    IgnoreNulls,
    RespectNulls,
}

impl fmt::Display for NullTreatment {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            NullTreatment::IgnoreNulls => "IGNORE NULLS",
            NullTreatment::RespectNulls => "RESPECT NULLS",
        })
    }
}

/// Specifies [WindowFrame]'s `start_bound` and `end_bound`
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum WindowFrameBound {
    /// `CURRENT ROW`
    CurrentRow,
    /// `<N> PRECEDING` or `UNBOUNDED PRECEDING`
    Preceding(Option<Box<Expr>>),
    /// `<N> FOLLOWING` or `UNBOUNDED FOLLOWING`.
    Following(Option<Box<Expr>>),
}

impl fmt::Display for WindowFrameBound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            WindowFrameBound::CurrentRow => f.write_str("CURRENT ROW"),
            WindowFrameBound::Preceding(None) => f.write_str("UNBOUNDED PRECEDING"),
            WindowFrameBound::Following(None) => f.write_str("UNBOUNDED FOLLOWING"),
            WindowFrameBound::Preceding(Some(n)) => write!(f, "{n} PRECEDING"),
            WindowFrameBound::Following(Some(n)) => write!(f, "{n} FOLLOWING"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum AddDropSync {
    ADD,
    DROP,
    SYNC,
}

impl fmt::Display for AddDropSync {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AddDropSync::SYNC => f.write_str("SYNC PARTITIONS"),
            AddDropSync::DROP => f.write_str("DROP PARTITIONS"),
            AddDropSync::ADD => f.write_str("ADD PARTITIONS"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum ShowCreateObject {
    Event,
    Function,
    Procedure,
    Table,
    Trigger,
    View,
}

impl fmt::Display for ShowCreateObject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ShowCreateObject::Event => f.write_str("EVENT"),
            ShowCreateObject::Function => f.write_str("FUNCTION"),
            ShowCreateObject::Procedure => f.write_str("PROCEDURE"),
            ShowCreateObject::Table => f.write_str("TABLE"),
            ShowCreateObject::Trigger => f.write_str("TRIGGER"),
            ShowCreateObject::View => f.write_str("VIEW"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum CommentObject {
    Column,
    Table,
    Extension,
    Schema,
    Database,
    User,
    Role,
}

impl fmt::Display for CommentObject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CommentObject::Column => f.write_str("COLUMN"),
            CommentObject::Table => f.write_str("TABLE"),
            CommentObject::Extension => f.write_str("EXTENSION"),
            CommentObject::Schema => f.write_str("SCHEMA"),
            CommentObject::Database => f.write_str("DATABASE"),
            CommentObject::User => f.write_str("USER"),
            CommentObject::Role => f.write_str("ROLE"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum Password {
    Password(Expr),
    NullPassword,
}

/// Represents an expression assignment within a variable `DECLARE` statement.
///
/// Examples:
/// ```sql
/// DECLARE variable_name := 42
/// DECLARE variable_name DEFAULT 42
/// ```
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum DeclareAssignment {
    /// Plain expression specified.
    Expr(Box<Expr>),

    /// Expression assigned via the `DEFAULT` keyword
    Default(Box<Expr>),

    /// Expression assigned via the `:=` syntax
    ///
    /// Example:
    /// ```sql
    /// DECLARE variable_name := 42;
    /// ```
    DuckAssignment(Box<Expr>),

    /// Expression via the `FOR` keyword
    ///
    /// Example:
    /// ```sql
    /// DECLARE c1 CURSOR FOR res
    /// ```
    For(Box<Expr>),

    /// Expression via the `=` syntax.
    ///
    /// Example:
    /// ```sql
    /// DECLARE @variable AS INT = 100
    /// ```
    MsSqlAssignment(Box<Expr>),
}

impl fmt::Display for DeclareAssignment {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DeclareAssignment::Expr(expr) => {
                write!(f, "{expr}")
            }
            DeclareAssignment::Default(expr) => {
                write!(f, "DEFAULT {expr}")
            }
            DeclareAssignment::DuckAssignment(expr) => {
                write!(f, ":= {expr}")
            }
            DeclareAssignment::MsSqlAssignment(expr) => {
                write!(f, "= {expr}")
            }
            DeclareAssignment::For(expr) => {
                write!(f, "FOR {expr}")
            }
        }
    }
}

/// Represents the type of a `DECLARE` statement.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum DeclareType {
    /// Cursor variable type. e.g. [Snowflake] [Postgres]
    ///
    /// [Snowflake]: https://docs.snowflake.com/en/developer-guide/snowflake-scripting/cursors#declaring-a-cursor
    /// [Postgres]: https://www.postgresql.org/docs/current/plpgsql-cursors.html
    Cursor,

    /// Result set variable type. [Snowflake]
    ///
    /// Syntax:
    /// ```text
    /// <resultset_name> RESULTSET [ { DEFAULT | := } ( <query> ) ] ;
    /// ```
    /// [Snowflake]: https://docs.snowflake.com/en/sql-reference/snowflake-scripting/declare#resultset-declaration-syntax
    ResultSet,

    /// Exception declaration syntax. [Snowflake]
    ///
    /// Syntax:
    /// ```text
    /// <exception_name> EXCEPTION [ ( <exception_number> , '<exception_message>' ) ] ;
    /// ```
    /// [Snowflake]: https://docs.snowflake.com/en/sql-reference/snowflake-scripting/declare#exception-declaration-syntax
    Exception,
}

impl fmt::Display for DeclareType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DeclareType::Cursor => {
                write!(f, "CURSOR")
            }
            DeclareType::ResultSet => {
                write!(f, "RESULTSET")
            }
            DeclareType::Exception => {
                write!(f, "EXCEPTION")
            }
        }
    }
}

/// A `DECLARE` statement.
/// [Postgres] [Snowflake] [BigQuery]
///
/// Examples:
/// ```sql
/// DECLARE variable_name := 42
/// DECLARE liahona CURSOR FOR SELECT * FROM films;
/// ```
///
/// [Postgres]: https://www.postgresql.org/docs/current/sql-declare.html
/// [Snowflake]: https://docs.snowflake.com/en/sql-reference/snowflake-scripting/declare
/// [BigQuery]: https://cloud.google.com/bigquery/docs/reference/standard-sql/procedural-language#declare
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct Declare {
    /// The name(s) being declared.
    /// Example: `DECLARE a, b, c DEFAULT 42;
    pub names: Vec<Ident>,
    /// Data-type assigned to the declared variable.
    /// Example: `DECLARE x INT64 DEFAULT 42;
    pub data_type: Option<DataType>,
    /// Expression being assigned to the declared variable.
    pub assignment: Option<DeclareAssignment>,
    /// Represents the type of the declared variable.
    pub declare_type: Option<DeclareType>,
    /// Causes the cursor to return data in binary rather than in text format.
    pub binary: Option<bool>,
    /// None = Not specified
    /// Some(true) = INSENSITIVE
    /// Some(false) = ASENSITIVE
    pub sensitive: Option<bool>,
    /// None = Not specified
    /// Some(true) = SCROLL
    /// Some(false) = NO SCROLL
    pub scroll: Option<bool>,
    /// None = Not specified
    /// Some(true) = WITH HOLD, specifies that the cursor can continue to be used after the transaction that created it successfully commits
    /// Some(false) = WITHOUT HOLD, specifies that the cursor cannot be used outside of the transaction that created it
    pub hold: Option<bool>,
    /// `FOR <query>` clause in a CURSOR declaration.
    pub for_query: Option<Box<Query>>,
}

impl fmt::Display for Declare {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let Declare {
            names,
            data_type,
            assignment,
            declare_type,
            binary,
            sensitive,
            scroll,
            hold,
            for_query,
        } = self;
        write!(f, "{}", display_comma_separated(names))?;

        if let Some(true) = binary {
            write!(f, " BINARY")?;
        }

        if let Some(sensitive) = sensitive {
            if *sensitive {
                write!(f, " INSENSITIVE")?;
            } else {
                write!(f, " ASENSITIVE")?;
            }
        }

        if let Some(scroll) = scroll {
            if *scroll {
                write!(f, " SCROLL")?;
            } else {
                write!(f, " NO SCROLL")?;
            }
        }

        if let Some(declare_type) = declare_type {
            write!(f, " {declare_type}")?;
        }

        if let Some(hold) = hold {
            if *hold {
                write!(f, " WITH HOLD")?;
            } else {
                write!(f, " WITHOUT HOLD")?;
            }
        }

        if let Some(query) = for_query {
            write!(f, " FOR {query}")?;
        }

        if let Some(data_type) = data_type {
            write!(f, " {data_type}")?;
        }

        if let Some(expr) = assignment {
            write!(f, " {expr}")?;
        }
        Ok(())
    }
}

/// Sql options of a `CREATE TABLE` statement.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum CreateTableOptions {
    None,
    /// Options specified using the `WITH` keyword.
    /// e.g. `WITH (description = "123")`
    ///
    /// <https://www.postgresql.org/docs/current/sql-createtable.html>
    ///
    /// MSSQL supports more specific options that's not only key-value pairs.
    ///
    /// WITH (
    ///     DISTRIBUTION = ROUND_ROBIN,
    ///     CLUSTERED INDEX (column_a DESC, column_b)
    /// )
    ///
    /// <https://learn.microsoft.com/en-us/sql/t-sql/statements/create-table-azure-sql-data-warehouse?view=aps-pdw-2016-au7#syntax>
    With(Vec<SqlOption>),
    /// Options specified using the `OPTIONS` keyword.
    /// e.g. `OPTIONS(description = "123")`
    ///
    /// <https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#table_option_list>
    Options(Vec<SqlOption>),
}

impl fmt::Display for CreateTableOptions {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CreateTableOptions::With(with_options) => {
                write!(f, "WITH ({})", display_comma_separated(with_options))
            }
            CreateTableOptions::Options(options) => {
                write!(f, "OPTIONS({})", display_comma_separated(options))
            }
            CreateTableOptions::None => Ok(()),
        }
    }
}

/// A `FROM` clause within a `DELETE` statement.
///
/// Syntax
/// ```sql
/// [FROM] table
/// ```
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum FromTable {
    /// An explicit `FROM` keyword was specified.
    WithFromKeyword(Vec<TableWithJoins>),
    /// BigQuery: `FROM` keyword was omitted.
    /// <https://cloud.google.com/bigquery/docs/reference/standard-sql/dml-syntax#delete_statement>
    WithoutKeyword(Vec<TableWithJoins>),
}
impl Display for FromTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FromTable::WithFromKeyword(tables) => {
                write!(f, "FROM {}", display_comma_separated(tables))
            }
            FromTable::WithoutKeyword(tables) => {
                write!(f, "{}", display_comma_separated(tables))
            }
        }
    }
}

/// Policy type for a `CREATE POLICY` statement.
/// ```sql
/// AS [ PERMISSIVE | RESTRICTIVE ]
/// ```
/// [PostgreSQL](https://www.postgresql.org/docs/current/sql-createpolicy.html)
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum CreatePolicyType {
    Permissive,
    Restrictive,
}

/// Policy command for a `CREATE POLICY` statement.
/// ```sql
/// FOR [ALL | SELECT | INSERT | UPDATE | DELETE]
/// ```
/// [PostgreSQL](https://www.postgresql.org/docs/current/sql-createpolicy.html)
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum CreatePolicyCommand {
    All,
    Select,
    Insert,
    Update,
    Delete,
}

/// A top-level statement (SELECT, INSERT, CREATE, etc.)
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "visitor",
    derive(Visit, VisitMut),
    visit(with = "visit_statement")
)]
pub enum Statement {
    /// ```sql
    /// ANALYZE
    /// ```
    /// Analyze (Hive)
    Analyze {
        #[cfg_attr(feature = "visitor", visit(with = "visit_relation"))]
        table_name: ObjectName,
        partitions: Option<Vec<Expr>>,
        for_columns: bool,
        columns: Vec<Ident>,
        cache_metadata: bool,
        noscan: bool,
        compute_statistics: bool,
    },
    /// ```sql
    /// TRUNCATE
    /// ```
    /// Truncate (Hive)
    Truncate {
        table_names: Vec<TruncateTableTarget>,
        partitions: Option<Vec<Expr>>,
        /// TABLE - optional keyword;
        table: bool,
        /// Postgres-specific option
        /// [ TRUNCATE TABLE ONLY ]
        only: bool,
        /// Postgres-specific option
        /// [ RESTART IDENTITY | CONTINUE IDENTITY ]
        identity: Option<TruncateIdentityOption>,
        /// Postgres-specific option
        /// [ CASCADE | RESTRICT ]
        cascade: Option<TruncateCascadeOption>,
        /// ClickHouse-specific option
        /// [ ON CLUSTER cluster_name ]
        ///
        /// [ClickHouse](https://clickhouse.com/docs/en/sql-reference/statements/truncate/)
        on_cluster: Option<Ident>,
    },
    /// ```sql
    /// MSCK
    /// ```
    /// Msck (Hive)
    Msck {
        #[cfg_attr(feature = "visitor", visit(with = "visit_relation"))]
        table_name: ObjectName,
        repair: bool,
        partition_action: Option<AddDropSync>,
    },
    /// ```sql
    /// SELECT
    /// ```
    Query(Box<Query>),
    /// ```sql
    /// INSERT
    /// ```
    Insert(Insert),
    /// ```sql
    /// INSTALL
    /// ```
    Install {
        /// Only for DuckDB
        extension_name: Ident,
    },
    /// ```sql
    /// LOAD
    /// ```
    Load {
        /// Only for DuckDB
        extension_name: Ident,
    },
    // TODO: Support ROW FORMAT
    Directory {
        overwrite: bool,
        local: bool,
        path: String,
        file_format: Option<FileFormat>,
        source: Box<Query>,
    },
    /// ```sql
    /// CALL <function>
    /// ```
    Call(Function),
    /// ```sql
    /// COPY [TO | FROM] ...
    /// ```
    Copy {
        /// The source of 'COPY TO', or the target of 'COPY FROM'
        source: CopySource,
        /// If true, is a 'COPY TO' statement. If false is a 'COPY FROM'
        to: bool,
        /// The target of 'COPY TO', or the source of 'COPY FROM'
        target: CopyTarget,
        /// WITH options (from PostgreSQL version 9.0)
        options: Vec<CopyOption>,
        /// WITH options (before PostgreSQL version 9.0)
        legacy_options: Vec<CopyLegacyOption>,
        /// VALUES a vector of values to be copied
        values: Vec<Option<String>>,
    },
    /// ```sql
    /// COPY INTO
    /// ```
    /// See <https://docs.snowflake.com/en/sql-reference/sql/copy-into-table>
    /// Copy Into syntax available for Snowflake is different than the one implemented in
    /// Postgres. Although they share common prefix, it is reasonable to implement them
    /// in different enums. This can be refactored later once custom dialects
    /// are allowed to have custom Statements.
    CopyIntoSnowflake {
        into: ObjectName,
        from_stage: ObjectName,
        from_stage_alias: Option<Ident>,
        stage_params: StageParamsObject,
        from_transformations: Option<Vec<StageLoadSelectItem>>,
        files: Option<Vec<String>>,
        pattern: Option<String>,
        file_format: DataLoadingOptions,
        copy_options: DataLoadingOptions,
        validation_mode: Option<String>,
    },
    /// ```sql
    /// CLOSE
    /// ```
    /// Closes the portal underlying an open cursor.
    Close {
        /// Cursor name
        cursor: CloseCursor,
    },
    /// ```sql
    /// UPDATE
    /// ```
    Update {
        /// TABLE
        table: TableWithJoins,
        /// Column assignments
        assignments: Vec<Assignment>,
        /// Table which provide value to be set
        from: Option<TableWithJoins>,
        /// WHERE
        selection: Option<Expr>,
        /// RETURNING
        returning: Option<Vec<SelectItem>>,
        /// SQLite-specific conflict resolution clause
        or: Option<SqliteOnConflict>,
    },
    /// ```sql
    /// DELETE
    /// ```
    Delete(Delete),
    /// ```sql
    /// CREATE VIEW
    /// ```
    CreateView {
        or_replace: bool,
        materialized: bool,
        /// View name
        name: ObjectName,
        columns: Vec<ViewColumnDef>,
        query: Box<Query>,
        options: CreateTableOptions,
        cluster_by: Vec<Ident>,
        /// Snowflake: Views can have comments in Snowflake.
        /// <https://docs.snowflake.com/en/sql-reference/sql/create-view#syntax>
        comment: Option<String>,
        /// if true, has RedShift [`WITH NO SCHEMA BINDING`] clause <https://docs.aws.amazon.com/redshift/latest/dg/r_CREATE_VIEW.html>
        with_no_schema_binding: bool,
        /// if true, has SQLite `IF NOT EXISTS` clause <https://www.sqlite.org/lang_createview.html>
        if_not_exists: bool,
        /// if true, has SQLite `TEMP` or `TEMPORARY` clause <https://www.sqlite.org/lang_createview.html>
        temporary: bool,
        /// if not None, has Clickhouse `TO` clause, specify the table into which to insert results
        /// <https://clickhouse.com/docs/en/sql-reference/statements/create/view#materialized-view>
        to: Option<ObjectName>,
    },
    /// ```sql
    /// CREATE TABLE
    /// ```
    CreateTable(CreateTable),
    /// ```sql
    /// CREATE VIRTUAL TABLE .. USING <module_name> (<module_args>)`
    /// ```
    /// Sqlite specific statement
    CreateVirtualTable {
        #[cfg_attr(feature = "visitor", visit(with = "visit_relation"))]
        name: ObjectName,
        if_not_exists: bool,
        module_name: Ident,
        module_args: Vec<Ident>,
    },
    /// ```sql
    /// `CREATE INDEX`
    /// ```
    CreateIndex(CreateIndex),
    /// ```sql
    /// CREATE ROLE
    /// ```
    /// See [postgres](https://www.postgresql.org/docs/current/sql-createrole.html)
    CreateRole {
        names: Vec<ObjectName>,
        if_not_exists: bool,
        // Postgres
        login: Option<bool>,
        inherit: Option<bool>,
        bypassrls: Option<bool>,
        password: Option<Password>,
        superuser: Option<bool>,
        create_db: Option<bool>,
        create_role: Option<bool>,
        replication: Option<bool>,
        connection_limit: Option<Expr>,
        valid_until: Option<Expr>,
        in_role: Vec<Ident>,
        in_group: Vec<Ident>,
        role: Vec<Ident>,
        user: Vec<Ident>,
        admin: Vec<Ident>,
        // MSSQL
        authorization_owner: Option<ObjectName>,
    },
    /// ```sql
    /// CREATE SECRET
    /// ```
    /// See [duckdb](https://duckdb.org/docs/sql/statements/create_secret.html)
    CreateSecret {
        or_replace: bool,
        temporary: Option<bool>,
        if_not_exists: bool,
        name: Option<Ident>,
        storage_specifier: Option<Ident>,
        secret_type: Ident,
        options: Vec<SecretOption>,
    },
    /// ```sql
    /// CREATE POLICY
    /// ```
    /// See [PostgreSQL](https://www.postgresql.org/docs/current/sql-createpolicy.html)
    CreatePolicy {
        name: Ident,
        #[cfg_attr(feature = "visitor", visit(with = "visit_relation"))]
        table_name: ObjectName,
        policy_type: Option<CreatePolicyType>,
        command: Option<CreatePolicyCommand>,
        to: Option<Vec<Owner>>,
        using: Option<Expr>,
        with_check: Option<Expr>,
    },
    /// ```sql
    /// ALTER TABLE
    /// ```
    AlterTable {
        /// Table name
        #[cfg_attr(feature = "visitor", visit(with = "visit_relation"))]
        name: ObjectName,
        if_exists: bool,
        only: bool,
        operations: Vec<AlterTableOperation>,
        location: Option<HiveSetLocation>,
        /// ClickHouse dialect supports `ON CLUSTER` clause for ALTER TABLE
        /// For example: `ALTER TABLE table_name ON CLUSTER cluster_name ADD COLUMN c UInt32`
        /// [ClickHouse](https://clickhouse.com/docs/en/sql-reference/statements/alter/update)
        on_cluster: Option<Ident>,
    },
    /// ```sql
    /// ALTER INDEX
    /// ```
    AlterIndex {
        name: ObjectName,
        operation: AlterIndexOperation,
    },
    /// ```sql
    /// ALTER VIEW
    /// ```
    AlterView {
        /// View name
        #[cfg_attr(feature = "visitor", visit(with = "visit_relation"))]
        name: ObjectName,
        columns: Vec<Ident>,
        query: Box<Query>,
        with_options: Vec<SqlOption>,
    },
    /// ```sql
    /// ALTER ROLE
    /// ```
    AlterRole {
        name: Ident,
        operation: AlterRoleOperation,
    },
    /// ```sql
    /// ALTER POLICY <NAME> ON <TABLE NAME> [<OPERATION>]
    /// ```
    /// (Postgresql-specific)
    AlterPolicy {
        name: Ident,
        #[cfg_attr(feature = "visitor", visit(with = "visit_relation"))]
        table_name: ObjectName,
        operation: AlterPolicyOperation,
    },
    /// ```sql
    /// ATTACH DATABASE 'path/to/file' AS alias
    /// ```
    /// (SQLite-specific)
    AttachDatabase {
        /// The name to bind to the newly attached database
        schema_name: Ident,
        /// An expression that indicates the path to the database file
        database_file_name: Expr,
        /// true if the syntax is 'ATTACH DATABASE', false if it's just 'ATTACH'
        database: bool,
    },
    /// (DuckDB-specific)
    /// ```sql
    /// ATTACH 'sqlite_file.db' AS sqlite_db (READ_ONLY, TYPE SQLITE);
    /// ```
    /// See <https://duckdb.org/docs/sql/statements/attach.html>
    AttachDuckDBDatabase {
        if_not_exists: bool,
        /// true if the syntax is 'ATTACH DATABASE', false if it's just 'ATTACH'
        database: bool,
        /// An expression that indicates the path to the database file
        database_path: Ident,
        database_alias: Option<Ident>,
        attach_options: Vec<AttachDuckDBDatabaseOption>,
    },
    /// (DuckDB-specific)
    /// ```sql
    /// DETACH db_alias;
    /// ```
    /// See <https://duckdb.org/docs/sql/statements/attach.html>
    DetachDuckDBDatabase {
        if_exists: bool,
        /// true if the syntax is 'DETACH DATABASE', false if it's just 'DETACH'
        database: bool,
        database_alias: Ident,
    },
    /// ```sql
    /// DROP [TABLE, VIEW, ...]
    /// ```
    Drop {
        /// The type of the object to drop: TABLE, VIEW, etc.
        object_type: ObjectType,
        /// An optional `IF EXISTS` clause. (Non-standard.)
        if_exists: bool,
        /// One or more objects to drop. (ANSI SQL requires exactly one.)
        names: Vec<ObjectName>,
        /// Whether `CASCADE` was specified. This will be `false` when
        /// `RESTRICT` or no drop behavior at all was specified.
        cascade: bool,
        /// Whether `RESTRICT` was specified. This will be `false` when
        /// `CASCADE` or no drop behavior at all was specified.
        restrict: bool,
        /// Hive allows you specify whether the table's stored data will be
        /// deleted along with the dropped table
        purge: bool,
        /// MySQL-specific "TEMPORARY" keyword
        temporary: bool,
    },
    /// ```sql
    /// DROP FUNCTION
    /// ```
    DropFunction {
        if_exists: bool,
        /// One or more function to drop
        func_desc: Vec<FunctionDesc>,
        /// `CASCADE` or `RESTRICT`
        option: Option<ReferentialAction>,
    },
    /// ```sql
    /// DROP PROCEDURE
    /// ```
    DropProcedure {
        if_exists: bool,
        /// One or more function to drop
        proc_desc: Vec<FunctionDesc>,
        /// `CASCADE` or `RESTRICT`
        option: Option<ReferentialAction>,
    },
    /// ```sql
    /// DROP SECRET
    /// ```
    DropSecret {
        if_exists: bool,
        temporary: Option<bool>,
        name: Ident,
        storage_specifier: Option<Ident>,
    },
    ///```sql
    /// DROP POLICY
    /// ```
    /// See [PostgreSQL](https://www.postgresql.org/docs/current/sql-droppolicy.html)
    DropPolicy {
        if_exists: bool,
        name: Ident,
        table_name: ObjectName,
        option: Option<ReferentialAction>,
    },
    /// ```sql
    /// DECLARE
    /// ```
    /// Declare Cursor Variables
    ///
    /// Note: this is a PostgreSQL-specific statement,
    /// but may also compatible with other SQL.
    Declare { stmts: Vec<Declare> },
    /// ```sql
    /// CREATE EXTENSION [ IF NOT EXISTS ] extension_name
    ///     [ WITH ] [ SCHEMA schema_name ]
    ///              [ VERSION version ]
    ///              [ CASCADE ]
    /// ```
    ///
    /// Note: this is a PostgreSQL-specific statement,
    CreateExtension {
        name: Ident,
        if_not_exists: bool,
        cascade: bool,
        schema: Option<Ident>,
        version: Option<Ident>,
    },
    /// ```sql
    /// FETCH
    /// ```
    /// Retrieve rows from a query using a cursor
    ///
    /// Note: this is a PostgreSQL-specific statement,
    /// but may also compatible with other SQL.
    Fetch {
        /// Cursor name
        name: Ident,
        direction: FetchDirection,
        /// Optional, It's possible to fetch rows form cursor to the table
        into: Option<ObjectName>,
    },
    /// ```sql
    /// FLUSH [NO_WRITE_TO_BINLOG | LOCAL] flush_option [, flush_option] ... | tables_option
    /// ```
    ///
    /// Note: this is a Mysql-specific statement,
    /// but may also compatible with other SQL.
    Flush {
        object_type: FlushType,
        location: Option<FlushLocation>,
        channel: Option<String>,
        read_lock: bool,
        export: bool,
        tables: Vec<ObjectName>,
    },
    /// ```sql
    /// DISCARD [ ALL | PLANS | SEQUENCES | TEMPORARY | TEMP ]
    /// ```
    ///
    /// Note: this is a PostgreSQL-specific statement,
    /// but may also compatible with other SQL.
    Discard { object_type: DiscardObject },
    /// ```sql
    /// SET [ SESSION | LOCAL ] ROLE role_name
    /// ```
    ///
    /// Sets session state. Examples: [ANSI][1], [Postgresql][2], [MySQL][3], and [Oracle][4]
    ///
    /// [1]: https://jakewheat.github.io/sql-overview/sql-2016-foundation-grammar.html#set-role-statement
    /// [2]: https://www.postgresql.org/docs/14/sql-set-role.html
    /// [3]: https://dev.mysql.com/doc/refman/8.0/en/set-role.html
    /// [4]: https://docs.oracle.com/cd/B19306_01/server.102/b14200/statements_10004.htm
    SetRole {
        /// Non-ANSI optional identifier to inform if the role is defined inside the current session (`SESSION`) or transaction (`LOCAL`).
        context_modifier: ContextModifier,
        /// Role name. If NONE is specified, then the current role name is removed.
        role_name: Option<Ident>,
    },
    /// ```sql
    /// SET <variable> = expression;
    /// SET (variable[, ...]) = (expression[, ...]);
    /// ```
    ///
    /// Note: this is not a standard SQL statement, but it is supported by at
    /// least MySQL and PostgreSQL. Not all MySQL-specific syntactic forms are
    /// supported yet.
    SetVariable {
        local: bool,
        hivevar: bool,
        variables: OneOrManyWithParens<ObjectName>,
        value: Vec<Expr>,
    },
    /// ```sql
    /// SET TIME ZONE <value>
    /// ```
    ///
    /// Note: this is a PostgreSQL-specific statements
    /// `SET TIME ZONE <value>` is an alias for `SET timezone TO <value>` in PostgreSQL
    SetTimeZone { local: bool, value: Expr },
    /// ```sql
    /// SET NAMES 'charset_name' [COLLATE 'collation_name']
    /// ```
    ///
    /// Note: this is a MySQL-specific statement.
    SetNames {
        charset_name: String,
        collation_name: Option<String>,
    },
    /// ```sql
    /// SET NAMES DEFAULT
    /// ```
    ///
    /// Note: this is a MySQL-specific statement.
    SetNamesDefault {},
    /// `SHOW FUNCTIONS`
    ///
    /// Note: this is a Presto-specific statement.
    ShowFunctions { filter: Option<ShowStatementFilter> },
    /// ```sql
    /// SHOW <variable>
    /// ```
    ///
    /// Note: this is a PostgreSQL-specific statement.
    ShowVariable { variable: Vec<Ident> },
    /// ```sql
    /// SHOW [GLOBAL | SESSION] STATUS [LIKE 'pattern' | WHERE expr]
    /// ```
    ///
    /// Note: this is a MySQL-specific statement.
    ShowStatus {
        filter: Option<ShowStatementFilter>,
        global: bool,
        session: bool,
    },
    /// ```sql
    /// SHOW VARIABLES
    /// ```
    ///
    /// Note: this is a MySQL-specific statement.
    ShowVariables {
        filter: Option<ShowStatementFilter>,
        global: bool,
        session: bool,
    },
    /// ```sql
    /// SHOW CREATE TABLE
    /// ```
    ///
    /// Note: this is a MySQL-specific statement.
    ShowCreate {
        obj_type: ShowCreateObject,
        obj_name: ObjectName,
    },
    /// ```sql
    /// SHOW COLUMNS
    /// ```
    ShowColumns {
        extended: bool,
        full: bool,
        show_options: ShowStatementOptions,
    },
    /// ```sql
    /// SHOW DATABASES
    /// ```
    ShowDatabases {
        terse: bool,
        history: bool,
        show_options: ShowStatementOptions,
    },
    /// ```sql
    /// SHOW SCHEMAS
    /// ```
    ShowSchemas {
        terse: bool,
        history: bool,
        show_options: ShowStatementOptions,
    },
    /// ```sql
    /// SHOW TABLES
    /// ```
    ShowTables {
        terse: bool,
        history: bool,
        extended: bool,
        full: bool,
        external: bool,
        show_options: ShowStatementOptions,
    },
    /// ```sql
    /// SHOW VIEWS
    /// ```
    ShowViews {
        terse: bool,
        materialized: bool,
        show_options: ShowStatementOptions,
    },
    /// ```sql
    /// SHOW COLLATION
    /// ```
    ///
    /// Note: this is a MySQL-specific statement.
    ShowCollation { filter: Option<ShowStatementFilter> },
    /// ```sql
    /// `USE ...`
    /// ```
    Use(Use),
    /// ```sql
    /// START  [ TRANSACTION | WORK ] | START TRANSACTION } ...
    /// ```
    /// If `begin` is false.
    ///
    /// ```sql
    /// `BEGIN  [ TRANSACTION | WORK ] | START TRANSACTION } ...`
    /// ```
    /// If `begin` is true
    StartTransaction {
        modes: Vec<TransactionMode>,
        begin: bool,
        transaction: Option<BeginTransactionKind>,
        /// Only for SQLite
        modifier: Option<TransactionModifier>,
    },
    /// ```sql
    /// SET TRANSACTION ...
    /// ```
    SetTransaction {
        modes: Vec<TransactionMode>,
        snapshot: Option<Value>,
        session: bool,
    },
    /// ```sql
    /// COMMENT ON ...
    /// ```
    ///
    /// Note: this is a PostgreSQL-specific statement.
    Comment {
        object_type: CommentObject,
        object_name: ObjectName,
        comment: Option<String>,
        /// An optional `IF EXISTS` clause. (Non-standard.)
        /// See <https://docs.snowflake.com/en/sql-reference/sql/comment>
        if_exists: bool,
    },
    /// ```sql
    /// COMMIT [ TRANSACTION | WORK ] [ AND [ NO ] CHAIN ]
    /// ```
    Commit { chain: bool },
    /// ```sql
    /// ROLLBACK [ TRANSACTION | WORK ] [ AND [ NO ] CHAIN ] [ TO [ SAVEPOINT ] savepoint_name ]
    /// ```
    Rollback {
        chain: bool,
        savepoint: Option<Ident>,
    },
    /// ```sql
    /// CREATE SCHEMA
    /// ```
    CreateSchema {
        /// `<schema name> | AUTHORIZATION <schema authorization identifier>  | <schema name>  AUTHORIZATION <schema authorization identifier>`
        schema_name: SchemaName,
        if_not_exists: bool,
    },
    /// ```sql
    /// CREATE DATABASE
    /// ```
    CreateDatabase {
        db_name: ObjectName,
        if_not_exists: bool,
        location: Option<String>,
        managed_location: Option<String>,
    },
    /// ```sql
    /// CREATE FUNCTION
    /// ```
    ///
    /// Supported variants:
    /// 1. [Hive](https://cwiki.apache.org/confluence/display/hive/languagemanual+ddl#LanguageManualDDL-Create/Drop/ReloadFunction)
    /// 2. [Postgres](https://www.postgresql.org/docs/15/sql-createfunction.html)
    /// 3. [BigQuery](https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#create_function_statement)
    CreateFunction(CreateFunction),
    /// CREATE TRIGGER
    ///
    /// Examples:
    ///
    /// ```sql
    /// CREATE TRIGGER trigger_name
    /// BEFORE INSERT ON table_name
    /// FOR EACH ROW
    /// EXECUTE FUNCTION trigger_function();
    /// ```
    ///
    /// Postgres: <https://www.postgresql.org/docs/current/sql-createtrigger.html>
    CreateTrigger {
        /// The `OR REPLACE` clause is used to re-create the trigger if it already exists.
        ///
        /// Example:
        /// ```sql
        /// CREATE OR REPLACE TRIGGER trigger_name
        /// AFTER INSERT ON table_name
        /// FOR EACH ROW
        /// EXECUTE FUNCTION trigger_function();
        /// ```
        or_replace: bool,
        /// The `CONSTRAINT` keyword is used to create a trigger as a constraint.
        is_constraint: bool,
        /// The name of the trigger to be created.
        name: ObjectName,
        /// Determines whether the function is called before, after, or instead of the event.
        ///
        /// Example of BEFORE:
        ///
        /// ```sql
        /// CREATE TRIGGER trigger_name
        /// BEFORE INSERT ON table_name
        /// FOR EACH ROW
        /// EXECUTE FUNCTION trigger_function();
        /// ```
        ///
        /// Example of AFTER:
        ///
        /// ```sql
        /// CREATE TRIGGER trigger_name
        /// AFTER INSERT ON table_name
        /// FOR EACH ROW
        /// EXECUTE FUNCTION trigger_function();
        /// ```
        ///
        /// Example of INSTEAD OF:
        ///
        /// ```sql
        /// CREATE TRIGGER trigger_name
        /// INSTEAD OF INSERT ON table_name
        /// FOR EACH ROW
        /// EXECUTE FUNCTION trigger_function();
        /// ```
        period: TriggerPeriod,
        /// Multiple events can be specified using OR, such as `INSERT`, `UPDATE`, `DELETE`, or `TRUNCATE`.
        events: Vec<TriggerEvent>,
        /// The table on which the trigger is to be created.
        table_name: ObjectName,
        /// The optional referenced table name that can be referenced via
        /// the `FROM` keyword.
        referenced_table_name: Option<ObjectName>,
        /// This keyword immediately precedes the declaration of one or two relation names that provide access to the transition relations of the triggering statement.
        referencing: Vec<TriggerReferencing>,
        /// This specifies whether the trigger function should be fired once for
        /// every row affected by the trigger event, or just once per SQL statement.
        trigger_object: TriggerObject,
        /// Whether to include the `EACH` term of the `FOR EACH`, as it is optional syntax.
        include_each: bool,
        ///  Triggering conditions
        condition: Option<Expr>,
        /// Execute logic block
        exec_body: TriggerExecBody,
        /// The characteristic of the trigger, which include whether the trigger is `DEFERRABLE`, `INITIALLY DEFERRED`, or `INITIALLY IMMEDIATE`,
        characteristics: Option<ConstraintCharacteristics>,
    },
    /// DROP TRIGGER
    ///
    /// ```sql
    /// DROP TRIGGER [ IF EXISTS ] name ON table_name [ CASCADE | RESTRICT ]
    /// ```
    ///
    DropTrigger {
        if_exists: bool,
        trigger_name: ObjectName,
        table_name: ObjectName,
        /// `CASCADE` or `RESTRICT`
        option: Option<ReferentialAction>,
    },
    /// ```sql
    /// CREATE PROCEDURE
    /// ```
    CreateProcedure {
        or_alter: bool,
        name: ObjectName,
        params: Option<Vec<ProcedureParam>>,
        body: Vec<Statement>,
    },
    /// ```sql
    /// CREATE MACRO
    /// ```
    ///
    /// Supported variants:
    /// 1. [DuckDB](https://duckdb.org/docs/sql/statements/create_macro)
    CreateMacro {
        or_replace: bool,
        temporary: bool,
        name: ObjectName,
        args: Option<Vec<MacroArg>>,
        definition: MacroDefinition,
    },
    /// ```sql
    /// CREATE STAGE
    /// ```
    /// See <https://docs.snowflake.com/en/sql-reference/sql/create-stage>
    CreateStage {
        or_replace: bool,
        temporary: bool,
        if_not_exists: bool,
        name: ObjectName,
        stage_params: StageParamsObject,
        directory_table_params: DataLoadingOptions,
        file_format: DataLoadingOptions,
        copy_options: DataLoadingOptions,
        comment: Option<String>,
    },
    /// ```sql
    /// ASSERT <condition> [AS <message>]
    /// ```
    Assert {
        condition: Expr,
        message: Option<Expr>,
    },
    /// ```sql
    /// GRANT privileges ON objects TO grantees
    /// ```
    Grant {
        privileges: Privileges,
        objects: GrantObjects,
        grantees: Vec<Ident>,
        with_grant_option: bool,
        granted_by: Option<Ident>,
    },
    /// ```sql
    /// REVOKE privileges ON objects FROM grantees
    /// ```
    Revoke {
        privileges: Privileges,
        objects: GrantObjects,
        grantees: Vec<Ident>,
        granted_by: Option<Ident>,
        cascade: bool,
    },
    /// ```sql
    /// DEALLOCATE [ PREPARE ] { name | ALL }
    /// ```
    ///
    /// Note: this is a PostgreSQL-specific statement.
    Deallocate { name: Ident, prepare: bool },
    /// ```sql
    /// EXECUTE name [ ( parameter [, ...] ) ] [USING <expr>]
    /// ```
    ///
    /// Note: this statement is supported by Postgres and MSSQL, with slight differences in syntax.
    ///
    /// Postgres: <https://www.postgresql.org/docs/current/sql-execute.html>
    /// MSSQL: <https://learn.microsoft.com/en-us/sql/relational-databases/stored-procedures/execute-a-stored-procedure>
    Execute {
        name: ObjectName,
        parameters: Vec<Expr>,
        has_parentheses: bool,
        using: Vec<Expr>,
    },
    /// ```sql
    /// PREPARE name [ ( data_type [, ...] ) ] AS statement
    /// ```
    ///
    /// Note: this is a PostgreSQL-specific statement.
    Prepare {
        name: Ident,
        data_types: Vec<DataType>,
        statement: Box<Statement>,
    },
    /// ```sql
    /// KILL [CONNECTION | QUERY | MUTATION]
    /// ```
    ///
    /// See <https://clickhouse.com/docs/en/sql-reference/statements/kill/>
    /// See <https://dev.mysql.com/doc/refman/8.0/en/kill.html>
    Kill {
        modifier: Option<KillType>,
        // processlist_id
        id: u64,
    },
    /// ```sql
    /// [EXPLAIN | DESC | DESCRIBE] TABLE
    /// ```
    /// Note: this is a MySQL-specific statement. See <https://dev.mysql.com/doc/refman/8.0/en/explain.html>
    ExplainTable {
        /// `EXPLAIN | DESC | DESCRIBE`
        describe_alias: DescribeAlias,
        /// Hive style `FORMATTED | EXTENDED`
        hive_format: Option<HiveDescribeFormat>,
        /// Snowflake and ClickHouse support `DESC|DESCRIBE TABLE <table_name>` syntax
        ///
        /// [Snowflake](https://docs.snowflake.com/en/sql-reference/sql/desc-table.html)
        /// [ClickHouse](https://clickhouse.com/docs/en/sql-reference/statements/describe-table)
        has_table_keyword: bool,
        /// Table name
        #[cfg_attr(feature = "visitor", visit(with = "visit_relation"))]
        table_name: ObjectName,
    },
    /// ```sql
    /// [EXPLAIN | DESC | DESCRIBE]  <statement>
    /// ```
    Explain {
        /// `EXPLAIN | DESC | DESCRIBE`
        describe_alias: DescribeAlias,
        /// Carry out the command and show actual run times and other statistics.
        analyze: bool,
        // Display additional information regarding the plan.
        verbose: bool,
        /// `EXPLAIN QUERY PLAN`
        /// Display the query plan without running the query.
        ///
        /// [SQLite](https://sqlite.org/lang_explain.html)
        query_plan: bool,
        /// A SQL query that specifies what to explain
        statement: Box<Statement>,
        /// Optional output format of explain
        format: Option<AnalyzeFormat>,
        /// Postgres style utility options, `(analyze, verbose true)`
        options: Option<Vec<UtilityOption>>,
    },
    /// ```sql
    /// SAVEPOINT
    /// ```
    /// Define a new savepoint within the current transaction
    Savepoint { name: Ident },
    /// ```sql
    /// RELEASE [ SAVEPOINT ] savepoint_name
    /// ```
    ReleaseSavepoint { name: Ident },
    /// A `MERGE` statement.
    ///
    /// ```sql
    /// MERGE INTO <target_table> USING <source> ON <join_expr> { matchedClause | notMatchedClause } [ ... ]
    /// ```
    /// [Snowflake](https://docs.snowflake.com/en/sql-reference/sql/merge)
    /// [BigQuery](https://cloud.google.com/bigquery/docs/reference/standard-sql/dml-syntax#merge_statement)
    Merge {
        /// optional INTO keyword
        into: bool,
        /// Specifies the table to merge
        table: TableFactor,
        /// Specifies the table or subquery to join with the target table
        source: TableFactor,
        /// Specifies the expression on which to join the target table and source
        on: Box<Expr>,
        /// Specifies the actions to perform when values match or do not match.
        clauses: Vec<MergeClause>,
    },
    /// ```sql
    /// CACHE [ FLAG ] TABLE <table_name> [ OPTIONS('K1' = 'V1', 'K2' = V2) ] [ AS ] [ <query> ]
    /// ```
    ///
    /// See [Spark SQL docs] for more details.
    ///
    /// [Spark SQL docs]: https://docs.databricks.com/spark/latest/spark-sql/language-manual/sql-ref-syntax-aux-cache-cache-table.html
    Cache {
        /// Table flag
        table_flag: Option<ObjectName>,
        /// Table name

        #[cfg_attr(feature = "visitor", visit(with = "visit_relation"))]
        table_name: ObjectName,
        has_as: bool,
        /// Table confs
        options: Vec<SqlOption>,
        /// Cache table as a Query
        query: Option<Box<Query>>,
    },
    /// ```sql
    /// UNCACHE TABLE [ IF EXISTS ]  <table_name>
    /// ```
    UNCache {
        /// Table name
        #[cfg_attr(feature = "visitor", visit(with = "visit_relation"))]
        table_name: ObjectName,
        if_exists: bool,
    },
    /// ```sql
    /// CREATE [ { TEMPORARY | TEMP } ] SEQUENCE [ IF NOT EXISTS ] <sequence_name>
    /// ```
    /// Define a new sequence:
    CreateSequence {
        temporary: bool,
        if_not_exists: bool,
        name: ObjectName,
        data_type: Option<DataType>,
        sequence_options: Vec<SequenceOptions>,
        owned_by: Option<ObjectName>,
    },
    /// ```sql
    /// CREATE TYPE <name>
    /// ```
    CreateType {
        name: ObjectName,
        representation: UserDefinedTypeRepresentation,
    },
    /// ```sql
    /// PRAGMA <schema-name>.<pragma-name> = <pragma-value>
    /// ```
    Pragma {
        name: ObjectName,
        value: Option<Value>,
        is_eq: bool,
    },
    /// ```sql
    /// LOCK TABLES <table_name> [READ [LOCAL] | [LOW_PRIORITY] WRITE]
    /// ```
    /// Note: this is a MySQL-specific statement. See <https://dev.mysql.com/doc/refman/8.0/en/lock-tables.html>
    LockTables { tables: Vec<LockTable> },
    /// ```sql
    /// UNLOCK TABLES
    /// ```
    /// Note: this is a MySQL-specific statement. See <https://dev.mysql.com/doc/refman/8.0/en/lock-tables.html>
    UnlockTables,
    /// ```sql
    /// UNLOAD(statement) TO <destination> [ WITH options ]
    /// ```
    /// See Redshift <https://docs.aws.amazon.com/redshift/latest/dg/r_UNLOAD.html> and
    // Athena <https://docs.aws.amazon.com/athena/latest/ug/unload.html>
    Unload {
        query: Box<Query>,
        to: Ident,
        with: Vec<SqlOption>,
    },
    /// ```sql
    /// OPTIMIZE TABLE [db.]name [ON CLUSTER cluster] [PARTITION partition | PARTITION ID 'partition_id'] [FINAL] [DEDUPLICATE [BY expression]]
    /// ```
    ///
    /// See ClickHouse <https://clickhouse.com/docs/en/sql-reference/statements/optimize>
    OptimizeTable {
        name: ObjectName,
        on_cluster: Option<Ident>,
        partition: Option<Partition>,
        include_final: bool,
        deduplicate: Option<Deduplicate>,
    },
    /// ```sql
    /// LISTEN
    /// ```
    /// listen for a notification channel
    ///
    /// See Postgres <https://www.postgresql.org/docs/current/sql-listen.html>
    LISTEN { channel: Ident },
    /// ```sql
    /// UNLISTEN
    /// ```
    /// stop listening for a notification
    ///
    /// See Postgres <https://www.postgresql.org/docs/current/sql-unlisten.html>
    UNLISTEN { channel: Ident },
    /// ```sql
    /// NOTIFY channel [ , payload ]
    /// ```
    /// send a notification event together with an optional “payload” string to channel
    ///
    /// See Postgres <https://www.postgresql.org/docs/current/sql-notify.html>
    NOTIFY {
        channel: Ident,
        payload: Option<String>,
    },
    /// ```sql
    /// LOAD DATA [LOCAL] INPATH 'filepath' [OVERWRITE] INTO TABLE tablename
    /// [PARTITION (partcol1=val1, partcol2=val2 ...)]
    /// [INPUTFORMAT 'inputformat' SERDE 'serde']
    /// ```
    /// Loading files into tables
    ///
    /// See Hive <https://cwiki.apache.org/confluence/pages/viewpage.action?pageId=27362036#LanguageManualDML-Loadingfilesintotables>
    LoadData {
        local: bool,
        inpath: String,
        overwrite: bool,
        table_name: ObjectName,
        partitioned: Option<Vec<Expr>>,
        table_format: Option<HiveLoadDataFormat>,
    },
}

impl fmt::Display for Statement {
    // Clippy thinks this function is too complicated, but it is painful to
    // split up without extracting structs for each `Statement` variant.
    #[allow(clippy::cognitive_complexity)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Statement::Flush {
                object_type,
                location,
                channel,
                read_lock,
                export,
                tables,
            } => {
                write!(f, "FLUSH")?;
                if let Some(location) = location {
                    write!(f, " {location}")?;
                }
                write!(f, " {object_type}")?;

                if let Some(channel) = channel {
                    write!(f, " FOR CHANNEL {channel}")?;
                }

                write!(
                    f,
                    "{tables}{read}{export}",
                    tables = if !tables.is_empty() {
                        " ".to_string() + &display_comma_separated(tables).to_string()
                    } else {
                        "".to_string()
                    },
                    export = if *export { " FOR EXPORT" } else { "" },
                    read = if *read_lock { " WITH READ LOCK" } else { "" }
                )
            }
            Statement::Kill { modifier, id } => {
                write!(f, "KILL ")?;

                if let Some(m) = modifier {
                    write!(f, "{m} ")?;
                }

                write!(f, "{id}")
            }
            Statement::ExplainTable {
                describe_alias,
                hive_format,
                has_table_keyword,
                table_name,
            } => {
                write!(f, "{describe_alias} ")?;

                if let Some(format) = hive_format {
                    write!(f, "{} ", format)?;
                }
                if *has_table_keyword {
                    write!(f, "TABLE ")?;
                }

                write!(f, "{table_name}")
            }
            Statement::Explain {
                describe_alias,
                verbose,
                analyze,
                query_plan,
                statement,
                format,
                options,
            } => {
                write!(f, "{describe_alias} ")?;

                if *query_plan {
                    write!(f, "QUERY PLAN ")?;
                }
                if *analyze {
                    write!(f, "ANALYZE ")?;
                }

                if *verbose {
                    write!(f, "VERBOSE ")?;
                }

                if let Some(format) = format {
                    write!(f, "FORMAT {format} ")?;
                }

                if let Some(options) = options {
                    write!(f, "({}) ", display_comma_separated(options))?;
                }

                write!(f, "{statement}")
            }
            Statement::Query(s) => write!(f, "{s}"),
            Statement::Declare { stmts } => {
                write!(f, "DECLARE ")?;
                write!(f, "{}", display_separated(stmts, "; "))
            }
            Statement::Fetch {
                name,
                direction,
                into,
            } => {
                write!(f, "FETCH {direction} ")?;

                write!(f, "IN {name}")?;

                if let Some(into) = into {
                    write!(f, " INTO {into}")?;
                }

                Ok(())
            }
            Statement::Directory {
                overwrite,
                local,
                path,
                file_format,
                source,
            } => {
                write!(
                    f,
                    "INSERT{overwrite}{local} DIRECTORY '{path}'",
                    overwrite = if *overwrite { " OVERWRITE" } else { "" },
                    local = if *local { " LOCAL" } else { "" },
                    path = path
                )?;
                if let Some(ref ff) = file_format {
                    write!(f, " STORED AS {ff}")?
                }
                write!(f, " {source}")
            }
            Statement::Msck {
                table_name,
                repair,
                partition_action,
            } => {
                write!(
                    f,
                    "MSCK {repair}TABLE {table}",
                    repair = if *repair { "REPAIR " } else { "" },
                    table = table_name
                )?;
                if let Some(pa) = partition_action {
                    write!(f, " {pa}")?;
                }
                Ok(())
            }
            Statement::Truncate {
                table_names,
                partitions,
                table,
                only,
                identity,
                cascade,
                on_cluster,
            } => {
                let table = if *table { "TABLE " } else { "" };
                let only = if *only { "ONLY " } else { "" };

                write!(
                    f,
                    "TRUNCATE {table}{only}{table_names}",
                    table_names = display_comma_separated(table_names)
                )?;

                if let Some(identity) = identity {
                    match identity {
                        TruncateIdentityOption::Restart => write!(f, " RESTART IDENTITY")?,
                        TruncateIdentityOption::Continue => write!(f, " CONTINUE IDENTITY")?,
                    }
                }
                if let Some(cascade) = cascade {
                    match cascade {
                        TruncateCascadeOption::Cascade => write!(f, " CASCADE")?,
                        TruncateCascadeOption::Restrict => write!(f, " RESTRICT")?,
                    }
                }

                if let Some(ref parts) = partitions {
                    if !parts.is_empty() {
                        write!(f, " PARTITION ({})", display_comma_separated(parts))?;
                    }
                }
                if let Some(on_cluster) = on_cluster {
                    write!(f, " ON CLUSTER {on_cluster}")?;
                }
                Ok(())
            }
            Statement::AttachDatabase {
                schema_name,
                database_file_name,
                database,
            } => {
                let keyword = if *database { "DATABASE " } else { "" };
                write!(f, "ATTACH {keyword}{database_file_name} AS {schema_name}")
            }
            Statement::AttachDuckDBDatabase {
                if_not_exists,
                database,
                database_path,
                database_alias,
                attach_options,
            } => {
                write!(
                    f,
                    "ATTACH{database}{if_not_exists} {database_path}",
                    database = if *database { " DATABASE" } else { "" },
                    if_not_exists = if *if_not_exists { " IF NOT EXISTS" } else { "" },
                )?;
                if let Some(alias) = database_alias {
                    write!(f, " AS {alias}")?;
                }
                if !attach_options.is_empty() {
                    write!(f, " ({})", display_comma_separated(attach_options))?;
                }
                Ok(())
            }
            Statement::DetachDuckDBDatabase {
                if_exists,
                database,
                database_alias,
            } => {
                write!(
                    f,
                    "DETACH{database}{if_exists} {database_alias}",
                    database = if *database { " DATABASE" } else { "" },
                    if_exists = if *if_exists { " IF EXISTS" } else { "" },
                )?;
                Ok(())
            }
            Statement::Analyze {
                table_name,
                partitions,
                for_columns,
                columns,
                cache_metadata,
                noscan,
                compute_statistics,
            } => {
                write!(f, "ANALYZE TABLE {table_name}")?;
                if let Some(ref parts) = partitions {
                    if !parts.is_empty() {
                        write!(f, " PARTITION ({})", display_comma_separated(parts))?;
                    }
                }

                if *compute_statistics {
                    write!(f, " COMPUTE STATISTICS")?;
                }
                if *noscan {
                    write!(f, " NOSCAN")?;
                }
                if *cache_metadata {
                    write!(f, " CACHE METADATA")?;
                }
                if *for_columns {
                    write!(f, " FOR COLUMNS")?;
                    if !columns.is_empty() {
                        write!(f, " {}", display_comma_separated(columns))?;
                    }
                }
                Ok(())
            }
            Statement::Insert(insert) => write!(f, "{insert}"),
            Statement::Install {
                extension_name: name,
            } => write!(f, "INSTALL {name}"),

            Statement::Load {
                extension_name: name,
            } => write!(f, "LOAD {name}"),

            Statement::Call(function) => write!(f, "CALL {function}"),

            Statement::Copy {
                source,
                to,
                target,
                options,
                legacy_options,
                values,
            } => {
                write!(f, "COPY")?;
                match source {
                    CopySource::Query(query) => write!(f, " ({query})")?,
                    CopySource::Table {
                        table_name,
                        columns,
                    } => {
                        write!(f, " {table_name}")?;
                        if !columns.is_empty() {
                            write!(f, " ({})", display_comma_separated(columns))?;
                        }
                    }
                }
                write!(f, " {} {}", if *to { "TO" } else { "FROM" }, target)?;
                if !options.is_empty() {
                    write!(f, " ({})", display_comma_separated(options))?;
                }
                if !legacy_options.is_empty() {
                    write!(f, " {}", display_separated(legacy_options, " "))?;
                }
                if !values.is_empty() {
                    writeln!(f, ";")?;
                    let mut delim = "";
                    for v in values {
                        write!(f, "{delim}")?;
                        delim = "\t";
                        if let Some(v) = v {
                            write!(f, "{v}")?;
                        } else {
                            write!(f, "\\N")?;
                        }
                    }
                    write!(f, "\n\\.")?;
                }
                Ok(())
            }
            Statement::Update {
                table,
                assignments,
                from,
                selection,
                returning,
                or,
            } => {
                write!(f, "UPDATE ")?;
                if let Some(or) = or {
                    write!(f, "{or} ")?;
                }
                write!(f, "{table}")?;
                if !assignments.is_empty() {
                    write!(f, " SET {}", display_comma_separated(assignments))?;
                }
                if let Some(from) = from {
                    write!(f, " FROM {from}")?;
                }
                if let Some(selection) = selection {
                    write!(f, " WHERE {selection}")?;
                }
                if let Some(returning) = returning {
                    write!(f, " RETURNING {}", display_comma_separated(returning))?;
                }
                Ok(())
            }
            Statement::Delete(delete) => write!(f, "{delete}"),
            Statement::Close { cursor } => {
                write!(f, "CLOSE {cursor}")?;

                Ok(())
            }
            Statement::CreateDatabase {
                db_name,
                if_not_exists,
                location,
                managed_location,
            } => {
                write!(f, "CREATE DATABASE")?;
                if *if_not_exists {
                    write!(f, " IF NOT EXISTS")?;
                }
                write!(f, " {db_name}")?;
                if let Some(l) = location {
                    write!(f, " LOCATION '{l}'")?;
                }
                if let Some(ml) = managed_location {
                    write!(f, " MANAGEDLOCATION '{ml}'")?;
                }
                Ok(())
            }
            Statement::CreateFunction(create_function) => create_function.fmt(f),
            Statement::CreateTrigger {
                or_replace,
                is_constraint,
                name,
                period,
                events,
                table_name,
                referenced_table_name,
                referencing,
                trigger_object,
                condition,
                include_each,
                exec_body,
                characteristics,
            } => {
                write!(
                    f,
                    "CREATE {or_replace}{is_constraint}TRIGGER {name} {period}",
                    or_replace = if *or_replace { "OR REPLACE " } else { "" },
                    is_constraint = if *is_constraint { "CONSTRAINT " } else { "" },
                )?;

                if !events.is_empty() {
                    write!(f, " {}", display_separated(events, " OR "))?;
                }
                write!(f, " ON {table_name}")?;

                if let Some(referenced_table_name) = referenced_table_name {
                    write!(f, " FROM {referenced_table_name}")?;
                }

                if let Some(characteristics) = characteristics {
                    write!(f, " {characteristics}")?;
                }

                if !referencing.is_empty() {
                    write!(f, " REFERENCING {}", display_separated(referencing, " "))?;
                }

                if *include_each {
                    write!(f, " FOR EACH {trigger_object}")?;
                } else {
                    write!(f, " FOR {trigger_object}")?;
                }
                if let Some(condition) = condition {
                    write!(f, " WHEN {condition}")?;
                }
                write!(f, " EXECUTE {exec_body}")
            }
            Statement::DropTrigger {
                if_exists,
                trigger_name,
                table_name,
                option,
            } => {
                write!(f, "DROP TRIGGER")?;
                if *if_exists {
                    write!(f, " IF EXISTS")?;
                }
                write!(f, " {trigger_name} ON {table_name}")?;
                if let Some(option) = option {
                    write!(f, " {option}")?;
                }
                Ok(())
            }
            Statement::CreateProcedure {
                name,
                or_alter,
                params,
                body,
            } => {
                write!(
                    f,
                    "CREATE {or_alter}PROCEDURE {name}",
                    or_alter = if *or_alter { "OR ALTER " } else { "" },
                    name = name
                )?;

                if let Some(p) = params {
                    if !p.is_empty() {
                        write!(f, " ({})", display_comma_separated(p))?;
                    }
                }
                write!(
                    f,
                    " AS BEGIN {body} END",
                    body = display_separated(body, "; ")
                )
            }
            Statement::CreateMacro {
                or_replace,
                temporary,
                name,
                args,
                definition,
            } => {
                write!(
                    f,
                    "CREATE {or_replace}{temp}MACRO {name}",
                    temp = if *temporary { "TEMPORARY " } else { "" },
                    or_replace = if *or_replace { "OR REPLACE " } else { "" },
                )?;
                if let Some(args) = args {
                    write!(f, "({})", display_comma_separated(args))?;
                }
                match definition {
                    MacroDefinition::Expr(expr) => write!(f, " AS {expr}")?,
                    MacroDefinition::Table(query) => write!(f, " AS TABLE {query}")?,
                }
                Ok(())
            }
            Statement::CreateView {
                name,
                or_replace,
                columns,
                query,
                materialized,
                options,
                cluster_by,
                comment,
                with_no_schema_binding,
                if_not_exists,
                temporary,
                to,
            } => {
                write!(
                    f,
                    "CREATE {or_replace}{materialized}{temporary}VIEW {if_not_exists}{name}{to}",
                    or_replace = if *or_replace { "OR REPLACE " } else { "" },
                    materialized = if *materialized { "MATERIALIZED " } else { "" },
                    name = name,
                    temporary = if *temporary { "TEMPORARY " } else { "" },
                    if_not_exists = if *if_not_exists { "IF NOT EXISTS " } else { "" },
                    to = to
                        .as_ref()
                        .map(|to| format!(" TO {to}"))
                        .unwrap_or_default()
                )?;
                if !columns.is_empty() {
                    write!(f, " ({})", display_comma_separated(columns))?;
                }
                if matches!(options, CreateTableOptions::With(_)) {
                    write!(f, " {options}")?;
                }
                if let Some(comment) = comment {
                    write!(
                        f,
                        " COMMENT = '{}'",
                        value::escape_single_quote_string(comment)
                    )?;
                }
                if !cluster_by.is_empty() {
                    write!(f, " CLUSTER BY ({})", display_comma_separated(cluster_by))?;
                }
                if matches!(options, CreateTableOptions::Options(_)) {
                    write!(f, " {options}")?;
                }
                write!(f, " AS {query}")?;
                if *with_no_schema_binding {
                    write!(f, " WITH NO SCHEMA BINDING")?;
                }
                Ok(())
            }
            Statement::CreateTable(create_table) => create_table.fmt(f),
            Statement::LoadData {
                local,
                inpath,
                overwrite,
                table_name,
                partitioned,
                table_format,
            } => {
                write!(
                    f,
                    "LOAD DATA {local}INPATH '{inpath}' {overwrite}INTO TABLE {table_name}",
                    local = if *local { "LOCAL " } else { "" },
                    inpath = inpath,
                    overwrite = if *overwrite { "OVERWRITE " } else { "" },
                    table_name = table_name,
                )?;
                if let Some(ref parts) = &partitioned {
                    if !parts.is_empty() {
                        write!(f, " PARTITION ({})", display_comma_separated(parts))?;
                    }
                }
                if let Some(HiveLoadDataFormat {
                    serde,
                    input_format,
                }) = &table_format
                {
                    write!(f, " INPUTFORMAT {input_format} SERDE {serde}")?;
                }
                Ok(())
            }
            Statement::CreateVirtualTable {
                name,
                if_not_exists,
                module_name,
                module_args,
            } => {
                write!(
                    f,
                    "CREATE VIRTUAL TABLE {if_not_exists}{name} USING {module_name}",
                    if_not_exists = if *if_not_exists { "IF NOT EXISTS " } else { "" },
                    name = name,
                    module_name = module_name
                )?;
                if !module_args.is_empty() {
                    write!(f, " ({})", display_comma_separated(module_args))?;
                }
                Ok(())
            }
            Statement::CreateIndex(create_index) => create_index.fmt(f),
            Statement::CreateExtension {
                name,
                if_not_exists,
                cascade,
                schema,
                version,
            } => {
                write!(
                    f,
                    "CREATE EXTENSION {if_not_exists}{name}",
                    if_not_exists = if *if_not_exists { "IF NOT EXISTS " } else { "" }
                )?;
                if *cascade || schema.is_some() || version.is_some() {
                    write!(f, " WITH")?;

                    if let Some(name) = schema {
                        write!(f, " SCHEMA {name}")?;
                    }
                    if let Some(version) = version {
                        write!(f, " VERSION {version}")?;
                    }
                    if *cascade {
                        write!(f, " CASCADE")?;
                    }
                }

                Ok(())
            }
            Statement::CreateRole {
                names,
                if_not_exists,
                inherit,
                login,
                bypassrls,
                password,
                create_db,
                create_role,
                superuser,
                replication,
                connection_limit,
                valid_until,
                in_role,
                in_group,
                role,
                user,
                admin,
                authorization_owner,
            } => {
                write!(
                    f,
                    "CREATE ROLE {if_not_exists}{names}{superuser}{create_db}{create_role}{inherit}{login}{replication}{bypassrls}",
                    if_not_exists = if *if_not_exists { "IF NOT EXISTS " } else { "" },
                    names = display_separated(names, ", "),
                    superuser = match *superuser {
                        Some(true) => " SUPERUSER",
                        Some(false) => " NOSUPERUSER",
                        None => ""
                    },
                    create_db = match *create_db {
                        Some(true) => " CREATEDB",
                        Some(false) => " NOCREATEDB",
                        None => ""
                    },
                    create_role = match *create_role {
                        Some(true) => " CREATEROLE",
                        Some(false) => " NOCREATEROLE",
                        None => ""
                    },
                    inherit = match *inherit {
                        Some(true) => " INHERIT",
                        Some(false) => " NOINHERIT",
                        None => ""
                    },
                    login = match *login {
                        Some(true) => " LOGIN",
                        Some(false) => " NOLOGIN",
                        None => ""
                    },
                    replication = match *replication {
                        Some(true) => " REPLICATION",
                        Some(false) => " NOREPLICATION",
                        None => ""
                    },
                    bypassrls = match *bypassrls {
                        Some(true) => " BYPASSRLS",
                        Some(false) => " NOBYPASSRLS",
                        None => ""
                    }
                )?;
                if let Some(limit) = connection_limit {
                    write!(f, " CONNECTION LIMIT {limit}")?;
                }
                match password {
                    Some(Password::Password(pass)) => write!(f, " PASSWORD {pass}"),
                    Some(Password::NullPassword) => write!(f, " PASSWORD NULL"),
                    None => Ok(()),
                }?;
                if let Some(until) = valid_until {
                    write!(f, " VALID UNTIL {until}")?;
                }
                if !in_role.is_empty() {
                    write!(f, " IN ROLE {}", display_comma_separated(in_role))?;
                }
                if !in_group.is_empty() {
                    write!(f, " IN GROUP {}", display_comma_separated(in_group))?;
                }
                if !role.is_empty() {
                    write!(f, " ROLE {}", display_comma_separated(role))?;
                }
                if !user.is_empty() {
                    write!(f, " USER {}", display_comma_separated(user))?;
                }
                if !admin.is_empty() {
                    write!(f, " ADMIN {}", display_comma_separated(admin))?;
                }
                if let Some(owner) = authorization_owner {
                    write!(f, " AUTHORIZATION {owner}")?;
                }
                Ok(())
            }
            Statement::CreateSecret {
                or_replace,
                temporary,
                if_not_exists,
                name,
                storage_specifier,
                secret_type,
                options,
            } => {
                write!(
                    f,
                    "CREATE {or_replace}",
                    or_replace = if *or_replace { "OR REPLACE " } else { "" },
                )?;
                if let Some(t) = temporary {
                    write!(f, "{}", if *t { "TEMPORARY " } else { "PERSISTENT " })?;
                }
                write!(
                    f,
                    "SECRET {if_not_exists}",
                    if_not_exists = if *if_not_exists { "IF NOT EXISTS " } else { "" },
                )?;
                if let Some(n) = name {
                    write!(f, "{n} ")?;
                };
                if let Some(s) = storage_specifier {
                    write!(f, "IN {s} ")?;
                }
                write!(f, "( TYPE {secret_type}",)?;
                if !options.is_empty() {
                    write!(f, ", {o}", o = display_comma_separated(options))?;
                }
                write!(f, " )")?;
                Ok(())
            }
            Statement::CreatePolicy {
                name,
                table_name,
                policy_type,
                command,
                to,
                using,
                with_check,
            } => {
                write!(f, "CREATE POLICY {name} ON {table_name}")?;

                if let Some(policy_type) = policy_type {
                    match policy_type {
                        CreatePolicyType::Permissive => write!(f, " AS PERMISSIVE")?,
                        CreatePolicyType::Restrictive => write!(f, " AS RESTRICTIVE")?,
                    }
                }

                if let Some(command) = command {
                    match command {
                        CreatePolicyCommand::All => write!(f, " FOR ALL")?,
                        CreatePolicyCommand::Select => write!(f, " FOR SELECT")?,
                        CreatePolicyCommand::Insert => write!(f, " FOR INSERT")?,
                        CreatePolicyCommand::Update => write!(f, " FOR UPDATE")?,
                        CreatePolicyCommand::Delete => write!(f, " FOR DELETE")?,
                    }
                }

                if let Some(to) = to {
                    write!(f, " TO {}", display_comma_separated(to))?;
                }

                if let Some(using) = using {
                    write!(f, " USING ({using})")?;
                }

                if let Some(with_check) = with_check {
                    write!(f, " WITH CHECK ({with_check})")?;
                }

                Ok(())
            }
            Statement::AlterTable {
                name,
                if_exists,
                only,
                operations,
                location,
                on_cluster,
            } => {
                write!(f, "ALTER TABLE ")?;
                if *if_exists {
                    write!(f, "IF EXISTS ")?;
                }
                if *only {
                    write!(f, "ONLY ")?;
                }
                write!(f, "{name} ", name = name)?;
                if let Some(cluster) = on_cluster {
                    write!(f, "ON CLUSTER {cluster} ")?;
                }
                write!(
                    f,
                    "{operations}",
                    operations = display_comma_separated(operations)
                )?;
                if let Some(loc) = location {
                    write!(f, " {loc}")?
                }
                Ok(())
            }
            Statement::AlterIndex { name, operation } => {
                write!(f, "ALTER INDEX {name} {operation}")
            }
            Statement::AlterView {
                name,
                columns,
                query,
                with_options,
            } => {
                write!(f, "ALTER VIEW {name}")?;
                if !with_options.is_empty() {
                    write!(f, " WITH ({})", display_comma_separated(with_options))?;
                }
                if !columns.is_empty() {
                    write!(f, " ({})", display_comma_separated(columns))?;
                }
                write!(f, " AS {query}")
            }
            Statement::AlterRole { name, operation } => {
                write!(f, "ALTER ROLE {name} {operation}")
            }
            Statement::AlterPolicy {
                name,
                table_name,
                operation,
            } => {
                write!(f, "ALTER POLICY {name} ON {table_name}{operation}")
            }
            Statement::Drop {
                object_type,
                if_exists,
                names,
                cascade,
                restrict,
                purge,
                temporary,
            } => write!(
                f,
                "DROP {}{}{} {}{}{}{}",
                if *temporary { "TEMPORARY " } else { "" },
                object_type,
                if *if_exists { " IF EXISTS" } else { "" },
                display_comma_separated(names),
                if *cascade { " CASCADE" } else { "" },
                if *restrict { " RESTRICT" } else { "" },
                if *purge { " PURGE" } else { "" }
            ),
            Statement::DropFunction {
                if_exists,
                func_desc,
                option,
            } => {
                write!(
                    f,
                    "DROP FUNCTION{} {}",
                    if *if_exists { " IF EXISTS" } else { "" },
                    display_comma_separated(func_desc),
                )?;
                if let Some(op) = option {
                    write!(f, " {op}")?;
                }
                Ok(())
            }
            Statement::DropProcedure {
                if_exists,
                proc_desc,
                option,
            } => {
                write!(
                    f,
                    "DROP PROCEDURE{} {}",
                    if *if_exists { " IF EXISTS" } else { "" },
                    display_comma_separated(proc_desc),
                )?;
                if let Some(op) = option {
                    write!(f, " {op}")?;
                }
                Ok(())
            }
            Statement::DropSecret {
                if_exists,
                temporary,
                name,
                storage_specifier,
            } => {
                write!(f, "DROP ")?;
                if let Some(t) = temporary {
                    write!(f, "{}", if *t { "TEMPORARY " } else { "PERSISTENT " })?;
                }
                write!(
                    f,
                    "SECRET {if_exists}{name}",
                    if_exists = if *if_exists { "IF EXISTS " } else { "" },
                )?;
                if let Some(s) = storage_specifier {
                    write!(f, " FROM {s}")?;
                }
                Ok(())
            }
            Statement::DropPolicy {
                if_exists,
                name,
                table_name,
                option,
            } => {
                write!(f, "DROP POLICY")?;
                if *if_exists {
                    write!(f, " IF EXISTS")?;
                }
                write!(f, " {name} ON {table_name}")?;
                if let Some(option) = option {
                    write!(f, " {option}")?;
                }
                Ok(())
            }
            Statement::Discard { object_type } => {
                write!(f, "DISCARD {object_type}")?;
                Ok(())
            }
            Self::SetRole {
                context_modifier,
                role_name,
            } => {
                let role_name = role_name.clone().unwrap_or_else(|| Ident::new("NONE"));
                write!(f, "SET{context_modifier} ROLE {role_name}")
            }
            Statement::SetVariable {
                local,
                variables,
                hivevar,
                value,
            } => {
                f.write_str("SET ")?;
                if *local {
                    f.write_str("LOCAL ")?;
                }
                let parenthesized = matches!(variables, OneOrManyWithParens::Many(_));
                write!(
                    f,
                    "{hivevar}{name} = {l_paren}{value}{r_paren}",
                    hivevar = if *hivevar { "HIVEVAR:" } else { "" },
                    name = variables,
                    l_paren = parenthesized.then_some("(").unwrap_or_default(),
                    value = display_comma_separated(value),
                    r_paren = parenthesized.then_some(")").unwrap_or_default(),
                )
            }
            Statement::SetTimeZone { local, value } => {
                f.write_str("SET ")?;
                if *local {
                    f.write_str("LOCAL ")?;
                }
                write!(f, "TIME ZONE {value}")
            }
            Statement::SetNames {
                charset_name,
                collation_name,
            } => {
                f.write_str("SET NAMES ")?;
                f.write_str(charset_name)?;

                if let Some(collation) = collation_name {
                    f.write_str(" COLLATE ")?;
                    f.write_str(collation)?;
                };

                Ok(())
            }
            Statement::SetNamesDefault {} => {
                f.write_str("SET NAMES DEFAULT")?;

                Ok(())
            }
            Statement::ShowVariable { variable } => {
                write!(f, "SHOW")?;
                if !variable.is_empty() {
                    write!(f, " {}", display_separated(variable, " "))?;
                }
                Ok(())
            }
            Statement::ShowStatus {
                filter,
                global,
                session,
            } => {
                write!(f, "SHOW")?;
                if *global {
                    write!(f, " GLOBAL")?;
                }
                if *session {
                    write!(f, " SESSION")?;
                }
                write!(f, " STATUS")?;
                if filter.is_some() {
                    write!(f, " {}", filter.as_ref().unwrap())?;
                }
                Ok(())
            }
            Statement::ShowVariables {
                filter,
                global,
                session,
            } => {
                write!(f, "SHOW")?;
                if *global {
                    write!(f, " GLOBAL")?;
                }
                if *session {
                    write!(f, " SESSION")?;
                }
                write!(f, " VARIABLES")?;
                if filter.is_some() {
                    write!(f, " {}", filter.as_ref().unwrap())?;
                }
                Ok(())
            }
            Statement::ShowCreate { obj_type, obj_name } => {
                write!(f, "SHOW CREATE {obj_type} {obj_name}",)?;
                Ok(())
            }
            Statement::ShowColumns {
                extended,
                full,
                show_options,
            } => {
                write!(
                    f,
                    "SHOW {extended}{full}COLUMNS{show_options}",
                    extended = if *extended { "EXTENDED " } else { "" },
                    full = if *full { "FULL " } else { "" },
                )?;
                Ok(())
            }
            Statement::ShowDatabases {
                terse,
                history,
                show_options,
            } => {
                write!(
                    f,
                    "SHOW {terse}DATABASES{history}{show_options}",
                    terse = if *terse { "TERSE " } else { "" },
                    history = if *history { " HISTORY" } else { "" },
                )?;
                Ok(())
            }
            Statement::ShowSchemas {
                terse,
                history,
                show_options,
            } => {
                write!(
                    f,
                    "SHOW {terse}SCHEMAS{history}{show_options}",
                    terse = if *terse { "TERSE " } else { "" },
                    history = if *history { " HISTORY" } else { "" },
                )?;
                Ok(())
            }
            Statement::ShowTables {
                terse,
                history,
                extended,
                full,
                external,
                show_options,
            } => {
                write!(
                    f,
                    "SHOW {terse}{extended}{full}{external}TABLES{history}{show_options}",
                    terse = if *terse { "TERSE " } else { "" },
                    extended = if *extended { "EXTENDED " } else { "" },
                    full = if *full { "FULL " } else { "" },
                    external = if *external { "EXTERNAL " } else { "" },
                    history = if *history { " HISTORY" } else { "" },
                )?;
                Ok(())
            }
            Statement::ShowViews {
                terse,
                materialized,
                show_options,
            } => {
                write!(
                    f,
                    "SHOW {terse}{materialized}VIEWS{show_options}",
                    terse = if *terse { "TERSE " } else { "" },
                    materialized = if *materialized { "MATERIALIZED " } else { "" }
                )?;
                Ok(())
            }
            Statement::ShowFunctions { filter } => {
                write!(f, "SHOW FUNCTIONS")?;
                if let Some(filter) = filter {
                    write!(f, " {filter}")?;
                }
                Ok(())
            }
            Statement::Use(use_expr) => use_expr.fmt(f),
            Statement::ShowCollation { filter } => {
                write!(f, "SHOW COLLATION")?;
                if let Some(filter) = filter {
                    write!(f, " {filter}")?;
                }
                Ok(())
            }
            Statement::StartTransaction {
                modes,
                begin: syntax_begin,
                transaction,
                modifier,
            } => {
                if *syntax_begin {
                    if let Some(modifier) = *modifier {
                        write!(f, "BEGIN {}", modifier)?;
                    } else {
                        write!(f, "BEGIN")?;
                    }
                } else {
                    write!(f, "START")?;
                }
                if let Some(transaction) = transaction {
                    write!(f, " {transaction}")?;
                }
                if !modes.is_empty() {
                    write!(f, " {}", display_comma_separated(modes))?;
                }
                Ok(())
            }
            Statement::SetTransaction {
                modes,
                snapshot,
                session,
            } => {
                if *session {
                    write!(f, "SET SESSION CHARACTERISTICS AS TRANSACTION")?;
                } else {
                    write!(f, "SET TRANSACTION")?;
                }
                if !modes.is_empty() {
                    write!(f, " {}", display_comma_separated(modes))?;
                }
                if let Some(snapshot_id) = snapshot {
                    write!(f, " SNAPSHOT {snapshot_id}")?;
                }
                Ok(())
            }
            Statement::Commit { chain } => {
                write!(f, "COMMIT{}", if *chain { " AND CHAIN" } else { "" },)
            }
            Statement::Rollback { chain, savepoint } => {
                write!(f, "ROLLBACK")?;

                if *chain {
                    write!(f, " AND CHAIN")?;
                }

                if let Some(savepoint) = savepoint {
                    write!(f, " TO SAVEPOINT {savepoint}")?;
                }

                Ok(())
            }
            Statement::CreateSchema {
                schema_name,
                if_not_exists,
            } => write!(
                f,
                "CREATE SCHEMA {if_not_exists}{name}",
                if_not_exists = if *if_not_exists { "IF NOT EXISTS " } else { "" },
                name = schema_name
            ),
            Statement::Assert { condition, message } => {
                write!(f, "ASSERT {condition}")?;
                if let Some(m) = message {
                    write!(f, " AS {m}")?;
                }
                Ok(())
            }
            Statement::Grant {
                privileges,
                objects,
                grantees,
                with_grant_option,
                granted_by,
            } => {
                write!(f, "GRANT {privileges} ")?;
                write!(f, "ON {objects} ")?;
                write!(f, "TO {}", display_comma_separated(grantees))?;
                if *with_grant_option {
                    write!(f, " WITH GRANT OPTION")?;
                }
                if let Some(grantor) = granted_by {
                    write!(f, " GRANTED BY {grantor}")?;
                }
                Ok(())
            }
            Statement::Revoke {
                privileges,
                objects,
                grantees,
                granted_by,
                cascade,
            } => {
                write!(f, "REVOKE {privileges} ")?;
                write!(f, "ON {objects} ")?;
                write!(f, "FROM {}", display_comma_separated(grantees))?;
                if let Some(grantor) = granted_by {
                    write!(f, " GRANTED BY {grantor}")?;
                }
                write!(f, " {}", if *cascade { "CASCADE" } else { "RESTRICT" })?;
                Ok(())
            }
            Statement::Deallocate { name, prepare } => write!(
                f,
                "DEALLOCATE {prepare}{name}",
                prepare = if *prepare { "PREPARE " } else { "" },
                name = name,
            ),
            Statement::Execute {
                name,
                parameters,
                has_parentheses,
                using,
            } => {
                let (open, close) = if *has_parentheses {
                    ("(", ")")
                } else {
                    (if parameters.is_empty() { "" } else { " " }, "")
                };
                write!(
                    f,
                    "EXECUTE {name}{open}{}{close}",
                    display_comma_separated(parameters),
                )?;
                if !using.is_empty() {
                    write!(f, " USING {}", display_comma_separated(using))?;
                };
                Ok(())
            }
            Statement::Prepare {
                name,
                data_types,
                statement,
            } => {
                write!(f, "PREPARE {name} ")?;
                if !data_types.is_empty() {
                    write!(f, "({}) ", display_comma_separated(data_types))?;
                }
                write!(f, "AS {statement}")
            }
            Statement::Comment {
                object_type,
                object_name,
                comment,
                if_exists,
            } => {
                write!(f, "COMMENT ")?;
                if *if_exists {
                    write!(f, "IF EXISTS ")?
                };
                write!(f, "ON {object_type} {object_name} IS ")?;
                if let Some(c) = comment {
                    write!(f, "'{c}'")
                } else {
                    write!(f, "NULL")
                }
            }
            Statement::Savepoint { name } => {
                write!(f, "SAVEPOINT ")?;
                write!(f, "{name}")
            }
            Statement::ReleaseSavepoint { name } => {
                write!(f, "RELEASE SAVEPOINT {name}")
            }
            Statement::Merge {
                into,
                table,
                source,
                on,
                clauses,
            } => {
                write!(
                    f,
                    "MERGE{int} {table} USING {source} ",
                    int = if *into { " INTO" } else { "" }
                )?;
                write!(f, "ON {on} ")?;
                write!(f, "{}", display_separated(clauses, " "))
            }
            Statement::Cache {
                table_name,
                table_flag,
                has_as,
                options,
                query,
            } => {
                if let Some(table_flag) = table_flag {
                    write!(f, "CACHE {table_flag} TABLE {table_name}")?;
                } else {
                    write!(f, "CACHE TABLE {table_name}")?;
                }

                if !options.is_empty() {
                    write!(f, " OPTIONS({})", display_comma_separated(options))?;
                }

                match (*has_as, query) {
                    (true, Some(query)) => write!(f, " AS {query}"),
                    (true, None) => f.write_str(" AS"),
                    (false, Some(query)) => write!(f, " {query}"),
                    (false, None) => Ok(()),
                }
            }
            Statement::UNCache {
                table_name,
                if_exists,
            } => {
                if *if_exists {
                    write!(f, "UNCACHE TABLE IF EXISTS {table_name}")
                } else {
                    write!(f, "UNCACHE TABLE {table_name}")
                }
            }
            Statement::CreateSequence {
                temporary,
                if_not_exists,
                name,
                data_type,
                sequence_options,
                owned_by,
            } => {
                let as_type: String = if let Some(dt) = data_type.as_ref() {
                    //Cannot use format!(" AS {}", dt), due to format! is not available in --target thumbv6m-none-eabi
                    // " AS ".to_owned() + &dt.to_string()
                    [" AS ", &dt.to_string()].concat()
                } else {
                    "".to_string()
                };
                write!(
                    f,
                    "CREATE {temporary}SEQUENCE {if_not_exists}{name}{as_type}",
                    if_not_exists = if *if_not_exists { "IF NOT EXISTS " } else { "" },
                    temporary = if *temporary { "TEMPORARY " } else { "" },
                    name = name,
                    as_type = as_type
                )?;
                for sequence_option in sequence_options {
                    write!(f, "{sequence_option}")?;
                }
                if let Some(ob) = owned_by.as_ref() {
                    write!(f, " OWNED BY {ob}")?;
                }
                write!(f, "")
            }
            Statement::CreateStage {
                or_replace,
                temporary,
                if_not_exists,
                name,
                stage_params,
                directory_table_params,
                file_format,
                copy_options,
                comment,
                ..
            } => {
                write!(
                    f,
                    "CREATE {or_replace}{temp}STAGE {if_not_exists}{name}{stage_params}",
                    temp = if *temporary { "TEMPORARY " } else { "" },
                    or_replace = if *or_replace { "OR REPLACE " } else { "" },
                    if_not_exists = if *if_not_exists { "IF NOT EXISTS " } else { "" },
                )?;
                if !directory_table_params.options.is_empty() {
                    write!(f, " DIRECTORY=({})", directory_table_params)?;
                }
                if !file_format.options.is_empty() {
                    write!(f, " FILE_FORMAT=({})", file_format)?;
                }
                if !copy_options.options.is_empty() {
                    write!(f, " COPY_OPTIONS=({})", copy_options)?;
                }
                if comment.is_some() {
                    write!(f, " COMMENT='{}'", comment.as_ref().unwrap())?;
                }
                Ok(())
            }
            Statement::CopyIntoSnowflake {
                into,
                from_stage,
                from_stage_alias,
                stage_params,
                from_transformations,
                files,
                pattern,
                file_format,
                copy_options,
                validation_mode,
            } => {
                write!(f, "COPY INTO {}", into)?;
                if from_transformations.is_none() {
                    // Standard data load
                    write!(f, " FROM {}{}", from_stage, stage_params)?;
                    if from_stage_alias.as_ref().is_some() {
                        write!(f, " AS {}", from_stage_alias.as_ref().unwrap())?;
                    }
                } else {
                    // Data load with transformation
                    write!(
                        f,
                        " FROM (SELECT {} FROM {}{}",
                        display_separated(from_transformations.as_ref().unwrap(), ", "),
                        from_stage,
                        stage_params,
                    )?;
                    if from_stage_alias.as_ref().is_some() {
                        write!(f, " AS {}", from_stage_alias.as_ref().unwrap())?;
                    }
                    write!(f, ")")?;
                }
                if files.is_some() {
                    write!(
                        f,
                        " FILES = ('{}')",
                        display_separated(files.as_ref().unwrap(), "', '")
                    )?;
                }
                if pattern.is_some() {
                    write!(f, " PATTERN = '{}'", pattern.as_ref().unwrap())?;
                }
                if !file_format.options.is_empty() {
                    write!(f, " FILE_FORMAT=({})", file_format)?;
                }
                if !copy_options.options.is_empty() {
                    write!(f, " COPY_OPTIONS=({})", copy_options)?;
                }
                if validation_mode.is_some() {
                    write!(
                        f,
                        " VALIDATION_MODE = {}",
                        validation_mode.as_ref().unwrap()
                    )?;
                }
                Ok(())
            }
            Statement::CreateType {
                name,
                representation,
            } => {
                write!(f, "CREATE TYPE {name} AS {representation}")
            }
            Statement::Pragma { name, value, is_eq } => {
                write!(f, "PRAGMA {name}")?;
                if value.is_some() {
                    let val = value.as_ref().unwrap();
                    if *is_eq {
                        write!(f, " = {val}")?;
                    } else {
                        write!(f, "({val})")?;
                    }
                }
                Ok(())
            }
            Statement::LockTables { tables } => {
                write!(f, "LOCK TABLES {}", display_comma_separated(tables))
            }
            Statement::UnlockTables => {
                write!(f, "UNLOCK TABLES")
            }
            Statement::Unload { query, to, with } => {
                write!(f, "UNLOAD({query}) TO {to}")?;

                if !with.is_empty() {
                    write!(f, " WITH ({})", display_comma_separated(with))?;
                }

                Ok(())
            }
            Statement::OptimizeTable {
                name,
                on_cluster,
                partition,
                include_final,
                deduplicate,
            } => {
                write!(f, "OPTIMIZE TABLE {name}")?;
                if let Some(on_cluster) = on_cluster {
                    write!(f, " ON CLUSTER {on_cluster}", on_cluster = on_cluster)?;
                }
                if let Some(partition) = partition {
                    write!(f, " {partition}", partition = partition)?;
                }
                if *include_final {
                    write!(f, " FINAL")?;
                }
                if let Some(deduplicate) = deduplicate {
                    write!(f, " {deduplicate}")?;
                }
                Ok(())
            }
            Statement::LISTEN { channel } => {
                write!(f, "LISTEN {channel}")?;
                Ok(())
            }
            Statement::UNLISTEN { channel } => {
                write!(f, "UNLISTEN {channel}")?;
                Ok(())
            }
            Statement::NOTIFY { channel, payload } => {
                write!(f, "NOTIFY {channel}")?;
                if let Some(payload) = payload {
                    write!(f, ", '{payload}'")?;
                }
                Ok(())
            }
        }
    }
}

/// Can use to describe options in create sequence or table column type identity
/// ```sql
/// [ INCREMENT [ BY ] increment ]
///     [ MINVALUE minvalue | NO MINVALUE ] [ MAXVALUE maxvalue | NO MAXVALUE ]
///     [ START [ WITH ] start ] [ CACHE cache ] [ [ NO ] CYCLE ]
/// ```
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum SequenceOptions {
    IncrementBy(Expr, bool),
    MinValue(Option<Expr>),
    MaxValue(Option<Expr>),
    StartWith(Expr, bool),
    Cache(Expr),
    Cycle(bool),
}

impl fmt::Display for SequenceOptions {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SequenceOptions::IncrementBy(increment, by) => {
                write!(
                    f,
                    " INCREMENT{by} {increment}",
                    by = if *by { " BY" } else { "" },
                    increment = increment
                )
            }
            SequenceOptions::MinValue(Some(expr)) => {
                write!(f, " MINVALUE {expr}")
            }
            SequenceOptions::MinValue(None) => {
                write!(f, " NO MINVALUE")
            }
            SequenceOptions::MaxValue(Some(expr)) => {
                write!(f, " MAXVALUE {expr}")
            }
            SequenceOptions::MaxValue(None) => {
                write!(f, " NO MAXVALUE")
            }
            SequenceOptions::StartWith(start, with) => {
                write!(
                    f,
                    " START{with} {start}",
                    with = if *with { " WITH" } else { "" },
                    start = start
                )
            }
            SequenceOptions::Cache(cache) => {
                write!(f, " CACHE {}", *cache)
            }
            SequenceOptions::Cycle(no) => {
                write!(f, " {}CYCLE", if *no { "NO " } else { "" })
            }
        }
    }
}

/// Target of a `TRUNCATE TABLE` command
///
/// Note this is its own struct because `visit_relation` requires an `ObjectName` (not a `Vec<ObjectName>`)
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TruncateTableTarget {
    /// name of the table being truncated
    #[cfg_attr(feature = "visitor", visit(with = "visit_relation"))]
    pub name: ObjectName,
}

impl fmt::Display for TruncateTableTarget {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// PostgreSQL identity option for TRUNCATE table
/// [ RESTART IDENTITY | CONTINUE IDENTITY ]
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum TruncateIdentityOption {
    Restart,
    Continue,
}

/// PostgreSQL cascade option for TRUNCATE table
/// [ CASCADE | RESTRICT ]
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum TruncateCascadeOption {
    Cascade,
    Restrict,
}

/// Transaction started with [ TRANSACTION | WORK ]
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum BeginTransactionKind {
    Transaction,
    Work,
}

impl Display for BeginTransactionKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BeginTransactionKind::Transaction => write!(f, "TRANSACTION"),
            BeginTransactionKind::Work => write!(f, "WORK"),
        }
    }
}

/// Can use to describe options in  create sequence or table column type identity
/// [ MINVALUE minvalue | NO MINVALUE ] [ MAXVALUE maxvalue | NO MAXVALUE ]
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum MinMaxValue {
    // clause is not specified
    Empty,
    // NO MINVALUE/NO MAXVALUE
    None,
    // MINVALUE <expr> / MAXVALUE <expr>
    Some(Expr),
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
#[non_exhaustive]
pub enum OnInsert {
    /// ON DUPLICATE KEY UPDATE (MySQL when the key already exists, then execute an update instead)
    DuplicateKeyUpdate(Vec<Assignment>),
    /// ON CONFLICT is a PostgreSQL and Sqlite extension
    OnConflict(OnConflict),
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct InsertAliases {
    pub row_alias: ObjectName,
    pub col_aliases: Option<Vec<Ident>>,
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct OnConflict {
    pub conflict_target: Option<ConflictTarget>,
    pub action: OnConflictAction,
}
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum ConflictTarget {
    Columns(Vec<Ident>),
    OnConstraint(ObjectName),
}
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum OnConflictAction {
    DoNothing,
    DoUpdate(DoUpdate),
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct DoUpdate {
    /// Column assignments
    pub assignments: Vec<Assignment>,
    /// WHERE
    pub selection: Option<Expr>,
}

impl fmt::Display for OnInsert {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::DuplicateKeyUpdate(expr) => write!(
                f,
                " ON DUPLICATE KEY UPDATE {}",
                display_comma_separated(expr)
            ),
            Self::OnConflict(o) => write!(f, "{o}"),
        }
    }
}
impl fmt::Display for OnConflict {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, " ON CONFLICT")?;
        if let Some(target) = &self.conflict_target {
            write!(f, "{target}")?;
        }
        write!(f, " {}", self.action)
    }
}
impl fmt::Display for ConflictTarget {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ConflictTarget::Columns(cols) => write!(f, "({})", display_comma_separated(cols)),
            ConflictTarget::OnConstraint(name) => write!(f, " ON CONSTRAINT {name}"),
        }
    }
}
impl fmt::Display for OnConflictAction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::DoNothing => write!(f, "DO NOTHING"),
            Self::DoUpdate(do_update) => {
                write!(f, "DO UPDATE")?;
                if !do_update.assignments.is_empty() {
                    write!(
                        f,
                        " SET {}",
                        display_comma_separated(&do_update.assignments)
                    )?;
                }
                if let Some(selection) = &do_update.selection {
                    write!(f, " WHERE {selection}")?;
                }
                Ok(())
            }
        }
    }
}

/// Privileges granted in a GRANT statement or revoked in a REVOKE statement.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum Privileges {
    /// All privileges applicable to the object type
    All {
        /// Optional keyword from the spec, ignored in practice
        with_privileges_keyword: bool,
    },
    /// Specific privileges (e.g. `SELECT`, `INSERT`)
    Actions(Vec<Action>),
}

impl fmt::Display for Privileges {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Privileges::All {
                with_privileges_keyword,
            } => {
                write!(
                    f,
                    "ALL{}",
                    if *with_privileges_keyword {
                        " PRIVILEGES"
                    } else {
                        ""
                    }
                )
            }
            Privileges::Actions(actions) => {
                write!(f, "{}", display_comma_separated(actions))
            }
        }
    }
}

/// Specific direction for FETCH statement
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum FetchDirection {
    Count { limit: Value },
    Next,
    Prior,
    First,
    Last,
    Absolute { limit: Value },
    Relative { limit: Value },
    All,
    // FORWARD
    // FORWARD count
    Forward { limit: Option<Value> },
    ForwardAll,
    // BACKWARD
    // BACKWARD count
    Backward { limit: Option<Value> },
    BackwardAll,
}

impl fmt::Display for FetchDirection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FetchDirection::Count { limit } => f.write_str(&limit.to_string())?,
            FetchDirection::Next => f.write_str("NEXT")?,
            FetchDirection::Prior => f.write_str("PRIOR")?,
            FetchDirection::First => f.write_str("FIRST")?,
            FetchDirection::Last => f.write_str("LAST")?,
            FetchDirection::Absolute { limit } => {
                f.write_str("ABSOLUTE ")?;
                f.write_str(&limit.to_string())?;
            }
            FetchDirection::Relative { limit } => {
                f.write_str("RELATIVE ")?;
                f.write_str(&limit.to_string())?;
            }
            FetchDirection::All => f.write_str("ALL")?,
            FetchDirection::Forward { limit } => {
                f.write_str("FORWARD")?;

                if let Some(l) = limit {
                    f.write_str(" ")?;
                    f.write_str(&l.to_string())?;
                }
            }
            FetchDirection::ForwardAll => f.write_str("FORWARD ALL")?,
            FetchDirection::Backward { limit } => {
                f.write_str("BACKWARD")?;

                if let Some(l) = limit {
                    f.write_str(" ")?;
                    f.write_str(&l.to_string())?;
                }
            }
            FetchDirection::BackwardAll => f.write_str("BACKWARD ALL")?,
        };

        Ok(())
    }
}

/// A privilege on a database object (table, sequence, etc.).
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum Action {
    Connect,
    Create,
    Delete,
    Execute,
    Insert { columns: Option<Vec<Ident>> },
    References { columns: Option<Vec<Ident>> },
    Select { columns: Option<Vec<Ident>> },
    Temporary,
    Trigger,
    Truncate,
    Update { columns: Option<Vec<Ident>> },
    Usage,
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Action::Connect => f.write_str("CONNECT")?,
            Action::Create => f.write_str("CREATE")?,
            Action::Delete => f.write_str("DELETE")?,
            Action::Execute => f.write_str("EXECUTE")?,
            Action::Insert { .. } => f.write_str("INSERT")?,
            Action::References { .. } => f.write_str("REFERENCES")?,
            Action::Select { .. } => f.write_str("SELECT")?,
            Action::Temporary => f.write_str("TEMPORARY")?,
            Action::Trigger => f.write_str("TRIGGER")?,
            Action::Truncate => f.write_str("TRUNCATE")?,
            Action::Update { .. } => f.write_str("UPDATE")?,
            Action::Usage => f.write_str("USAGE")?,
        };
        match self {
            Action::Insert { columns }
            | Action::References { columns }
            | Action::Select { columns }
            | Action::Update { columns } => {
                if let Some(columns) = columns {
                    write!(f, " ({})", display_comma_separated(columns))?;
                }
            }
            _ => (),
        };
        Ok(())
    }
}

/// Objects on which privileges are granted in a GRANT statement.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum GrantObjects {
    /// Grant privileges on `ALL SEQUENCES IN SCHEMA <schema_name> [, ...]`
    AllSequencesInSchema { schemas: Vec<ObjectName> },
    /// Grant privileges on `ALL TABLES IN SCHEMA <schema_name> [, ...]`
    AllTablesInSchema { schemas: Vec<ObjectName> },
    /// Grant privileges on specific schemas
    Schemas(Vec<ObjectName>),
    /// Grant privileges on specific sequences
    Sequences(Vec<ObjectName>),
    /// Grant privileges on specific tables
    Tables(Vec<ObjectName>),
}

impl fmt::Display for GrantObjects {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GrantObjects::Sequences(sequences) => {
                write!(f, "SEQUENCE {}", display_comma_separated(sequences))
            }
            GrantObjects::Schemas(schemas) => {
                write!(f, "SCHEMA {}", display_comma_separated(schemas))
            }
            GrantObjects::Tables(tables) => {
                write!(f, "{}", display_comma_separated(tables))
            }
            GrantObjects::AllSequencesInSchema { schemas } => {
                write!(
                    f,
                    "ALL SEQUENCES IN SCHEMA {}",
                    display_comma_separated(schemas)
                )
            }
            GrantObjects::AllTablesInSchema { schemas } => {
                write!(
                    f,
                    "ALL TABLES IN SCHEMA {}",
                    display_comma_separated(schemas)
                )
            }
        }
    }
}

/// SQL assignment `foo = expr` as used in SQLUpdate
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct Assignment {
    pub target: AssignmentTarget,
    pub value: Expr,
}

impl fmt::Display for Assignment {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} = {}", self.target, self.value)
    }
}

/// Left-hand side of an assignment in an UPDATE statement,
/// e.g. `foo` in `foo = 5` (ColumnName assignment) or
/// `(a, b)` in `(a, b) = (1, 2)` (Tuple assignment).
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum AssignmentTarget {
    /// A single column
    ColumnName(ObjectName),
    /// A tuple of columns
    Tuple(Vec<ObjectName>),
}

impl fmt::Display for AssignmentTarget {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AssignmentTarget::ColumnName(column) => write!(f, "{}", column),
            AssignmentTarget::Tuple(columns) => write!(f, "({})", display_comma_separated(columns)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum FunctionArgExpr {
    Expr(Expr),
    /// Qualified wildcard, e.g. `alias.*` or `schema.table.*`.
    QualifiedWildcard(ObjectName),
    /// An unqualified `*`
    Wildcard,
}

impl From<Expr> for FunctionArgExpr {
    fn from(wildcard_expr: Expr) -> Self {
        match wildcard_expr {
            Expr::QualifiedWildcard(prefix, _) => Self::QualifiedWildcard(prefix),
            Expr::Wildcard(_) => Self::Wildcard,
            expr => Self::Expr(expr),
        }
    }
}

impl fmt::Display for FunctionArgExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FunctionArgExpr::Expr(expr) => write!(f, "{expr}"),
            FunctionArgExpr::QualifiedWildcard(prefix) => write!(f, "{prefix}.*"),
            FunctionArgExpr::Wildcard => f.write_str("*"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
/// Operator used to separate function arguments
pub enum FunctionArgOperator {
    /// function(arg1 = value1)
    Equals,
    /// function(arg1 => value1)
    RightArrow,
    /// function(arg1 := value1)
    Assignment,
    /// function(arg1 : value1)
    Colon,
    /// function(arg1 VALUE value1)
    Value,
}

impl fmt::Display for FunctionArgOperator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FunctionArgOperator::Equals => f.write_str("="),
            FunctionArgOperator::RightArrow => f.write_str("=>"),
            FunctionArgOperator::Assignment => f.write_str(":="),
            FunctionArgOperator::Colon => f.write_str(":"),
            FunctionArgOperator::Value => f.write_str("VALUE"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum FunctionArg {
    /// `name` is identifier
    ///
    /// Enabled when `Dialect::supports_named_fn_args_with_expr_name` returns 'false'
    Named {
        name: Ident,
        arg: FunctionArgExpr,
        operator: FunctionArgOperator,
    },
    /// `name` is arbitrary expression
    ///
    /// Enabled when `Dialect::supports_named_fn_args_with_expr_name` returns 'true'
    ExprNamed {
        name: Expr,
        arg: FunctionArgExpr,
        operator: FunctionArgOperator,
    },
    Unnamed(FunctionArgExpr),
}

impl fmt::Display for FunctionArg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FunctionArg::Named {
                name,
                arg,
                operator,
            } => write!(f, "{name} {operator} {arg}"),
            FunctionArg::ExprNamed {
                name,
                arg,
                operator,
            } => write!(f, "{name} {operator} {arg}"),
            FunctionArg::Unnamed(unnamed_arg) => write!(f, "{unnamed_arg}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum CloseCursor {
    All,
    Specific { name: Ident },
}

impl fmt::Display for CloseCursor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CloseCursor::All => write!(f, "ALL"),
            CloseCursor::Specific { name } => write!(f, "{name}"),
        }
    }
}

/// A function call
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct Function {
    pub name: ObjectName,
    /// Flags whether this function call uses the [ODBC syntax].
    ///
    /// Example:
    /// ```sql
    /// SELECT {fn CONCAT('foo', 'bar')}
    /// ```
    ///
    /// [ODBC syntax]: https://learn.microsoft.com/en-us/sql/odbc/reference/develop-app/scalar-function-calls?view=sql-server-2017
    pub uses_odbc_syntax: bool,
    /// The parameters to the function, including any options specified within the
    /// delimiting parentheses.
    ///
    /// Example:
    /// ```plaintext
    /// HISTOGRAM(0.5, 0.6)(x, y)
    /// ```
    ///
    /// [ClickHouse](https://clickhouse.com/docs/en/sql-reference/aggregate-functions/parametric-functions)
    pub parameters: FunctionArguments,
    /// The arguments to the function, including any options specified within the
    /// delimiting parentheses.
    pub args: FunctionArguments,
    /// e.g. `x > 5` in `COUNT(x) FILTER (WHERE x > 5)`
    pub filter: Option<Box<Expr>>,
    /// Indicates how `NULL`s should be handled in the calculation.
    ///
    /// Example:
    /// ```plaintext
    /// FIRST_VALUE( <expr> ) [ { IGNORE | RESPECT } NULLS ] OVER ...
    /// ```
    ///
    /// [Snowflake](https://docs.snowflake.com/en/sql-reference/functions/first_value)
    pub null_treatment: Option<NullTreatment>,
    /// The `OVER` clause, indicating a window function call.
    pub over: Option<WindowType>,
    /// A clause used with certain aggregate functions to control the ordering
    /// within grouped sets before the function is applied.
    ///
    /// Syntax:
    /// ```plaintext
    /// <aggregate_function>(expression) WITHIN GROUP (ORDER BY key [ASC | DESC], ...)
    /// ```
    pub within_group: Vec<OrderByExpr>,
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.uses_odbc_syntax {
            write!(f, "{{fn ")?;
        }

        write!(f, "{}{}{}", self.name, self.parameters, self.args)?;

        if !self.within_group.is_empty() {
            write!(
                f,
                " WITHIN GROUP (ORDER BY {})",
                display_comma_separated(&self.within_group)
            )?;
        }

        if let Some(filter_cond) = &self.filter {
            write!(f, " FILTER (WHERE {filter_cond})")?;
        }

        if let Some(null_treatment) = &self.null_treatment {
            write!(f, " {null_treatment}")?;
        }

        if let Some(o) = &self.over {
            write!(f, " OVER {o}")?;
        }

        if self.uses_odbc_syntax {
            write!(f, "}}")?;
        }

        Ok(())
    }
}

/// The arguments passed to a function call.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum FunctionArguments {
    /// Used for special functions like `CURRENT_TIMESTAMP` that are invoked
    /// without parentheses.
    None,
    /// On some dialects, a subquery can be passed without surrounding
    /// parentheses if it's the sole argument to the function.
    Subquery(Box<Query>),
    /// A normal function argument list, including any clauses within it such as
    /// `DISTINCT` or `ORDER BY`.
    List(FunctionArgumentList),
}

impl fmt::Display for FunctionArguments {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FunctionArguments::None => Ok(()),
            FunctionArguments::Subquery(query) => write!(f, "({})", query),
            FunctionArguments::List(args) => write!(f, "({})", args),
        }
    }
}

/// This represents everything inside the parentheses when calling a function.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct FunctionArgumentList {
    /// `[ ALL | DISTINCT ]`
    pub duplicate_treatment: Option<DuplicateTreatment>,
    /// The function arguments.
    pub args: Vec<FunctionArg>,
    /// Additional clauses specified within the argument list.
    pub clauses: Vec<FunctionArgumentClause>,
}

impl fmt::Display for FunctionArgumentList {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(duplicate_treatment) = self.duplicate_treatment {
            write!(f, "{} ", duplicate_treatment)?;
        }
        write!(f, "{}", display_comma_separated(&self.args))?;
        if !self.clauses.is_empty() {
            if !self.args.is_empty() {
                write!(f, " ")?;
            }
            write!(f, "{}", display_separated(&self.clauses, " "))?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum FunctionArgumentClause {
    /// Indicates how `NULL`s should be handled in the calculation, e.g. in `FIRST_VALUE` on [BigQuery].
    ///
    /// Syntax:
    /// ```plaintext
    /// { IGNORE | RESPECT } NULLS ]
    /// ```
    ///
    /// [BigQuery]: https://cloud.google.com/bigquery/docs/reference/standard-sql/navigation_functions#first_value
    IgnoreOrRespectNulls(NullTreatment),
    /// Specifies the the ordering for some ordered set aggregates, e.g. `ARRAY_AGG` on [BigQuery].
    ///
    /// [BigQuery]: https://cloud.google.com/bigquery/docs/reference/standard-sql/aggregate_functions#array_agg
    OrderBy(Vec<OrderByExpr>),
    /// Specifies a limit for the `ARRAY_AGG` and `ARRAY_CONCAT_AGG` functions on BigQuery.
    Limit(Expr),
    /// Specifies the behavior on overflow of the `LISTAGG` function.
    ///
    /// See <https://trino.io/docs/current/functions/aggregate.html>.
    OnOverflow(ListAggOnOverflow),
    /// Specifies a minimum or maximum bound on the input to [`ANY_VALUE`] on BigQuery.
    ///
    /// Syntax:
    /// ```plaintext
    /// HAVING { MAX | MIN } expression
    /// ```
    ///
    /// [`ANY_VALUE`]: https://cloud.google.com/bigquery/docs/reference/standard-sql/aggregate_functions#any_value
    Having(HavingBound),
    /// The `SEPARATOR` clause to the [`GROUP_CONCAT`] function in MySQL.
    ///
    /// [`GROUP_CONCAT`]: https://dev.mysql.com/doc/refman/8.0/en/aggregate-functions.html#function_group-concat
    Separator(Value),
    /// The json-null-clause to the [`JSON_ARRAY`]/[`JSON_OBJECT`] function in MSSQL.
    ///
    /// [`JSON_ARRAY`]: <https://learn.microsoft.com/en-us/sql/t-sql/functions/json-array-transact-sql?view=sql-server-ver16>
    /// [`JSON_OBJECT`]: <https://learn.microsoft.com/en-us/sql/t-sql/functions/json-object-transact-sql?view=sql-server-ver16>
    JsonNullClause(JsonNullClause),
}

impl fmt::Display for FunctionArgumentClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FunctionArgumentClause::IgnoreOrRespectNulls(null_treatment) => {
                write!(f, "{}", null_treatment)
            }
            FunctionArgumentClause::OrderBy(order_by) => {
                write!(f, "ORDER BY {}", display_comma_separated(order_by))
            }
            FunctionArgumentClause::Limit(limit) => write!(f, "LIMIT {limit}"),
            FunctionArgumentClause::OnOverflow(on_overflow) => write!(f, "{on_overflow}"),
            FunctionArgumentClause::Having(bound) => write!(f, "{bound}"),
            FunctionArgumentClause::Separator(sep) => write!(f, "SEPARATOR {sep}"),
            FunctionArgumentClause::JsonNullClause(null_clause) => write!(f, "{null_clause}"),
        }
    }
}

/// A method call
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct Method {
    pub expr: Box<Expr>,
    // always non-empty
    pub method_chain: Vec<Function>,
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}.{}",
            self.expr,
            display_separated(&self.method_chain, ".")
        )
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum DuplicateTreatment {
    /// Perform the calculation only unique values.
    Distinct,
    /// Retain all duplicate values (the default).
    All,
}

impl fmt::Display for DuplicateTreatment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DuplicateTreatment::Distinct => write!(f, "DISTINCT"),
            DuplicateTreatment::All => write!(f, "ALL"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum AnalyzeFormat {
    TEXT,
    GRAPHVIZ,
    JSON,
}

impl fmt::Display for AnalyzeFormat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            AnalyzeFormat::TEXT => "TEXT",
            AnalyzeFormat::GRAPHVIZ => "GRAPHVIZ",
            AnalyzeFormat::JSON => "JSON",
        })
    }
}

/// External table's available file format
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum FileFormat {
    TEXTFILE,
    SEQUENCEFILE,
    ORC,
    PARQUET,
    AVRO,
    RCFILE,
    JSONFILE,
}

impl fmt::Display for FileFormat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::FileFormat::*;
        f.write_str(match self {
            TEXTFILE => "TEXTFILE",
            SEQUENCEFILE => "SEQUENCEFILE",
            ORC => "ORC",
            PARQUET => "PARQUET",
            AVRO => "AVRO",
            RCFILE => "RCFILE",
            JSONFILE => "JSONFILE",
        })
    }
}

/// The `ON OVERFLOW` clause of a LISTAGG invocation
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum ListAggOnOverflow {
    /// `ON OVERFLOW ERROR`
    Error,

    /// `ON OVERFLOW TRUNCATE [ <filler> ] WITH[OUT] COUNT`
    Truncate {
        filler: Option<Box<Expr>>,
        with_count: bool,
    },
}

impl fmt::Display for ListAggOnOverflow {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ON OVERFLOW")?;
        match self {
            ListAggOnOverflow::Error => write!(f, " ERROR"),
            ListAggOnOverflow::Truncate { filler, with_count } => {
                write!(f, " TRUNCATE")?;
                if let Some(filler) = filler {
                    write!(f, " {filler}")?;
                }
                if *with_count {
                    write!(f, " WITH")?;
                } else {
                    write!(f, " WITHOUT")?;
                }
                write!(f, " COUNT")
            }
        }
    }
}

/// The `HAVING` clause in a call to `ANY_VALUE` on BigQuery.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct HavingBound(pub HavingBoundKind, pub Expr);

impl fmt::Display for HavingBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HAVING {} {}", self.0, self.1)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum HavingBoundKind {
    Min,
    Max,
}

impl fmt::Display for HavingBoundKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HavingBoundKind::Min => write!(f, "MIN"),
            HavingBoundKind::Max => write!(f, "MAX"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum ObjectType {
    Table,
    View,
    Index,
    Schema,
    Database,
    Role,
    Sequence,
    Stage,
    Type,
}

impl fmt::Display for ObjectType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            ObjectType::Table => "TABLE",
            ObjectType::View => "VIEW",
            ObjectType::Index => "INDEX",
            ObjectType::Schema => "SCHEMA",
            ObjectType::Database => "DATABASE",
            ObjectType::Role => "ROLE",
            ObjectType::Sequence => "SEQUENCE",
            ObjectType::Stage => "STAGE",
            ObjectType::Type => "TYPE",
        })
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum KillType {
    Connection,
    Query,
    Mutation,
}

impl fmt::Display for KillType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            // MySQL
            KillType::Connection => "CONNECTION",
            KillType::Query => "QUERY",
            // Clickhouse supports Mutation
            KillType::Mutation => "MUTATION",
        })
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum HiveDistributionStyle {
    PARTITIONED {
        columns: Vec<ColumnDef>,
    },
    SKEWED {
        columns: Vec<ColumnDef>,
        on: Vec<ColumnDef>,
        stored_as_directories: bool,
    },
    NONE,
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum HiveRowFormat {
    SERDE { class: String },
    DELIMITED { delimiters: Vec<HiveRowDelimiter> },
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct HiveLoadDataFormat {
    pub serde: Expr,
    pub input_format: Expr,
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct HiveRowDelimiter {
    pub delimiter: HiveDelimiter,
    pub char: Ident,
}

impl fmt::Display for HiveRowDelimiter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ", self.delimiter)?;
        write!(f, "{}", self.char)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum HiveDelimiter {
    FieldsTerminatedBy,
    FieldsEscapedBy,
    CollectionItemsTerminatedBy,
    MapKeysTerminatedBy,
    LinesTerminatedBy,
    NullDefinedAs,
}

impl fmt::Display for HiveDelimiter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use HiveDelimiter::*;
        f.write_str(match self {
            FieldsTerminatedBy => "FIELDS TERMINATED BY",
            FieldsEscapedBy => "ESCAPED BY",
            CollectionItemsTerminatedBy => "COLLECTION ITEMS TERMINATED BY",
            MapKeysTerminatedBy => "MAP KEYS TERMINATED BY",
            LinesTerminatedBy => "LINES TERMINATED BY",
            NullDefinedAs => "NULL DEFINED AS",
        })
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum HiveDescribeFormat {
    Extended,
    Formatted,
}

impl fmt::Display for HiveDescribeFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use HiveDescribeFormat::*;
        f.write_str(match self {
            Extended => "EXTENDED",
            Formatted => "FORMATTED",
        })
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum DescribeAlias {
    Describe,
    Explain,
    Desc,
}

impl fmt::Display for DescribeAlias {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use DescribeAlias::*;
        f.write_str(match self {
            Describe => "DESCRIBE",
            Explain => "EXPLAIN",
            Desc => "DESC",
        })
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
#[allow(clippy::large_enum_variant)]
pub enum HiveIOFormat {
    IOF {
        input_format: Expr,
        output_format: Expr,
    },
    FileFormat {
        format: FileFormat,
    },
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct HiveFormat {
    pub row_format: Option<HiveRowFormat>,
    pub serde_properties: Option<Vec<SqlOption>>,
    pub storage: Option<HiveIOFormat>,
    pub location: Option<String>,
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct ClusteredIndex {
    pub name: Ident,
    pub asc: Option<bool>,
}

impl fmt::Display for ClusteredIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)?;
        match self.asc {
            Some(true) => write!(f, " ASC"),
            Some(false) => write!(f, " DESC"),
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum TableOptionsClustered {
    ColumnstoreIndex,
    ColumnstoreIndexOrder(Vec<Ident>),
    Index(Vec<ClusteredIndex>),
}

impl fmt::Display for TableOptionsClustered {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TableOptionsClustered::ColumnstoreIndex => {
                write!(f, "CLUSTERED COLUMNSTORE INDEX")
            }
            TableOptionsClustered::ColumnstoreIndexOrder(values) => {
                write!(
                    f,
                    "CLUSTERED COLUMNSTORE INDEX ORDER ({})",
                    display_comma_separated(values)
                )
            }
            TableOptionsClustered::Index(values) => {
                write!(f, "CLUSTERED INDEX ({})", display_comma_separated(values))
            }
        }
    }
}

/// Specifies which partition the boundary values on table partitioning belongs to.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum PartitionRangeDirection {
    Left,
    Right,
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum SqlOption {
    /// Clustered represents the clustered version of table storage for MSSQL.
    ///
    /// <https://learn.microsoft.com/en-us/sql/t-sql/statements/create-table-azure-sql-data-warehouse?view=aps-pdw-2016-au7#TableOptions>
    Clustered(TableOptionsClustered),
    /// Single identifier options, e.g. `HEAP` for MSSQL.
    ///
    /// <https://learn.microsoft.com/en-us/sql/t-sql/statements/create-table-azure-sql-data-warehouse?view=aps-pdw-2016-au7#TableOptions>
    Ident(Ident),
    /// Any option that consists of a key value pair where the value is an expression. e.g.
    ///
    ///   WITH(DISTRIBUTION = ROUND_ROBIN)
    KeyValue { key: Ident, value: Expr },
    /// One or more table partitions and represents which partition the boundary values belong to,
    /// e.g.
    ///
    ///   PARTITION (id RANGE LEFT FOR VALUES (10, 20, 30, 40))
    ///
    /// <https://learn.microsoft.com/en-us/sql/t-sql/statements/create-table-azure-sql-data-warehouse?view=aps-pdw-2016-au7#TablePartitionOptions>
    Partition {
        column_name: Ident,
        range_direction: Option<PartitionRangeDirection>,
        for_values: Vec<Expr>,
    },
}

impl fmt::Display for SqlOption {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SqlOption::Clustered(c) => write!(f, "{}", c),
            SqlOption::Ident(ident) => {
                write!(f, "{}", ident)
            }
            SqlOption::KeyValue { key: name, value } => {
                write!(f, "{} = {}", name, value)
            }
            SqlOption::Partition {
                column_name,
                range_direction,
                for_values,
            } => {
                let direction = match range_direction {
                    Some(PartitionRangeDirection::Left) => " LEFT",
                    Some(PartitionRangeDirection::Right) => " RIGHT",
                    None => "",
                };

                write!(
                    f,
                    "PARTITION ({} RANGE{} FOR VALUES ({}))",
                    column_name,
                    direction,
                    display_comma_separated(for_values)
                )
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct SecretOption {
    pub key: Ident,
    pub value: Ident,
}

impl fmt::Display for SecretOption {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {}", self.key, self.value)
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum AttachDuckDBDatabaseOption {
    ReadOnly(Option<bool>),
    Type(Ident),
}

impl fmt::Display for AttachDuckDBDatabaseOption {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AttachDuckDBDatabaseOption::ReadOnly(Some(true)) => write!(f, "READ_ONLY true"),
            AttachDuckDBDatabaseOption::ReadOnly(Some(false)) => write!(f, "READ_ONLY false"),
            AttachDuckDBDatabaseOption::ReadOnly(None) => write!(f, "READ_ONLY"),
            AttachDuckDBDatabaseOption::Type(t) => write!(f, "TYPE {}", t),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum TransactionMode {
    AccessMode(TransactionAccessMode),
    IsolationLevel(TransactionIsolationLevel),
}

impl fmt::Display for TransactionMode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use TransactionMode::*;
        match self {
            AccessMode(access_mode) => write!(f, "{access_mode}"),
            IsolationLevel(iso_level) => write!(f, "ISOLATION LEVEL {iso_level}"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum TransactionAccessMode {
    ReadOnly,
    ReadWrite,
}

impl fmt::Display for TransactionAccessMode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use TransactionAccessMode::*;
        f.write_str(match self {
            ReadOnly => "READ ONLY",
            ReadWrite => "READ WRITE",
        })
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum TransactionIsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

impl fmt::Display for TransactionIsolationLevel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use TransactionIsolationLevel::*;
        f.write_str(match self {
            ReadUncommitted => "READ UNCOMMITTED",
            ReadCommitted => "READ COMMITTED",
            RepeatableRead => "REPEATABLE READ",
            Serializable => "SERIALIZABLE",
        })
    }
}

/// SQLite specific syntax
///
/// <https://sqlite.org/lang_transaction.html>
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum TransactionModifier {
    Deferred,
    Immediate,
    Exclusive,
}

impl fmt::Display for TransactionModifier {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use TransactionModifier::*;
        f.write_str(match self {
            Deferred => "DEFERRED",
            Immediate => "IMMEDIATE",
            Exclusive => "EXCLUSIVE",
        })
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum ShowStatementFilter {
    Like(String),
    ILike(String),
    Where(Expr),
    NoKeyword(String),
}

impl fmt::Display for ShowStatementFilter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ShowStatementFilter::*;
        match self {
            Like(pattern) => write!(f, "LIKE '{}'", value::escape_single_quote_string(pattern)),
            ILike(pattern) => write!(f, "ILIKE {}", value::escape_single_quote_string(pattern)),
            Where(expr) => write!(f, "WHERE {expr}"),
            NoKeyword(pattern) => write!(f, "'{}'", value::escape_single_quote_string(pattern)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum ShowStatementInClause {
    IN,
    FROM,
}

impl fmt::Display for ShowStatementInClause {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ShowStatementInClause::*;
        match self {
            FROM => write!(f, "FROM"),
            IN => write!(f, "IN"),
        }
    }
}

/// Sqlite specific syntax
///
/// See [Sqlite documentation](https://sqlite.org/lang_conflict.html)
/// for more details.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum SqliteOnConflict {
    Rollback,
    Abort,
    Fail,
    Ignore,
    Replace,
}

impl fmt::Display for SqliteOnConflict {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use SqliteOnConflict::*;
        match self {
            Rollback => write!(f, "OR ROLLBACK"),
            Abort => write!(f, "OR ABORT"),
            Fail => write!(f, "OR FAIL"),
            Ignore => write!(f, "OR IGNORE"),
            Replace => write!(f, "OR REPLACE"),
        }
    }
}

/// Mysql specific syntax
///
/// See [Mysql documentation](https://dev.mysql.com/doc/refman/8.0/en/replace.html)
/// See [Mysql documentation](https://dev.mysql.com/doc/refman/8.0/en/insert.html)
/// for more details.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum MysqlInsertPriority {
    LowPriority,
    Delayed,
    HighPriority,
}

impl fmt::Display for crate::ast::MysqlInsertPriority {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use MysqlInsertPriority::*;
        match self {
            LowPriority => write!(f, "LOW_PRIORITY"),
            Delayed => write!(f, "DELAYED"),
            HighPriority => write!(f, "HIGH_PRIORITY"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum CopySource {
    Table {
        /// The name of the table to copy from.
        table_name: ObjectName,
        /// A list of column names to copy. Empty list means that all columns
        /// are copied.
        columns: Vec<Ident>,
    },
    Query(Box<Query>),
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum CopyTarget {
    Stdin,
    Stdout,
    File {
        /// The path name of the input or output file.
        filename: String,
    },
    Program {
        /// A command to execute
        command: String,
    },
}

impl fmt::Display for CopyTarget {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use CopyTarget::*;
        match self {
            Stdin { .. } => write!(f, "STDIN"),
            Stdout => write!(f, "STDOUT"),
            File { filename } => write!(f, "'{}'", value::escape_single_quote_string(filename)),
            Program { command } => write!(
                f,
                "PROGRAM '{}'",
                value::escape_single_quote_string(command)
            ),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum OnCommit {
    DeleteRows,
    PreserveRows,
    Drop,
}

/// An option in `COPY` statement.
///
/// <https://www.postgresql.org/docs/14/sql-copy.html>
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum CopyOption {
    /// FORMAT format_name
    Format(Ident),
    /// FREEZE \[ boolean \]
    Freeze(bool),
    /// DELIMITER 'delimiter_character'
    Delimiter(char),
    /// NULL 'null_string'
    Null(String),
    /// HEADER \[ boolean \]
    Header(bool),
    /// QUOTE 'quote_character'
    Quote(char),
    /// ESCAPE 'escape_character'
    Escape(char),
    /// FORCE_QUOTE { ( column_name [, ...] ) | * }
    ForceQuote(Vec<Ident>),
    /// FORCE_NOT_NULL ( column_name [, ...] )
    ForceNotNull(Vec<Ident>),
    /// FORCE_NULL ( column_name [, ...] )
    ForceNull(Vec<Ident>),
    /// ENCODING 'encoding_name'
    Encoding(String),
}

impl fmt::Display for CopyOption {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use CopyOption::*;
        match self {
            Format(name) => write!(f, "FORMAT {name}"),
            Freeze(true) => write!(f, "FREEZE"),
            Freeze(false) => write!(f, "FREEZE FALSE"),
            Delimiter(char) => write!(f, "DELIMITER '{char}'"),
            Null(string) => write!(f, "NULL '{}'", value::escape_single_quote_string(string)),
            Header(true) => write!(f, "HEADER"),
            Header(false) => write!(f, "HEADER FALSE"),
            Quote(char) => write!(f, "QUOTE '{char}'"),
            Escape(char) => write!(f, "ESCAPE '{char}'"),
            ForceQuote(columns) => write!(f, "FORCE_QUOTE ({})", display_comma_separated(columns)),
            ForceNotNull(columns) => {
                write!(f, "FORCE_NOT_NULL ({})", display_comma_separated(columns))
            }
            ForceNull(columns) => write!(f, "FORCE_NULL ({})", display_comma_separated(columns)),
            Encoding(name) => write!(f, "ENCODING '{}'", value::escape_single_quote_string(name)),
        }
    }
}

/// An option in `COPY` statement before PostgreSQL version 9.0.
///
/// <https://www.postgresql.org/docs/8.4/sql-copy.html>
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum CopyLegacyOption {
    /// BINARY
    Binary,
    /// DELIMITER \[ AS \] 'delimiter_character'
    Delimiter(char),
    /// NULL \[ AS \] 'null_string'
    Null(String),
    /// CSV ...
    Csv(Vec<CopyLegacyCsvOption>),
}

impl fmt::Display for CopyLegacyOption {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use CopyLegacyOption::*;
        match self {
            Binary => write!(f, "BINARY"),
            Delimiter(char) => write!(f, "DELIMITER '{char}'"),
            Null(string) => write!(f, "NULL '{}'", value::escape_single_quote_string(string)),
            Csv(opts) => write!(f, "CSV {}", display_separated(opts, " ")),
        }
    }
}

/// A `CSV` option in `COPY` statement before PostgreSQL version 9.0.
///
/// <https://www.postgresql.org/docs/8.4/sql-copy.html>
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum CopyLegacyCsvOption {
    /// HEADER
    Header,
    /// QUOTE \[ AS \] 'quote_character'
    Quote(char),
    /// ESCAPE \[ AS \] 'escape_character'
    Escape(char),
    /// FORCE QUOTE { column_name [, ...] | * }
    ForceQuote(Vec<Ident>),
    /// FORCE NOT NULL column_name [, ...]
    ForceNotNull(Vec<Ident>),
}

impl fmt::Display for CopyLegacyCsvOption {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use CopyLegacyCsvOption::*;
        match self {
            Header => write!(f, "HEADER"),
            Quote(char) => write!(f, "QUOTE '{char}'"),
            Escape(char) => write!(f, "ESCAPE '{char}'"),
            ForceQuote(columns) => write!(f, "FORCE QUOTE {}", display_comma_separated(columns)),
            ForceNotNull(columns) => {
                write!(f, "FORCE NOT NULL {}", display_comma_separated(columns))
            }
        }
    }
}

/// Variant of `WHEN` clause used within a `MERGE` Statement.
///
/// Example:
/// ```sql
/// MERGE INTO T USING U ON FALSE WHEN MATCHED THEN DELETE
/// ```
/// [Snowflake](https://docs.snowflake.com/en/sql-reference/sql/merge)
/// [BigQuery](https://cloud.google.com/bigquery/docs/reference/standard-sql/dml-syntax#merge_statement)
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum MergeClauseKind {
    /// `WHEN MATCHED`
    Matched,
    /// `WHEN NOT MATCHED`
    NotMatched,
    /// `WHEN MATCHED BY TARGET`
    ///
    /// [BigQuery](https://cloud.google.com/bigquery/docs/reference/standard-sql/dml-syntax#merge_statement)
    NotMatchedByTarget,
    /// `WHEN MATCHED BY SOURCE`
    ///
    /// [BigQuery](https://cloud.google.com/bigquery/docs/reference/standard-sql/dml-syntax#merge_statement)
    NotMatchedBySource,
}

impl Display for MergeClauseKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MergeClauseKind::Matched => write!(f, "MATCHED"),
            MergeClauseKind::NotMatched => write!(f, "NOT MATCHED"),
            MergeClauseKind::NotMatchedByTarget => write!(f, "NOT MATCHED BY TARGET"),
            MergeClauseKind::NotMatchedBySource => write!(f, "NOT MATCHED BY SOURCE"),
        }
    }
}

/// The type of expression used to insert rows within a `MERGE` statement.
///
/// [Snowflake](https://docs.snowflake.com/en/sql-reference/sql/merge)
/// [BigQuery](https://cloud.google.com/bigquery/docs/reference/standard-sql/dml-syntax#merge_statement)
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum MergeInsertKind {
    /// The insert expression is defined from an explicit `VALUES` clause
    ///
    /// Example:
    /// ```sql
    /// INSERT VALUES(product, quantity)
    /// ```
    Values(Values),
    /// The insert expression is defined using only the `ROW` keyword.
    ///
    /// Example:
    /// ```sql
    /// INSERT ROW
    /// ```
    /// [BigQuery](https://cloud.google.com/bigquery/docs/reference/standard-sql/dml-syntax#merge_statement)
    Row,
}

impl Display for MergeInsertKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MergeInsertKind::Values(values) => {
                write!(f, "{values}")
            }
            MergeInsertKind::Row => {
                write!(f, "ROW")
            }
        }
    }
}

/// The expression used to insert rows within a `MERGE` statement.
///
/// Examples
/// ```sql
/// INSERT (product, quantity) VALUES(product, quantity)
/// INSERT ROW
/// ```
///
/// [Snowflake](https://docs.snowflake.com/en/sql-reference/sql/merge)
/// [BigQuery](https://cloud.google.com/bigquery/docs/reference/standard-sql/dml-syntax#merge_statement)
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct MergeInsertExpr {
    /// Columns (if any) specified by the insert.
    ///
    /// Example:
    /// ```sql
    /// INSERT (product, quantity) VALUES(product, quantity)
    /// INSERT (product, quantity) ROW
    /// ```
    pub columns: Vec<Ident>,
    /// The insert type used by the statement.
    pub kind: MergeInsertKind,
}

impl Display for MergeInsertExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if !self.columns.is_empty() {
            write!(f, "({}) ", display_comma_separated(self.columns.as_slice()))?;
        }
        write!(f, "{}", self.kind)
    }
}

/// Underlying statement of a when clause within a `MERGE` Statement
///
/// Example
/// ```sql
/// INSERT (product, quantity) VALUES(product, quantity)
/// ```
///
/// [Snowflake](https://docs.snowflake.com/en/sql-reference/sql/merge)
/// [BigQuery](https://cloud.google.com/bigquery/docs/reference/standard-sql/dml-syntax#merge_statement)
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum MergeAction {
    /// An `INSERT` clause
    ///
    /// Example:
    /// ```sql
    /// INSERT (product, quantity) VALUES(product, quantity)
    /// ```
    Insert(MergeInsertExpr),
    /// An `UPDATE` clause
    ///
    /// Example:
    /// ```sql
    /// UPDATE SET quantity = T.quantity + S.quantity
    /// ```
    Update { assignments: Vec<Assignment> },
    /// A plain `DELETE` clause
    Delete,
}

impl Display for MergeAction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MergeAction::Insert(insert) => {
                write!(f, "INSERT {insert}")
            }
            MergeAction::Update { assignments } => {
                write!(f, "UPDATE SET {}", display_comma_separated(assignments))
            }
            MergeAction::Delete => {
                write!(f, "DELETE")
            }
        }
    }
}

/// A when clause within a `MERGE` Statement
///
/// Example:
/// ```sql
/// WHEN NOT MATCHED BY SOURCE AND product LIKE '%washer%' THEN DELETE
/// ```
/// [Snowflake](https://docs.snowflake.com/en/sql-reference/sql/merge)
/// [BigQuery](https://cloud.google.com/bigquery/docs/reference/standard-sql/dml-syntax#merge_statement)
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct MergeClause {
    pub clause_kind: MergeClauseKind,
    pub predicate: Option<Expr>,
    pub action: MergeAction,
}

impl Display for MergeClause {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let MergeClause {
            clause_kind,
            predicate,
            action,
        } = self;

        write!(f, "WHEN {clause_kind}")?;
        if let Some(pred) = predicate {
            write!(f, " AND {pred}")?;
        }
        write!(f, " THEN {action}")
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum DiscardObject {
    ALL,
    PLANS,
    SEQUENCES,
    TEMP,
}

impl fmt::Display for DiscardObject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DiscardObject::ALL => f.write_str("ALL"),
            DiscardObject::PLANS => f.write_str("PLANS"),
            DiscardObject::SEQUENCES => f.write_str("SEQUENCES"),
            DiscardObject::TEMP => f.write_str("TEMP"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum FlushType {
    BinaryLogs,
    EngineLogs,
    ErrorLogs,
    GeneralLogs,
    Hosts,
    Logs,
    Privileges,
    OptimizerCosts,
    RelayLogs,
    SlowLogs,
    Status,
    UserResources,
    Tables,
}

impl fmt::Display for FlushType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FlushType::BinaryLogs => f.write_str("BINARY LOGS"),
            FlushType::EngineLogs => f.write_str("ENGINE LOGS"),
            FlushType::ErrorLogs => f.write_str("ERROR LOGS"),
            FlushType::GeneralLogs => f.write_str("GENERAL LOGS"),
            FlushType::Hosts => f.write_str("HOSTS"),
            FlushType::Logs => f.write_str("LOGS"),
            FlushType::Privileges => f.write_str("PRIVILEGES"),
            FlushType::OptimizerCosts => f.write_str("OPTIMIZER_COSTS"),
            FlushType::RelayLogs => f.write_str("RELAY LOGS"),
            FlushType::SlowLogs => f.write_str("SLOW LOGS"),
            FlushType::Status => f.write_str("STATUS"),
            FlushType::UserResources => f.write_str("USER_RESOURCES"),
            FlushType::Tables => f.write_str("TABLES"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum FlushLocation {
    NoWriteToBinlog,
    Local,
}

impl fmt::Display for FlushLocation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FlushLocation::NoWriteToBinlog => f.write_str("NO_WRITE_TO_BINLOG"),
            FlushLocation::Local => f.write_str("LOCAL"),
        }
    }
}

/// Optional context modifier for statements that can be or `LOCAL`, or `SESSION`.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum ContextModifier {
    /// No context defined. Each dialect defines the default in this scenario.
    None,
    /// `LOCAL` identifier, usually related to transactional states.
    Local,
    /// `SESSION` identifier
    Session,
}

impl fmt::Display for ContextModifier {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::None => {
                write!(f, "")
            }
            Self::Local => {
                write!(f, " LOCAL")
            }
            Self::Session => {
                write!(f, " SESSION")
            }
        }
    }
}

/// Function describe in DROP FUNCTION.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DropFunctionOption {
    Restrict,
    Cascade,
}

impl fmt::Display for DropFunctionOption {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DropFunctionOption::Restrict => write!(f, "RESTRICT "),
            DropFunctionOption::Cascade => write!(f, "CASCADE  "),
        }
    }
}

/// Generic function description for DROP FUNCTION and CREATE TRIGGER.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct FunctionDesc {
    pub name: ObjectName,
    pub args: Option<Vec<OperateFunctionArg>>,
}

impl fmt::Display for FunctionDesc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if let Some(args) = &self.args {
            write!(f, "({})", display_comma_separated(args))?;
        }
        Ok(())
    }
}

/// Function argument in CREATE OR DROP FUNCTION.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct OperateFunctionArg {
    pub mode: Option<ArgMode>,
    pub name: Option<Ident>,
    pub data_type: DataType,
    pub default_expr: Option<Expr>,
}

impl OperateFunctionArg {
    /// Returns an unnamed argument.
    pub fn unnamed(data_type: DataType) -> Self {
        Self {
            mode: None,
            name: None,
            data_type,
            default_expr: None,
        }
    }

    /// Returns an argument with name.
    pub fn with_name(name: &str, data_type: DataType) -> Self {
        Self {
            mode: None,
            name: Some(name.into()),
            data_type,
            default_expr: None,
        }
    }
}

impl fmt::Display for OperateFunctionArg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(mode) = &self.mode {
            write!(f, "{mode} ")?;
        }
        if let Some(name) = &self.name {
            write!(f, "{name} ")?;
        }
        write!(f, "{}", self.data_type)?;
        if let Some(default_expr) = &self.default_expr {
            write!(f, " = {default_expr}")?;
        }
        Ok(())
    }
}

/// The mode of an argument in CREATE FUNCTION.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum ArgMode {
    In,
    Out,
    InOut,
}

impl fmt::Display for ArgMode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ArgMode::In => write!(f, "IN"),
            ArgMode::Out => write!(f, "OUT"),
            ArgMode::InOut => write!(f, "INOUT"),
        }
    }
}

/// These attributes inform the query optimizer about the behavior of the function.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum FunctionBehavior {
    Immutable,
    Stable,
    Volatile,
}

impl fmt::Display for FunctionBehavior {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FunctionBehavior::Immutable => write!(f, "IMMUTABLE"),
            FunctionBehavior::Stable => write!(f, "STABLE"),
            FunctionBehavior::Volatile => write!(f, "VOLATILE"),
        }
    }
}

/// These attributes describe the behavior of the function when called with a null argument.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum FunctionCalledOnNull {
    CalledOnNullInput,
    ReturnsNullOnNullInput,
    Strict,
}

impl fmt::Display for FunctionCalledOnNull {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FunctionCalledOnNull::CalledOnNullInput => write!(f, "CALLED ON NULL INPUT"),
            FunctionCalledOnNull::ReturnsNullOnNullInput => write!(f, "RETURNS NULL ON NULL INPUT"),
            FunctionCalledOnNull::Strict => write!(f, "STRICT"),
        }
    }
}

/// If it is safe for PostgreSQL to call the function from multiple threads at once
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum FunctionParallel {
    Unsafe,
    Restricted,
    Safe,
}

impl fmt::Display for FunctionParallel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FunctionParallel::Unsafe => write!(f, "PARALLEL UNSAFE"),
            FunctionParallel::Restricted => write!(f, "PARALLEL RESTRICTED"),
            FunctionParallel::Safe => write!(f, "PARALLEL SAFE"),
        }
    }
}

/// [BigQuery] Determinism specifier used in a UDF definition.
///
/// [BigQuery]: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#syntax_11
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum FunctionDeterminismSpecifier {
    Deterministic,
    NotDeterministic,
}

impl fmt::Display for FunctionDeterminismSpecifier {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FunctionDeterminismSpecifier::Deterministic => {
                write!(f, "DETERMINISTIC")
            }
            FunctionDeterminismSpecifier::NotDeterministic => {
                write!(f, "NOT DETERMINISTIC")
            }
        }
    }
}

/// Represent the expression body of a `CREATE FUNCTION` statement as well as
/// where within the statement, the body shows up.
///
/// [BigQuery]: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#syntax_11
/// [Postgres]: https://www.postgresql.org/docs/15/sql-createfunction.html
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum CreateFunctionBody {
    /// A function body expression using the 'AS' keyword and shows up
    /// before any `OPTIONS` clause.
    ///
    /// Example:
    /// ```sql
    /// CREATE FUNCTION myfunc(x FLOAT64, y FLOAT64) RETURNS FLOAT64
    /// AS (x * y)
    /// OPTIONS(description="desc");
    /// ```
    ///
    /// [BigQuery]: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#syntax_11
    AsBeforeOptions(Expr),
    /// A function body expression using the 'AS' keyword and shows up
    /// after any `OPTIONS` clause.
    ///
    /// Example:
    /// ```sql
    /// CREATE FUNCTION myfunc(x FLOAT64, y FLOAT64) RETURNS FLOAT64
    /// OPTIONS(description="desc")
    /// AS (x * y);
    /// ```
    ///
    /// [BigQuery]: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#syntax_11
    AsAfterOptions(Expr),
    /// Function body expression using the 'RETURN' keyword.
    ///
    /// Example:
    /// ```sql
    /// CREATE FUNCTION myfunc(a INTEGER, IN b INTEGER = 1) RETURNS INTEGER
    /// LANGUAGE SQL
    /// RETURN a + b;
    /// ```
    ///
    /// [Postgres]: https://www.postgresql.org/docs/current/sql-createfunction.html
    Return(Expr),
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum CreateFunctionUsing {
    Jar(String),
    File(String),
    Archive(String),
}

impl fmt::Display for CreateFunctionUsing {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "USING ")?;
        match self {
            CreateFunctionUsing::Jar(uri) => write!(f, "JAR '{uri}'"),
            CreateFunctionUsing::File(uri) => write!(f, "FILE '{uri}'"),
            CreateFunctionUsing::Archive(uri) => write!(f, "ARCHIVE '{uri}'"),
        }
    }
}

/// `NAME = <EXPR>` arguments for DuckDB macros
///
/// See [Create Macro - DuckDB](https://duckdb.org/docs/sql/statements/create_macro)
/// for more details
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct MacroArg {
    pub name: Ident,
    pub default_expr: Option<Expr>,
}

impl MacroArg {
    /// Returns an argument with name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            default_expr: None,
        }
    }
}

impl fmt::Display for MacroArg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if let Some(default_expr) = &self.default_expr {
            write!(f, " := {default_expr}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum MacroDefinition {
    Expr(Expr),
    Table(Box<Query>),
}

impl fmt::Display for MacroDefinition {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MacroDefinition::Expr(expr) => write!(f, "{expr}")?,
            MacroDefinition::Table(query) => write!(f, "{query}")?,
        }
        Ok(())
    }
}

/// Schema possible naming variants ([1]).
///
/// [1]: https://jakewheat.github.io/sql-overview/sql-2016-foundation-grammar.html#schema-definition
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum SchemaName {
    /// Only schema name specified: `<schema name>`.
    Simple(ObjectName),
    /// Only authorization identifier specified: `AUTHORIZATION <schema authorization identifier>`.
    UnnamedAuthorization(Ident),
    /// Both schema name and authorization identifier specified: `<schema name>  AUTHORIZATION <schema authorization identifier>`.
    NamedAuthorization(ObjectName, Ident),
}

impl fmt::Display for SchemaName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchemaName::Simple(name) => {
                write!(f, "{name}")
            }
            SchemaName::UnnamedAuthorization(authorization) => {
                write!(f, "AUTHORIZATION {authorization}")
            }
            SchemaName::NamedAuthorization(name, authorization) => {
                write!(f, "{name} AUTHORIZATION {authorization}")
            }
        }
    }
}

/// Fulltext search modifiers ([1]).
///
/// [1]: https://dev.mysql.com/doc/refman/8.0/en/fulltext-search.html#function_match
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum SearchModifier {
    /// `IN NATURAL LANGUAGE MODE`.
    InNaturalLanguageMode,
    /// `IN NATURAL LANGUAGE MODE WITH QUERY EXPANSION`.
    InNaturalLanguageModeWithQueryExpansion,
    ///`IN BOOLEAN MODE`.
    InBooleanMode,
    ///`WITH QUERY EXPANSION`.
    WithQueryExpansion,
}

impl fmt::Display for SearchModifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InNaturalLanguageMode => {
                write!(f, "IN NATURAL LANGUAGE MODE")?;
            }
            Self::InNaturalLanguageModeWithQueryExpansion => {
                write!(f, "IN NATURAL LANGUAGE MODE WITH QUERY EXPANSION")?;
            }
            Self::InBooleanMode => {
                write!(f, "IN BOOLEAN MODE")?;
            }
            Self::WithQueryExpansion => {
                write!(f, "WITH QUERY EXPANSION")?;
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct LockTable {
    pub table: Ident,
    pub alias: Option<Ident>,
    pub lock_type: LockTableType,
}

impl fmt::Display for LockTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            table: tbl_name,
            alias,
            lock_type,
        } = self;

        write!(f, "{tbl_name} ")?;
        if let Some(alias) = alias {
            write!(f, "AS {alias} ")?;
        }
        write!(f, "{lock_type}")?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum LockTableType {
    Read { local: bool },
    Write { low_priority: bool },
}

impl fmt::Display for LockTableType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { local } => {
                write!(f, "READ")?;
                if *local {
                    write!(f, " LOCAL")?;
                }
            }
            Self::Write { low_priority } => {
                if *low_priority {
                    write!(f, "LOW_PRIORITY ")?;
                }
                write!(f, "WRITE")?;
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct HiveSetLocation {
    pub has_set: bool,
    pub location: Ident,
}

impl fmt::Display for HiveSetLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.has_set {
            write!(f, "SET ")?;
        }
        write!(f, "LOCATION {}", self.location)
    }
}

/// MySQL `ALTER TABLE` only  [FIRST | AFTER column_name]
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum MySQLColumnPosition {
    First,
    After(Ident),
}

impl Display for MySQLColumnPosition {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MySQLColumnPosition::First => Ok(write!(f, "FIRST")?),
            MySQLColumnPosition::After(ident) => {
                let column_name = &ident.value;
                Ok(write!(f, "AFTER {column_name}")?)
            }
        }
    }
}

/// Engine of DB. Some warehouse has parameters of engine, e.g. [clickhouse]
///
/// [clickhouse]: https://clickhouse.com/docs/en/engines/table-engines
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct TableEngine {
    pub name: String,
    pub parameters: Option<Vec<Ident>>,
}

impl Display for TableEngine {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)?;

        if let Some(parameters) = self.parameters.as_ref() {
            write!(f, "({})", display_comma_separated(parameters))?;
        }

        Ok(())
    }
}

/// Snowflake `WITH ROW ACCESS POLICY policy_name ON (identifier, ...)`
///
/// <https://docs.snowflake.com/en/sql-reference/sql/create-table>
/// <https://docs.snowflake.com/en/user-guide/security-row-intro>
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct RowAccessPolicy {
    pub policy: ObjectName,
    pub on: Vec<Ident>,
}

impl RowAccessPolicy {
    pub fn new(policy: ObjectName, on: Vec<Ident>) -> Self {
        Self { policy, on }
    }
}

impl Display for RowAccessPolicy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "WITH ROW ACCESS POLICY {} ON ({})",
            self.policy,
            display_comma_separated(self.on.as_slice())
        )
    }
}

/// Snowflake `WITH TAG ( tag_name = '<tag_value>', ...)`
///
/// <https://docs.snowflake.com/en/sql-reference/sql/create-table>
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct Tag {
    pub key: Ident,
    pub value: String,
}

impl Tag {
    pub fn new(key: Ident, value: String) -> Self {
        Self { key, value }
    }
}

impl Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}='{}'", self.key, self.value)
    }
}

/// Helper to indicate if a comment includes the `=` in the display form
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum CommentDef {
    /// Includes `=` when printing the comment, as `COMMENT = 'comment'`
    /// Does not include `=` when printing the comment, as `COMMENT 'comment'`
    WithEq(String),
    WithoutEq(String),
    // For Hive dialect, the table comment is after the column definitions without `=`,
    // so we need to add an extra variant to allow to identify this case when displaying.
    // [Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-CreateTable)
    AfterColumnDefsWithoutEq(String),
}

impl Display for CommentDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CommentDef::WithEq(comment)
            | CommentDef::WithoutEq(comment)
            | CommentDef::AfterColumnDefsWithoutEq(comment) => write!(f, "{comment}"),
        }
    }
}

/// Helper to indicate if a collection should be wrapped by a symbol in the display form
///
/// [`Display`] is implemented for every [`Vec<T>`] where `T: Display`.
/// The string output is a comma separated list for the vec items
///
/// # Examples
/// ```
/// # use sqlparser::ast::WrappedCollection;
/// let items = WrappedCollection::Parentheses(vec!["one", "two", "three"]);
/// assert_eq!("(one, two, three)", items.to_string());
///
/// let items = WrappedCollection::NoWrapping(vec!["one", "two", "three"]);
/// assert_eq!("one, two, three", items.to_string());
/// ```
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum WrappedCollection<T> {
    /// Print the collection without wrapping symbols, as `item, item, item`
    NoWrapping(T),
    /// Wraps the collection in Parentheses, as `(item, item, item)`
    Parentheses(T),
}

impl<T> Display for WrappedCollection<Vec<T>>
where
    T: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WrappedCollection::NoWrapping(inner) => {
                write!(f, "{}", display_comma_separated(inner.as_slice()))
            }
            WrappedCollection::Parentheses(inner) => {
                write!(f, "({})", display_comma_separated(inner.as_slice()))
            }
        }
    }
}

/// Represents a single PostgreSQL utility option.
///
/// A utility option is a key-value pair where the key is an identifier (IDENT) and the value
/// can be one of the following:
/// - A number with an optional sign (`+` or `-`). Example: `+10`, `-10.2`, `3`
/// - A non-keyword string. Example: `option1`, `'option2'`, `"option3"`
/// - keyword: `TRUE`, `FALSE`, `ON` (`off` is also accept).
/// - Empty. Example: `ANALYZE` (identifier only)
///
/// Utility options are used in various PostgreSQL DDL statements, including statements such as
/// `CLUSTER`, `EXPLAIN`, `VACUUM`, and `REINDEX`. These statements format options as `( option [, ...] )`.
///
/// [CLUSTER](https://www.postgresql.org/docs/current/sql-cluster.html)
/// [EXPLAIN](https://www.postgresql.org/docs/current/sql-explain.html)
/// [VACUUM](https://www.postgresql.org/docs/current/sql-vacuum.html)
/// [REINDEX](https://www.postgresql.org/docs/current/sql-reindex.html)
///
/// For example, the `EXPLAIN` AND `VACUUM` statements with options might look like this:
/// ```sql
/// EXPLAIN (ANALYZE, VERBOSE TRUE, FORMAT TEXT) SELECT * FROM my_table;
///
/// VACCUM (VERBOSE, ANALYZE ON, PARALLEL 10) my_table;
/// ```
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct UtilityOption {
    pub name: Ident,
    pub arg: Option<Expr>,
}

impl Display for UtilityOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref arg) = self.arg {
            write!(f, "{} {}", self.name, arg)
        } else {
            write!(f, "{}", self.name)
        }
    }
}

/// Represents the different options available for `SHOW`
/// statements to filter the results. Example from Snowflake:
/// <https://docs.snowflake.com/en/sql-reference/sql/show-tables>
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct ShowStatementOptions {
    pub show_in: Option<ShowStatementIn>,
    pub starts_with: Option<Value>,
    pub limit: Option<Expr>,
    pub limit_from: Option<Value>,
    pub filter_position: Option<ShowStatementFilterPosition>,
}

impl Display for ShowStatementOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (like_in_infix, like_in_suffix) = match &self.filter_position {
            Some(ShowStatementFilterPosition::Infix(filter)) => {
                (format!(" {filter}"), "".to_string())
            }
            Some(ShowStatementFilterPosition::Suffix(filter)) => {
                ("".to_string(), format!(" {filter}"))
            }
            None => ("".to_string(), "".to_string()),
        };
        write!(
            f,
            "{like_in_infix}{show_in}{starts_with}{limit}{from}{like_in_suffix}",
            show_in = match &self.show_in {
                Some(i) => format!(" {i}"),
                None => String::new(),
            },
            starts_with = match &self.starts_with {
                Some(s) => format!(" STARTS WITH {s}"),
                None => String::new(),
            },
            limit = match &self.limit {
                Some(l) => format!(" LIMIT {l}"),
                None => String::new(),
            },
            from = match &self.limit_from {
                Some(f) => format!(" FROM {f}"),
                None => String::new(),
            }
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum ShowStatementFilterPosition {
    Infix(ShowStatementFilter), // For example: SHOW COLUMNS LIKE '%name%' IN TABLE tbl
    Suffix(ShowStatementFilter), // For example: SHOW COLUMNS IN tbl LIKE '%name%'
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum ShowStatementInParentType {
    Account,
    Database,
    Schema,
    Table,
    View,
}

impl fmt::Display for ShowStatementInParentType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ShowStatementInParentType::Account => write!(f, "ACCOUNT"),
            ShowStatementInParentType::Database => write!(f, "DATABASE"),
            ShowStatementInParentType::Schema => write!(f, "SCHEMA"),
            ShowStatementInParentType::Table => write!(f, "TABLE"),
            ShowStatementInParentType::View => write!(f, "VIEW"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct ShowStatementIn {
    pub clause: ShowStatementInClause,
    pub parent_type: Option<ShowStatementInParentType>,
    #[cfg_attr(feature = "visitor", visit(with = "visit_relation"))]
    pub parent_name: Option<ObjectName>,
}

impl fmt::Display for ShowStatementIn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.clause)?;
        if let Some(parent_type) = &self.parent_type {
            write!(f, " {}", parent_type)?;
        }
        if let Some(parent_name) = &self.parent_name {
            write!(f, " {}", parent_name)?;
        }
        Ok(())
    }
}

/// MSSQL's json null clause
///
/// ```plaintext
/// <json_null_clause> ::=
///       NULL ON NULL
///     | ABSENT ON NULL
/// ```
///
/// <https://learn.microsoft.com/en-us/sql/t-sql/functions/json-object-transact-sql?view=sql-server-ver16#json_null_clause>
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum JsonNullClause {
    NullOnNull,
    AbsentOnNull,
}

impl Display for JsonNullClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsonNullClause::NullOnNull => write!(f, "NULL ON NULL"),
            JsonNullClause::AbsentOnNull => write!(f, "ABSENT ON NULL"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_frame_default() {
        let window_frame = WindowFrame::default();
        assert_eq!(WindowFrameBound::Preceding(None), window_frame.start_bound);
    }

    #[test]
    fn test_grouping_sets_display() {
        // a and b in different group
        let grouping_sets = Expr::GroupingSets(vec![
            vec![Expr::Identifier(Ident::new("a"))],
            vec![Expr::Identifier(Ident::new("b"))],
        ]);
        assert_eq!("GROUPING SETS ((a), (b))", format!("{grouping_sets}"));

        // a and b in the same group
        let grouping_sets = Expr::GroupingSets(vec![vec![
            Expr::Identifier(Ident::new("a")),
            Expr::Identifier(Ident::new("b")),
        ]]);
        assert_eq!("GROUPING SETS ((a, b))", format!("{grouping_sets}"));

        // (a, b) and (c, d) in different group
        let grouping_sets = Expr::GroupingSets(vec![
            vec![
                Expr::Identifier(Ident::new("a")),
                Expr::Identifier(Ident::new("b")),
            ],
            vec![
                Expr::Identifier(Ident::new("c")),
                Expr::Identifier(Ident::new("d")),
            ],
        ]);
        assert_eq!("GROUPING SETS ((a, b), (c, d))", format!("{grouping_sets}"));
    }

    #[test]
    fn test_rollup_display() {
        let rollup = Expr::Rollup(vec![vec![Expr::Identifier(Ident::new("a"))]]);
        assert_eq!("ROLLUP (a)", format!("{rollup}"));

        let rollup = Expr::Rollup(vec![vec![
            Expr::Identifier(Ident::new("a")),
            Expr::Identifier(Ident::new("b")),
        ]]);
        assert_eq!("ROLLUP ((a, b))", format!("{rollup}"));

        let rollup = Expr::Rollup(vec![
            vec![Expr::Identifier(Ident::new("a"))],
            vec![Expr::Identifier(Ident::new("b"))],
        ]);
        assert_eq!("ROLLUP (a, b)", format!("{rollup}"));

        let rollup = Expr::Rollup(vec![
            vec![Expr::Identifier(Ident::new("a"))],
            vec![
                Expr::Identifier(Ident::new("b")),
                Expr::Identifier(Ident::new("c")),
            ],
            vec![Expr::Identifier(Ident::new("d"))],
        ]);
        assert_eq!("ROLLUP (a, (b, c), d)", format!("{rollup}"));
    }

    #[test]
    fn test_cube_display() {
        let cube = Expr::Cube(vec![vec![Expr::Identifier(Ident::new("a"))]]);
        assert_eq!("CUBE (a)", format!("{cube}"));

        let cube = Expr::Cube(vec![vec![
            Expr::Identifier(Ident::new("a")),
            Expr::Identifier(Ident::new("b")),
        ]]);
        assert_eq!("CUBE ((a, b))", format!("{cube}"));

        let cube = Expr::Cube(vec![
            vec![Expr::Identifier(Ident::new("a"))],
            vec![Expr::Identifier(Ident::new("b"))],
        ]);
        assert_eq!("CUBE (a, b)", format!("{cube}"));

        let cube = Expr::Cube(vec![
            vec![Expr::Identifier(Ident::new("a"))],
            vec![
                Expr::Identifier(Ident::new("b")),
                Expr::Identifier(Ident::new("c")),
            ],
            vec![Expr::Identifier(Ident::new("d"))],
        ]);
        assert_eq!("CUBE (a, (b, c), d)", format!("{cube}"));
    }

    #[test]
    fn test_interval_display() {
        let interval = Expr::Interval(Interval {
            value: Box::new(Expr::Value(Value::SingleQuotedString(String::from(
                "123:45.67",
            )))),
            leading_field: Some(DateTimeField::Minute),
            leading_precision: Some(10),
            last_field: Some(DateTimeField::Second),
            fractional_seconds_precision: Some(9),
        });
        assert_eq!(
            "INTERVAL '123:45.67' MINUTE (10) TO SECOND (9)",
            format!("{interval}"),
        );

        let interval = Expr::Interval(Interval {
            value: Box::new(Expr::Value(Value::SingleQuotedString(String::from("5")))),
            leading_field: Some(DateTimeField::Second),
            leading_precision: Some(1),
            last_field: None,
            fractional_seconds_precision: Some(3),
        });
        assert_eq!("INTERVAL '5' SECOND (1, 3)", format!("{interval}"));
    }

    #[test]
    fn test_one_or_many_with_parens_deref() {
        use core::ops::Index;

        let one = OneOrManyWithParens::One("a");

        assert_eq!(one.deref(), &["a"]);
        assert_eq!(<OneOrManyWithParens<_> as Deref>::deref(&one), &["a"]);

        assert_eq!(one[0], "a");
        assert_eq!(one.index(0), &"a");
        assert_eq!(
            <<OneOrManyWithParens<_> as Deref>::Target as Index<usize>>::index(&one, 0),
            &"a"
        );

        assert_eq!(one.len(), 1);
        assert_eq!(<OneOrManyWithParens<_> as Deref>::Target::len(&one), 1);

        let many1 = OneOrManyWithParens::Many(vec!["b"]);

        assert_eq!(many1.deref(), &["b"]);
        assert_eq!(<OneOrManyWithParens<_> as Deref>::deref(&many1), &["b"]);

        assert_eq!(many1[0], "b");
        assert_eq!(many1.index(0), &"b");
        assert_eq!(
            <<OneOrManyWithParens<_> as Deref>::Target as Index<usize>>::index(&many1, 0),
            &"b"
        );

        assert_eq!(many1.len(), 1);
        assert_eq!(<OneOrManyWithParens<_> as Deref>::Target::len(&many1), 1);

        let many2 = OneOrManyWithParens::Many(vec!["c", "d"]);

        assert_eq!(many2.deref(), &["c", "d"]);
        assert_eq!(
            <OneOrManyWithParens<_> as Deref>::deref(&many2),
            &["c", "d"]
        );

        assert_eq!(many2[0], "c");
        assert_eq!(many2.index(0), &"c");
        assert_eq!(
            <<OneOrManyWithParens<_> as Deref>::Target as Index<usize>>::index(&many2, 0),
            &"c"
        );

        assert_eq!(many2[1], "d");
        assert_eq!(many2.index(1), &"d");
        assert_eq!(
            <<OneOrManyWithParens<_> as Deref>::Target as Index<usize>>::index(&many2, 1),
            &"d"
        );

        assert_eq!(many2.len(), 2);
        assert_eq!(<OneOrManyWithParens<_> as Deref>::Target::len(&many2), 2);
    }

    #[test]
    fn test_one_or_many_with_parens_as_ref() {
        let one = OneOrManyWithParens::One("a");

        assert_eq!(one.as_ref(), &["a"]);
        assert_eq!(<OneOrManyWithParens<_> as AsRef<_>>::as_ref(&one), &["a"]);

        let many1 = OneOrManyWithParens::Many(vec!["b"]);

        assert_eq!(many1.as_ref(), &["b"]);
        assert_eq!(<OneOrManyWithParens<_> as AsRef<_>>::as_ref(&many1), &["b"]);

        let many2 = OneOrManyWithParens::Many(vec!["c", "d"]);

        assert_eq!(many2.as_ref(), &["c", "d"]);
        assert_eq!(
            <OneOrManyWithParens<_> as AsRef<_>>::as_ref(&many2),
            &["c", "d"]
        );
    }

    #[test]
    fn test_one_or_many_with_parens_ref_into_iter() {
        let one = OneOrManyWithParens::One("a");

        assert_eq!(Vec::from_iter(&one), vec![&"a"]);

        let many1 = OneOrManyWithParens::Many(vec!["b"]);

        assert_eq!(Vec::from_iter(&many1), vec![&"b"]);

        let many2 = OneOrManyWithParens::Many(vec!["c", "d"]);

        assert_eq!(Vec::from_iter(&many2), vec![&"c", &"d"]);
    }

    #[test]
    fn test_one_or_many_with_parens_value_into_iter() {
        use core::iter::once;

        //tests that our iterator implemented methods behaves exactly as it's inner iterator, at every step up to n calls to next/next_back
        fn test_steps<I>(ours: OneOrManyWithParens<usize>, inner: I, n: usize)
        where
            I: IntoIterator<Item = usize, IntoIter: DoubleEndedIterator + Clone> + Clone,
        {
            fn checks<I>(ours: OneOrManyWithParensIntoIter<usize>, inner: I)
            where
                I: Iterator<Item = usize> + Clone + DoubleEndedIterator,
            {
                assert_eq!(ours.size_hint(), inner.size_hint());
                assert_eq!(ours.clone().count(), inner.clone().count());

                assert_eq!(
                    ours.clone().fold(1, |a, v| a + v),
                    inner.clone().fold(1, |a, v| a + v)
                );

                assert_eq!(Vec::from_iter(ours.clone()), Vec::from_iter(inner.clone()));
                assert_eq!(
                    Vec::from_iter(ours.clone().rev()),
                    Vec::from_iter(inner.clone().rev())
                );
            }

            let mut ours_next = ours.clone().into_iter();
            let mut inner_next = inner.clone().into_iter();

            for _ in 0..n {
                checks(ours_next.clone(), inner_next.clone());

                assert_eq!(ours_next.next(), inner_next.next());
            }

            let mut ours_next_back = ours.clone().into_iter();
            let mut inner_next_back = inner.clone().into_iter();

            for _ in 0..n {
                checks(ours_next_back.clone(), inner_next_back.clone());

                assert_eq!(ours_next_back.next_back(), inner_next_back.next_back());
            }

            let mut ours_mixed = ours.clone().into_iter();
            let mut inner_mixed = inner.clone().into_iter();

            for i in 0..n {
                checks(ours_mixed.clone(), inner_mixed.clone());

                if i % 2 == 0 {
                    assert_eq!(ours_mixed.next_back(), inner_mixed.next_back());
                } else {
                    assert_eq!(ours_mixed.next(), inner_mixed.next());
                }
            }

            let mut ours_mixed2 = ours.into_iter();
            let mut inner_mixed2 = inner.into_iter();

            for i in 0..n {
                checks(ours_mixed2.clone(), inner_mixed2.clone());

                if i % 2 == 0 {
                    assert_eq!(ours_mixed2.next(), inner_mixed2.next());
                } else {
                    assert_eq!(ours_mixed2.next_back(), inner_mixed2.next_back());
                }
            }
        }

        test_steps(OneOrManyWithParens::One(1), once(1), 3);
        test_steps(OneOrManyWithParens::Many(vec![2]), vec![2], 3);
        test_steps(OneOrManyWithParens::Many(vec![3, 4]), vec![3, 4], 4);
    }
}
