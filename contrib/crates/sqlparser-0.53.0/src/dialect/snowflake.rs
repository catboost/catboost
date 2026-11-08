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

#[cfg(not(feature = "std"))]
use crate::alloc::string::ToString;
use crate::ast::helpers::stmt_create_table::CreateTableBuilder;
use crate::ast::helpers::stmt_data_loading::{
    DataLoadingOption, DataLoadingOptionType, DataLoadingOptions, StageLoadSelectItem,
    StageParamsObject,
};
use crate::ast::{
    ColumnOption, ColumnPolicy, ColumnPolicyProperty, Ident, IdentityParameters, IdentityProperty,
    IdentityPropertyFormatKind, IdentityPropertyKind, IdentityPropertyOrder, ObjectName,
    RowAccessPolicy, Statement, TagsColumnOption, WrappedCollection,
};
use crate::dialect::{Dialect, Precedence};
use crate::keywords::Keyword;
use crate::parser::{Parser, ParserError};
use crate::tokenizer::Token;
#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::{format, vec};

use super::keywords::RESERVED_FOR_IDENTIFIER;

/// A [`Dialect`] for [Snowflake](https://www.snowflake.com/)
#[derive(Debug, Default)]
pub struct SnowflakeDialect;

impl Dialect for SnowflakeDialect {
    // see https://docs.snowflake.com/en/sql-reference/identifiers-syntax.html
    fn is_identifier_start(&self, ch: char) -> bool {
        ch.is_ascii_lowercase() || ch.is_ascii_uppercase() || ch == '_'
    }

    fn supports_projection_trailing_commas(&self) -> bool {
        true
    }

    // Snowflake supports double-dot notation when the schema name is not specified
    // In this case the default PUBLIC schema is used
    //
    // see https://docs.snowflake.com/en/sql-reference/name-resolution#resolution-when-schema-omitted-double-dot-notation
    fn supports_object_name_double_dot_notation(&self) -> bool {
        true
    }

    fn is_identifier_part(&self, ch: char) -> bool {
        ch.is_ascii_lowercase()
            || ch.is_ascii_uppercase()
            || ch.is_ascii_digit()
            || ch == '$'
            || ch == '_'
    }

    // See https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical#escape_sequences
    fn supports_string_literal_backslash_escape(&self) -> bool {
        true
    }

    fn supports_within_after_array_aggregation(&self) -> bool {
        true
    }

    fn supports_connect_by(&self) -> bool {
        true
    }

    fn supports_match_recognize(&self) -> bool {
        true
    }

    // Snowflake uses this syntax for "object constants" (the values of which
    // are not actually required to be constants).
    //
    // https://docs.snowflake.com/en/sql-reference/data-types-semistructured#label-object-constant
    fn supports_dictionary_syntax(&self) -> bool {
        true
    }

    // Snowflake doesn't document this but `FIRST_VALUE(arg, { IGNORE | RESPECT } NULLS)`
    // works (i.e. inside the argument list instead of after).
    fn supports_window_function_null_treatment_arg(&self) -> bool {
        true
    }

    /// See [doc](https://docs.snowflake.com/en/sql-reference/sql/set#syntax)
    fn supports_parenthesized_set_variables(&self) -> bool {
        true
    }

    /// See [doc](https://docs.snowflake.com/en/sql-reference/sql/comment)
    fn supports_comment_on(&self) -> bool {
        true
    }

    fn parse_statement(&self, parser: &mut Parser) -> Option<Result<Statement, ParserError>> {
        if parser.parse_keyword(Keyword::CREATE) {
            // possibly CREATE STAGE
            //[ OR  REPLACE ]
            let or_replace = parser.parse_keywords(&[Keyword::OR, Keyword::REPLACE]);
            // LOCAL | GLOBAL
            let global = match parser.parse_one_of_keywords(&[Keyword::LOCAL, Keyword::GLOBAL]) {
                Some(Keyword::LOCAL) => Some(false),
                Some(Keyword::GLOBAL) => Some(true),
                _ => None,
            };

            let mut temporary = false;
            let mut volatile = false;
            let mut transient = false;

            match parser.parse_one_of_keywords(&[
                Keyword::TEMP,
                Keyword::TEMPORARY,
                Keyword::VOLATILE,
                Keyword::TRANSIENT,
            ]) {
                Some(Keyword::TEMP | Keyword::TEMPORARY) => temporary = true,
                Some(Keyword::VOLATILE) => volatile = true,
                Some(Keyword::TRANSIENT) => transient = true,
                _ => {}
            }

            if parser.parse_keyword(Keyword::STAGE) {
                // OK - this is CREATE STAGE statement
                return Some(parse_create_stage(or_replace, temporary, parser));
            } else if parser.parse_keyword(Keyword::TABLE) {
                return Some(parse_create_table(
                    or_replace, global, temporary, volatile, transient, parser,
                ));
            } else {
                // need to go back with the cursor
                let mut back = 1;
                if or_replace {
                    back += 2
                }
                if temporary {
                    back += 1
                }
                for _i in 0..back {
                    parser.prev_token();
                }
            }
        }
        if parser.parse_keywords(&[Keyword::COPY, Keyword::INTO]) {
            // COPY INTO
            return Some(parse_copy_into(parser));
        }

        None
    }

    fn parse_column_option(
        &self,
        parser: &mut Parser,
    ) -> Result<Option<Result<Option<ColumnOption>, ParserError>>, ParserError> {
        parser.maybe_parse(|parser| {
            let with = parser.parse_keyword(Keyword::WITH);

            if parser.parse_keyword(Keyword::IDENTITY) {
                Ok(parse_identity_property(parser)
                    .map(|p| Some(ColumnOption::Identity(IdentityPropertyKind::Identity(p)))))
            } else if parser.parse_keyword(Keyword::AUTOINCREMENT) {
                Ok(parse_identity_property(parser).map(|p| {
                    Some(ColumnOption::Identity(IdentityPropertyKind::Autoincrement(
                        p,
                    )))
                }))
            } else if parser.parse_keywords(&[Keyword::MASKING, Keyword::POLICY]) {
                Ok(parse_column_policy_property(parser, with)
                    .map(|p| Some(ColumnOption::Policy(ColumnPolicy::MaskingPolicy(p)))))
            } else if parser.parse_keywords(&[Keyword::PROJECTION, Keyword::POLICY]) {
                Ok(parse_column_policy_property(parser, with)
                    .map(|p| Some(ColumnOption::Policy(ColumnPolicy::ProjectionPolicy(p)))))
            } else if parser.parse_keywords(&[Keyword::TAG]) {
                Ok(parse_column_tags(parser, with).map(|p| Some(ColumnOption::Tags(p))))
            } else {
                Err(ParserError::ParserError("not found match".to_string()))
            }
        })
    }

    fn get_next_precedence(&self, parser: &Parser) -> Option<Result<u8, ParserError>> {
        let token = parser.peek_token();
        // Snowflake supports the `:` cast operator unlike other dialects
        match token.token {
            Token::Colon => Some(Ok(self.prec_value(Precedence::DoubleColon))),
            _ => None,
        }
    }

    fn describe_requires_table_keyword(&self) -> bool {
        true
    }

    fn allow_extract_custom(&self) -> bool {
        true
    }

    fn allow_extract_single_quotes(&self) -> bool {
        true
    }

    /// Snowflake expects the `LIKE` option before the `IN` option,
    /// for example: <https://docs.snowflake.com/en/sql-reference/sql/show-views#syntax>
    fn supports_show_like_before_in(&self) -> bool {
        true
    }

    fn is_reserved_for_identifier(&self, kw: Keyword) -> bool {
        // Unreserve some keywords that Snowflake accepts as identifiers
        // See: https://docs.snowflake.com/en/sql-reference/reserved-keywords
        if matches!(kw, Keyword::INTERVAL) {
            false
        } else {
            RESERVED_FOR_IDENTIFIER.contains(&kw)
        }
    }
}

/// Parse snowflake create table statement.
/// <https://docs.snowflake.com/en/sql-reference/sql/create-table>
pub fn parse_create_table(
    or_replace: bool,
    global: Option<bool>,
    temporary: bool,
    volatile: bool,
    transient: bool,
    parser: &mut Parser,
) -> Result<Statement, ParserError> {
    let if_not_exists = parser.parse_keywords(&[Keyword::IF, Keyword::NOT, Keyword::EXISTS]);
    let table_name = parser.parse_object_name(false)?;

    let mut builder = CreateTableBuilder::new(table_name)
        .or_replace(or_replace)
        .if_not_exists(if_not_exists)
        .temporary(temporary)
        .transient(transient)
        .volatile(volatile)
        .global(global)
        .hive_formats(Some(Default::default()));

    // Snowflake does not enforce order of the parameters in the statement. The parser needs to
    // parse the statement in a loop.
    //
    // "CREATE TABLE x COPY GRANTS (c INT)" and "CREATE TABLE x (c INT) COPY GRANTS" are both
    // accepted by Snowflake

    loop {
        let next_token = parser.next_token();
        match &next_token.token {
            Token::Word(word) => match word.keyword {
                Keyword::COPY => {
                    parser.expect_keyword(Keyword::GRANTS)?;
                    builder = builder.copy_grants(true);
                }
                Keyword::COMMENT => {
                    // Rewind the COMMENT keyword
                    parser.prev_token();
                    builder = builder.comment(parser.parse_optional_inline_comment()?);
                }
                Keyword::AS => {
                    let query = parser.parse_query()?;
                    builder = builder.query(Some(query));
                    break;
                }
                Keyword::CLONE => {
                    let clone = parser.parse_object_name(false).ok();
                    builder = builder.clone_clause(clone);
                    break;
                }
                Keyword::LIKE => {
                    let like = parser.parse_object_name(false).ok();
                    builder = builder.like(like);
                    break;
                }
                Keyword::CLUSTER => {
                    parser.expect_keyword(Keyword::BY)?;
                    parser.expect_token(&Token::LParen)?;
                    let cluster_by = Some(WrappedCollection::Parentheses(
                        parser.parse_comma_separated(|p| p.parse_identifier(false))?,
                    ));
                    parser.expect_token(&Token::RParen)?;

                    builder = builder.cluster_by(cluster_by)
                }
                Keyword::ENABLE_SCHEMA_EVOLUTION => {
                    parser.expect_token(&Token::Eq)?;
                    let enable_schema_evolution =
                        match parser.parse_one_of_keywords(&[Keyword::TRUE, Keyword::FALSE]) {
                            Some(Keyword::TRUE) => true,
                            Some(Keyword::FALSE) => false,
                            _ => {
                                return parser.expected("TRUE or FALSE", next_token);
                            }
                        };

                    builder = builder.enable_schema_evolution(Some(enable_schema_evolution));
                }
                Keyword::CHANGE_TRACKING => {
                    parser.expect_token(&Token::Eq)?;
                    let change_tracking =
                        match parser.parse_one_of_keywords(&[Keyword::TRUE, Keyword::FALSE]) {
                            Some(Keyword::TRUE) => true,
                            Some(Keyword::FALSE) => false,
                            _ => {
                                return parser.expected("TRUE or FALSE", next_token);
                            }
                        };

                    builder = builder.change_tracking(Some(change_tracking));
                }
                Keyword::DATA_RETENTION_TIME_IN_DAYS => {
                    parser.expect_token(&Token::Eq)?;
                    let data_retention_time_in_days = parser.parse_literal_uint()?;
                    builder =
                        builder.data_retention_time_in_days(Some(data_retention_time_in_days));
                }
                Keyword::MAX_DATA_EXTENSION_TIME_IN_DAYS => {
                    parser.expect_token(&Token::Eq)?;
                    let max_data_extension_time_in_days = parser.parse_literal_uint()?;
                    builder = builder
                        .max_data_extension_time_in_days(Some(max_data_extension_time_in_days));
                }
                Keyword::DEFAULT_DDL_COLLATION => {
                    parser.expect_token(&Token::Eq)?;
                    let default_ddl_collation = parser.parse_literal_string()?;
                    builder = builder.default_ddl_collation(Some(default_ddl_collation));
                }
                // WITH is optional, we just verify that next token is one of the expected ones and
                // fallback to the default match statement
                Keyword::WITH => {
                    parser.expect_one_of_keywords(&[
                        Keyword::AGGREGATION,
                        Keyword::TAG,
                        Keyword::ROW,
                    ])?;
                    parser.prev_token();
                }
                Keyword::AGGREGATION => {
                    parser.expect_keyword(Keyword::POLICY)?;
                    let aggregation_policy = parser.parse_object_name(false)?;
                    builder = builder.with_aggregation_policy(Some(aggregation_policy));
                }
                Keyword::ROW => {
                    parser.expect_keywords(&[Keyword::ACCESS, Keyword::POLICY])?;
                    let policy = parser.parse_object_name(false)?;
                    parser.expect_keyword(Keyword::ON)?;
                    parser.expect_token(&Token::LParen)?;
                    let columns = parser.parse_comma_separated(|p| p.parse_identifier(false))?;
                    parser.expect_token(&Token::RParen)?;

                    builder =
                        builder.with_row_access_policy(Some(RowAccessPolicy::new(policy, columns)))
                }
                Keyword::TAG => {
                    parser.expect_token(&Token::LParen)?;
                    let tags = parser.parse_comma_separated(Parser::parse_tag)?;
                    parser.expect_token(&Token::RParen)?;
                    builder = builder.with_tags(Some(tags));
                }
                _ => {
                    return parser.expected("end of statement", next_token);
                }
            },
            Token::LParen => {
                parser.prev_token();
                let (columns, constraints) = parser.parse_columns()?;
                builder = builder.columns(columns).constraints(constraints);
            }
            Token::EOF => {
                if builder.columns.is_empty() {
                    return Err(ParserError::ParserError(
                        "unexpected end of input".to_string(),
                    ));
                }

                break;
            }
            Token::SemiColon => {
                if builder.columns.is_empty() {
                    return Err(ParserError::ParserError(
                        "unexpected end of input".to_string(),
                    ));
                }

                parser.prev_token();
                break;
            }
            _ => {
                return parser.expected("end of statement", next_token);
            }
        }
    }

    Ok(builder.build())
}

pub fn parse_create_stage(
    or_replace: bool,
    temporary: bool,
    parser: &mut Parser,
) -> Result<Statement, ParserError> {
    //[ IF NOT EXISTS ]
    let if_not_exists = parser.parse_keywords(&[Keyword::IF, Keyword::NOT, Keyword::EXISTS]);
    let name = parser.parse_object_name(false)?;
    let mut directory_table_params = Vec::new();
    let mut file_format = Vec::new();
    let mut copy_options = Vec::new();
    let mut comment = None;

    // [ internalStageParams | externalStageParams ]
    let stage_params = parse_stage_params(parser)?;

    // [ directoryTableParams ]
    if parser.parse_keyword(Keyword::DIRECTORY) {
        parser.expect_token(&Token::Eq)?;
        directory_table_params = parse_parentheses_options(parser)?;
    }

    // [ file_format]
    if parser.parse_keyword(Keyword::FILE_FORMAT) {
        parser.expect_token(&Token::Eq)?;
        file_format = parse_parentheses_options(parser)?;
    }

    // [ copy_options ]
    if parser.parse_keyword(Keyword::COPY_OPTIONS) {
        parser.expect_token(&Token::Eq)?;
        copy_options = parse_parentheses_options(parser)?;
    }

    // [ comment ]
    if parser.parse_keyword(Keyword::COMMENT) {
        parser.expect_token(&Token::Eq)?;
        comment = Some(match parser.next_token().token {
            Token::SingleQuotedString(word) => Ok(word),
            _ => parser.expected("a comment statement", parser.peek_token()),
        }?)
    }

    Ok(Statement::CreateStage {
        or_replace,
        temporary,
        if_not_exists,
        name,
        stage_params,
        directory_table_params: DataLoadingOptions {
            options: directory_table_params,
        },
        file_format: DataLoadingOptions {
            options: file_format,
        },
        copy_options: DataLoadingOptions {
            options: copy_options,
        },
        comment,
    })
}

pub fn parse_stage_name_identifier(parser: &mut Parser) -> Result<Ident, ParserError> {
    let mut ident = String::new();
    while let Some(next_token) = parser.next_token_no_skip() {
        match &next_token.token {
            Token::Whitespace(_) => break,
            Token::Period => {
                parser.prev_token();
                break;
            }
            Token::RParen => {
                parser.prev_token();
                break;
            }
            Token::AtSign => ident.push('@'),
            Token::Tilde => ident.push('~'),
            Token::Mod => ident.push('%'),
            Token::Div => ident.push('/'),
            Token::Word(w) => ident.push_str(&w.value),
            _ => return parser.expected("stage name identifier", parser.peek_token()),
        }
    }
    Ok(Ident::new(ident))
}

pub fn parse_snowflake_stage_name(parser: &mut Parser) -> Result<ObjectName, ParserError> {
    match parser.next_token().token {
        Token::AtSign => {
            parser.prev_token();
            let mut idents = vec![];
            loop {
                idents.push(parse_stage_name_identifier(parser)?);
                if !parser.consume_token(&Token::Period) {
                    break;
                }
            }
            Ok(ObjectName(idents))
        }
        _ => {
            parser.prev_token();
            Ok(parser.parse_object_name(false)?)
        }
    }
}

pub fn parse_copy_into(parser: &mut Parser) -> Result<Statement, ParserError> {
    let into: ObjectName = parse_snowflake_stage_name(parser)?;
    let mut files: Vec<String> = vec![];
    let mut from_transformations: Option<Vec<StageLoadSelectItem>> = None;
    let from_stage_alias;
    let from_stage: ObjectName;
    let stage_params: StageParamsObject;

    parser.expect_keyword(Keyword::FROM)?;
    // check if data load transformations are present
    match parser.next_token().token {
        Token::LParen => {
            // data load with transformations
            parser.expect_keyword(Keyword::SELECT)?;
            from_transformations = parse_select_items_for_data_load(parser)?;

            parser.expect_keyword(Keyword::FROM)?;
            from_stage = parse_snowflake_stage_name(parser)?;
            stage_params = parse_stage_params(parser)?;

            // as
            from_stage_alias = if parser.parse_keyword(Keyword::AS) {
                Some(match parser.next_token().token {
                    Token::Word(w) => Ok(Ident::new(w.value)),
                    _ => parser.expected("stage alias", parser.peek_token()),
                }?)
            } else {
                None
            };
            parser.expect_token(&Token::RParen)?;
        }
        _ => {
            parser.prev_token();
            from_stage = parse_snowflake_stage_name(parser)?;
            stage_params = parse_stage_params(parser)?;

            // as
            from_stage_alias = if parser.parse_keyword(Keyword::AS) {
                Some(match parser.next_token().token {
                    Token::Word(w) => Ok(Ident::new(w.value)),
                    _ => parser.expected("stage alias", parser.peek_token()),
                }?)
            } else {
                None
            };
        }
    };

    // [ files ]
    if parser.parse_keyword(Keyword::FILES) {
        parser.expect_token(&Token::Eq)?;
        parser.expect_token(&Token::LParen)?;
        let mut continue_loop = true;
        while continue_loop {
            continue_loop = false;
            let next_token = parser.next_token();
            match next_token.token {
                Token::SingleQuotedString(s) => files.push(s),
                _ => parser.expected("file token", next_token)?,
            };
            if parser.next_token().token.eq(&Token::Comma) {
                continue_loop = true;
            } else {
                parser.prev_token(); // not a comma, need to go back
            }
        }
        parser.expect_token(&Token::RParen)?;
    }

    // [ pattern ]
    let mut pattern = None;
    if parser.parse_keyword(Keyword::PATTERN) {
        parser.expect_token(&Token::Eq)?;
        let next_token = parser.next_token();
        pattern = Some(match next_token.token {
            Token::SingleQuotedString(s) => s,
            _ => parser.expected("pattern", next_token)?,
        });
    }

    // [ file_format]
    let mut file_format = Vec::new();
    if parser.parse_keyword(Keyword::FILE_FORMAT) {
        parser.expect_token(&Token::Eq)?;
        file_format = parse_parentheses_options(parser)?;
    }

    // [ copy_options ]
    let mut copy_options = Vec::new();
    if parser.parse_keyword(Keyword::COPY_OPTIONS) {
        parser.expect_token(&Token::Eq)?;
        copy_options = parse_parentheses_options(parser)?;
    }

    // [ VALIDATION_MODE ]
    let mut validation_mode = None;
    if parser.parse_keyword(Keyword::VALIDATION_MODE) {
        parser.expect_token(&Token::Eq)?;
        validation_mode = Some(parser.next_token().token.to_string());
    }

    Ok(Statement::CopyIntoSnowflake {
        into,
        from_stage,
        from_stage_alias,
        stage_params,
        from_transformations,
        files: if files.is_empty() { None } else { Some(files) },
        pattern,
        file_format: DataLoadingOptions {
            options: file_format,
        },
        copy_options: DataLoadingOptions {
            options: copy_options,
        },
        validation_mode,
    })
}

fn parse_select_items_for_data_load(
    parser: &mut Parser,
) -> Result<Option<Vec<StageLoadSelectItem>>, ParserError> {
    // [<alias>.]$<file_col_num>[.<element>] [ , [<alias>.]$<file_col_num>[.<element>] ... ]
    let mut select_items: Vec<StageLoadSelectItem> = vec![];
    loop {
        let mut alias: Option<Ident> = None;
        let mut file_col_num: i32 = 0;
        let mut element: Option<Ident> = None;
        let mut item_as: Option<Ident> = None;

        let next_token = parser.next_token();
        match next_token.token {
            Token::Placeholder(w) => {
                file_col_num = w.to_string().split_off(1).parse::<i32>().map_err(|e| {
                    ParserError::ParserError(format!("Could not parse '{w}' as i32: {e}"))
                })?;
                Ok(())
            }
            Token::Word(w) => {
                alias = Some(Ident::new(w.value));
                Ok(())
            }
            _ => parser.expected("alias or file_col_num", next_token),
        }?;

        if alias.is_some() {
            parser.expect_token(&Token::Period)?;
            // now we get col_num token
            let col_num_token = parser.next_token();
            match col_num_token.token {
                Token::Placeholder(w) => {
                    file_col_num = w.to_string().split_off(1).parse::<i32>().map_err(|e| {
                        ParserError::ParserError(format!("Could not parse '{w}' as i32: {e}"))
                    })?;
                    Ok(())
                }
                _ => parser.expected("file_col_num", col_num_token),
            }?;
        }

        // try extracting optional element
        match parser.next_token().token {
            Token::Colon => {
                // parse element
                element = Some(Ident::new(match parser.next_token().token {
                    Token::Word(w) => Ok(w.value),
                    _ => parser.expected("file_col_num", parser.peek_token()),
                }?));
            }
            _ => {
                // element not present move back
                parser.prev_token();
            }
        }

        // as
        if parser.parse_keyword(Keyword::AS) {
            item_as = Some(match parser.next_token().token {
                Token::Word(w) => Ok(Ident::new(w.value)),
                _ => parser.expected("column item alias", parser.peek_token()),
            }?);
        }

        select_items.push(StageLoadSelectItem {
            alias,
            file_col_num,
            element,
            item_as,
        });

        match parser.next_token().token {
            Token::Comma => {
                // continue
            }
            _ => {
                parser.prev_token(); // need to move back
                break;
            }
        }
    }
    Ok(Some(select_items))
}

fn parse_stage_params(parser: &mut Parser) -> Result<StageParamsObject, ParserError> {
    let (mut url, mut storage_integration, mut endpoint) = (None, None, None);
    let mut encryption: DataLoadingOptions = DataLoadingOptions { options: vec![] };
    let mut credentials: DataLoadingOptions = DataLoadingOptions { options: vec![] };

    // URL
    if parser.parse_keyword(Keyword::URL) {
        parser.expect_token(&Token::Eq)?;
        url = Some(match parser.next_token().token {
            Token::SingleQuotedString(word) => Ok(word),
            _ => parser.expected("a URL statement", parser.peek_token()),
        }?)
    }

    // STORAGE INTEGRATION
    if parser.parse_keyword(Keyword::STORAGE_INTEGRATION) {
        parser.expect_token(&Token::Eq)?;
        storage_integration = Some(parser.next_token().token.to_string());
    }

    // ENDPOINT
    if parser.parse_keyword(Keyword::ENDPOINT) {
        parser.expect_token(&Token::Eq)?;
        endpoint = Some(match parser.next_token().token {
            Token::SingleQuotedString(word) => Ok(word),
            _ => parser.expected("an endpoint statement", parser.peek_token()),
        }?)
    }

    // CREDENTIALS
    if parser.parse_keyword(Keyword::CREDENTIALS) {
        parser.expect_token(&Token::Eq)?;
        credentials = DataLoadingOptions {
            options: parse_parentheses_options(parser)?,
        };
    }

    // ENCRYPTION
    if parser.parse_keyword(Keyword::ENCRYPTION) {
        parser.expect_token(&Token::Eq)?;
        encryption = DataLoadingOptions {
            options: parse_parentheses_options(parser)?,
        };
    }

    Ok(StageParamsObject {
        url,
        encryption,
        endpoint,
        storage_integration,
        credentials,
    })
}

/// Parses options provided within parentheses like:
/// ( ENABLE = { TRUE | FALSE }
///      [ AUTO_REFRESH = { TRUE | FALSE } ]
///      [ REFRESH_ON_CREATE =  { TRUE | FALSE } ]
///      [ NOTIFICATION_INTEGRATION = '<notification_integration_name>' ] )
///
fn parse_parentheses_options(parser: &mut Parser) -> Result<Vec<DataLoadingOption>, ParserError> {
    let mut options: Vec<DataLoadingOption> = Vec::new();

    parser.expect_token(&Token::LParen)?;
    loop {
        match parser.next_token().token {
            Token::RParen => break,
            Token::Word(key) => {
                parser.expect_token(&Token::Eq)?;
                if parser.parse_keyword(Keyword::TRUE) {
                    options.push(DataLoadingOption {
                        option_name: key.value,
                        option_type: DataLoadingOptionType::BOOLEAN,
                        value: "TRUE".to_string(),
                    });
                    Ok(())
                } else if parser.parse_keyword(Keyword::FALSE) {
                    options.push(DataLoadingOption {
                        option_name: key.value,
                        option_type: DataLoadingOptionType::BOOLEAN,
                        value: "FALSE".to_string(),
                    });
                    Ok(())
                } else {
                    match parser.next_token().token {
                        Token::SingleQuotedString(value) => {
                            options.push(DataLoadingOption {
                                option_name: key.value,
                                option_type: DataLoadingOptionType::STRING,
                                value,
                            });
                            Ok(())
                        }
                        Token::Word(word) => {
                            options.push(DataLoadingOption {
                                option_name: key.value,
                                option_type: DataLoadingOptionType::ENUM,
                                value: word.value,
                            });
                            Ok(())
                        }
                        _ => parser.expected("expected option value", parser.peek_token()),
                    }
                }
            }
            _ => parser.expected("another option or ')'", parser.peek_token()),
        }?;
    }
    Ok(options)
}

/// Parsing a property of identity or autoincrement column option
/// Syntax:
/// ```sql
/// [ (seed , increment) | START num INCREMENT num ] [ ORDER | NOORDER ]
/// ```
/// [Snowflake]: https://docs.snowflake.com/en/sql-reference/sql/create-table
fn parse_identity_property(parser: &mut Parser) -> Result<IdentityProperty, ParserError> {
    let parameters = if parser.consume_token(&Token::LParen) {
        let seed = parser.parse_number()?;
        parser.expect_token(&Token::Comma)?;
        let increment = parser.parse_number()?;
        parser.expect_token(&Token::RParen)?;

        Some(IdentityPropertyFormatKind::FunctionCall(
            IdentityParameters { seed, increment },
        ))
    } else if parser.parse_keyword(Keyword::START) {
        let seed = parser.parse_number()?;
        parser.expect_keyword(Keyword::INCREMENT)?;
        let increment = parser.parse_number()?;

        Some(IdentityPropertyFormatKind::StartAndIncrement(
            IdentityParameters { seed, increment },
        ))
    } else {
        None
    };
    let order = match parser.parse_one_of_keywords(&[Keyword::ORDER, Keyword::NOORDER]) {
        Some(Keyword::ORDER) => Some(IdentityPropertyOrder::Order),
        Some(Keyword::NOORDER) => Some(IdentityPropertyOrder::NoOrder),
        _ => None,
    };
    Ok(IdentityProperty { parameters, order })
}

/// Parsing a policy property of column option
/// Syntax:
/// ```sql
/// <policy_name> [ USING ( <col_name> , <cond_col1> , ... )
/// ```
/// [Snowflake]: https://docs.snowflake.com/en/sql-reference/sql/create-table
fn parse_column_policy_property(
    parser: &mut Parser,
    with: bool,
) -> Result<ColumnPolicyProperty, ParserError> {
    let policy_name = parser.parse_identifier(false)?;
    let using_columns = if parser.parse_keyword(Keyword::USING) {
        parser.expect_token(&Token::LParen)?;
        let columns = parser.parse_comma_separated(|p| p.parse_identifier(false))?;
        parser.expect_token(&Token::RParen)?;
        Some(columns)
    } else {
        None
    };

    Ok(ColumnPolicyProperty {
        with,
        policy_name,
        using_columns,
    })
}

/// Parsing tags list of column
/// Syntax:
/// ```sql
/// ( <tag_name> = '<tag_value>' [ , <tag_name> = '<tag_value>' , ... ] )
/// ```
/// [Snowflake]: https://docs.snowflake.com/en/sql-reference/sql/create-table
fn parse_column_tags(parser: &mut Parser, with: bool) -> Result<TagsColumnOption, ParserError> {
    parser.expect_token(&Token::LParen)?;
    let tags = parser.parse_comma_separated(Parser::parse_tag)?;
    parser.expect_token(&Token::RParen)?;

    Ok(TagsColumnOption { with, tags })
}
