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
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec::Vec,
};

use core::fmt::{self, Display};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "visitor")]
use sqlparser_derive::{Visit, VisitMut};

pub use super::ddl::{ColumnDef, TableConstraint};

use super::{
    display_comma_separated, display_separated, ClusteredBy, CommentDef, Expr, FileFormat,
    FromTable, HiveDistributionStyle, HiveFormat, HiveIOFormat, HiveRowFormat, Ident,
    InsertAliases, MysqlInsertPriority, ObjectName, OnCommit, OnInsert, OneOrManyWithParens,
    OrderByExpr, Query, RowAccessPolicy, SelectItem, SqlOption, SqliteOnConflict, TableEngine,
    TableWithJoins, Tag, WrappedCollection,
};

/// CREATE INDEX statement.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct CreateIndex {
    /// index name
    pub name: Option<ObjectName>,
    #[cfg_attr(feature = "visitor", visit(with = "visit_relation"))]
    pub table_name: ObjectName,
    pub using: Option<Ident>,
    pub columns: Vec<OrderByExpr>,
    pub unique: bool,
    pub concurrently: bool,
    pub if_not_exists: bool,
    pub include: Vec<Ident>,
    pub nulls_distinct: Option<bool>,
    /// WITH clause: <https://www.postgresql.org/docs/current/sql-createindex.html>
    pub with: Vec<Expr>,
    pub predicate: Option<Expr>,
}

impl Display for CreateIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "CREATE {unique}INDEX {concurrently}{if_not_exists}",
            unique = if self.unique { "UNIQUE " } else { "" },
            concurrently = if self.concurrently {
                "CONCURRENTLY "
            } else {
                ""
            },
            if_not_exists = if self.if_not_exists {
                "IF NOT EXISTS "
            } else {
                ""
            },
        )?;
        if let Some(value) = &self.name {
            write!(f, "{value} ")?;
        }
        write!(f, "ON {}", self.table_name)?;
        if let Some(value) = &self.using {
            write!(f, " USING {value} ")?;
        }
        write!(f, "({})", display_separated(&self.columns, ","))?;
        if !self.include.is_empty() {
            write!(f, " INCLUDE ({})", display_separated(&self.include, ","))?;
        }
        if let Some(value) = self.nulls_distinct {
            if value {
                write!(f, " NULLS DISTINCT")?;
            } else {
                write!(f, " NULLS NOT DISTINCT")?;
            }
        }
        if !self.with.is_empty() {
            write!(f, " WITH ({})", display_comma_separated(&self.with))?;
        }
        if let Some(predicate) = &self.predicate {
            write!(f, " WHERE {predicate}")?;
        }
        Ok(())
    }
}

/// CREATE TABLE statement.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct CreateTable {
    pub or_replace: bool,
    pub temporary: bool,
    pub external: bool,
    pub global: Option<bool>,
    pub if_not_exists: bool,
    pub transient: bool,
    pub volatile: bool,
    /// Table name
    #[cfg_attr(feature = "visitor", visit(with = "visit_relation"))]
    pub name: ObjectName,
    /// Optional schema
    pub columns: Vec<ColumnDef>,
    pub constraints: Vec<TableConstraint>,
    pub hive_distribution: HiveDistributionStyle,
    pub hive_formats: Option<HiveFormat>,
    pub table_properties: Vec<SqlOption>,
    pub with_options: Vec<SqlOption>,
    pub file_format: Option<FileFormat>,
    pub location: Option<String>,
    pub query: Option<Box<Query>>,
    pub without_rowid: bool,
    pub like: Option<ObjectName>,
    pub clone: Option<ObjectName>,
    pub engine: Option<TableEngine>,
    pub comment: Option<CommentDef>,
    pub auto_increment_offset: Option<u32>,
    pub default_charset: Option<String>,
    pub collation: Option<String>,
    pub on_commit: Option<OnCommit>,
    /// ClickHouse "ON CLUSTER" clause:
    /// <https://clickhouse.com/docs/en/sql-reference/distributed-ddl/>
    pub on_cluster: Option<Ident>,
    /// ClickHouse "PRIMARY KEY " clause.
    /// <https://clickhouse.com/docs/en/sql-reference/statements/create/table/>
    pub primary_key: Option<Box<Expr>>,
    /// ClickHouse "ORDER BY " clause. Note that omitted ORDER BY is different
    /// than empty (represented as ()), the latter meaning "no sorting".
    /// <https://clickhouse.com/docs/en/sql-reference/statements/create/table/>
    pub order_by: Option<OneOrManyWithParens<Expr>>,
    /// BigQuery: A partition expression for the table.
    /// <https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#partition_expression>
    pub partition_by: Option<Box<Expr>>,
    /// BigQuery: Table clustering column list.
    /// <https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#table_option_list>
    pub cluster_by: Option<WrappedCollection<Vec<Ident>>>,
    /// Hive: Table clustering column list.
    /// <https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-CreateTable>
    pub clustered_by: Option<ClusteredBy>,
    /// BigQuery: Table options list.
    /// <https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#table_option_list>
    pub options: Option<Vec<SqlOption>>,
    /// SQLite "STRICT" clause.
    /// if the "STRICT" table-option keyword is added to the end, after the closing ")",
    /// then strict typing rules apply to that table.
    pub strict: bool,
    /// Snowflake "COPY GRANTS" clause
    /// <https://docs.snowflake.com/en/sql-reference/sql/create-table>
    pub copy_grants: bool,
    /// Snowflake "ENABLE_SCHEMA_EVOLUTION" clause
    /// <https://docs.snowflake.com/en/sql-reference/sql/create-table>
    pub enable_schema_evolution: Option<bool>,
    /// Snowflake "CHANGE_TRACKING" clause
    /// <https://docs.snowflake.com/en/sql-reference/sql/create-table>
    pub change_tracking: Option<bool>,
    /// Snowflake "DATA_RETENTION_TIME_IN_DAYS" clause
    /// <https://docs.snowflake.com/en/sql-reference/sql/create-table>
    pub data_retention_time_in_days: Option<u64>,
    /// Snowflake "MAX_DATA_EXTENSION_TIME_IN_DAYS" clause
    /// <https://docs.snowflake.com/en/sql-reference/sql/create-table>
    pub max_data_extension_time_in_days: Option<u64>,
    /// Snowflake "DEFAULT_DDL_COLLATION" clause
    /// <https://docs.snowflake.com/en/sql-reference/sql/create-table>
    pub default_ddl_collation: Option<String>,
    /// Snowflake "WITH AGGREGATION POLICY" clause
    /// <https://docs.snowflake.com/en/sql-reference/sql/create-table>
    pub with_aggregation_policy: Option<ObjectName>,
    /// Snowflake "WITH ROW ACCESS POLICY" clause
    /// <https://docs.snowflake.com/en/sql-reference/sql/create-table>
    pub with_row_access_policy: Option<RowAccessPolicy>,
    /// Snowflake "WITH TAG" clause
    /// <https://docs.snowflake.com/en/sql-reference/sql/create-table>
    pub with_tags: Option<Vec<Tag>>,
}

impl Display for CreateTable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // We want to allow the following options
        // Empty column list, allowed by PostgreSQL:
        //   `CREATE TABLE t ()`
        // No columns provided for CREATE TABLE AS:
        //   `CREATE TABLE t AS SELECT a from t2`
        // Columns provided for CREATE TABLE AS:
        //   `CREATE TABLE t (a INT) AS SELECT a from t2`
        write!(
            f,
            "CREATE {or_replace}{external}{global}{temporary}{transient}{volatile}TABLE {if_not_exists}{name}",
            or_replace = if self.or_replace { "OR REPLACE " } else { "" },
            external = if self.external { "EXTERNAL " } else { "" },
            global = self.global
                .map(|global| {
                    if global {
                        "GLOBAL "
                    } else {
                        "LOCAL "
                    }
                })
                .unwrap_or(""),
            if_not_exists = if self.if_not_exists { "IF NOT EXISTS " } else { "" },
            temporary = if self.temporary { "TEMPORARY " } else { "" },
            transient = if self.transient { "TRANSIENT " } else { "" },
            volatile = if self.volatile { "VOLATILE " } else { "" },
            name = self.name,
        )?;
        if let Some(on_cluster) = &self.on_cluster {
            write!(f, " ON CLUSTER {}", on_cluster)?;
        }
        if !self.columns.is_empty() || !self.constraints.is_empty() {
            write!(f, " ({}", display_comma_separated(&self.columns))?;
            if !self.columns.is_empty() && !self.constraints.is_empty() {
                write!(f, ", ")?;
            }
            write!(f, "{})", display_comma_separated(&self.constraints))?;
        } else if self.query.is_none() && self.like.is_none() && self.clone.is_none() {
            // PostgreSQL allows `CREATE TABLE t ();`, but requires empty parens
            write!(f, " ()")?;
        }

        // Hive table comment should be after column definitions, please refer to:
        // [Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-CreateTable)
        if let Some(CommentDef::AfterColumnDefsWithoutEq(comment)) = &self.comment {
            write!(f, " COMMENT '{comment}'")?;
        }

        // Only for SQLite
        if self.without_rowid {
            write!(f, " WITHOUT ROWID")?;
        }

        // Only for Hive
        if let Some(l) = &self.like {
            write!(f, " LIKE {l}")?;
        }

        if let Some(c) = &self.clone {
            write!(f, " CLONE {c}")?;
        }

        match &self.hive_distribution {
            HiveDistributionStyle::PARTITIONED { columns } => {
                write!(f, " PARTITIONED BY ({})", display_comma_separated(columns))?;
            }
            HiveDistributionStyle::SKEWED {
                columns,
                on,
                stored_as_directories,
            } => {
                write!(
                    f,
                    " SKEWED BY ({})) ON ({})",
                    display_comma_separated(columns),
                    display_comma_separated(on)
                )?;
                if *stored_as_directories {
                    write!(f, " STORED AS DIRECTORIES")?;
                }
            }
            _ => (),
        }

        if let Some(clustered_by) = &self.clustered_by {
            write!(f, " {clustered_by}")?;
        }

        if let Some(HiveFormat {
            row_format,
            serde_properties,
            storage,
            location,
        }) = &self.hive_formats
        {
            match row_format {
                Some(HiveRowFormat::SERDE { class }) => write!(f, " ROW FORMAT SERDE '{class}'")?,
                Some(HiveRowFormat::DELIMITED { delimiters }) => {
                    write!(f, " ROW FORMAT DELIMITED")?;
                    if !delimiters.is_empty() {
                        write!(f, " {}", display_separated(delimiters, " "))?;
                    }
                }
                None => (),
            }
            match storage {
                Some(HiveIOFormat::IOF {
                    input_format,
                    output_format,
                }) => write!(
                    f,
                    " STORED AS INPUTFORMAT {input_format} OUTPUTFORMAT {output_format}"
                )?,
                Some(HiveIOFormat::FileFormat { format }) if !self.external => {
                    write!(f, " STORED AS {format}")?
                }
                _ => (),
            }
            if let Some(serde_properties) = serde_properties.as_ref() {
                write!(
                    f,
                    " WITH SERDEPROPERTIES ({})",
                    display_comma_separated(serde_properties)
                )?;
            }
            if !self.external {
                if let Some(loc) = location {
                    write!(f, " LOCATION '{loc}'")?;
                }
            }
        }
        if self.external {
            if let Some(file_format) = self.file_format {
                write!(f, " STORED AS {file_format}")?;
            }
            write!(f, " LOCATION '{}'", self.location.as_ref().unwrap())?;
        }
        if !self.table_properties.is_empty() {
            write!(
                f,
                " TBLPROPERTIES ({})",
                display_comma_separated(&self.table_properties)
            )?;
        }
        if !self.with_options.is_empty() {
            write!(f, " WITH ({})", display_comma_separated(&self.with_options))?;
        }
        if let Some(engine) = &self.engine {
            write!(f, " ENGINE={engine}")?;
        }
        if let Some(comment_def) = &self.comment {
            match comment_def {
                CommentDef::WithEq(comment) => {
                    write!(f, " COMMENT = '{comment}'")?;
                }
                CommentDef::WithoutEq(comment) => {
                    write!(f, " COMMENT '{comment}'")?;
                }
                // For CommentDef::AfterColumnDefsWithoutEq will be displayed after column definition
                CommentDef::AfterColumnDefsWithoutEq(_) => (),
            }
        }

        if let Some(auto_increment_offset) = self.auto_increment_offset {
            write!(f, " AUTO_INCREMENT {auto_increment_offset}")?;
        }
        if let Some(primary_key) = &self.primary_key {
            write!(f, " PRIMARY KEY {}", primary_key)?;
        }
        if let Some(order_by) = &self.order_by {
            write!(f, " ORDER BY {}", order_by)?;
        }
        if let Some(partition_by) = self.partition_by.as_ref() {
            write!(f, " PARTITION BY {partition_by}")?;
        }
        if let Some(cluster_by) = self.cluster_by.as_ref() {
            write!(f, " CLUSTER BY {cluster_by}")?;
        }

        if let Some(options) = self.options.as_ref() {
            write!(
                f,
                " OPTIONS({})",
                display_comma_separated(options.as_slice())
            )?;
        }

        if self.copy_grants {
            write!(f, " COPY GRANTS")?;
        }

        if let Some(is_enabled) = self.enable_schema_evolution {
            write!(
                f,
                " ENABLE_SCHEMA_EVOLUTION={}",
                if is_enabled { "TRUE" } else { "FALSE" }
            )?;
        }

        if let Some(is_enabled) = self.change_tracking {
            write!(
                f,
                " CHANGE_TRACKING={}",
                if is_enabled { "TRUE" } else { "FALSE" }
            )?;
        }

        if let Some(data_retention_time_in_days) = self.data_retention_time_in_days {
            write!(
                f,
                " DATA_RETENTION_TIME_IN_DAYS={data_retention_time_in_days}",
            )?;
        }

        if let Some(max_data_extension_time_in_days) = self.max_data_extension_time_in_days {
            write!(
                f,
                " MAX_DATA_EXTENSION_TIME_IN_DAYS={max_data_extension_time_in_days}",
            )?;
        }

        if let Some(default_ddl_collation) = &self.default_ddl_collation {
            write!(f, " DEFAULT_DDL_COLLATION='{default_ddl_collation}'",)?;
        }

        if let Some(with_aggregation_policy) = &self.with_aggregation_policy {
            write!(f, " WITH AGGREGATION POLICY {with_aggregation_policy}",)?;
        }

        if let Some(row_access_policy) = &self.with_row_access_policy {
            write!(f, " {row_access_policy}",)?;
        }

        if let Some(tag) = &self.with_tags {
            write!(f, " WITH TAG ({})", display_comma_separated(tag.as_slice()))?;
        }

        if let Some(default_charset) = &self.default_charset {
            write!(f, " DEFAULT CHARSET={default_charset}")?;
        }
        if let Some(collation) = &self.collation {
            write!(f, " COLLATE={collation}")?;
        }

        if self.on_commit.is_some() {
            let on_commit = match self.on_commit {
                Some(OnCommit::DeleteRows) => "ON COMMIT DELETE ROWS",
                Some(OnCommit::PreserveRows) => "ON COMMIT PRESERVE ROWS",
                Some(OnCommit::Drop) => "ON COMMIT DROP",
                None => "",
            };
            write!(f, " {on_commit}")?;
        }
        if self.strict {
            write!(f, " STRICT")?;
        }
        if let Some(query) = &self.query {
            write!(f, " AS {query}")?;
        }
        Ok(())
    }
}

/// INSERT statement.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct Insert {
    /// Only for Sqlite
    pub or: Option<SqliteOnConflict>,
    /// Only for mysql
    pub ignore: bool,
    /// INTO - optional keyword
    pub into: bool,
    /// TABLE
    #[cfg_attr(feature = "visitor", visit(with = "visit_relation"))]
    pub table_name: ObjectName,
    /// table_name as foo (for PostgreSQL)
    pub table_alias: Option<Ident>,
    /// COLUMNS
    pub columns: Vec<Ident>,
    /// Overwrite (Hive)
    pub overwrite: bool,
    /// A SQL query that specifies what to insert
    pub source: Option<Box<Query>>,
    /// partitioned insert (Hive)
    pub partitioned: Option<Vec<Expr>>,
    /// Columns defined after PARTITION
    pub after_columns: Vec<Ident>,
    /// whether the insert has the table keyword (Hive)
    pub table: bool,
    pub on: Option<OnInsert>,
    /// RETURNING
    pub returning: Option<Vec<SelectItem>>,
    /// Only for mysql
    pub replace_into: bool,
    /// Only for mysql
    pub priority: Option<MysqlInsertPriority>,
    /// Only for mysql
    pub insert_alias: Option<InsertAliases>,
}

impl Display for Insert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let table_name = if let Some(alias) = &self.table_alias {
            format!("{0} AS {alias}", self.table_name)
        } else {
            self.table_name.to_string()
        };

        if let Some(on_conflict) = self.or {
            write!(f, "INSERT {on_conflict} INTO {table_name} ")?;
        } else {
            write!(
                f,
                "{start}",
                start = if self.replace_into {
                    "REPLACE"
                } else {
                    "INSERT"
                },
            )?;
            if let Some(priority) = self.priority {
                write!(f, " {priority}",)?;
            }

            write!(
                f,
                "{ignore}{over}{int}{tbl} {table_name} ",
                table_name = table_name,
                ignore = if self.ignore { " IGNORE" } else { "" },
                over = if self.overwrite { " OVERWRITE" } else { "" },
                int = if self.into { " INTO" } else { "" },
                tbl = if self.table { " TABLE" } else { "" },
            )?;
        }
        if !self.columns.is_empty() {
            write!(f, "({}) ", display_comma_separated(&self.columns))?;
        }
        if let Some(ref parts) = self.partitioned {
            if !parts.is_empty() {
                write!(f, "PARTITION ({}) ", display_comma_separated(parts))?;
            }
        }
        if !self.after_columns.is_empty() {
            write!(f, "({}) ", display_comma_separated(&self.after_columns))?;
        }

        if let Some(source) = &self.source {
            write!(f, "{source}")?;
        }

        if self.source.is_none() && self.columns.is_empty() {
            write!(f, "DEFAULT VALUES")?;
        }

        if let Some(insert_alias) = &self.insert_alias {
            write!(f, " AS {0}", insert_alias.row_alias)?;

            if let Some(col_aliases) = &insert_alias.col_aliases {
                if !col_aliases.is_empty() {
                    write!(f, " ({})", display_comma_separated(col_aliases))?;
                }
            }
        }

        if let Some(on) = &self.on {
            write!(f, "{on}")?;
        }

        if let Some(returning) = &self.returning {
            write!(f, " RETURNING {}", display_comma_separated(returning))?;
        }
        Ok(())
    }
}

/// DELETE statement.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct Delete {
    /// Multi tables delete are supported in mysql
    pub tables: Vec<ObjectName>,
    /// FROM
    pub from: FromTable,
    /// USING (Snowflake, Postgres, MySQL)
    pub using: Option<Vec<TableWithJoins>>,
    /// WHERE
    pub selection: Option<Expr>,
    /// RETURNING
    pub returning: Option<Vec<SelectItem>>,
    /// ORDER BY (MySQL)
    pub order_by: Vec<OrderByExpr>,
    /// LIMIT (MySQL)
    pub limit: Option<Expr>,
}

impl Display for Delete {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DELETE ")?;
        if !self.tables.is_empty() {
            write!(f, "{} ", display_comma_separated(&self.tables))?;
        }
        match &self.from {
            FromTable::WithFromKeyword(from) => {
                write!(f, "FROM {}", display_comma_separated(from))?;
            }
            FromTable::WithoutKeyword(from) => {
                write!(f, "{}", display_comma_separated(from))?;
            }
        }
        if let Some(using) = &self.using {
            write!(f, " USING {}", display_comma_separated(using))?;
        }
        if let Some(selection) = &self.selection {
            write!(f, " WHERE {selection}")?;
        }
        if let Some(returning) = &self.returning {
            write!(f, " RETURNING {}", display_comma_separated(returning))?;
        }
        if !self.order_by.is_empty() {
            write!(f, " ORDER BY {}", display_comma_separated(&self.order_by))?;
        }
        if let Some(limit) = &self.limit {
            write!(f, " LIMIT {limit}")?;
        }
        Ok(())
    }
}
