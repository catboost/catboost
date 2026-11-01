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

//! AST types specific to CREATE/ALTER variants of [`Statement`](crate::ast::Statement)
//! (commonly referred to as Data Definition Language, or DDL)

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};
use core::fmt::{self, Write};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "visitor")]
use sqlparser_derive::{Visit, VisitMut};

use crate::ast::value::escape_single_quote_string;
use crate::ast::{
    display_comma_separated, display_separated, CreateFunctionBody, CreateFunctionUsing, DataType,
    Expr, FunctionBehavior, FunctionCalledOnNull, FunctionDeterminismSpecifier, FunctionParallel,
    Ident, MySQLColumnPosition, ObjectName, OperateFunctionArg, OrderByExpr, ProjectionSelect,
    SequenceOptions, SqlOption, Tag, Value,
};
use crate::keywords::Keyword;
use crate::tokenizer::Token;

/// An `ALTER TABLE` (`Statement::AlterTable`) operation
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum AlterTableOperation {
    /// `ADD <table_constraint>`
    AddConstraint(TableConstraint),
    /// `ADD [COLUMN] [IF NOT EXISTS] <column_def>`
    AddColumn {
        /// `[COLUMN]`.
        column_keyword: bool,
        /// `[IF NOT EXISTS]`
        if_not_exists: bool,
        /// <column_def>.
        column_def: ColumnDef,
        /// MySQL `ALTER TABLE` only  [FIRST | AFTER column_name]
        column_position: Option<MySQLColumnPosition>,
    },
    /// `ADD PROJECTION [IF NOT EXISTS] name ( SELECT <COLUMN LIST EXPR> [GROUP BY] [ORDER BY])`
    ///
    /// Note: this is a ClickHouse-specific operation.
    /// Please refer to [ClickHouse](https://clickhouse.com/docs/en/sql-reference/statements/alter/projection#add-projection)
    AddProjection {
        if_not_exists: bool,
        name: Ident,
        select: ProjectionSelect,
    },

    /// `DROP PROJECTION [IF EXISTS] name`
    ///
    /// Note: this is a ClickHouse-specific operation.
    /// Please refer to [ClickHouse](https://clickhouse.com/docs/en/sql-reference/statements/alter/projection#drop-projection)
    DropProjection {
        if_exists: bool,
        name: Ident,
    },

    /// `MATERIALIZE PROJECTION [IF EXISTS] name [IN PARTITION partition_name]`
    ///
    ///  Note: this is a ClickHouse-specific operation.
    /// Please refer to [ClickHouse](https://clickhouse.com/docs/en/sql-reference/statements/alter/projection#materialize-projection)
    MaterializeProjection {
        if_exists: bool,
        name: Ident,
        partition: Option<Ident>,
    },

    /// `CLEAR PROJECTION [IF EXISTS] name [IN PARTITION partition_name]`
    ///
    /// Note: this is a ClickHouse-specific operation.
    /// Please refer to [ClickHouse](https://clickhouse.com/docs/en/sql-reference/statements/alter/projection#clear-projection)
    ClearProjection {
        if_exists: bool,
        name: Ident,
        partition: Option<Ident>,
    },

    /// `DISABLE ROW LEVEL SECURITY`
    ///
    /// Note: this is a PostgreSQL-specific operation.
    DisableRowLevelSecurity,
    /// `DISABLE RULE rewrite_rule_name`
    ///
    /// Note: this is a PostgreSQL-specific operation.
    DisableRule {
        name: Ident,
    },
    /// `DISABLE TRIGGER [ trigger_name | ALL | USER ]`
    ///
    /// Note: this is a PostgreSQL-specific operation.
    DisableTrigger {
        name: Ident,
    },
    /// `DROP CONSTRAINT [ IF EXISTS ] <name>`
    DropConstraint {
        if_exists: bool,
        name: Ident,
        cascade: bool,
    },
    /// `DROP [ COLUMN ] [ IF EXISTS ] <column_name> [ CASCADE ]`
    DropColumn {
        column_name: Ident,
        if_exists: bool,
        cascade: bool,
    },
    /// `ATTACH PART|PARTITION <partition_expr>`
    /// Note: this is a ClickHouse-specific operation, please refer to
    /// [ClickHouse](https://clickhouse.com/docs/en/sql-reference/statements/alter/pakrtition#attach-partitionpart)
    AttachPartition {
        // PART is not a short form of PARTITION, it's a separate keyword
        // which represents a physical file on disk and partition is a logical entity.
        partition: Partition,
    },
    /// `DETACH PART|PARTITION <partition_expr>`
    /// Note: this is a ClickHouse-specific operation, please refer to
    /// [ClickHouse](https://clickhouse.com/docs/en/sql-reference/statements/alter/partition#detach-partitionpart)
    DetachPartition {
        // See `AttachPartition` for more details
        partition: Partition,
    },
    /// `FREEZE PARTITION <partition_expr>`
    /// Note: this is a ClickHouse-specific operation, please refer to
    /// [ClickHouse](https://clickhouse.com/docs/en/sql-reference/statements/alter/partition#freeze-partition)
    FreezePartition {
        partition: Partition,
        with_name: Option<Ident>,
    },
    /// `UNFREEZE PARTITION <partition_expr>`
    /// Note: this is a ClickHouse-specific operation, please refer to
    /// [ClickHouse](https://clickhouse.com/docs/en/sql-reference/statements/alter/partition#unfreeze-partition)
    UnfreezePartition {
        partition: Partition,
        with_name: Option<Ident>,
    },
    /// `DROP PRIMARY KEY`
    ///
    /// Note: this is a MySQL-specific operation.
    DropPrimaryKey,
    /// `ENABLE ALWAYS RULE rewrite_rule_name`
    ///
    /// Note: this is a PostgreSQL-specific operation.
    EnableAlwaysRule {
        name: Ident,
    },
    /// `ENABLE ALWAYS TRIGGER trigger_name`
    ///
    /// Note: this is a PostgreSQL-specific operation.
    EnableAlwaysTrigger {
        name: Ident,
    },
    /// `ENABLE REPLICA RULE rewrite_rule_name`
    ///
    /// Note: this is a PostgreSQL-specific operation.
    EnableReplicaRule {
        name: Ident,
    },
    /// `ENABLE REPLICA TRIGGER trigger_name`
    ///
    /// Note: this is a PostgreSQL-specific operation.
    EnableReplicaTrigger {
        name: Ident,
    },
    /// `ENABLE ROW LEVEL SECURITY`
    ///
    /// Note: this is a PostgreSQL-specific operation.
    EnableRowLevelSecurity,
    /// `ENABLE RULE rewrite_rule_name`
    ///
    /// Note: this is a PostgreSQL-specific operation.
    EnableRule {
        name: Ident,
    },
    /// `ENABLE TRIGGER [ trigger_name | ALL | USER ]`
    ///
    /// Note: this is a PostgreSQL-specific operation.
    EnableTrigger {
        name: Ident,
    },
    /// `RENAME TO PARTITION (partition=val)`
    RenamePartitions {
        old_partitions: Vec<Expr>,
        new_partitions: Vec<Expr>,
    },
    /// Add Partitions
    AddPartitions {
        if_not_exists: bool,
        new_partitions: Vec<Partition>,
    },
    DropPartitions {
        partitions: Vec<Expr>,
        if_exists: bool,
    },
    /// `RENAME [ COLUMN ] <old_column_name> TO <new_column_name>`
    RenameColumn {
        old_column_name: Ident,
        new_column_name: Ident,
    },
    /// `RENAME TO <table_name>`
    RenameTable {
        table_name: ObjectName,
    },
    // CHANGE [ COLUMN ] <old_name> <new_name> <data_type> [ <options> ]
    ChangeColumn {
        old_name: Ident,
        new_name: Ident,
        data_type: DataType,
        options: Vec<ColumnOption>,
        /// MySQL `ALTER TABLE` only  [FIRST | AFTER column_name]
        column_position: Option<MySQLColumnPosition>,
    },
    // CHANGE [ COLUMN ] <col_name> <data_type> [ <options> ]
    ModifyColumn {
        col_name: Ident,
        data_type: DataType,
        options: Vec<ColumnOption>,
        /// MySQL `ALTER TABLE` only  [FIRST | AFTER column_name]
        column_position: Option<MySQLColumnPosition>,
    },
    /// `RENAME CONSTRAINT <old_constraint_name> TO <new_constraint_name>`
    ///
    /// Note: this is a PostgreSQL-specific operation.
    RenameConstraint {
        old_name: Ident,
        new_name: Ident,
    },
    /// `ALTER [ COLUMN ]`
    AlterColumn {
        column_name: Ident,
        op: AlterColumnOperation,
    },
    /// 'SWAP WITH <table_name>'
    ///
    /// Note: this is Snowflake specific <https://docs.snowflake.com/en/sql-reference/sql/alter-table>
    SwapWith {
        table_name: ObjectName,
    },
    /// 'SET TBLPROPERTIES ( { property_key [ = ] property_val } [, ...] )'
    SetTblProperties {
        table_properties: Vec<SqlOption>,
    },
    /// `OWNER TO { <new_owner> | CURRENT_ROLE | CURRENT_USER | SESSION_USER }`
    ///
    /// Note: this is PostgreSQL-specific <https://www.postgresql.org/docs/current/sql-altertable.html>
    OwnerTo {
        new_owner: Owner,
    },
    /// Snowflake table clustering options
    /// <https://docs.snowflake.com/en/sql-reference/sql/alter-table#clustering-actions-clusteringaction>
    ClusterBy {
        exprs: Vec<Expr>,
    },
    DropClusteringKey,
    SuspendRecluster,
    ResumeRecluster,
}

/// An `ALTER Policy` (`Statement::AlterPolicy`) operation
///
/// [PostgreSQL Documentation](https://www.postgresql.org/docs/current/sql-altertable.html)
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum AlterPolicyOperation {
    Rename {
        new_name: Ident,
    },
    Apply {
        to: Option<Vec<Owner>>,
        using: Option<Expr>,
        with_check: Option<Expr>,
    },
}

impl fmt::Display for AlterPolicyOperation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AlterPolicyOperation::Rename { new_name } => {
                write!(f, " RENAME TO {new_name}")
            }
            AlterPolicyOperation::Apply {
                to,
                using,
                with_check,
            } => {
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
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum Owner {
    Ident(Ident),
    CurrentRole,
    CurrentUser,
    SessionUser,
}

impl fmt::Display for Owner {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Owner::Ident(ident) => write!(f, "{}", ident),
            Owner::CurrentRole => write!(f, "CURRENT_ROLE"),
            Owner::CurrentUser => write!(f, "CURRENT_USER"),
            Owner::SessionUser => write!(f, "SESSION_USER"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum AlterIndexOperation {
    RenameIndex { index_name: ObjectName },
}

impl fmt::Display for AlterTableOperation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AlterTableOperation::AddPartitions {
                if_not_exists,
                new_partitions,
            } => write!(
                f,
                "ADD{ine} {}",
                display_separated(new_partitions, " "),
                ine = if *if_not_exists { " IF NOT EXISTS" } else { "" }
            ),
            AlterTableOperation::AddConstraint(c) => write!(f, "ADD {c}"),
            AlterTableOperation::AddColumn {
                column_keyword,
                if_not_exists,
                column_def,
                column_position,
            } => {
                write!(f, "ADD")?;
                if *column_keyword {
                    write!(f, " COLUMN")?;
                }
                if *if_not_exists {
                    write!(f, " IF NOT EXISTS")?;
                }
                write!(f, " {column_def}")?;

                if let Some(position) = column_position {
                    write!(f, " {position}")?;
                }

                Ok(())
            }
            AlterTableOperation::AddProjection {
                if_not_exists,
                name,
                select: query,
            } => {
                write!(f, "ADD PROJECTION")?;
                if *if_not_exists {
                    write!(f, " IF NOT EXISTS")?;
                }
                write!(f, " {} ({})", name, query)
            }
            AlterTableOperation::DropProjection { if_exists, name } => {
                write!(f, "DROP PROJECTION")?;
                if *if_exists {
                    write!(f, " IF EXISTS")?;
                }
                write!(f, " {}", name)
            }
            AlterTableOperation::MaterializeProjection {
                if_exists,
                name,
                partition,
            } => {
                write!(f, "MATERIALIZE PROJECTION")?;
                if *if_exists {
                    write!(f, " IF EXISTS")?;
                }
                write!(f, " {}", name)?;
                if let Some(partition) = partition {
                    write!(f, " IN PARTITION {}", partition)?;
                }
                Ok(())
            }
            AlterTableOperation::ClearProjection {
                if_exists,
                name,
                partition,
            } => {
                write!(f, "CLEAR PROJECTION")?;
                if *if_exists {
                    write!(f, " IF EXISTS")?;
                }
                write!(f, " {}", name)?;
                if let Some(partition) = partition {
                    write!(f, " IN PARTITION {}", partition)?;
                }
                Ok(())
            }
            AlterTableOperation::AlterColumn { column_name, op } => {
                write!(f, "ALTER COLUMN {column_name} {op}")
            }
            AlterTableOperation::DisableRowLevelSecurity => {
                write!(f, "DISABLE ROW LEVEL SECURITY")
            }
            AlterTableOperation::DisableRule { name } => {
                write!(f, "DISABLE RULE {name}")
            }
            AlterTableOperation::DisableTrigger { name } => {
                write!(f, "DISABLE TRIGGER {name}")
            }
            AlterTableOperation::DropPartitions {
                partitions,
                if_exists,
            } => write!(
                f,
                "DROP{ie} PARTITION ({})",
                display_comma_separated(partitions),
                ie = if *if_exists { " IF EXISTS" } else { "" }
            ),
            AlterTableOperation::DropConstraint {
                if_exists,
                name,
                cascade,
            } => {
                write!(
                    f,
                    "DROP CONSTRAINT {}{}{}",
                    if *if_exists { "IF EXISTS " } else { "" },
                    name,
                    if *cascade { " CASCADE" } else { "" },
                )
            }
            AlterTableOperation::DropPrimaryKey => write!(f, "DROP PRIMARY KEY"),
            AlterTableOperation::DropColumn {
                column_name,
                if_exists,
                cascade,
            } => write!(
                f,
                "DROP COLUMN {}{}{}",
                if *if_exists { "IF EXISTS " } else { "" },
                column_name,
                if *cascade { " CASCADE" } else { "" }
            ),
            AlterTableOperation::AttachPartition { partition } => {
                write!(f, "ATTACH {partition}")
            }
            AlterTableOperation::DetachPartition { partition } => {
                write!(f, "DETACH {partition}")
            }
            AlterTableOperation::EnableAlwaysRule { name } => {
                write!(f, "ENABLE ALWAYS RULE {name}")
            }
            AlterTableOperation::EnableAlwaysTrigger { name } => {
                write!(f, "ENABLE ALWAYS TRIGGER {name}")
            }
            AlterTableOperation::EnableReplicaRule { name } => {
                write!(f, "ENABLE REPLICA RULE {name}")
            }
            AlterTableOperation::EnableReplicaTrigger { name } => {
                write!(f, "ENABLE REPLICA TRIGGER {name}")
            }
            AlterTableOperation::EnableRowLevelSecurity => {
                write!(f, "ENABLE ROW LEVEL SECURITY")
            }
            AlterTableOperation::EnableRule { name } => {
                write!(f, "ENABLE RULE {name}")
            }
            AlterTableOperation::EnableTrigger { name } => {
                write!(f, "ENABLE TRIGGER {name}")
            }
            AlterTableOperation::RenamePartitions {
                old_partitions,
                new_partitions,
            } => write!(
                f,
                "PARTITION ({}) RENAME TO PARTITION ({})",
                display_comma_separated(old_partitions),
                display_comma_separated(new_partitions)
            ),
            AlterTableOperation::RenameColumn {
                old_column_name,
                new_column_name,
            } => write!(f, "RENAME COLUMN {old_column_name} TO {new_column_name}"),
            AlterTableOperation::RenameTable { table_name } => {
                write!(f, "RENAME TO {table_name}")
            }
            AlterTableOperation::ChangeColumn {
                old_name,
                new_name,
                data_type,
                options,
                column_position,
            } => {
                write!(f, "CHANGE COLUMN {old_name} {new_name} {data_type}")?;
                if !options.is_empty() {
                    write!(f, " {}", display_separated(options, " "))?;
                }
                if let Some(position) = column_position {
                    write!(f, " {position}")?;
                }

                Ok(())
            }
            AlterTableOperation::ModifyColumn {
                col_name,
                data_type,
                options,
                column_position,
            } => {
                write!(f, "MODIFY COLUMN {col_name} {data_type}")?;
                if !options.is_empty() {
                    write!(f, " {}", display_separated(options, " "))?;
                }
                if let Some(position) = column_position {
                    write!(f, " {position}")?;
                }

                Ok(())
            }
            AlterTableOperation::RenameConstraint { old_name, new_name } => {
                write!(f, "RENAME CONSTRAINT {old_name} TO {new_name}")
            }
            AlterTableOperation::SwapWith { table_name } => {
                write!(f, "SWAP WITH {table_name}")
            }
            AlterTableOperation::OwnerTo { new_owner } => {
                write!(f, "OWNER TO {new_owner}")
            }
            AlterTableOperation::SetTblProperties { table_properties } => {
                write!(
                    f,
                    "SET TBLPROPERTIES({})",
                    display_comma_separated(table_properties)
                )
            }
            AlterTableOperation::FreezePartition {
                partition,
                with_name,
            } => {
                write!(f, "FREEZE {partition}")?;
                if let Some(name) = with_name {
                    write!(f, " WITH NAME {name}")?;
                }
                Ok(())
            }
            AlterTableOperation::UnfreezePartition {
                partition,
                with_name,
            } => {
                write!(f, "UNFREEZE {partition}")?;
                if let Some(name) = with_name {
                    write!(f, " WITH NAME {name}")?;
                }
                Ok(())
            }
            AlterTableOperation::ClusterBy { exprs } => {
                write!(f, "CLUSTER BY ({})", display_comma_separated(exprs))?;
                Ok(())
            }
            AlterTableOperation::DropClusteringKey => {
                write!(f, "DROP CLUSTERING KEY")?;
                Ok(())
            }
            AlterTableOperation::SuspendRecluster => {
                write!(f, "SUSPEND RECLUSTER")?;
                Ok(())
            }
            AlterTableOperation::ResumeRecluster => {
                write!(f, "RESUME RECLUSTER")?;
                Ok(())
            }
        }
    }
}

impl fmt::Display for AlterIndexOperation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AlterIndexOperation::RenameIndex { index_name } => {
                write!(f, "RENAME TO {index_name}")
            }
        }
    }
}

/// An `ALTER COLUMN` (`Statement::AlterTable`) operation
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum AlterColumnOperation {
    /// `SET NOT NULL`
    SetNotNull,
    /// `DROP NOT NULL`
    DropNotNull,
    /// `SET DEFAULT <expr>`
    SetDefault { value: Expr },
    /// `DROP DEFAULT`
    DropDefault,
    /// `[SET DATA] TYPE <data_type> [USING <expr>]`
    SetDataType {
        data_type: DataType,
        /// PostgreSQL specific
        using: Option<Expr>,
    },
    /// `ADD GENERATED { ALWAYS | BY DEFAULT } AS IDENTITY [ ( sequence_options ) ]`
    ///
    /// Note: this is a PostgreSQL-specific operation.
    AddGenerated {
        generated_as: Option<GeneratedAs>,
        sequence_options: Option<Vec<SequenceOptions>>,
    },
}

impl fmt::Display for AlterColumnOperation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AlterColumnOperation::SetNotNull => write!(f, "SET NOT NULL",),
            AlterColumnOperation::DropNotNull => write!(f, "DROP NOT NULL",),
            AlterColumnOperation::SetDefault { value } => {
                write!(f, "SET DEFAULT {value}")
            }
            AlterColumnOperation::DropDefault {} => {
                write!(f, "DROP DEFAULT")
            }
            AlterColumnOperation::SetDataType { data_type, using } => {
                if let Some(expr) = using {
                    write!(f, "SET DATA TYPE {data_type} USING {expr}")
                } else {
                    write!(f, "SET DATA TYPE {data_type}")
                }
            }
            AlterColumnOperation::AddGenerated {
                generated_as,
                sequence_options,
            } => {
                let generated_as = match generated_as {
                    Some(GeneratedAs::Always) => " ALWAYS",
                    Some(GeneratedAs::ByDefault) => " BY DEFAULT",
                    _ => "",
                };

                write!(f, "ADD GENERATED{generated_as} AS IDENTITY",)?;
                if let Some(options) = sequence_options {
                    write!(f, " (")?;

                    for sequence_option in options {
                        write!(f, "{sequence_option}")?;
                    }

                    write!(f, " )")?;
                }
                Ok(())
            }
        }
    }
}

/// A table-level constraint, specified in a `CREATE TABLE` or an
/// `ALTER TABLE ADD <constraint>` statement.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum TableConstraint {
    /// MySQL [definition][1] for `UNIQUE` constraints statements:\
    /// * `[CONSTRAINT [<name>]] UNIQUE <index_type_display> [<index_name>] [index_type] (<columns>) <index_options>`
    ///
    /// where:
    /// * [index_type][2] is `USING {BTREE | HASH}`
    /// * [index_options][3] is `{index_type | COMMENT 'string' | ... %currently unsupported stmts% } ...`
    /// * [index_type_display][4] is `[INDEX | KEY]`
    ///
    /// [1]: https://dev.mysql.com/doc/refman/8.3/en/create-table.html
    /// [2]: IndexType
    /// [3]: IndexOption
    /// [4]: KeyOrIndexDisplay
    Unique {
        /// Constraint name.
        ///
        /// Can be not the same as `index_name`
        name: Option<Ident>,
        /// Index name
        index_name: Option<Ident>,
        /// Whether the type is followed by the keyword `KEY`, `INDEX`, or no keyword at all.
        index_type_display: KeyOrIndexDisplay,
        /// Optional `USING` of [index type][1] statement before columns.
        ///
        /// [1]: IndexType
        index_type: Option<IndexType>,
        /// Identifiers of the columns that are unique.
        columns: Vec<Ident>,
        index_options: Vec<IndexOption>,
        characteristics: Option<ConstraintCharacteristics>,
        /// Optional Postgres nulls handling: `[ NULLS [ NOT ] DISTINCT ]`
        nulls_distinct: NullsDistinctOption,
    },
    /// MySQL [definition][1] for `PRIMARY KEY` constraints statements:\
    /// * `[CONSTRAINT [<name>]] PRIMARY KEY [index_name] [index_type] (<columns>) <index_options>`
    ///
    /// Actually the specification have no `[index_name]` but the next query will complete successfully:
    /// ```sql
    /// CREATE TABLE unspec_table (
    ///   xid INT NOT NULL,
    ///   CONSTRAINT p_name PRIMARY KEY index_name USING BTREE (xid)
    /// );
    /// ```
    ///
    /// where:
    /// * [index_type][2] is `USING {BTREE | HASH}`
    /// * [index_options][3] is `{index_type | COMMENT 'string' | ... %currently unsupported stmts% } ...`
    ///
    /// [1]: https://dev.mysql.com/doc/refman/8.3/en/create-table.html
    /// [2]: IndexType
    /// [3]: IndexOption
    PrimaryKey {
        /// Constraint name.
        ///
        /// Can be not the same as `index_name`
        name: Option<Ident>,
        /// Index name
        index_name: Option<Ident>,
        /// Optional `USING` of [index type][1] statement before columns.
        ///
        /// [1]: IndexType
        index_type: Option<IndexType>,
        /// Identifiers of the columns that form the primary key.
        columns: Vec<Ident>,
        index_options: Vec<IndexOption>,
        characteristics: Option<ConstraintCharacteristics>,
    },
    /// A referential integrity constraint (`[ CONSTRAINT <name> ] FOREIGN KEY (<columns>)
    /// REFERENCES <foreign_table> (<referred_columns>)
    /// { [ON DELETE <referential_action>] [ON UPDATE <referential_action>] |
    ///   [ON UPDATE <referential_action>] [ON DELETE <referential_action>]
    /// }`).
    ForeignKey {
        name: Option<Ident>,
        columns: Vec<Ident>,
        foreign_table: ObjectName,
        referred_columns: Vec<Ident>,
        on_delete: Option<ReferentialAction>,
        on_update: Option<ReferentialAction>,
        characteristics: Option<ConstraintCharacteristics>,
    },
    /// `[ CONSTRAINT <name> ] CHECK (<expr>)`
    Check {
        name: Option<Ident>,
        expr: Box<Expr>,
    },
    /// MySQLs [index definition][1] for index creation. Not present on ANSI so, for now, the usage
    /// is restricted to MySQL, as no other dialects that support this syntax were found.
    ///
    /// `{INDEX | KEY} [index_name] [index_type] (key_part,...) [index_option]...`
    ///
    /// [1]: https://dev.mysql.com/doc/refman/8.0/en/create-table.html
    Index {
        /// Whether this index starts with KEY (true) or INDEX (false), to maintain the same syntax.
        display_as_key: bool,
        /// Index name.
        name: Option<Ident>,
        /// Optional [index type][1].
        ///
        /// [1]: IndexType
        index_type: Option<IndexType>,
        /// Referred column identifier list.
        columns: Vec<Ident>,
    },
    /// MySQLs [fulltext][1] definition. Since the [`SPATIAL`][2] definition is exactly the same,
    /// and MySQL displays both the same way, it is part of this definition as well.
    ///
    /// Supported syntax:
    ///
    /// ```markdown
    /// {FULLTEXT | SPATIAL} [INDEX | KEY] [index_name] (key_part,...)
    ///
    /// key_part: col_name
    /// ```
    ///
    /// [1]: https://dev.mysql.com/doc/refman/8.0/en/fulltext-natural-language.html
    /// [2]: https://dev.mysql.com/doc/refman/8.0/en/spatial-types.html
    FulltextOrSpatial {
        /// Whether this is a `FULLTEXT` (true) or `SPATIAL` (false) definition.
        fulltext: bool,
        /// Whether the type is followed by the keyword `KEY`, `INDEX`, or no keyword at all.
        index_type_display: KeyOrIndexDisplay,
        /// Optional index name.
        opt_index_name: Option<Ident>,
        /// Referred column identifier list.
        columns: Vec<Ident>,
    },
}

impl fmt::Display for TableConstraint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TableConstraint::Unique {
                name,
                index_name,
                index_type_display,
                index_type,
                columns,
                index_options,
                characteristics,
                nulls_distinct,
            } => {
                write!(
                    f,
                    "{}UNIQUE{nulls_distinct}{index_type_display:>}{}{} ({})",
                    display_constraint_name(name),
                    display_option_spaced(index_name),
                    display_option(" USING ", "", index_type),
                    display_comma_separated(columns),
                )?;

                if !index_options.is_empty() {
                    write!(f, " {}", display_separated(index_options, " "))?;
                }

                write!(f, "{}", display_option_spaced(characteristics))?;
                Ok(())
            }
            TableConstraint::PrimaryKey {
                name,
                index_name,
                index_type,
                columns,
                index_options,
                characteristics,
            } => {
                write!(
                    f,
                    "{}PRIMARY KEY{}{} ({})",
                    display_constraint_name(name),
                    display_option_spaced(index_name),
                    display_option(" USING ", "", index_type),
                    display_comma_separated(columns),
                )?;

                if !index_options.is_empty() {
                    write!(f, " {}", display_separated(index_options, " "))?;
                }

                write!(f, "{}", display_option_spaced(characteristics))?;
                Ok(())
            }
            TableConstraint::ForeignKey {
                name,
                columns,
                foreign_table,
                referred_columns,
                on_delete,
                on_update,
                characteristics,
            } => {
                write!(
                    f,
                    "{}FOREIGN KEY ({}) REFERENCES {}({})",
                    display_constraint_name(name),
                    display_comma_separated(columns),
                    foreign_table,
                    display_comma_separated(referred_columns),
                )?;
                if let Some(action) = on_delete {
                    write!(f, " ON DELETE {action}")?;
                }
                if let Some(action) = on_update {
                    write!(f, " ON UPDATE {action}")?;
                }
                if let Some(characteristics) = characteristics {
                    write!(f, " {}", characteristics)?;
                }
                Ok(())
            }
            TableConstraint::Check { name, expr } => {
                write!(f, "{}CHECK ({})", display_constraint_name(name), expr)
            }
            TableConstraint::Index {
                display_as_key,
                name,
                index_type,
                columns,
            } => {
                write!(f, "{}", if *display_as_key { "KEY" } else { "INDEX" })?;
                if let Some(name) = name {
                    write!(f, " {name}")?;
                }
                if let Some(index_type) = index_type {
                    write!(f, " USING {index_type}")?;
                }
                write!(f, " ({})", display_comma_separated(columns))?;

                Ok(())
            }
            Self::FulltextOrSpatial {
                fulltext,
                index_type_display,
                opt_index_name,
                columns,
            } => {
                if *fulltext {
                    write!(f, "FULLTEXT")?;
                } else {
                    write!(f, "SPATIAL")?;
                }

                write!(f, "{index_type_display:>}")?;

                if let Some(name) = opt_index_name {
                    write!(f, " {name}")?;
                }

                write!(f, " ({})", display_comma_separated(columns))?;

                Ok(())
            }
        }
    }
}

/// Representation whether a definition can can contains the KEY or INDEX keywords with the same
/// meaning.
///
/// This enum initially is directed to `FULLTEXT`,`SPATIAL`, and `UNIQUE` indexes on create table
/// statements of `MySQL` [(1)].
///
/// [1]: https://dev.mysql.com/doc/refman/8.0/en/create-table.html
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum KeyOrIndexDisplay {
    /// Nothing to display
    None,
    /// Display the KEY keyword
    Key,
    /// Display the INDEX keyword
    Index,
}

impl KeyOrIndexDisplay {
    pub fn is_none(self) -> bool {
        matches!(self, Self::None)
    }
}

impl fmt::Display for KeyOrIndexDisplay {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let left_space = matches!(f.align(), Some(fmt::Alignment::Right));

        if left_space && !self.is_none() {
            f.write_char(' ')?
        }

        match self {
            KeyOrIndexDisplay::None => {
                write!(f, "")
            }
            KeyOrIndexDisplay::Key => {
                write!(f, "KEY")
            }
            KeyOrIndexDisplay::Index => {
                write!(f, "INDEX")
            }
        }
    }
}

/// Indexing method used by that index.
///
/// This structure isn't present on ANSI, but is found at least in [`MySQL` CREATE TABLE][1],
/// [`MySQL` CREATE INDEX][2], and [Postgresql CREATE INDEX][3] statements.
///
/// [1]: https://dev.mysql.com/doc/refman/8.0/en/create-table.html
/// [2]: https://dev.mysql.com/doc/refman/8.0/en/create-index.html
/// [3]: https://www.postgresql.org/docs/14/sql-createindex.html
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum IndexType {
    BTree,
    Hash,
    // TODO add Postgresql's possible indexes
}

impl fmt::Display for IndexType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree => write!(f, "BTREE"),
            Self::Hash => write!(f, "HASH"),
        }
    }
}

/// MySQLs index option.
///
/// This structure used here [`MySQL` CREATE TABLE][1], [`MySQL` CREATE INDEX][2].
///
/// [1]: https://dev.mysql.com/doc/refman/8.3/en/create-table.html
/// [2]: https://dev.mysql.com/doc/refman/8.3/en/create-index.html
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum IndexOption {
    Using(IndexType),
    Comment(String),
}

impl fmt::Display for IndexOption {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Using(index_type) => write!(f, "USING {index_type}"),
            Self::Comment(s) => write!(f, "COMMENT '{s}'"),
        }
    }
}

/// [Postgres] unique index nulls handling option: `[ NULLS [ NOT ] DISTINCT ]`
///
/// [Postgres]: https://www.postgresql.org/docs/17/sql-altertable.html
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum NullsDistinctOption {
    /// Not specified
    None,
    /// NULLS DISTINCT
    Distinct,
    /// NULLS NOT DISTINCT
    NotDistinct,
}

impl fmt::Display for NullsDistinctOption {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::None => Ok(()),
            Self::Distinct => write!(f, " NULLS DISTINCT"),
            Self::NotDistinct => write!(f, " NULLS NOT DISTINCT"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct ProcedureParam {
    pub name: Ident,
    pub data_type: DataType,
}

impl fmt::Display for ProcedureParam {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {}", self.name, self.data_type)
    }
}

/// SQL column definition
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct ColumnDef {
    pub name: Ident,
    pub data_type: DataType,
    pub collation: Option<ObjectName>,
    pub options: Vec<ColumnOptionDef>,
}

impl fmt::Display for ColumnDef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.data_type == DataType::Unspecified {
            write!(f, "{}", self.name)?;
        } else {
            write!(f, "{} {}", self.name, self.data_type)?;
        }
        if let Some(collation) = &self.collation {
            write!(f, " COLLATE {collation}")?;
        }
        for option in &self.options {
            write!(f, " {option}")?;
        }
        Ok(())
    }
}

/// Column definition specified in a `CREATE VIEW` statement.
///
/// Syntax
/// ```markdown
/// <name> [data_type][OPTIONS(option, ...)]
///
/// option: <name> = <value>
/// ```
///
/// Examples:
/// ```sql
/// name
/// age OPTIONS(description = "age column", tag = "prod")
/// amount COMMENT 'The total amount for the order line'
/// created_at DateTime64
/// ```
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct ViewColumnDef {
    pub name: Ident,
    pub data_type: Option<DataType>,
    pub options: Option<Vec<ColumnOption>>,
}

impl fmt::Display for ViewColumnDef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if let Some(data_type) = self.data_type.as_ref() {
            write!(f, " {}", data_type)?;
        }
        if let Some(options) = self.options.as_ref() {
            write!(f, " {}", display_comma_separated(options.as_slice()))?;
        }
        Ok(())
    }
}

/// An optionally-named `ColumnOption`: `[ CONSTRAINT <name> ] <column-option>`.
///
/// Note that implementations are substantially more permissive than the ANSI
/// specification on what order column options can be presented in, and whether
/// they are allowed to be named. The specification distinguishes between
/// constraints (NOT NULL, UNIQUE, PRIMARY KEY, and CHECK), which can be named
/// and can appear in any order, and other options (DEFAULT, GENERATED), which
/// cannot be named and must appear in a fixed order. `PostgreSQL`, however,
/// allows preceding any option with `CONSTRAINT <name>`, even those that are
/// not really constraints, like NULL and DEFAULT. MSSQL is less permissive,
/// allowing DEFAULT, UNIQUE, PRIMARY KEY and CHECK to be named, but not NULL or
/// NOT NULL constraints (the last of which is in violation of the spec).
///
/// For maximum flexibility, we don't distinguish between constraint and
/// non-constraint options, lumping them all together under the umbrella of
/// "column options," and we allow any column option to be named.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct ColumnOptionDef {
    pub name: Option<Ident>,
    pub option: ColumnOption,
}

impl fmt::Display for ColumnOptionDef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", display_constraint_name(&self.name), self.option)
    }
}

/// Identity is a column option for defining an identity or autoincrement column in a `CREATE TABLE` statement.
/// Syntax
/// ```sql
/// { IDENTITY | AUTOINCREMENT } [ (seed , increment) | START num INCREMENT num ] [ ORDER | NOORDER ]
/// ```
/// [MS SQL Server]: https://learn.microsoft.com/en-us/sql/t-sql/statements/create-table-transact-sql-identity-property
/// [Snowflake]: https://docs.snowflake.com/en/sql-reference/sql/create-table
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum IdentityPropertyKind {
    /// An identity property declared via the `AUTOINCREMENT` key word
    /// Example:
    /// ```sql
    ///  AUTOINCREMENT(100, 1) NOORDER
    ///  AUTOINCREMENT START 100 INCREMENT 1 ORDER
    /// ```
    /// [Snowflake]: https://docs.snowflake.com/en/sql-reference/sql/create-table
    Autoincrement(IdentityProperty),
    /// An identity property declared via the `IDENTITY` key word
    /// Example, [MS SQL Server] or [Snowflake]:
    /// ```sql
    ///  IDENTITY(100, 1)
    /// ```
    /// [Snowflake]
    /// ```sql
    ///  IDENTITY(100, 1) ORDER
    ///  IDENTITY START 100 INCREMENT 1 NOORDER
    /// ```
    /// [MS SQL Server]: https://learn.microsoft.com/en-us/sql/t-sql/statements/create-table-transact-sql-identity-property
    /// [Snowflake]: https://docs.snowflake.com/en/sql-reference/sql/create-table
    Identity(IdentityProperty),
}

impl fmt::Display for IdentityPropertyKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (command, property) = match self {
            IdentityPropertyKind::Identity(property) => ("IDENTITY", property),
            IdentityPropertyKind::Autoincrement(property) => ("AUTOINCREMENT", property),
        };
        write!(f, "{command}")?;
        if let Some(parameters) = &property.parameters {
            write!(f, "{parameters}")?;
        }
        if let Some(order) = &property.order {
            write!(f, "{order}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct IdentityProperty {
    pub parameters: Option<IdentityPropertyFormatKind>,
    pub order: Option<IdentityPropertyOrder>,
}

/// A format of parameters of identity column.
///
/// It is [Snowflake] specific.
/// Syntax
/// ```sql
/// (seed , increment) | START num INCREMENT num
/// ```
/// [MS SQL Server] uses one way of representing these parameters.
/// Syntax
/// ```sql
/// (seed , increment)
/// ```
/// [MS SQL Server]: https://learn.microsoft.com/en-us/sql/t-sql/statements/create-table-transact-sql-identity-property
/// [Snowflake]: https://docs.snowflake.com/en/sql-reference/sql/create-table
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum IdentityPropertyFormatKind {
    /// A parameters of identity column declared like parameters of function call
    /// Example:
    /// ```sql
    ///  (100, 1)
    /// ```
    /// [MS SQL Server]: https://learn.microsoft.com/en-us/sql/t-sql/statements/create-table-transact-sql-identity-property
    /// [Snowflake]: https://docs.snowflake.com/en/sql-reference/sql/create-table
    FunctionCall(IdentityParameters),
    /// A parameters of identity column declared with keywords `START` and `INCREMENT`
    /// Example:
    /// ```sql
    ///  START 100 INCREMENT 1
    /// ```
    /// [Snowflake]: https://docs.snowflake.com/en/sql-reference/sql/create-table
    StartAndIncrement(IdentityParameters),
}

impl fmt::Display for IdentityPropertyFormatKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IdentityPropertyFormatKind::FunctionCall(parameters) => {
                write!(f, "({}, {})", parameters.seed, parameters.increment)
            }
            IdentityPropertyFormatKind::StartAndIncrement(parameters) => {
                write!(
                    f,
                    " START {} INCREMENT {}",
                    parameters.seed, parameters.increment
                )
            }
        }
    }
}
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct IdentityParameters {
    pub seed: Expr,
    pub increment: Expr,
}

/// The identity column option specifies how values are generated for the auto-incremented column, either in increasing or decreasing order.
/// Syntax
/// ```sql
/// ORDER | NOORDER
/// ```
/// [Snowflake]: https://docs.snowflake.com/en/sql-reference/sql/create-table
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum IdentityPropertyOrder {
    Order,
    NoOrder,
}

impl fmt::Display for IdentityPropertyOrder {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IdentityPropertyOrder::Order => write!(f, " ORDER"),
            IdentityPropertyOrder::NoOrder => write!(f, " NOORDER"),
        }
    }
}

/// Column policy that identify a security policy of access to a column.
/// Syntax
/// ```sql
/// [ WITH ] MASKING POLICY <policy_name> [ USING ( <col_name> , <cond_col1> , ... ) ]
/// [ WITH ] PROJECTION POLICY <policy_name>
/// ```
/// [Snowflake]: https://docs.snowflake.com/en/sql-reference/sql/create-table
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum ColumnPolicy {
    MaskingPolicy(ColumnPolicyProperty),
    ProjectionPolicy(ColumnPolicyProperty),
}

impl fmt::Display for ColumnPolicy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (command, property) = match self {
            ColumnPolicy::MaskingPolicy(property) => ("MASKING POLICY", property),
            ColumnPolicy::ProjectionPolicy(property) => ("PROJECTION POLICY", property),
        };
        if property.with {
            write!(f, "WITH ")?;
        }
        write!(f, "{command} {}", property.policy_name)?;
        if let Some(using_columns) = &property.using_columns {
            write!(f, " USING ({})", display_comma_separated(using_columns))?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct ColumnPolicyProperty {
    /// This flag indicates that the column policy option is declared using the `WITH` prefix.
    /// Example
    /// ```sql
    /// WITH PROJECTION POLICY sample_policy
    /// ```
    /// [Snowflake]: https://docs.snowflake.com/en/sql-reference/sql/create-table
    pub with: bool,
    pub policy_name: Ident,
    pub using_columns: Option<Vec<Ident>>,
}

/// Tags option of column
/// Syntax
/// ```sql
/// [ WITH ] TAG ( <tag_name> = '<tag_value>' [ , <tag_name> = '<tag_value>' , ... ] )
/// ```
/// [Snowflake]: https://docs.snowflake.com/en/sql-reference/sql/create-table
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct TagsColumnOption {
    /// This flag indicates that the tags option is declared using the `WITH` prefix.
    /// Example:
    /// ```sql
    /// WITH TAG (A = 'Tag A')
    /// ```
    /// [Snowflake]: https://docs.snowflake.com/en/sql-reference/sql/create-table
    pub with: bool,
    pub tags: Vec<Tag>,
}

impl fmt::Display for TagsColumnOption {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.with {
            write!(f, "WITH ")?;
        }
        write!(f, "TAG ({})", display_comma_separated(&self.tags))?;
        Ok(())
    }
}

/// `ColumnOption`s are modifiers that follow a column definition in a `CREATE
/// TABLE` statement.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum ColumnOption {
    /// `NULL`
    Null,
    /// `NOT NULL`
    NotNull,
    /// `DEFAULT <restricted-expr>`
    Default(Expr),

    /// `MATERIALIZE <expr>`
    /// Syntax: `b INT MATERIALIZE (a + 1)`
    ///
    /// [ClickHouse](https://clickhouse.com/docs/en/sql-reference/statements/create/table#default_values)
    Materialized(Expr),
    /// `EPHEMERAL [<expr>]`
    ///
    /// [ClickHouse](https://clickhouse.com/docs/en/sql-reference/statements/create/table#default_values)
    Ephemeral(Option<Expr>),
    /// `ALIAS <expr>`
    ///
    /// [ClickHouse](https://clickhouse.com/docs/en/sql-reference/statements/create/table#default_values)
    Alias(Expr),

    /// `{ PRIMARY KEY | UNIQUE } [<constraint_characteristics>]`
    Unique {
        is_primary: bool,
        characteristics: Option<ConstraintCharacteristics>,
    },
    /// A referential integrity constraint (`[FOREIGN KEY REFERENCES
    /// <foreign_table> (<referred_columns>)
    /// { [ON DELETE <referential_action>] [ON UPDATE <referential_action>] |
    ///   [ON UPDATE <referential_action>] [ON DELETE <referential_action>]
    /// }
    /// [<constraint_characteristics>]
    /// `).
    ForeignKey {
        foreign_table: ObjectName,
        referred_columns: Vec<Ident>,
        on_delete: Option<ReferentialAction>,
        on_update: Option<ReferentialAction>,
        characteristics: Option<ConstraintCharacteristics>,
    },
    /// `CHECK (<expr>)`
    Check(Expr),
    /// Dialect-specific options, such as:
    /// - MySQL's `AUTO_INCREMENT` or SQLite's `AUTOINCREMENT`
    /// - ...
    DialectSpecific(Vec<Token>),
    CharacterSet(ObjectName),
    Comment(String),
    OnUpdate(Expr),
    /// `Generated`s are modifiers that follow a column definition in a `CREATE
    /// TABLE` statement.
    Generated {
        generated_as: GeneratedAs,
        sequence_options: Option<Vec<SequenceOptions>>,
        generation_expr: Option<Expr>,
        generation_expr_mode: Option<GeneratedExpressionMode>,
        /// false if 'GENERATED ALWAYS' is skipped (option starts with AS)
        generated_keyword: bool,
    },
    /// BigQuery specific: Explicit column options in a view [1] or table [2]
    /// Syntax
    /// ```sql
    /// OPTIONS(description="field desc")
    /// ```
    /// [1]: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#view_column_option_list
    /// [2]: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#column_option_list
    Options(Vec<SqlOption>),
    /// Creates an identity or an autoincrement column in a table.
    /// Syntax
    /// ```sql
    /// { IDENTITY | AUTOINCREMENT } [ (seed , increment) | START num INCREMENT num ] [ ORDER | NOORDER ]
    /// ```
    /// [MS SQL Server]: https://learn.microsoft.com/en-us/sql/t-sql/statements/create-table-transact-sql-identity-property
    /// [Snowflake]: https://docs.snowflake.com/en/sql-reference/sql/create-table
    Identity(IdentityPropertyKind),
    /// SQLite specific: ON CONFLICT option on column definition
    /// <https://www.sqlite.org/lang_conflict.html>
    OnConflict(Keyword),
    /// Snowflake specific: an option of specifying security masking or projection policy to set on a column.
    /// Syntax:
    /// ```sql
    /// [ WITH ] MASKING POLICY <policy_name> [ USING ( <col_name> , <cond_col1> , ... ) ]
    /// [ WITH ] PROJECTION POLICY <policy_name>
    /// ```
    /// [Snowflake]: https://docs.snowflake.com/en/sql-reference/sql/create-table
    Policy(ColumnPolicy),
    /// Snowflake specific: Specifies the tag name and the tag string value.
    /// Syntax:
    /// ```sql
    /// [ WITH ] TAG ( <tag_name> = '<tag_value>' [ , <tag_name> = '<tag_value>' , ... ] )
    /// ```
    /// [Snowflake]: https://docs.snowflake.com/en/sql-reference/sql/create-table
    Tags(TagsColumnOption),
}

impl fmt::Display for ColumnOption {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ColumnOption::*;
        match self {
            Null => write!(f, "NULL"),
            NotNull => write!(f, "NOT NULL"),
            Default(expr) => write!(f, "DEFAULT {expr}"),
            Materialized(expr) => write!(f, "MATERIALIZED {expr}"),
            Ephemeral(expr) => {
                if let Some(e) = expr {
                    write!(f, "EPHEMERAL {e}")
                } else {
                    write!(f, "EPHEMERAL")
                }
            }
            Alias(expr) => write!(f, "ALIAS {expr}"),
            Unique {
                is_primary,
                characteristics,
            } => {
                write!(f, "{}", if *is_primary { "PRIMARY KEY" } else { "UNIQUE" })?;
                if let Some(characteristics) = characteristics {
                    write!(f, " {}", characteristics)?;
                }
                Ok(())
            }
            ForeignKey {
                foreign_table,
                referred_columns,
                on_delete,
                on_update,
                characteristics,
            } => {
                write!(f, "REFERENCES {foreign_table}")?;
                if !referred_columns.is_empty() {
                    write!(f, " ({})", display_comma_separated(referred_columns))?;
                }
                if let Some(action) = on_delete {
                    write!(f, " ON DELETE {action}")?;
                }
                if let Some(action) = on_update {
                    write!(f, " ON UPDATE {action}")?;
                }
                if let Some(characteristics) = characteristics {
                    write!(f, " {}", characteristics)?;
                }
                Ok(())
            }
            Check(expr) => write!(f, "CHECK ({expr})"),
            DialectSpecific(val) => write!(f, "{}", display_separated(val, " ")),
            CharacterSet(n) => write!(f, "CHARACTER SET {n}"),
            Comment(v) => write!(f, "COMMENT '{}'", escape_single_quote_string(v)),
            OnUpdate(expr) => write!(f, "ON UPDATE {expr}"),
            Generated {
                generated_as,
                sequence_options,
                generation_expr,
                generation_expr_mode,
                generated_keyword,
            } => {
                if let Some(expr) = generation_expr {
                    let modifier = match generation_expr_mode {
                        None => "",
                        Some(GeneratedExpressionMode::Virtual) => " VIRTUAL",
                        Some(GeneratedExpressionMode::Stored) => " STORED",
                    };
                    if *generated_keyword {
                        write!(f, "GENERATED ALWAYS AS ({expr}){modifier}")?;
                    } else {
                        write!(f, "AS ({expr}){modifier}")?;
                    }
                    Ok(())
                } else {
                    // Like Postgres - generated from sequence
                    let when = match generated_as {
                        GeneratedAs::Always => "ALWAYS",
                        GeneratedAs::ByDefault => "BY DEFAULT",
                        // ExpStored goes with an expression, handled above
                        GeneratedAs::ExpStored => unreachable!(),
                    };
                    write!(f, "GENERATED {when} AS IDENTITY")?;
                    if sequence_options.is_some() {
                        let so = sequence_options.as_ref().unwrap();
                        if !so.is_empty() {
                            write!(f, " (")?;
                        }
                        for sequence_option in so {
                            write!(f, "{sequence_option}")?;
                        }
                        if !so.is_empty() {
                            write!(f, " )")?;
                        }
                    }
                    Ok(())
                }
            }
            Options(options) => {
                write!(f, "OPTIONS({})", display_comma_separated(options))
            }
            Identity(parameters) => {
                write!(f, "{parameters}")
            }
            OnConflict(keyword) => {
                write!(f, "ON CONFLICT {:?}", keyword)?;
                Ok(())
            }
            Policy(parameters) => {
                write!(f, "{parameters}")
            }
            Tags(tags) => {
                write!(f, "{tags}")
            }
        }
    }
}

/// `GeneratedAs`s are modifiers that follow a column option in a `generated`.
/// 'ExpStored' is used for a column generated from an expression and stored.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum GeneratedAs {
    Always,
    ByDefault,
    ExpStored,
}

/// `GeneratedExpressionMode`s are modifiers that follow an expression in a `generated`.
/// No modifier is typically the same as Virtual.
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum GeneratedExpressionMode {
    Virtual,
    Stored,
}

#[must_use]
fn display_constraint_name(name: &'_ Option<Ident>) -> impl fmt::Display + '_ {
    struct ConstraintName<'a>(&'a Option<Ident>);
    impl fmt::Display for ConstraintName<'_> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            if let Some(name) = self.0 {
                write!(f, "CONSTRAINT {name} ")?;
            }
            Ok(())
        }
    }
    ConstraintName(name)
}

/// If `option` is
/// * `Some(inner)` => create display struct for `"{prefix}{inner}{postfix}"`
/// * `_` => do nothing
#[must_use]
fn display_option<'a, T: fmt::Display>(
    prefix: &'a str,
    postfix: &'a str,
    option: &'a Option<T>,
) -> impl fmt::Display + 'a {
    struct OptionDisplay<'a, T>(&'a str, &'a str, &'a Option<T>);
    impl<T: fmt::Display> fmt::Display for OptionDisplay<'_, T> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            if let Some(inner) = self.2 {
                let (prefix, postfix) = (self.0, self.1);
                write!(f, "{prefix}{inner}{postfix}")?;
            }
            Ok(())
        }
    }
    OptionDisplay(prefix, postfix, option)
}

/// If `option` is
/// * `Some(inner)` => create display struct for `" {inner}"`
/// * `_` => do nothing
#[must_use]
fn display_option_spaced<T: fmt::Display>(option: &Option<T>) -> impl fmt::Display + '_ {
    display_option(" ", "", option)
}

/// `<constraint_characteristics> = [ DEFERRABLE | NOT DEFERRABLE ] [ INITIALLY DEFERRED | INITIALLY IMMEDIATE ] [ ENFORCED | NOT ENFORCED ]`
///
/// Used in UNIQUE and foreign key constraints. The individual settings may occur in any order.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Default, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct ConstraintCharacteristics {
    /// `[ DEFERRABLE | NOT DEFERRABLE ]`
    pub deferrable: Option<bool>,
    /// `[ INITIALLY DEFERRED | INITIALLY IMMEDIATE ]`
    pub initially: Option<DeferrableInitial>,
    /// `[ ENFORCED | NOT ENFORCED ]`
    pub enforced: Option<bool>,
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum DeferrableInitial {
    /// `INITIALLY IMMEDIATE`
    Immediate,
    /// `INITIALLY DEFERRED`
    Deferred,
}

impl ConstraintCharacteristics {
    fn deferrable_text(&self) -> Option<&'static str> {
        self.deferrable.map(|deferrable| {
            if deferrable {
                "DEFERRABLE"
            } else {
                "NOT DEFERRABLE"
            }
        })
    }

    fn initially_immediate_text(&self) -> Option<&'static str> {
        self.initially
            .map(|initially_immediate| match initially_immediate {
                DeferrableInitial::Immediate => "INITIALLY IMMEDIATE",
                DeferrableInitial::Deferred => "INITIALLY DEFERRED",
            })
    }

    fn enforced_text(&self) -> Option<&'static str> {
        self.enforced.map(
            |enforced| {
                if enforced {
                    "ENFORCED"
                } else {
                    "NOT ENFORCED"
                }
            },
        )
    }
}

impl fmt::Display for ConstraintCharacteristics {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let deferrable = self.deferrable_text();
        let initially_immediate = self.initially_immediate_text();
        let enforced = self.enforced_text();

        match (deferrable, initially_immediate, enforced) {
            (None, None, None) => Ok(()),
            (None, None, Some(enforced)) => write!(f, "{enforced}"),
            (None, Some(initial), None) => write!(f, "{initial}"),
            (None, Some(initial), Some(enforced)) => write!(f, "{initial} {enforced}"),
            (Some(deferrable), None, None) => write!(f, "{deferrable}"),
            (Some(deferrable), None, Some(enforced)) => write!(f, "{deferrable} {enforced}"),
            (Some(deferrable), Some(initial), None) => write!(f, "{deferrable} {initial}"),
            (Some(deferrable), Some(initial), Some(enforced)) => {
                write!(f, "{deferrable} {initial} {enforced}")
            }
        }
    }
}

/// `<referential_action> =
/// { RESTRICT | CASCADE | SET NULL | NO ACTION | SET DEFAULT }`
///
/// Used in foreign key constraints in `ON UPDATE` and `ON DELETE` options.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum ReferentialAction {
    Restrict,
    Cascade,
    SetNull,
    NoAction,
    SetDefault,
}

impl fmt::Display for ReferentialAction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            ReferentialAction::Restrict => "RESTRICT",
            ReferentialAction::Cascade => "CASCADE",
            ReferentialAction::SetNull => "SET NULL",
            ReferentialAction::NoAction => "NO ACTION",
            ReferentialAction::SetDefault => "SET DEFAULT",
        })
    }
}

/// SQL user defined type definition
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum UserDefinedTypeRepresentation {
    Composite {
        attributes: Vec<UserDefinedTypeCompositeAttributeDef>,
    },
    /// Note: this is PostgreSQL-specific. See <https://www.postgresql.org/docs/current/sql-createtype.html>
    Enum { labels: Vec<Ident> },
}

impl fmt::Display for UserDefinedTypeRepresentation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            UserDefinedTypeRepresentation::Composite { attributes } => {
                write!(f, "({})", display_comma_separated(attributes))
            }
            UserDefinedTypeRepresentation::Enum { labels } => {
                write!(f, "ENUM ({})", display_comma_separated(labels))
            }
        }
    }
}

/// SQL user defined type attribute definition
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct UserDefinedTypeCompositeAttributeDef {
    pub name: Ident,
    pub data_type: DataType,
    pub collation: Option<ObjectName>,
}

impl fmt::Display for UserDefinedTypeCompositeAttributeDef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {}", self.name, self.data_type)?;
        if let Some(collation) = &self.collation {
            write!(f, " COLLATE {collation}")?;
        }
        Ok(())
    }
}

/// PARTITION statement used in ALTER TABLE et al. such as in Hive and ClickHouse SQL.
/// For example, ClickHouse's OPTIMIZE TABLE supports syntax like PARTITION ID 'partition_id' and PARTITION expr.
/// [ClickHouse](https://clickhouse.com/docs/en/sql-reference/statements/optimize)
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum Partition {
    Identifier(Ident),
    Expr(Expr),
    /// ClickHouse supports PART expr which represents physical partition in disk.
    /// [ClickHouse](https://clickhouse.com/docs/en/sql-reference/statements/alter/partition#attach-partitionpart)
    Part(Expr),
    Partitions(Vec<Expr>),
}

impl fmt::Display for Partition {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Partition::Identifier(id) => write!(f, "PARTITION ID {id}"),
            Partition::Expr(expr) => write!(f, "PARTITION {expr}"),
            Partition::Part(expr) => write!(f, "PART {expr}"),
            Partition::Partitions(partitions) => {
                write!(f, "PARTITION ({})", display_comma_separated(partitions))
            }
        }
    }
}

/// DEDUPLICATE statement used in OPTIMIZE TABLE et al. such as in ClickHouse SQL
/// [ClickHouse](https://clickhouse.com/docs/en/sql-reference/statements/optimize)
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum Deduplicate {
    All,
    ByExpression(Expr),
}

impl fmt::Display for Deduplicate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Deduplicate::All => write!(f, "DEDUPLICATE"),
            Deduplicate::ByExpression(expr) => write!(f, "DEDUPLICATE BY {expr}"),
        }
    }
}

/// Hive supports `CLUSTERED BY` statement in `CREATE TABLE`.
/// Syntax: `CLUSTERED BY (col_name, ...) [SORTED BY (col_name [ASC|DESC], ...)] INTO num_buckets BUCKETS`
///
/// [Hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+DDL#LanguageManualDDL-CreateTable)
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct ClusteredBy {
    pub columns: Vec<Ident>,
    pub sorted_by: Option<Vec<OrderByExpr>>,
    pub num_buckets: Value,
}

impl fmt::Display for ClusteredBy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "CLUSTERED BY ({})",
            display_comma_separated(&self.columns)
        )?;
        if let Some(ref sorted_by) = self.sorted_by {
            write!(f, " SORTED BY ({})", display_comma_separated(sorted_by))?;
        }
        write!(f, " INTO {} BUCKETS", self.num_buckets)
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct CreateFunction {
    pub or_replace: bool,
    pub temporary: bool,
    pub if_not_exists: bool,
    pub name: ObjectName,
    pub args: Option<Vec<OperateFunctionArg>>,
    pub return_type: Option<DataType>,
    /// The expression that defines the function.
    ///
    /// Examples:
    /// ```sql
    /// AS ((SELECT 1))
    /// AS "console.log();"
    /// ```
    pub function_body: Option<CreateFunctionBody>,
    /// Behavior attribute for the function
    ///
    /// IMMUTABLE | STABLE | VOLATILE
    ///
    /// [Postgres](https://www.postgresql.org/docs/current/sql-createfunction.html)
    pub behavior: Option<FunctionBehavior>,
    /// CALLED ON NULL INPUT | RETURNS NULL ON NULL INPUT | STRICT
    ///
    /// [Postgres](https://www.postgresql.org/docs/current/sql-createfunction.html)
    pub called_on_null: Option<FunctionCalledOnNull>,
    /// PARALLEL { UNSAFE | RESTRICTED | SAFE }
    ///
    /// [Postgres](https://www.postgresql.org/docs/current/sql-createfunction.html)
    pub parallel: Option<FunctionParallel>,
    /// USING ... (Hive only)
    pub using: Option<CreateFunctionUsing>,
    /// Language used in a UDF definition.
    ///
    /// Example:
    /// ```sql
    /// CREATE FUNCTION foo() LANGUAGE js AS "console.log();"
    /// ```
    /// [BigQuery](https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#create_a_javascript_udf)
    pub language: Option<Ident>,
    /// Determinism keyword used for non-sql UDF definitions.
    ///
    /// [BigQuery](https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#syntax_11)
    pub determinism_specifier: Option<FunctionDeterminismSpecifier>,
    /// List of options for creating the function.
    ///
    /// [BigQuery](https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#syntax_11)
    pub options: Option<Vec<SqlOption>>,
    /// Connection resource for a remote function.
    ///
    /// Example:
    /// ```sql
    /// CREATE FUNCTION foo()
    /// RETURNS FLOAT64
    /// REMOTE WITH CONNECTION us.myconnection
    /// ```
    /// [BigQuery](https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#create_a_remote_function)
    pub remote_connection: Option<ObjectName>,
}

impl fmt::Display for CreateFunction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "CREATE {or_replace}{temp}FUNCTION {if_not_exists}{name}",
            name = self.name,
            temp = if self.temporary { "TEMPORARY " } else { "" },
            or_replace = if self.or_replace { "OR REPLACE " } else { "" },
            if_not_exists = if self.if_not_exists {
                "IF NOT EXISTS "
            } else {
                ""
            },
        )?;
        if let Some(args) = &self.args {
            write!(f, "({})", display_comma_separated(args))?;
        }
        if let Some(return_type) = &self.return_type {
            write!(f, " RETURNS {return_type}")?;
        }
        if let Some(determinism_specifier) = &self.determinism_specifier {
            write!(f, " {determinism_specifier}")?;
        }
        if let Some(language) = &self.language {
            write!(f, " LANGUAGE {language}")?;
        }
        if let Some(behavior) = &self.behavior {
            write!(f, " {behavior}")?;
        }
        if let Some(called_on_null) = &self.called_on_null {
            write!(f, " {called_on_null}")?;
        }
        if let Some(parallel) = &self.parallel {
            write!(f, " {parallel}")?;
        }
        if let Some(remote_connection) = &self.remote_connection {
            write!(f, " REMOTE WITH CONNECTION {remote_connection}")?;
        }
        if let Some(CreateFunctionBody::AsBeforeOptions(function_body)) = &self.function_body {
            write!(f, " AS {function_body}")?;
        }
        if let Some(CreateFunctionBody::Return(function_body)) = &self.function_body {
            write!(f, " RETURN {function_body}")?;
        }
        if let Some(using) = &self.using {
            write!(f, " {using}")?;
        }
        if let Some(options) = &self.options {
            write!(
                f,
                " OPTIONS({})",
                display_comma_separated(options.as_slice())
            )?;
        }
        if let Some(CreateFunctionBody::AsAfterOptions(function_body)) = &self.function_body {
            write!(f, " AS {function_body}")?;
        }
        Ok(())
    }
}
