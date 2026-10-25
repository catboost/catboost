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

//! Recursive visitors for ast Nodes. See [`Visitor`] for more details.

use crate::ast::{Expr, ObjectName, Query, Statement, TableFactor};
use core::ops::ControlFlow;

/// A type that can be visited by a [`Visitor`]. See [`Visitor`] for
/// recursively visiting parsed SQL statements.
///
/// # Note
///
/// This trait should be automatically derived for sqlparser AST nodes
/// using the [Visit](sqlparser_derive::Visit) proc macro.
///
/// ```text
/// #[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
/// ```
pub trait Visit {
    fn visit<V: Visitor>(&self, visitor: &mut V) -> ControlFlow<V::Break>;
}

/// A type that can be visited by a [`VisitorMut`]. See [`VisitorMut`] for
/// recursively visiting parsed SQL statements.
///
/// # Note
///
/// This trait should be automatically derived for sqlparser AST nodes
/// using the [VisitMut](sqlparser_derive::VisitMut) proc macro.
///
/// ```text
/// #[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
/// ```
pub trait VisitMut {
    fn visit<V: VisitorMut>(&mut self, visitor: &mut V) -> ControlFlow<V::Break>;
}

impl<T: Visit> Visit for Option<T> {
    fn visit<V: Visitor>(&self, visitor: &mut V) -> ControlFlow<V::Break> {
        if let Some(s) = self {
            s.visit(visitor)?;
        }
        ControlFlow::Continue(())
    }
}

impl<T: Visit> Visit for Vec<T> {
    fn visit<V: Visitor>(&self, visitor: &mut V) -> ControlFlow<V::Break> {
        for v in self {
            v.visit(visitor)?;
        }
        ControlFlow::Continue(())
    }
}

impl<T: Visit> Visit for Box<T> {
    fn visit<V: Visitor>(&self, visitor: &mut V) -> ControlFlow<V::Break> {
        T::visit(self, visitor)
    }
}

impl<T: VisitMut> VisitMut for Option<T> {
    fn visit<V: VisitorMut>(&mut self, visitor: &mut V) -> ControlFlow<V::Break> {
        if let Some(s) = self {
            s.visit(visitor)?;
        }
        ControlFlow::Continue(())
    }
}

impl<T: VisitMut> VisitMut for Vec<T> {
    fn visit<V: VisitorMut>(&mut self, visitor: &mut V) -> ControlFlow<V::Break> {
        for v in self {
            v.visit(visitor)?;
        }
        ControlFlow::Continue(())
    }
}

impl<T: VisitMut> VisitMut for Box<T> {
    fn visit<V: VisitorMut>(&mut self, visitor: &mut V) -> ControlFlow<V::Break> {
        T::visit(self, visitor)
    }
}

macro_rules! visit_noop {
    ($($t:ty),+) => {
        $(impl Visit for $t {
            fn visit<V: Visitor>(&self, _visitor: &mut V) -> ControlFlow<V::Break> {
               ControlFlow::Continue(())
            }
        })+
        $(impl VisitMut for $t {
            fn visit<V: VisitorMut>(&mut self, _visitor: &mut V) -> ControlFlow<V::Break> {
               ControlFlow::Continue(())
            }
        })+
    };
}

visit_noop!(u8, u16, u32, u64, i8, i16, i32, i64, char, bool, String);

#[cfg(feature = "bigdecimal")]
visit_noop!(bigdecimal::BigDecimal);

/// A visitor that can be used to walk an AST tree.
///
/// `pre_visit_` methods are invoked before visiting all children of the
/// node and `post_visit_` methods are invoked after visiting all
/// children of the node.
///
/// # See also
///
/// These methods provide a more concise way of visiting nodes of a certain type:
/// * [visit_relations]
/// * [visit_expressions]
/// * [visit_statements]
///
/// # Example
/// ```
/// # use sqlparser::parser::Parser;
/// # use sqlparser::dialect::GenericDialect;
/// # use sqlparser::ast::{Visit, Visitor, ObjectName, Expr};
/// # use core::ops::ControlFlow;
/// // A structure that records statements and relations
/// #[derive(Default)]
/// struct V {
///    visited: Vec<String>,
/// }
///
/// // Visit relations and exprs before children are visited (depth first walk)
/// // Note you can also visit statements and visit exprs after children have been visited
/// impl Visitor for V {
///   type Break = ();
///
///   fn pre_visit_relation(&mut self, relation: &ObjectName) -> ControlFlow<Self::Break> {
///     self.visited.push(format!("PRE: RELATION: {}", relation));
///     ControlFlow::Continue(())
///   }
///
///   fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<Self::Break> {
///     self.visited.push(format!("PRE: EXPR: {}", expr));
///     ControlFlow::Continue(())
///   }
/// }
///
/// let sql = "SELECT a FROM foo where x IN (SELECT y FROM bar)";
/// let statements = Parser::parse_sql(&GenericDialect{}, sql)
///    .unwrap();
///
/// // Drive the visitor through the AST
/// let mut visitor = V::default();
/// statements.visit(&mut visitor);
///
/// // The visitor has visited statements and expressions in pre-traversal order
/// let expected : Vec<_> = [
///   "PRE: EXPR: a",
///   "PRE: RELATION: foo",
///   "PRE: EXPR: x IN (SELECT y FROM bar)",
///   "PRE: EXPR: x",
///   "PRE: EXPR: y",
///   "PRE: RELATION: bar",
/// ]
///   .into_iter().map(|s| s.to_string()).collect();
///
/// assert_eq!(visitor.visited, expected);
/// ```
pub trait Visitor {
    /// Type returned when the recursion returns early.
    type Break;

    /// Invoked for any queries that appear in the AST before visiting children
    fn pre_visit_query(&mut self, _query: &Query) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    /// Invoked for any queries that appear in the AST after visiting children
    fn post_visit_query(&mut self, _query: &Query) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    /// Invoked for any relations (e.g. tables) that appear in the AST before visiting children
    fn pre_visit_relation(&mut self, _relation: &ObjectName) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    /// Invoked for any relations (e.g. tables) that appear in the AST after visiting children
    fn post_visit_relation(&mut self, _relation: &ObjectName) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    /// Invoked for any table factors that appear in the AST before visiting children
    fn pre_visit_table_factor(&mut self, _table_factor: &TableFactor) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    /// Invoked for any table factors that appear in the AST after visiting children
    fn post_visit_table_factor(&mut self, _table_factor: &TableFactor) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    /// Invoked for any expressions that appear in the AST before visiting children
    fn pre_visit_expr(&mut self, _expr: &Expr) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    /// Invoked for any expressions that appear in the AST
    fn post_visit_expr(&mut self, _expr: &Expr) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    /// Invoked for any statements that appear in the AST before visiting children
    fn pre_visit_statement(&mut self, _statement: &Statement) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    /// Invoked for any statements that appear in the AST after visiting children
    fn post_visit_statement(&mut self, _statement: &Statement) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
}

/// A visitor that can be used to mutate an AST tree.
///
/// `pre_visit_` methods are invoked before visiting all children of the
/// node and `post_visit_` methods are invoked after visiting all
/// children of the node.
///
/// # See also
///
/// These methods provide a more concise way of visiting nodes of a certain type:
/// * [visit_relations_mut]
/// * [visit_expressions_mut]
/// * [visit_statements_mut]
///
/// # Example
/// ```
/// # use sqlparser::parser::Parser;
/// # use sqlparser::dialect::GenericDialect;
/// # use sqlparser::ast::{VisitMut, VisitorMut, ObjectName, Expr, Ident};
/// # use core::ops::ControlFlow;
///
/// // A visitor that replaces "to_replace" with "replaced" in all expressions
/// struct Replacer;
///
/// // Visit each expression after its children have been visited
/// impl VisitorMut for Replacer {
///   type Break = ();
///
///   fn post_visit_expr(&mut self, expr: &mut Expr) -> ControlFlow<Self::Break> {
///     if let Expr::Identifier(Ident{ value, ..}) = expr {
///         *value = value.replace("to_replace", "replaced")
///     }
///     ControlFlow::Continue(())
///   }
/// }
///
/// let sql = "SELECT to_replace FROM foo where to_replace IN (SELECT to_replace FROM bar)";
/// let mut statements = Parser::parse_sql(&GenericDialect{}, sql).unwrap();
///
/// // Drive the visitor through the AST
/// statements.visit(&mut Replacer);
///
/// assert_eq!(statements[0].to_string(), "SELECT replaced FROM foo WHERE replaced IN (SELECT replaced FROM bar)");
/// ```
pub trait VisitorMut {
    /// Type returned when the recursion returns early.
    type Break;

    /// Invoked for any queries that appear in the AST before visiting children
    fn pre_visit_query(&mut self, _query: &mut Query) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    /// Invoked for any queries that appear in the AST after visiting children
    fn post_visit_query(&mut self, _query: &mut Query) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    /// Invoked for any relations (e.g. tables) that appear in the AST before visiting children
    fn pre_visit_relation(&mut self, _relation: &mut ObjectName) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    /// Invoked for any relations (e.g. tables) that appear in the AST after visiting children
    fn post_visit_relation(&mut self, _relation: &mut ObjectName) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    /// Invoked for any table factors that appear in the AST before visiting children
    fn pre_visit_table_factor(
        &mut self,
        _table_factor: &mut TableFactor,
    ) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    /// Invoked for any table factors that appear in the AST after visiting children
    fn post_visit_table_factor(
        &mut self,
        _table_factor: &mut TableFactor,
    ) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    /// Invoked for any expressions that appear in the AST before visiting children
    fn pre_visit_expr(&mut self, _expr: &mut Expr) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    /// Invoked for any expressions that appear in the AST
    fn post_visit_expr(&mut self, _expr: &mut Expr) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    /// Invoked for any statements that appear in the AST before visiting children
    fn pre_visit_statement(&mut self, _statement: &mut Statement) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    /// Invoked for any statements that appear in the AST after visiting children
    fn post_visit_statement(&mut self, _statement: &mut Statement) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
}

struct RelationVisitor<F>(F);

impl<E, F: FnMut(&ObjectName) -> ControlFlow<E>> Visitor for RelationVisitor<F> {
    type Break = E;

    fn pre_visit_relation(&mut self, relation: &ObjectName) -> ControlFlow<Self::Break> {
        self.0(relation)
    }
}

impl<E, F: FnMut(&mut ObjectName) -> ControlFlow<E>> VisitorMut for RelationVisitor<F> {
    type Break = E;

    fn post_visit_relation(&mut self, relation: &mut ObjectName) -> ControlFlow<Self::Break> {
        self.0(relation)
    }
}

/// Invokes the provided closure on all relations (e.g. table names) present in `v`
///
/// # Example
/// ```
/// # use sqlparser::parser::Parser;
/// # use sqlparser::dialect::GenericDialect;
/// # use sqlparser::ast::{visit_relations};
/// # use core::ops::ControlFlow;
/// let sql = "SELECT a FROM foo where x IN (SELECT y FROM bar)";
/// let statements = Parser::parse_sql(&GenericDialect{}, sql)
///    .unwrap();
///
/// // visit statements, capturing relations (table names)
/// let mut visited = vec![];
/// visit_relations(&statements, |relation| {
///   visited.push(format!("RELATION: {}", relation));
///   ControlFlow::<()>::Continue(())
/// });
///
/// let expected : Vec<_> = [
///   "RELATION: foo",
///   "RELATION: bar",
/// ]
///   .into_iter().map(|s| s.to_string()).collect();
///
/// assert_eq!(visited, expected);
/// ```
pub fn visit_relations<V, E, F>(v: &V, f: F) -> ControlFlow<E>
where
    V: Visit,
    F: FnMut(&ObjectName) -> ControlFlow<E>,
{
    let mut visitor = RelationVisitor(f);
    v.visit(&mut visitor)?;
    ControlFlow::Continue(())
}

/// Invokes the provided closure with a mutable reference to all relations (e.g. table names)
/// present in `v`.
///
/// When the closure mutates its argument, the new mutated relation will not be visited again.
///
/// # Example
/// ```
/// # use sqlparser::parser::Parser;
/// # use sqlparser::dialect::GenericDialect;
/// # use sqlparser::ast::{ObjectName, visit_relations_mut};
/// # use core::ops::ControlFlow;
/// let sql = "SELECT a FROM foo";
/// let mut statements = Parser::parse_sql(&GenericDialect{}, sql)
///    .unwrap();
///
/// // visit statements, renaming table foo to bar
/// visit_relations_mut(&mut statements, |table| {
///   table.0[0].value = table.0[0].value.replace("foo", "bar");
///   ControlFlow::<()>::Continue(())
/// });
///
/// assert_eq!(statements[0].to_string(), "SELECT a FROM bar");
/// ```
pub fn visit_relations_mut<V, E, F>(v: &mut V, f: F) -> ControlFlow<E>
where
    V: VisitMut,
    F: FnMut(&mut ObjectName) -> ControlFlow<E>,
{
    let mut visitor = RelationVisitor(f);
    v.visit(&mut visitor)?;
    ControlFlow::Continue(())
}

struct ExprVisitor<F>(F);

impl<E, F: FnMut(&Expr) -> ControlFlow<E>> Visitor for ExprVisitor<F> {
    type Break = E;

    fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<Self::Break> {
        self.0(expr)
    }
}

impl<E, F: FnMut(&mut Expr) -> ControlFlow<E>> VisitorMut for ExprVisitor<F> {
    type Break = E;

    fn post_visit_expr(&mut self, expr: &mut Expr) -> ControlFlow<Self::Break> {
        self.0(expr)
    }
}

/// Invokes the provided closure on all expressions (e.g. `1 + 2`) present in `v`
///
/// # Example
/// ```
/// # use sqlparser::parser::Parser;
/// # use sqlparser::dialect::GenericDialect;
/// # use sqlparser::ast::{visit_expressions};
/// # use core::ops::ControlFlow;
/// let sql = "SELECT a FROM foo where x IN (SELECT y FROM bar)";
/// let statements = Parser::parse_sql(&GenericDialect{}, sql)
///    .unwrap();
///
/// // visit all expressions
/// let mut visited = vec![];
/// visit_expressions(&statements, |expr| {
///   visited.push(format!("EXPR: {}", expr));
///   ControlFlow::<()>::Continue(())
/// });
///
/// let expected : Vec<_> = [
///   "EXPR: a",
///   "EXPR: x IN (SELECT y FROM bar)",
///   "EXPR: x",
///   "EXPR: y",
/// ]
///   .into_iter().map(|s| s.to_string()).collect();
///
/// assert_eq!(visited, expected);
/// ```
pub fn visit_expressions<V, E, F>(v: &V, f: F) -> ControlFlow<E>
where
    V: Visit,
    F: FnMut(&Expr) -> ControlFlow<E>,
{
    let mut visitor = ExprVisitor(f);
    v.visit(&mut visitor)?;
    ControlFlow::Continue(())
}

/// Invokes the provided closure iteratively with a mutable reference to all expressions
/// present in `v`.
///
/// This performs a depth-first search, so if the closure mutates the expression
///
/// # Example
///
/// ## Remove all select limits in sub-queries
/// ```
/// # use sqlparser::parser::Parser;
/// # use sqlparser::dialect::GenericDialect;
/// # use sqlparser::ast::{Expr, visit_expressions_mut, visit_statements_mut};
/// # use core::ops::ControlFlow;
/// let sql = "SELECT (SELECT y FROM z LIMIT 9) FROM t LIMIT 3";
/// let mut statements = Parser::parse_sql(&GenericDialect{}, sql).unwrap();
///
/// // Remove all select limits in sub-queries
/// visit_expressions_mut(&mut statements, |expr| {
///   if let Expr::Subquery(q) = expr {
///      q.limit = None
///   }
///   ControlFlow::<()>::Continue(())
/// });
///
/// assert_eq!(statements[0].to_string(), "SELECT (SELECT y FROM z) FROM t LIMIT 3");
/// ```
///
/// ## Wrap column name in function call
///
/// This demonstrates how to effectively replace an expression with another more complicated one
/// that references the original. This example avoids unnecessary allocations by using the
/// [`std::mem`] family of functions.
///
/// ```
/// # use sqlparser::parser::Parser;
/// # use sqlparser::dialect::GenericDialect;
/// # use sqlparser::ast::*;
/// # use core::ops::ControlFlow;
/// let sql = "SELECT x, y FROM t";
/// let mut statements = Parser::parse_sql(&GenericDialect{}, sql).unwrap();
///
/// visit_expressions_mut(&mut statements, |expr| {
///   if matches!(expr, Expr::Identifier(col_name) if col_name.value == "x") {
///     let old_expr = std::mem::replace(expr, Expr::Value(Value::Null));
///     *expr = Expr::Function(Function {
///           name: ObjectName(vec![Ident::new("f")]),
///           uses_odbc_syntax: false,
///           args: FunctionArguments::List(FunctionArgumentList {
///               duplicate_treatment: None,
///               args: vec![FunctionArg::Unnamed(FunctionArgExpr::Expr(old_expr))],
///               clauses: vec![],
///           }),
///           null_treatment: None,
///           filter: None,
///           over: None,
///           parameters: FunctionArguments::None,
///           within_group: vec![],
///      });
///   }
///   ControlFlow::<()>::Continue(())
/// });
///
/// assert_eq!(statements[0].to_string(), "SELECT f(x), y FROM t");
/// ```
pub fn visit_expressions_mut<V, E, F>(v: &mut V, f: F) -> ControlFlow<E>
where
    V: VisitMut,
    F: FnMut(&mut Expr) -> ControlFlow<E>,
{
    v.visit(&mut ExprVisitor(f))?;
    ControlFlow::Continue(())
}

struct StatementVisitor<F>(F);

impl<E, F: FnMut(&Statement) -> ControlFlow<E>> Visitor for StatementVisitor<F> {
    type Break = E;

    fn pre_visit_statement(&mut self, statement: &Statement) -> ControlFlow<Self::Break> {
        self.0(statement)
    }
}

impl<E, F: FnMut(&mut Statement) -> ControlFlow<E>> VisitorMut for StatementVisitor<F> {
    type Break = E;

    fn post_visit_statement(&mut self, statement: &mut Statement) -> ControlFlow<Self::Break> {
        self.0(statement)
    }
}

/// Invokes the provided closure iteratively with a mutable reference to all statements
/// present in `v` (e.g. `SELECT`, `CREATE TABLE`, etc).
///
/// # Example
/// ```
/// # use sqlparser::parser::Parser;
/// # use sqlparser::dialect::GenericDialect;
/// # use sqlparser::ast::{visit_statements};
/// # use core::ops::ControlFlow;
/// let sql = "SELECT a FROM foo where x IN (SELECT y FROM bar); CREATE TABLE baz(q int)";
/// let statements = Parser::parse_sql(&GenericDialect{}, sql)
///    .unwrap();
///
/// // visit all statements
/// let mut visited = vec![];
/// visit_statements(&statements, |stmt| {
///   visited.push(format!("STATEMENT: {}", stmt));
///   ControlFlow::<()>::Continue(())
/// });
///
/// let expected : Vec<_> = [
///   "STATEMENT: SELECT a FROM foo WHERE x IN (SELECT y FROM bar)",
///   "STATEMENT: CREATE TABLE baz (q INT)"
/// ]
///   .into_iter().map(|s| s.to_string()).collect();
///
/// assert_eq!(visited, expected);
/// ```
pub fn visit_statements<V, E, F>(v: &V, f: F) -> ControlFlow<E>
where
    V: Visit,
    F: FnMut(&Statement) -> ControlFlow<E>,
{
    let mut visitor = StatementVisitor(f);
    v.visit(&mut visitor)?;
    ControlFlow::Continue(())
}

/// Invokes the provided closure on all statements (e.g. `SELECT`, `CREATE TABLE`, etc) present in `v`
///
/// # Example
/// ```
/// # use sqlparser::parser::Parser;
/// # use sqlparser::dialect::GenericDialect;
/// # use sqlparser::ast::{Statement, visit_statements_mut};
/// # use core::ops::ControlFlow;
/// let sql = "SELECT x FROM foo LIMIT 9+$limit; SELECT * FROM t LIMIT f()";
/// let mut statements = Parser::parse_sql(&GenericDialect{}, sql).unwrap();
///
/// // Remove all select limits in outer statements (not in sub-queries)
/// visit_statements_mut(&mut statements, |stmt| {
///   if let Statement::Query(q) = stmt {
///      q.limit = None
///   }
///   ControlFlow::<()>::Continue(())
/// });
///
/// assert_eq!(statements[0].to_string(), "SELECT x FROM foo");
/// assert_eq!(statements[1].to_string(), "SELECT * FROM t");
/// ```
pub fn visit_statements_mut<V, E, F>(v: &mut V, f: F) -> ControlFlow<E>
where
    V: VisitMut,
    F: FnMut(&mut Statement) -> ControlFlow<E>,
{
    v.visit(&mut StatementVisitor(f))?;
    ControlFlow::Continue(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::GenericDialect;
    use crate::parser::Parser;
    use crate::tokenizer::Tokenizer;

    #[derive(Default)]
    struct TestVisitor {
        visited: Vec<String>,
    }

    impl Visitor for TestVisitor {
        type Break = ();

        /// Invoked for any queries that appear in the AST before visiting children
        fn pre_visit_query(&mut self, query: &Query) -> ControlFlow<Self::Break> {
            self.visited.push(format!("PRE: QUERY: {query}"));
            ControlFlow::Continue(())
        }

        /// Invoked for any queries that appear in the AST after visiting children
        fn post_visit_query(&mut self, query: &Query) -> ControlFlow<Self::Break> {
            self.visited.push(format!("POST: QUERY: {query}"));
            ControlFlow::Continue(())
        }

        fn pre_visit_relation(&mut self, relation: &ObjectName) -> ControlFlow<Self::Break> {
            self.visited.push(format!("PRE: RELATION: {relation}"));
            ControlFlow::Continue(())
        }

        fn post_visit_relation(&mut self, relation: &ObjectName) -> ControlFlow<Self::Break> {
            self.visited.push(format!("POST: RELATION: {relation}"));
            ControlFlow::Continue(())
        }

        fn pre_visit_table_factor(
            &mut self,
            table_factor: &TableFactor,
        ) -> ControlFlow<Self::Break> {
            self.visited
                .push(format!("PRE: TABLE FACTOR: {table_factor}"));
            ControlFlow::Continue(())
        }

        fn post_visit_table_factor(
            &mut self,
            table_factor: &TableFactor,
        ) -> ControlFlow<Self::Break> {
            self.visited
                .push(format!("POST: TABLE FACTOR: {table_factor}"));
            ControlFlow::Continue(())
        }

        fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<Self::Break> {
            self.visited.push(format!("PRE: EXPR: {expr}"));
            ControlFlow::Continue(())
        }

        fn post_visit_expr(&mut self, expr: &Expr) -> ControlFlow<Self::Break> {
            self.visited.push(format!("POST: EXPR: {expr}"));
            ControlFlow::Continue(())
        }

        fn pre_visit_statement(&mut self, statement: &Statement) -> ControlFlow<Self::Break> {
            self.visited.push(format!("PRE: STATEMENT: {statement}"));
            ControlFlow::Continue(())
        }

        fn post_visit_statement(&mut self, statement: &Statement) -> ControlFlow<Self::Break> {
            self.visited.push(format!("POST: STATEMENT: {statement}"));
            ControlFlow::Continue(())
        }
    }

    fn do_visit(sql: &str) -> Vec<String> {
        let dialect = GenericDialect {};
        let tokens = Tokenizer::new(&dialect, sql).tokenize().unwrap();
        let s = Parser::new(&dialect)
            .with_tokens(tokens)
            .parse_statement()
            .unwrap();

        let mut visitor = TestVisitor::default();
        s.visit(&mut visitor);
        visitor.visited
    }

    #[test]
    fn test_sql() {
        let tests = vec![
            (
                "SELECT * from table_name as my_table",
                vec![
                    "PRE: STATEMENT: SELECT * FROM table_name AS my_table",
                    "PRE: QUERY: SELECT * FROM table_name AS my_table",
                    "PRE: TABLE FACTOR: table_name AS my_table",
                    "PRE: RELATION: table_name",
                    "POST: RELATION: table_name",
                    "POST: TABLE FACTOR: table_name AS my_table",
                    "POST: QUERY: SELECT * FROM table_name AS my_table",
                    "POST: STATEMENT: SELECT * FROM table_name AS my_table",
                ],
            ),
            (
                "SELECT * from t1 join t2 on t1.id = t2.t1_id",
                vec![
                    "PRE: STATEMENT: SELECT * FROM t1 JOIN t2 ON t1.id = t2.t1_id",
                    "PRE: QUERY: SELECT * FROM t1 JOIN t2 ON t1.id = t2.t1_id",
                    "PRE: TABLE FACTOR: t1",
                    "PRE: RELATION: t1",
                    "POST: RELATION: t1",
                    "POST: TABLE FACTOR: t1",
                    "PRE: TABLE FACTOR: t2",
                    "PRE: RELATION: t2",
                    "POST: RELATION: t2",
                    "POST: TABLE FACTOR: t2",
                    "PRE: EXPR: t1.id = t2.t1_id",
                    "PRE: EXPR: t1.id",
                    "POST: EXPR: t1.id",
                    "PRE: EXPR: t2.t1_id",
                    "POST: EXPR: t2.t1_id",
                    "POST: EXPR: t1.id = t2.t1_id",
                    "POST: QUERY: SELECT * FROM t1 JOIN t2 ON t1.id = t2.t1_id",
                    "POST: STATEMENT: SELECT * FROM t1 JOIN t2 ON t1.id = t2.t1_id",
                ],
            ),
            (
                "SELECT * from t1 where EXISTS(SELECT column from t2)",
                vec![
                    "PRE: STATEMENT: SELECT * FROM t1 WHERE EXISTS (SELECT column FROM t2)",
                    "PRE: QUERY: SELECT * FROM t1 WHERE EXISTS (SELECT column FROM t2)",
                    "PRE: TABLE FACTOR: t1",
                    "PRE: RELATION: t1",
                    "POST: RELATION: t1",
                    "POST: TABLE FACTOR: t1",
                    "PRE: EXPR: EXISTS (SELECT column FROM t2)",
                    "PRE: QUERY: SELECT column FROM t2",
                    "PRE: EXPR: column",
                    "POST: EXPR: column",
                    "PRE: TABLE FACTOR: t2",
                    "PRE: RELATION: t2",
                    "POST: RELATION: t2",
                    "POST: TABLE FACTOR: t2",
                    "POST: QUERY: SELECT column FROM t2",
                    "POST: EXPR: EXISTS (SELECT column FROM t2)",
                    "POST: QUERY: SELECT * FROM t1 WHERE EXISTS (SELECT column FROM t2)",
                    "POST: STATEMENT: SELECT * FROM t1 WHERE EXISTS (SELECT column FROM t2)",
                ],
            ),
            (
                "SELECT * from t1 where EXISTS(SELECT column from t2)",
                vec![
                    "PRE: STATEMENT: SELECT * FROM t1 WHERE EXISTS (SELECT column FROM t2)",
                    "PRE: QUERY: SELECT * FROM t1 WHERE EXISTS (SELECT column FROM t2)",
                    "PRE: TABLE FACTOR: t1",
                    "PRE: RELATION: t1",
                    "POST: RELATION: t1",
                    "POST: TABLE FACTOR: t1",
                    "PRE: EXPR: EXISTS (SELECT column FROM t2)",
                    "PRE: QUERY: SELECT column FROM t2",
                    "PRE: EXPR: column",
                    "POST: EXPR: column",
                    "PRE: TABLE FACTOR: t2",
                    "PRE: RELATION: t2",
                    "POST: RELATION: t2",
                    "POST: TABLE FACTOR: t2",
                    "POST: QUERY: SELECT column FROM t2",
                    "POST: EXPR: EXISTS (SELECT column FROM t2)",
                    "POST: QUERY: SELECT * FROM t1 WHERE EXISTS (SELECT column FROM t2)",
                    "POST: STATEMENT: SELECT * FROM t1 WHERE EXISTS (SELECT column FROM t2)",
                ],
            ),
            (
                "SELECT * from t1 where EXISTS(SELECT column from t2) UNION SELECT * from t3",
                vec![
                    "PRE: STATEMENT: SELECT * FROM t1 WHERE EXISTS (SELECT column FROM t2) UNION SELECT * FROM t3",
                    "PRE: QUERY: SELECT * FROM t1 WHERE EXISTS (SELECT column FROM t2) UNION SELECT * FROM t3",
                    "PRE: TABLE FACTOR: t1",
                    "PRE: RELATION: t1",
                    "POST: RELATION: t1",
                    "POST: TABLE FACTOR: t1",
                    "PRE: EXPR: EXISTS (SELECT column FROM t2)",
                    "PRE: QUERY: SELECT column FROM t2",
                    "PRE: EXPR: column",
                    "POST: EXPR: column",
                    "PRE: TABLE FACTOR: t2",
                    "PRE: RELATION: t2",
                    "POST: RELATION: t2",
                    "POST: TABLE FACTOR: t2",
                    "POST: QUERY: SELECT column FROM t2",
                    "POST: EXPR: EXISTS (SELECT column FROM t2)",
                    "PRE: TABLE FACTOR: t3",
                    "PRE: RELATION: t3",
                    "POST: RELATION: t3",
                    "POST: TABLE FACTOR: t3",
                    "POST: QUERY: SELECT * FROM t1 WHERE EXISTS (SELECT column FROM t2) UNION SELECT * FROM t3",
                    "POST: STATEMENT: SELECT * FROM t1 WHERE EXISTS (SELECT column FROM t2) UNION SELECT * FROM t3",
                ],
            ),
            (
                concat!(
                    "SELECT * FROM monthly_sales ",
                    "PIVOT(SUM(a.amount) FOR a.MONTH IN ('JAN', 'FEB', 'MAR', 'APR')) AS p (c, d) ",
                    "ORDER BY EMPID"
                ),
                vec![
                    "PRE: STATEMENT: SELECT * FROM monthly_sales PIVOT(SUM(a.amount) FOR a.MONTH IN ('JAN', 'FEB', 'MAR', 'APR')) AS p (c, d) ORDER BY EMPID",
                    "PRE: QUERY: SELECT * FROM monthly_sales PIVOT(SUM(a.amount) FOR a.MONTH IN ('JAN', 'FEB', 'MAR', 'APR')) AS p (c, d) ORDER BY EMPID",
                    "PRE: TABLE FACTOR: monthly_sales PIVOT(SUM(a.amount) FOR a.MONTH IN ('JAN', 'FEB', 'MAR', 'APR')) AS p (c, d)",
                    "PRE: TABLE FACTOR: monthly_sales",
                    "PRE: RELATION: monthly_sales",
                    "POST: RELATION: monthly_sales",
                    "POST: TABLE FACTOR: monthly_sales",
                    "PRE: EXPR: SUM(a.amount)",
                    "PRE: EXPR: a.amount",
                    "POST: EXPR: a.amount",
                    "POST: EXPR: SUM(a.amount)",
                    "PRE: EXPR: 'JAN'",
                    "POST: EXPR: 'JAN'",
                    "PRE: EXPR: 'FEB'",
                    "POST: EXPR: 'FEB'",
                    "PRE: EXPR: 'MAR'",
                    "POST: EXPR: 'MAR'",
                    "PRE: EXPR: 'APR'",
                    "POST: EXPR: 'APR'",
                    "POST: TABLE FACTOR: monthly_sales PIVOT(SUM(a.amount) FOR a.MONTH IN ('JAN', 'FEB', 'MAR', 'APR')) AS p (c, d)",
                    "PRE: EXPR: EMPID",
                    "POST: EXPR: EMPID",
                    "POST: QUERY: SELECT * FROM monthly_sales PIVOT(SUM(a.amount) FOR a.MONTH IN ('JAN', 'FEB', 'MAR', 'APR')) AS p (c, d) ORDER BY EMPID",
                    "POST: STATEMENT: SELECT * FROM monthly_sales PIVOT(SUM(a.amount) FOR a.MONTH IN ('JAN', 'FEB', 'MAR', 'APR')) AS p (c, d) ORDER BY EMPID",
                ]
            ),
            (
                "SHOW COLUMNS FROM t1",
                vec![
                    "PRE: STATEMENT: SHOW COLUMNS FROM t1",
                    "PRE: RELATION: t1",
                    "POST: RELATION: t1",
                    "POST: STATEMENT: SHOW COLUMNS FROM t1",
                ],
            ),
        ];
        for (sql, expected) in tests {
            let actual = do_visit(sql);
            let actual: Vec<_> = actual.iter().map(|x| x.as_str()).collect();
            assert_eq!(actual, expected)
        }
    }
}
