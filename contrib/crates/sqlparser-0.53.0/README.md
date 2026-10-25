<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Extensible SQL Lexer and Parser for Rust

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Version](https://img.shields.io/crates/v/sqlparser.svg)](https://crates.io/crates/sqlparser)
[![Build Status](https://github.com/sqlparser-rs/sqlparser-rs/workflows/Rust/badge.svg?branch=main)](https://github.com/sqlparser-rs/sqlparser-rs/actions?query=workflow%3ARust+branch%3Amain)
[![Coverage Status](https://coveralls.io/repos/github/sqlparser-rs/sqlparser-rs/badge.svg?branch=main)](https://coveralls.io/github/sqlparser-rs/sqlparser-rs?branch=main)
[![Gitter Chat](https://badges.gitter.im/sqlparser-rs/community.svg)](https://gitter.im/sqlparser-rs/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

This crate contains a lexer and parser for SQL that conforms with the
[ANSI/ISO SQL standard][sql-standard] and other dialects. This crate
is used as a foundation for SQL query engines, vendor-specific
parsers, and various SQL analysis.

## Example

To parse a simple `SELECT` statement:

```rust
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

let sql = "SELECT a, b, 123, myfunc(b) \
           FROM table_1 \
           WHERE a > b AND b < 100 \
           ORDER BY a DESC, b";

let dialect = GenericDialect {}; // or AnsiDialect, or your own dialect ...

let ast = Parser::parse_sql(&dialect, sql).unwrap();

println!("AST: {:?}", ast);
```

This outputs

```rust
AST: [Query(Query { ctes: [], body: Select(Select { distinct: false, projection: [UnnamedExpr(Identifier("a")), UnnamedExpr(Identifier("b")), UnnamedExpr(Value(Long(123))), UnnamedExpr(Function(Function { name: ObjectName(["myfunc"]), args: [Identifier("b")], filter: None, over: None, distinct: false }))], from: [TableWithJoins { relation: Table { name: ObjectName(["table_1"]), alias: None, args: [], with_hints: [] }, joins: [] }], selection: Some(BinaryOp { left: BinaryOp { left: Identifier("a"), op: Gt, right: Identifier("b") }, op: And, right: BinaryOp { left: Identifier("b"), op: Lt, right: Value(Long(100)) } }), group_by: [], having: None }), order_by: [OrderByExpr { expr: Identifier("a"), asc: Some(false) }, OrderByExpr { expr: Identifier("b"), asc: None }], limit: None, offset: None, fetch: None })]
```


## Features

The following optional [crate  features](https://doc.rust-lang.org/cargo/reference/features.html) are available:

* `serde`: Adds [Serde](https://serde.rs/) support by implementing  `Serialize` and `Deserialize` for all AST nodes.
* `visitor`: Adds a `Visitor` capable of recursively walking the AST tree.


## Syntax vs Semantics

This crate provides only a syntax parser, and tries to avoid applying
any SQL semantics, and accepts queries that specific databases would
reject, even when using that Database's specific `Dialect`. For
example, `CREATE TABLE(x int, x int)` is accepted by this crate, even
though most SQL engines will reject this statement due to the repeated
column name `x`.

This crate avoids semantic analysis because it varies drastically
between dialects and implementations. If you want to do semantic
analysis, feel free to use this project as a base.

## Preserves Syntax Round Trip 

This crate allows users to recover the original SQL text (with comments removed,
normalized whitespace and keyword capitalization), which is useful for tools
that analyze and manipulate SQL.

This means that other than comments, whitespace and the capitalization of
keywords, the following should hold true for all SQL:

```rust
// Parse SQL
let ast = Parser::parse_sql(&GenericDialect, sql).unwrap();

// The original SQL text can be generated from the AST
assert_eq!(ast[0].to_string(), sql);
```

There are still some cases in this crate where different SQL with seemingly
similar semantics are represented with the same AST. We welcome PRs to fix such
issues and distinguish different syntaxes in the AST.


## Source Locations (Work in Progress)

This crate allows recovering source locations from AST nodes via the [Spanned]
trait, which can be used for advanced diagnostics tooling. Note that this
feature is a work in progress and many nodes report missing or inaccurate spans.
Please see [this ticket] for information on how to contribute missing
improvements.

[Spanned]: https://docs.rs/sqlparser/latest/sqlparser/ast/trait.Spanned.html
[this ticket]: https://github.com/apache/datafusion-sqlparser-rs/issues/1548

```rust
// Parse SQL
let ast = Parser::parse_sql(&GenericDialect, "SELECT A FROM B").unwrap();

// The source span can be retrieved with start and end locations
assert_eq!(ast[0].span(), Span {
  start: Location::of(1, 1),
  end: Location::of(1, 16),
});
```

## SQL compliance

SQL was first standardized in 1987, and revisions of the standard have been
published regularly since. Most revisions have added significant new features to
the language, and as a result no database claims to support the full breadth of
features. This parser currently supports most of the SQL-92 syntax, plus some
syntax from newer versions that have been explicitly requested, plus various
other dialect-specific syntax. Whenever possible, the [online SQL:2016
grammar][sql-2016-grammar] is used to guide what syntax to accept.

Unfortunately, stating anything more specific about compliance is difficult.
There is no publicly available test suite that can assess compliance
automatically, and doing so manually would strain the project's limited
resources. Still, we are interested in eventually supporting the full SQL
dialect, and we are slowly building out our own test suite.

If you are assessing whether this project will be suitable for your needs,
you'll likely need to experimentally verify whether it supports the subset of
SQL that you need. Please file issues about any unsupported queries that you
discover. Doing so helps us prioritize support for the portions of the standard
that are actually used. Note that if you urgently need support for a feature,
you will likely need to write the implementation yourself. See the
[Contributing](#Contributing) section for details.

## Command line

This crate contains a CLI program that can parse a file and dump the results as JSON:
```
$ cargo run --features json_example --example cli FILENAME.sql [--dialectname]
```

## Users

This parser is currently being used by the [DataFusion] query engine, [LocustDB],
[Ballista], [GlueSQL], [Opteryx], [Polars], [PRQL], [Qrlew], [JumpWire], and [ParadeDB].

If your project is using sqlparser-rs feel free to make a PR to add it
to this list.

## Design

The core expression parser uses the [Pratt Parser] design, which is a top-down
operator-precedence (TDOP) parser, while the surrounding SQL statement parser is
a traditional, hand-written recursive descent parser. Eli Bendersky has a good
[tutorial on TDOP parsers][tdop-tutorial], if you are interested in learning
more about the technique.

We are a fan of this design pattern over parser generators for the following
reasons:

- Code is simple to write and can be concise and elegant
- Performance is generally better than code generated by parser generators
- Debugging is much easier with hand-written code
- It is far easier to extend and make dialect-specific extensions
  compared to using a parser generator

### Supporting custom SQL dialects

This is a work in progress, but we have some notes on [writing a custom SQL
parser](docs/custom_sql_parser.md).

## Contributing

Contributions are highly encouraged! However, the bandwidth we have to
maintain this crate is limited. Please read the following sections carefully.

### New Syntax

The most commonly accepted PRs add support for or fix a bug in a feature in the
SQL standard, or a popular RDBMS, such as Microsoft SQL
Server or PostgreSQL, will likely be accepted after a brief
review.  Any SQL feature that is dialect specific should be parsed by *both* the relevant [`Dialect`] 
as well as [`GenericDialect`].

### Major API Changes

The current maintainers do not plan for any substantial changes to
this crate's API. PRs proposing major refactors
are not likely to be accepted.

### Testing

While we hope to review PRs in a reasonably
timely fashion, it may take a week or more. In order to speed the process,
please make sure the PR passes all CI checks, and includes tests
demonstrating your code works as intended (and to avoid
regressions). Remember to also test error paths.

PRs without tests will not be reviewed or merged.  Since the CI
ensures that `cargo test`, `cargo fmt`, and `cargo clippy`, pass you
should likely to run all three commands locally before submitting
your PR.

### Filing Issues

If you are unable to submit a patch, feel free to file an issue instead. Please
try to include:

  * some representative examples of the syntax you wish to support or fix;
  * the relevant bits of the [SQL grammar][sql-2016-grammar], if the syntax is
    part of SQL:2016; and
  * links to documentation for the feature for a few of the most popular
    databases that support it.

Unfortunately, if you need support for a feature, you will likely need to implement
it yourself, or file a well enough described ticket that another member of the community can do so.
Our goal as maintainers is to facilitate the integration
of various features from various contributors, but not to provide the
implementations ourselves, as we simply don't have the resources.

### Benchmarking

There are several micro benchmarks in the `sqlparser_bench` directory. 
You can run them with:

```
git checkout main
cd sqlparser_bench
cargo bench
git checkout <your branch>
cargo bench
```

## Licensing

All code in this repository is licensed under the [Apache Software License 2.0](LICENSE.txt).

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
licensed as above, without any additional terms or conditions.


[tdop-tutorial]: https://eli.thegreenplace.net/2010/01/02/top-down-operator-precedence-parsing
[`cargo fmt`]: https://github.com/rust-lang/rustfmt#on-the-stable-toolchain
[current issues]: https://github.com/sqlparser-rs/sqlparser-rs/issues
[DataFusion]: https://github.com/apache/arrow-datafusion
[LocustDB]: https://github.com/cswinter/LocustDB
[Ballista]: https://github.com/apache/arrow-ballista
[GlueSQL]: https://github.com/gluesql/gluesql
[Opteryx]: https://github.com/mabel-dev/opteryx
[Polars]: https://pola.rs/
[PRQL]: https://github.com/PRQL/prql
[Qrlew]: https://github.com/Qrlew/qrlew
[JumpWire]: https://github.com/extragoodlabs/jumpwire
[ParadeDB]: https://github.com/paradedb/paradedb
[Pratt Parser]: https://tdop.github.io/
[sql-2016-grammar]: https://jakewheat.github.io/sql-overview/sql-2016-foundation-grammar.html
[sql-standard]: https://en.wikipedia.org/wiki/ISO/IEC_9075
[`Dialect`]: https://docs.rs/sqlparser/latest/sqlparser/dialect/trait.Dialect.html
[`GenericDialect`]: https://docs.rs/sqlparser/latest/sqlparser/dialect/struct.GenericDialect.html
