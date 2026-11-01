# [Rust Skiplist](http://www.jpellis.me/projects/rust-skiplist)

[![crates.io](https://img.shields.io/crates/v/skiplist.svg)](https://crates.io/crates/skiplist)
[![crates.io](https://img.shields.io/crates/d/skiplist.svg)](https://crates.io/crates/skiplist)
[![Codecov branch](https://img.shields.io/codecov/c/github/JP-Ellis/rust-skiplist/master)](https://codecov.io/gh/JP-Ellis/rust-skiplist)
[![Build Status](https://img.shields.io/github/actions/workflow/status/JP-Ellis/rust-skiplist/rust.yml?branch=master)](https://github.com/JP-Ellis/rust-skiplist/actions)

A [skiplist](http://en.wikipedia.org/wiki/Skip_list) provides a way of storing
data with `log(i)` access, insertion and removal for an element in the `i`th
position.

There are three kinds of collections defined here:

-   **SkipList** This behaves like nearly any other double-ended list.
-   **OrderedSkipList** Ensures that the elements are always sorted. Still allows
  for access nodes at a given index.
-   **SkipMap** A map in which the keys are ordered.

Documentation can be found on [docs.rs](https://docs.rs/skiplist) and the cargo
crate can be found on [crates.io](https://crates.io/crates/skiplist).
