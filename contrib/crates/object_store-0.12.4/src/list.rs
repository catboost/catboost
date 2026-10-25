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

//! Paginated Listing

use super::Result;
use crate::ListResult;
use async_trait::async_trait;
use std::borrow::Cow;

/// Options for a paginated list request
#[derive(Debug, Default, Clone)]
pub struct PaginatedListOptions {
    /// Path to start listing from
    ///
    /// Note: Not all stores support this
    pub offset: Option<String>,

    /// A delimiter use to group keys with a common prefix
    ///
    /// Note: Some stores only support `/`
    pub delimiter: Option<Cow<'static, str>>,

    /// The maximum number of paths to return
    pub max_keys: Option<usize>,

    /// A page token from a previous request
    ///
    /// Note: Behaviour is implementation defined if the previous request
    /// used a different prefix or options
    pub page_token: Option<String>,

    /// Implementation-specific extensions. Intended for use by implementations
    /// that need to pass context-specific information (like tracing spans) via trait methods.
    ///
    /// These extensions are ignored entirely by backends offered through this crate.
    pub extensions: http::Extensions,
}

/// A [`ListResult`] with optional pagination token
#[derive(Debug)]
pub struct PaginatedListResult {
    /// The list result
    pub result: ListResult,
    /// If result set truncated, the pagination token to fetch next results
    pub page_token: Option<String>,
}

/// A low-level interface for interacting with paginated listing APIs
///
/// Most use-cases should prefer [`ObjectStore::list`] as this is supported by more
/// backends, including [`LocalFileSystem`], however, [`PaginatedListStore`] can be
/// used where stateless pagination or non-path segment based listing is required
///
/// [`ObjectStore::list`]: crate::ObjectStore::list
/// [`LocalFileSystem`]: crate::local::LocalFileSystem
#[async_trait]
pub trait PaginatedListStore: Send + Sync + 'static {
    /// Perform a paginated list request
    ///
    /// Note: the order of returned objects is not guaranteed and
    /// unlike [`ObjectStore::list`] a trailing delimiter is not
    /// automatically added to `prefix`
    ///
    /// [`ObjectStore::list`]: crate::ObjectStore::list
    async fn list_paginated(
        &self,
        prefix: Option<&str>,
        opts: PaginatedListOptions,
    ) -> Result<PaginatedListResult>;
}
