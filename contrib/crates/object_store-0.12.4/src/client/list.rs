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

use crate::client::pagination::stream_paginated;
use crate::list::{PaginatedListOptions, PaginatedListResult};
use crate::path::{Path, DELIMITER};
use crate::Result;
use crate::{ListResult, ObjectMeta};
use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use std::borrow::Cow;
use std::collections::BTreeSet;

/// A client that can perform paginated list requests
#[async_trait]
pub(crate) trait ListClient: Send + Sync + 'static {
    async fn list_request(
        &self,
        prefix: Option<&str>,
        options: PaginatedListOptions,
    ) -> Result<PaginatedListResult>;
}

/// Extension trait for [`ListClient`] that adds common listing functionality
#[async_trait]
pub(crate) trait ListClientExt {
    fn list_paginated(
        &self,
        prefix: Option<&Path>,
        delimiter: bool,
        offset: Option<&Path>,
    ) -> BoxStream<'static, Result<ListResult>>;

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>>;

    #[allow(unused)]
    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, Result<ObjectMeta>>;

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult>;
}

#[async_trait]
impl<T: ListClient + Clone> ListClientExt for T {
    fn list_paginated(
        &self,
        prefix: Option<&Path>,
        delimiter: bool,
        offset: Option<&Path>,
    ) -> BoxStream<'static, Result<ListResult>> {
        let offset = offset.map(|x| x.to_string());
        let prefix = prefix
            .filter(|x| !x.as_ref().is_empty())
            .map(|p| format!("{}{}", p.as_ref(), DELIMITER));
        stream_paginated(
            self.clone(),
            (prefix, offset),
            move |client, (prefix, offset), page_token| async move {
                let r = client
                    .list_request(
                        prefix.as_deref(),
                        PaginatedListOptions {
                            offset: offset.clone(),
                            delimiter: delimiter.then_some(Cow::Borrowed(DELIMITER)),
                            page_token,
                            ..Default::default()
                        },
                    )
                    .await?;
                Ok((r.result, (prefix, offset), r.page_token))
            },
        )
        .boxed()
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        self.list_paginated(prefix, false, None)
            .map_ok(|r| futures::stream::iter(r.objects.into_iter().map(Ok)))
            .try_flatten()
            .boxed()
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        self.list_paginated(prefix, false, Some(offset))
            .map_ok(|r| futures::stream::iter(r.objects.into_iter().map(Ok)))
            .try_flatten()
            .boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        let mut stream = self.list_paginated(prefix, true, None);

        let mut common_prefixes = BTreeSet::new();
        let mut objects = Vec::new();

        while let Some(result) = stream.next().await {
            let response = result?;
            common_prefixes.extend(response.common_prefixes.into_iter());
            objects.extend(response.objects.into_iter());
        }

        Ok(ListResult {
            common_prefixes: common_prefixes.into_iter().collect(),
            objects,
        })
    }
}
