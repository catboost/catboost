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

//! AST types specific to loading and unloading syntax, like one available in Snowflake which
//! contains: STAGE ddl operations, PUT upload or COPY INTO
//! See [this page](https://docs.snowflake.com/en/sql-reference/commands-data-loading) for more details.

#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::fmt;
use core::fmt::Formatter;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::ast::Ident;
#[cfg(feature = "visitor")]
use sqlparser_derive::{Visit, VisitMut};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct StageParamsObject {
    pub url: Option<String>,
    pub encryption: DataLoadingOptions,
    pub endpoint: Option<String>,
    pub storage_integration: Option<String>,
    pub credentials: DataLoadingOptions,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct DataLoadingOptions {
    pub options: Vec<DataLoadingOption>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub enum DataLoadingOptionType {
    STRING,
    BOOLEAN,
    ENUM,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct DataLoadingOption {
    pub option_name: String,
    pub option_type: DataLoadingOptionType,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "visitor", derive(Visit, VisitMut))]
pub struct StageLoadSelectItem {
    pub alias: Option<Ident>,
    pub file_col_num: i32,
    pub element: Option<Ident>,
    pub item_as: Option<Ident>,
}

impl fmt::Display for StageParamsObject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let url = &self.url.as_ref();
        let storage_integration = &self.storage_integration.as_ref();
        let endpoint = &self.endpoint.as_ref();

        if url.is_some() {
            write!(f, " URL='{}'", url.unwrap())?;
        }
        if storage_integration.is_some() {
            write!(f, " STORAGE_INTEGRATION={}", storage_integration.unwrap())?;
        }
        if endpoint.is_some() {
            write!(f, " ENDPOINT='{}'", endpoint.unwrap())?;
        }
        if !self.credentials.options.is_empty() {
            write!(f, " CREDENTIALS=({})", self.credentials)?;
        }
        if !self.encryption.options.is_empty() {
            write!(f, " ENCRYPTION=({})", self.encryption)?;
        }

        Ok(())
    }
}

impl fmt::Display for DataLoadingOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if !self.options.is_empty() {
            let mut first = false;
            for option in &self.options {
                if !first {
                    first = true;
                } else {
                    f.write_str(" ")?;
                }
                write!(f, "{}", option)?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for DataLoadingOption {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.option_type {
            DataLoadingOptionType::STRING => {
                write!(f, "{}='{}'", self.option_name, self.value)?;
            }
            DataLoadingOptionType::ENUM => {
                // single quote is omitted
                write!(f, "{}={}", self.option_name, self.value)?;
            }
            DataLoadingOptionType::BOOLEAN => {
                // single quote is omitted
                write!(f, "{}={}", self.option_name, self.value)?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for StageLoadSelectItem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.alias.is_some() {
            write!(f, "{}.", self.alias.as_ref().unwrap())?;
        }
        write!(f, "${}", self.file_col_num)?;
        if self.element.is_some() {
            write!(f, ":{}", self.element.as_ref().unwrap())?;
        }
        if self.item_as.is_some() {
            write!(f, " AS {}", self.item_as.as_ref().unwrap())?;
        }
        Ok(())
    }
}
