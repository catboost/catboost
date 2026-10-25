//! Composable pool services
//!
//! This module contains various concepts of a connection pool separated into
//! their own concerns. This allows for users to compose the layers, along with
//! any other layers, when constructing custom connection pools.

pub mod cache;
pub mod map;
pub mod negotiate;
pub mod singleton;
