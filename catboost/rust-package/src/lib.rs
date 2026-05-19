#[cfg(test)]
#[macro_use]
extern crate approx;

mod error;
pub use crate::error::{CatBoostError, CatBoostResult};

mod features;
pub use crate::features::{
    ObjectsOrderFeatures,
    EmptyFloatFeatures,
    EmptyCatFeatures,
    EmptyTextFeatures,
    EmptyEmbeddingFeatures
};

mod model;
pub use crate::model::Model;
