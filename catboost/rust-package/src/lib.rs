#[cfg(test)]
#[macro_use]
extern crate approx;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

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
