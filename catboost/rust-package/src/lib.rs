#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

mod error;
pub use crate::error::{CatBoostError, CatBoostResult};

mod model;
pub use crate::model::Model;
