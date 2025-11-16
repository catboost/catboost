// Minimal XGBoost-like compatibility shim for the CatBoost rust-package
// This file provides a Booster and DMatrix type with similar function
// signatures to many xgboost Rust wrappers. Fill in the TODOs where
// the underlying catboost model API is invoked.

use crate::error::CatBoostError;
use crate::features::ObjectsOrderFeatures;
use crate::model::Model as CatBoostModel;
use std::ffi::CStr;
use std::path::Path;

/// A simple DMatrix-like container for dense data (row-major).
/// Keep this minimal and extend later to accept sparse formats.
#[derive(Debug, Clone)]
pub struct DMatrix {
    pub data: Vec<f32>, // row-major, length = nrow * ncol
    pub nrow: usize,
    pub ncol: usize,
    pub missing: Option<f32>,
}

impl DMatrix {
    /// Create DMatrix from a dense slice. The caller must ensure len == nrow * ncol.
    pub fn from_dense(data: &[f32], nrow: usize, ncol: usize) -> Self {
        DMatrix {
            data: data.to_vec(),
            nrow,
            ncol,
            missing: None,
        }
    }

    /// Convenience constructor with explicit missing value sentinel
    pub fn from_dense_with_missing(data: &[f32], nrow: usize, ncol: usize, missing: f32) -> Self {
        DMatrix {
            data: data.to_vec(),
            nrow,
            ncol,
            missing: Some(missing),
        }
    }
}

/// A Booster wrapper that aims to match xgboost-like function signatures.
/// Internally holds the CatBoost model and delegates calls.
pub struct Booster {
    inner: CatBoostModel,
}

impl Booster {
    /// Load a model from file path (like xgboost::Booster::load_model)
    pub fn load_model<P: AsRef<Path>>(path: P) -> Result<Self, CatBoostError> {
        let model = CatBoostModel::load(path)?;
        Ok(Booster { inner: model })
    }

    /// Load from raw bytes if your CatBoost model loader supports it.
    pub fn from_buffer(buf: &[u8]) -> Result<Self, CatBoostError> {
        let model = CatBoostModel::load_buffer(&buf.to_vec())?;
        Ok(Booster { inner: model })
    }

    /// Predict on a single DMatrix returning a flattened Vec<f32> of predictions.
    /// For binary classification this would return one score per row; for multi-class it will
    /// be nrow * nclass elements (row-major).
    pub fn predict(&self, dmat: &DMatrix) -> Result<Vec<f32>, CatBoostError> {
        let float_features: Vec<Vec<f32>> = if dmat.nrow > 0 {
            dmat.data.chunks(dmat.ncol).map(|s| s.to_vec()).collect()
        } else {
            vec![]
        };

        let features = ObjectsOrderFeatures::new().with_float_features(float_features);

        let preds_f64 = self.inner.predict(features)?;
        Ok(preds_f64.into_iter().map(|x| x as f32).collect())
    }

    /// Save model to a file path
    pub fn save_model<P: AsRef<Path>>(&self, _path: P) -> Result<(), CatBoostError> {
        // CatBoostModel does not have a save method in the provided context.
        // If it were added, it would be called here.
        // For now, returning an error or making this a no-op.
        Err(CatBoostError {
            description: "save_model is not implemented".to_string(),
        })
    }

    /// Return number of features the model was trained with
    pub fn num_feature(&self) -> usize {
        self.inner.get_float_features_count()
            + self.inner.get_cat_features_count()
            + self.inner.get_text_features_count()
            + self.inner.get_embedding_features_count()
    }

    /// return feature names that were used in training
    pub fn get_feature_names(&self) -> Result<Vec<String>, CatBoostError> {
        self.inner.get_feature_names()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_booster_load_and_predict_err() {
        let booster = Booster::load_model("tmp/model.bin").unwrap();
        assert_eq!(booster.num_feature(), 4); // 3 float + 1 cat

        let test_data = vec![-10.0, 5.0, 753.0, 30.0, 1.0, 760.0, 40.0, 0.1, 705.0];
        let dmatrix = DMatrix::from_dense(&test_data, 3, 3);

        // The provided model expects categorical features too, which DMatrix doesn't support.
        // The predict call will fail. This test verifies that we can load and call predict.
        // To make it pass, we would need to extend DMatrix to support categorical features.
        let result = booster.predict(&dmatrix);
        assert!(result.is_err());
        assert!(result.unwrap_err().description.contains("cat feature"));
    }

    #[test]
    fn test_from_buffer() {
        let buffer = fs::read("tmp/model.bin").unwrap();
        let booster = Booster::from_buffer(&buffer).unwrap();
        assert_eq!(booster.num_feature(), 4);
    }

    #[test]
    fn test_feature_names() {
        let booster = Booster::load_model("tmp/model.bin").unwrap();
        let feature_names = booster.feature_names().unwrap();
        assert_eq!(feature_names, vec!["float1", "float2", "float3", "cat1"]);
    }
}
