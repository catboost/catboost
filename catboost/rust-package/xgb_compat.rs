// Minimal XGBoost-like compatibility shim for the CatBoost rust-package
// This file provides a Booster and DMatrix type with similar function
// signatures to many xgboost Rust wrappers. Fill in the TODOs where
// the underlying catboost model API is invoked.

use std::path::Path;
use crate::error::Error as CatboostError;
use crate::model::Model as CatBoostModel;

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
    pub fn load_model<P: AsRef<Path>>(path: P) -> Result<Self, CatboostError> {
        // TODO: replace with the actual model loading method from model.rs
        // e.g. CatBoostModel::load_from_file(path.as_ref()) or similar.
        let model = CatBoostModel::from_file(path.as_ref().to_str().unwrap())?;
        Ok(Booster { inner: model })
    }

    /// Load from raw bytes if your CatBoost model loader supports it.
    pub fn from_buffer(buf: &[u8]) -> Result<Self, CatboostError> {
        // TODO: implement using the catboost model API that accepts bytes
        let model = CatBoostModel::from_bytes(buf)?;
        Ok(Booster { inner: model })
    }

    /// Predict on a single DMatrix returning a flattened Vec<f32> of predictions.
    /// For binary classification this would return one score per row; for multi-class it will
    /// be nrow * nclass elements (row-major).
    pub fn predict(&self, dmat: &DMatrix) -> Result<Vec<f32>, CatboostError> {
        // TODO: map DMatrix to the input expected by CatBoostModel prediction API.
        // Example:
        //   - If CatBoostModel supports predict_from_dense(&[f32], nrow, ncol) -> Vec<f32>
        //   - Otherwise, build internal representation and call the FFI.
        let preds = self.inner.predict_dense(&dmat.data, dmat.nrow, dmat.ncol)?;
        Ok(preds)
    }

    /// Save model to a file path
    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<(), CatboostError> {
        self.inner.save_to_file(path.as_ref().to_str().unwrap())
    }

    /// Return number of features the model was trained with
    pub fn num_feature(&self) -> usize {
        // TODO: call the actual method on CatBoostModel
        self.inner.get_num_features()
    }

    /// Optionally return feature names if available
    pub fn feature_names(&self) -> Option<Vec<String>> {
        // TODO: call into catboost features API
        self.inner.get_feature_names()
    }
}

// Re-export common names so consumer code can do `use rust_xgboost_compat::Booster;`
pub use self::{Booster, DMatrix};