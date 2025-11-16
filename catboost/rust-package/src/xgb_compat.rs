// Minimal XGBoost-like compatibility shim for the CatBoost rust-package
// This file provides a Booster with similar function signatures to many 
// xgboost Rust wrappers. 

use crate::error::CatBoostError;
use crate::features::ObjectsOrderFeatures;
use crate::model::Model as CatBoostModel;
use std::ffi::CString;
use std::path::Path;

/// A Booster wrapper that aims to match xgboost-like function signatures.
/// Internally holds the CatBoost model and delegates calls.
pub struct Booster {
    inner: CatBoostModel,
}

impl Booster {
    /// Load a model from file path (like xgboost::Booster::load_model)
    pub fn load_model<P: AsRef<Path>>(path: P) -> Result<Self, CatBoostError> {
        let model = CatBoostModel::load(path)?;
        Ok(Booster {
            inner: model
        })
    }

    /// Load from raw bytes if your CatBoost model loader supports it.
    pub fn from_buffer(buf: &[u8]) -> Result<Self, CatBoostError> {
        let model = CatBoostModel::load_buffer(&buf.to_vec())?;
        Ok(Booster {
            inner: model
        })
    }

    /// Predict using the model. Use empty vectors for unused feature types.
    /// For binary classification this would return one score per row; for multi-class it will
    /// be nrow * nclass elements (row-major).
    pub fn predict(
        &self,
        float_features: &[Vec<f32>],
        cat_features: &[Vec<String>],
        text_features: &[Vec<String>],
        embedding_features: &[Vec<Vec<f32>>],
    ) -> Result<Vec<f64>, CatBoostError> {
        // The underlying API for text features requires a Vec<Vec<CString>>.
        let text_features_cstr: Vec<Vec<CString>> = text_features
            .iter()
            .map(|row| {
                row.iter()
                    .map(|s| CString::new(s.as_str()).unwrap())
                    .collect()
            })
            .collect();

        let features = ObjectsOrderFeatures {
            float_features,
            cat_features,
            text_features: text_features_cstr,
            embedding_features,
        };
        self.inner.predict(features)
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
    fn test_booster_load_and_predict() {
        let booster = Booster::load_model("tmp/model.bin").unwrap();
        assert_eq!(booster.num_feature(), 4); // 3 float + 1 cat

        let cat_features: Vec<Vec<String>> = vec![
            vec!["north".to_string()],
            vec!["south".to_string()],
            vec!["south".to_string()],
        ];

        let float_features = vec![
            vec![-10.0, 5.0, 753.0],
            vec![30.0, 1.0, 760.0],
            vec![40.0, 0.1, 705.0],
        ];

        let result = booster.predict(&float_features, &cat_features, &[], &[]);
        assert!(result.is_ok());
        let predictions = result.unwrap();
        assert_eq!(predictions.len(), 3);

        let expected = vec![
            0.9980003729960197,
            0.00249414628534181,
            -0.0013677527881450977,
        ];
        for (p, e) in predictions.iter().zip(expected.iter()) {
            assert!((p - e).abs() < 1e-6);
        }
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
        let feature_names = booster.get_feature_names().unwrap();
        assert_eq!(feature_names, vec!["0", "1", "wind direction", "3"]);
    }
}
