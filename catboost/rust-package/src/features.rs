
use std::ffi::CString;

#[derive(Default)]
pub struct EmptyFloatFeatures {
}

impl AsRef<[Vec<f32>]> for EmptyFloatFeatures {
    fn as_ref(&self) -> &[Vec<f32>] {
        &[]
    }
}

#[derive(Default)]
pub struct EmptyCatFeatures {
}

impl AsRef<[Vec<String>]> for EmptyCatFeatures {
    fn as_ref(&self) -> &[Vec<String>] {
        &[]
    }
}

#[derive(Default)]
pub struct EmptyTextFeatures {
}

impl AsRef<[Vec<CString>]> for EmptyTextFeatures {
    fn as_ref(&self) -> &[Vec<CString>] {
        &[]
    }
}

#[derive(Default)]
pub struct EmptyEmbeddingFeatures {
}

impl AsRef<[Vec<Vec<f32>>]> for EmptyEmbeddingFeatures {
    fn as_ref(&self) -> &[Vec<Vec<f32>>] {
        &[]
    }
}

 pub struct ObjectsOrderFeatures<
    /// must provide 2-level dereferencing to f32. Outer is by-object, inner is by float feature
    TFloatFeatures = EmptyFloatFeatures,

    /// must provide 2-level dereferencing to str. Outer is by-object, inner is by cat feature
    TCatFeatures = EmptyCatFeatures,

    /// must provide 2-level dereferencing to CStr. Outer is by-object, inner is by cat feature
    /// Note: CStr is used because that's what CatBoost's C API functions accept this format for now.
    TTextFeatures = EmptyTextFeatures,

    /// must provide 3-level dereferencing to f32. Levels are: by-object, by-embedding, index in embedding
    TEmbeddingFeatures = EmptyEmbeddingFeatures,
> {
  pub float_features: TFloatFeatures,
  pub cat_features: TCatFeatures,
  pub text_features: TTextFeatures,
  pub embedding_features: TEmbeddingFeatures
}


impl ObjectsOrderFeatures<EmptyFloatFeatures, EmptyCatFeatures, EmptyTextFeatures, EmptyEmbeddingFeatures>
{
    pub fn new() -> Self {
        ObjectsOrderFeatures{
            float_features: EmptyFloatFeatures{},
            cat_features: EmptyCatFeatures{},
            text_features: EmptyTextFeatures{},
            embedding_features: EmptyEmbeddingFeatures{}
        }
    }
}


/// `with_*_features` are convenience functions when you don't want to specify all types of features when you don't
///   need them.
/// They are necessary because Rust does not support default params.
/// See examples in model tests.
impl<TFloatFeatures, TCatFeatures, TTextFeatures, TEmbeddingFeatures>
    ObjectsOrderFeatures<TFloatFeatures, TCatFeatures, TTextFeatures, TEmbeddingFeatures>
{
    pub fn with_float_features<TNewFloatFeatures>(
        self,
        new_float_features: TNewFloatFeatures
    ) -> ObjectsOrderFeatures<TNewFloatFeatures, TCatFeatures, TTextFeatures, TEmbeddingFeatures> {
        ObjectsOrderFeatures{
            float_features: new_float_features,
            cat_features: self.cat_features,
            text_features: self.text_features,
            embedding_features: self.embedding_features
        }
    }

    pub fn with_cat_features<TNewCatFeatures>(
        self,
        new_cat_features: TNewCatFeatures
    ) -> ObjectsOrderFeatures<TFloatFeatures, TNewCatFeatures, TTextFeatures, TEmbeddingFeatures> {
        ObjectsOrderFeatures{
            float_features: self.float_features,
            cat_features: new_cat_features,
            text_features: self.text_features,
            embedding_features: self.embedding_features
        }
    }

    pub fn with_text_features<TNewTextFeatures>(
        self,
        new_text_features: TNewTextFeatures
    ) -> ObjectsOrderFeatures<TFloatFeatures, TCatFeatures, TNewTextFeatures, TEmbeddingFeatures> {
        ObjectsOrderFeatures{
            float_features: self.float_features,
            cat_features: self.cat_features,
            text_features: new_text_features,
            embedding_features: self.embedding_features
        }
    }

    pub fn with_embedding_features<TNewEmbeddingFeatures>(
        self,
        new_embedding_features: TNewEmbeddingFeatures
    ) -> ObjectsOrderFeatures<TFloatFeatures, TCatFeatures, TTextFeatures, TNewEmbeddingFeatures> {
        ObjectsOrderFeatures{
            float_features: self.float_features,
            cat_features: self.cat_features,
            text_features: self.text_features,
            embedding_features: new_embedding_features
        }
    }
}