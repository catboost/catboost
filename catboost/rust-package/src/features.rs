
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
    TFloatFeatures = EmptyFloatFeatures,
    TCatFeatures = EmptyCatFeatures,
    TTextFeatures = EmptyTextFeatures,
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