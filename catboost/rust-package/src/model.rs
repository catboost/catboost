use crate::error::{CatBoostError, CatBoostResult};
use crate::features::{
    ObjectsOrderFeatures,
    EmptyTextFeatures,
    EmptyEmbeddingFeatures
};
use catboost_sys;
use std::ffi::{CStr,CString};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

pub struct Model {
    handle: *mut catboost_sys::ModelCalcerHandle,
}

unsafe impl Send for Model {}

impl Model {
    fn new() -> Self {
        let model_handle = unsafe { catboost_sys::ModelCalcerCreate() };
        Model {
            handle: model_handle,
        }
    }

    /// Load a model from a file
    pub fn load<P: AsRef<Path>>(path: P) -> CatBoostResult<Self> {
        let model = Model::new();
        let path_c_str = CString::new(path.as_ref().as_os_str().as_bytes()).unwrap();
        CatBoostError::check_return_value(unsafe {
            catboost_sys::LoadFullModelFromFile(model.handle, path_c_str.as_ptr())
        })?;
        Ok(model)
    }

    /// Load a model from a buffer
    pub fn load_buffer<P: AsRef<Vec<u8>>>(buffer: P) -> CatBoostResult<Self> {
        let model = Model::new();
        CatBoostError::check_return_value(unsafe {
            catboost_sys::LoadFullModelFromBuffer(
                model.handle,
                buffer.as_ref().as_ptr() as *const std::os::raw::c_void,
                buffer.as_ref().len(),
            )
        })?;
        Ok(model)
    }

    fn set_or_check_object_count<
        TFeature,
        TObjectFeatures: AsRef<[TFeature]>,
        TFeatures: AsRef<[TObjectFeatures]>
    >
    (
        object_count: &mut Option<usize>,
        features: &TFeatures
    ) -> CatBoostResult<()> {
        let features_array_size = features.as_ref().len();
        if features_array_size > 0 {
            match object_count {
                Some(count) => {
                    if *count != features_array_size {
                        return Err(
                            CatBoostError{ description: "features arguments have different nonzero sizes".to_owned() }
                        )
                    }
                }
                None => {
                    object_count.replace(features_array_size);
                }
            }
        }
        Ok(())
    }

    /// Calculate raw model predictions
    pub fn predict<
        TObjectFloatFeatures: AsRef<[f32]>,
        TFloatFeatures: AsRef<[TObjectFloatFeatures]>,
        TCatFeatureString: AsRef<str>,
        TObjectCatFeatures: AsRef<[TCatFeatureString]>,
        TCatFeatures: AsRef<[TObjectCatFeatures]>,
        TTextFeatureString: AsRef<CStr>,
        TObjectTextFeatures: AsRef<[TTextFeatureString]>,
        TTextFeatures: AsRef<[TObjectTextFeatures]>,
        TEmbedding: AsRef<[f32]>,
        TObjectEmbeddingFeatures: AsRef<[TEmbedding]>,
        TEmbeddingFeatures: AsRef<[TObjectEmbeddingFeatures]>
    >(
        &self,
        features: ObjectsOrderFeatures<
            TFloatFeatures,
            TCatFeatures,
            TTextFeatures,
            TEmbeddingFeatures
        >
    ) -> CatBoostResult<Vec<f64>> {
        let mut object_count = None;
        Self::set_or_check_object_count(&mut object_count, &features.float_features)?;
        Self::set_or_check_object_count(&mut object_count, &features.cat_features)?;
        Self::set_or_check_object_count(&mut object_count, &features.text_features)?;
        Self::set_or_check_object_count(&mut object_count, &features.embedding_features)?;
        if object_count.is_none() {
            return Err(
                CatBoostError{ description: "all features arguments are empty".to_owned() }
            );
        }

        let mut float_features_ptr = features.float_features
            .as_ref()
            .iter()
            .map(|x| x.as_ref().as_ptr())
            .collect::<Vec<_>>();

        let hashed_cat_features =  features.cat_features
            .as_ref()
            .iter()
            .map(|doc_cat_features| {
                doc_cat_features
                    .as_ref()
                    .iter()
                    .map(|cat_feature| unsafe {
                        catboost_sys::GetStringCatFeatureHash(
                            cat_feature.as_ref().as_ptr() as *const std::os::raw::c_char,
                            cat_feature.as_ref().len(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut hashed_cat_features_ptr = hashed_cat_features
            .iter()
            .map(|x| x.as_ptr())
            .collect::<Vec<_>>();

        let mut text_features_ptr_storage = features.text_features
            .as_ref()
            .iter()
            .map(
                |object_text_features|
                    object_text_features.as_ref()
                        .iter()
                        .map(|text|
                            text.as_ref().as_ptr()
                        )
                        .collect::<Vec<_>>()
            )
            .collect::<Vec<_>>();

        let mut text_features_ptr = text_features_ptr_storage
            .iter_mut()
            .map(|object_texts_ptrs: &mut Vec<*const i8>| object_texts_ptrs.as_mut_ptr())
            .collect::<Vec<_>>();

        let mut embedding_dimensions = if features.embedding_features.as_ref().len() > 0 {
            features.embedding_features.as_ref()[0].as_ref().iter()
                .map(|x| x.as_ref().len())
                .collect::<Vec<_>>()
        } else {
            vec![]
        };

        let mut embedding_features_ptr_storage = features.embedding_features
            .as_ref()
            .iter()
            .map(
                |object_embeddings|
                    object_embeddings.as_ref()
                        .iter()
                        .map(|embedding|
                            embedding.as_ref().as_ptr()
                        )
                        .collect::<Vec<_>>()
            )
            .collect::<Vec<_>>();

        let mut embedding_features_ptr = embedding_features_ptr_storage
            .iter_mut()
            .map(|object_embeddings_ptrs: &mut Vec<*const f32>| object_embeddings_ptrs.as_mut_ptr())
            .collect::<Vec<_>>();

        let mut prediction = vec![0.0; object_count.unwrap() * self.get_dimensions_count()];
        CatBoostError::check_return_value(unsafe {
            catboost_sys::CalcModelPredictionWithHashedCatFeaturesAndTextAndEmbeddingFeatures(
                self.handle,
                object_count.unwrap(),
                float_features_ptr.as_mut_ptr(),
                if features.float_features.as_ref().is_empty() { 0 } else { features.float_features.as_ref()[0].as_ref().len() },
                hashed_cat_features_ptr.as_mut_ptr(),
                if features.cat_features.as_ref().is_empty() { 0 } else { features.cat_features.as_ref()[0].as_ref().len() },
                text_features_ptr.as_mut_ptr(),
                if features.text_features.as_ref().is_empty() { 0 } else { features.text_features.as_ref()[0].as_ref().len() },
                embedding_features_ptr.as_mut_ptr(),
                embedding_dimensions.as_mut_ptr(),
                embedding_dimensions.len(),
                prediction.as_mut_ptr(),
                prediction.len(),
            )
        })?;
        Ok(prediction)
    }

    /// Calculate raw model predictions on float features and string categorical feature values
    pub fn calc_model_prediction<
        TFloatFeature: AsRef<[f32]>,
        TFloatFeatures: AsRef<[TFloatFeature]>,
        TString: AsRef<str>,
        TCatFeature: AsRef<[TString]>,
        TCatFeatures: AsRef<[TCatFeature]>
    >
    (
        &self,
        float_features: TFloatFeatures,
        cat_features: TCatFeatures,
    ) -> CatBoostResult<Vec<f64>> {
        self.predict(
            ObjectsOrderFeatures{
                float_features: float_features,
                cat_features: cat_features,
                text_features: EmptyTextFeatures{},
                embedding_features: EmptyEmbeddingFeatures{}
            }
        )
    }

    /// Get expected float feature count for model
    pub fn get_float_features_count(&self) -> usize {
        unsafe { catboost_sys::GetFloatFeaturesCount(self.handle) }
    }

    /// Get expected categorical feature count for model
    pub fn get_cat_features_count(&self) -> usize {
        unsafe { catboost_sys::GetCatFeaturesCount(self.handle) }
    }

    /// Get expected text feature count for model
    pub fn get_text_features_count(&self) -> usize {
        unsafe { catboost_sys::GetTextFeaturesCount(self.handle) }
    }

    /// Get expected embedding feature count for model
    pub fn get_embedding_features_count(&self) -> usize {
        unsafe { catboost_sys::GetEmbeddingFeaturesCount(self.handle) }
    }

    /// Get number of trees in model
    pub fn get_tree_count(&self) -> usize {
        unsafe { catboost_sys::GetTreeCount(self.handle) }
    }

    /// Get number of dimensions in model
    pub fn get_dimensions_count(&self) -> usize {
        unsafe { catboost_sys::GetDimensionsCount(self.handle) }
    }

    pub fn enable_gpu_evaluation(&self) -> CatBoostResult<()> {
        CatBoostError::check_return_value( unsafe { catboost_sys::EnableGPUEvaluation(self.handle, 0) } )
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { catboost_sys::ModelCalcerDelete(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_model() {
        let model = Model::load("tmp/model.bin");
        assert!(model.is_ok());
    }

    #[test]
    fn load_model_buffer() {
        let buffer: Vec<u8> = read_fast("tmp/model.bin").unwrap();
        let model = Model::load_buffer(buffer);
        assert!(model.is_ok());
    }

    #[test]
    fn calc_prediction() {
        let model = Model::load("tmp/model.bin").unwrap();
        let prediction = model
            .calc_model_prediction(
                vec![
                    vec![-10.0, 5.0, 753.0],
                    vec![30.0, 1.0, 760.0],
                    vec![40.0, 0.1, 705.0],
                ],
                vec![
                    vec![String::from("north")],
                    vec![String::from("south")],
                    vec![String::from("south")],
                ],
            )
            .unwrap();

        assert_eq!(prediction[0], 0.9980003729960197);
        assert_eq!(prediction[1], 0.00249414628534181);
        assert_eq!(prediction[2], -0.0013677527881450977);

        let prediction = model
            .calc_model_prediction(
                &[
                    [-10.0, 5.0, 753.0],
                    [30.0, 1.0, 760.0],
                    [40.0, 0.1, 705.0],
                ],
                &[
                    ["north"],
                    ["south"],
                    ["south"],
                ],
            )
            .unwrap();

        assert_eq!(prediction[0], 0.9980003729960197);
        assert_eq!(prediction[1], 0.00249414628534181);
        assert_eq!(prediction[2], -0.0013677527881450977);
    }

    #[test]
    #[should_panic]
    fn calc_prediction_object_size_mismatch() {
        let model = Model::load("tmp/model.bin").unwrap();
        model.calc_model_prediction(
            vec![
                vec![-10.0, 5.0, 753.0],
                vec![30.0, 1.0, 760.0],
                vec![40.0, 0.1, 705.0],
            ],
            vec![
                vec![String::from("north")],
                vec![String::from("south")],
            ],
        )
        .unwrap();
    }

    fn split_string_with_floats(features: &str, delim: char) -> Vec<f32> {
        features.split(delim).map(|e| e.parse::<f32>().unwrap() ).collect::<Vec<_>>()
    }

    fn get_num_features(features: &str) -> Vec<f32> {
        split_string_with_floats(features, '\t')
    }

    fn get_categ_features(features: &str) -> Vec<&str> {
        features.split('\t').collect::<Vec<_>>()
    }

    fn get_text_features(features: &[&str]) -> Vec<CString> {
        features.iter().map(|s| CString::new(*s).unwrap() ).collect::<Vec<_>>()
    }

    fn get_embeddings(features: &str) -> Vec<f32> {
        split_string_with_floats(features, ';')
    }

    fn test_predict_model_with_num_features(on_gpu: bool) {
        let model = Model::load("../pytest/data/models/features_num__dataset_querywise.cbm").unwrap();

        if on_gpu {
            model.enable_gpu_evaluation().unwrap()
        }

        let prediction = model.predict(
            ObjectsOrderFeatures::new()
                .with_float_features(
                    &[
                        get_num_features(
                            "0.257727	0.0215909	0.171299	1	1	1	1	1	0	0	0	0	0	0	0.431373	0.935065	0.0208333	0.070824	1	0	0.313726	1	1	0	0.937724	0	1	0	0	0	0	0.0566038	0	0	1	0.73929	1	0.000505391	0.885819	0.000172727	0	0	0	0	0	0	0.153262	0.578118	0.222098	1"
                        ),
                        get_num_features(
                            "0.424438	0.164384	0.572649	1	1	0	1	0	0	1	0	0	0	0	0.360784	0.512195	0.0447049	0.0717587	1	1	0.321569	1	1	0	0.941214	0	1	0	0	0	0	0.275362	1	0.209302	1	0.391239	1	0.0143194	0.885819	0.00140996	0	0	0	0	0	0	0.357143	0.883721	0.820312	1"
                        ),
                        get_num_features(
                            "0.345548	0.248034	0.0853067	1	1	0	1	0	0	1	0	0	0	0	0.6	0.5	0.114292	0.071371	1	0	0.396078	1	1	0	0.939218	0	1	0	0	0	0	0.0384615	0	0	1	0	0	0	0.885819	0.00224069	0	0	0	0	0	0	0.588235	0.444444	0.608696	1"
                        ),
                        get_num_features(
                            "0.946305	0.139752	0.429885	1	1	1	1	1	0	1	0	0	0	0	0.811765	0.75	0.119755	0.071636	0	1	0.541176	1	1	0	0.993828	0	1	0	0	0	0	0.509804	1	0.000398	1	1.5	1	0.417659	0.885819	0.00117792	0	0	0	0	0	0	0.52962	0.958103	0.885843	1"
                        )
                    ]
                )
        ).unwrap();

        let expected_prediction = [
            0.08819508860736715,
            0.043193651033534904,
            -0.0019333444540111586,
            0.0836685835428004
        ];
        assert!( std::iter::zip(expected_prediction, prediction).all(|(l, r)| abs_diff_eq!(l, r, epsilon=1.0e-6)) )

    }

    fn test_predict_model_with_num_cat_features(on_gpu: bool) {
        let model = Model::load("../pytest/data/models/features_num_cat__dataset_adult.cbm").unwrap();

        if on_gpu {
            model.enable_gpu_evaluation().unwrap()
        }

        let prediction = model.predict(
            ObjectsOrderFeatures::new()
                .with_float_features(
                    &[
                        get_num_features("44.0	403782.0	11.0	0.0	0.0	45.0"),
                        get_num_features("19.0	208874.0	10.0	0.0	0.0	40.0"),
                        get_num_features("27.0	158647.0	10.0	0.0	0.0	40.0"),
                        get_num_features("18.0	115258.0	6.0	0.0	0.0	40.0")
                    ]
                )
                .with_cat_features(
                    &[
                        get_categ_features(
                            "0	n	1	Private	Assoc-voc	Divorced	Sales	Not-in-family	White	Male	United-States"
                        ),
                        get_categ_features(
                            "0	n	1	Private	Assoc-voc	Divorced	Sales	Not-in-family	White	Male	United-States"
                        ),
                        get_categ_features(
                            "0	n	1	Private	Assoc-voc	Divorced	Sales	Not-in-family	White	Male	United-States"
                        ),
                        get_categ_features(
                            "0	n	1	Private	Assoc-voc	Divorced	Sales	Not-in-family	White	Male	United-States"
                        )
                    ]
                )
        ).unwrap();

        let expected_prediction = [
            0.9876043724015489,
            0.9869589576174197,
            0.997648058191363,
            1.0147797830459155
        ];
        assert!( std::iter::zip(expected_prediction, prediction).all(|(l, r)| abs_diff_eq!(l, r, epsilon=1.0e-6)) )

    }

    fn test_predict_model_with_num_cat_text_features(on_gpu: bool) {
        let model = Model::load("../pytest/data/models/features_num_cat_text__dataset_rotten_tomatoes__binclass.cbm").unwrap();

        if on_gpu {
            model.enable_gpu_evaluation().unwrap()
        }

        let prediction = model.predict(
            ObjectsOrderFeatures::new()
                .with_float_features(
                    &[
                        &[20100514.0f32,
                          20100914.0f32,
                          20100514.0f32],
                        &[19950602.0f32,
                          19970925.0f32,
                          20130731.0f32],
                        &[19990521.0f32,
                          19991019.0f32,
                          20000101.0f32]
                    ]
                )
                .with_cat_features(
                    &[
                        [
                            "PG",
                            "Gary Winick",
                            "Jose Rivera|Tim Sullivan",
                            "1/4",
                            "rotten",
                            "Kyle Smith",
                            "New York Post"
                        ],
                        [
                            "PG-13",
                            "Clint Eastwood",
                            "Richard LaGravenese|Richard La Gravenese",
                            "4/5",
                            "fresh",
                            "Angie Errigo",
                            "Empire Magazine"
                        ],
                        [
                            "R",
                            "Amos Poe",
                            "Amos Poe",
                            "2.5/5",
                            "rotten",
                            "Marc Savlov",
                            "Austin Chronicle"
                        ]
                    ]
                )
                .with_text_features(
                    &[
                    get_text_features(
                        &[
                            "When a young American travels to the city of Verona, home of the star-crossed lover Juliet Capulet of Romeo and Juliet fame, she joins a group of volunteers who respond to letters to Juliet seeking advice about love. When she answers one letter dated 1957, she inspires its author to travel to Italy in search of her long-lost love and sets off a chain of events that will bring a love into both their lives, unlike anything they ever imagined.",
                            "Comedy Drama Romance",
                            "The story is as straight and obvious as raw spaghetti."
                        ]
                    ),
                    get_text_features(
                        &[
                            "The brief, illicit love affair between an Iowa housewife and a post-middle-age free-lance photographer is chronicled in this powerful romance based on the best-selling novella by Robert James Waller. The story begins as globetrotting National Geographic photographer Robert Kincaid journeys to Madison County in 1965 to film its lovely covered bridges. Upon his arrival, he stops by an old farmhouse to ask directions. There he encounters housewife, Francesca Johnson, whose spouse and two children are out of town. Thus begins their four-day affair, a liaison that fundamentally changes them both. Later Francesca chronicles the affair in a diary which her flabbergasted grown children read; never would they have expected their mother to be capable of the passion she experienced with Kincaid.",
                            "Drama Romance",
                            "Streep and Eastwood's chemistry makes the film."
                        ]
                    ),
                    get_text_features(
                        &[
                            "FROGS FOR SNAKES is a neo-noir comic thriller centering on the hopes and ambitions of a motley ensemble of New York theater actors, who moonlight as illegal money collectors. Eva, the only group member with genuine acting ability and common sense, is fed up with the pretense of this twisted underworld and dreams of leaving town with her five-year-old son, Augie. She yearns to move to the suburbs and simply \"be.\" Eva's soul mate in this strange, incestuous community is Quint, who owns the diner where she waitresses. In the meantime, Eva supports herself and her son by making collections for a loan shark, who just happens to be her ex-husband, Al Santana. Al, the gang's \"boss\", also fancies himself a theatrical impresario. He plans to stage a production of David Mamet's American Buffalo and has already cast two of the three male parts. Naturally, everyone wants a part, particularly the coveted role of Teach, in this classic production. Al is confronted by his driver/hitman/actor UB who is desperate for a role. He tells Al, \"I f",
                            "Action and Adventure Comedy Mystery and Suspense",
                            ""
                        ]
                    ),
                    ]
                )
        ).unwrap();

        let expected_prediction = [
            3.225051356124634,
            -3.525299913259707,
            -3.2694112073224884
        ];
        assert!( std::iter::zip(expected_prediction, prediction).all(|(l, r)| abs_diff_eq!(l, r, epsilon=1.0e-6)) )

    }


    fn test_predict_model_with_num_cat_text_embedding_features(on_gpu: bool) {
        let model = Model::load("../pytest/data/models/features_num_cat_text_emb__dataset_rotten_tomatoes__binclass.cbm").unwrap();

        if on_gpu {
            model.enable_gpu_evaluation().unwrap()
        }

        let prediction = model.predict(
            ObjectsOrderFeatures::new()
                .with_float_features(
                    &[
                        &[20100514.0f32,
                          20100914.0f32,
                          20100514.0f32],
                        &[19950602.0f32,
                          19970925.0f32,
                          20130731.0f32],
                        &[19990521.0f32,
                          19991019.0f32,
                          20000101.0f32]
                    ]
                )
                .with_cat_features(
                    &[
                        [
                            "PG",
                            "Gary Winick",
                            "Jose Rivera|Tim Sullivan",
                            "1/4",
                            "rotten",
                            "Kyle Smith",
                            "New York Post"
                        ],
                        [
                            "PG-13",
                            "Clint Eastwood",
                            "Richard LaGravenese|Richard La Gravenese",
                            "4/5",
                            "fresh",
                            "Angie Errigo",
                            "Empire Magazine"
                        ],
                        [
                            "R",
                            "Amos Poe",
                            "Amos Poe",
                            "2.5/5",
                            "rotten",
                            "Marc Savlov",
                            "Austin Chronicle"
                        ]
                    ]
                )
                .with_text_features(
                    &[
                    get_text_features(
                        &[
                            "When a young American travels to the city of Verona, home of the star-crossed lover Juliet Capulet of Romeo and Juliet fame, she joins a group of volunteers who respond to letters to Juliet seeking advice about love. When she answers one letter dated 1957, she inspires its author to travel to Italy in search of her long-lost love and sets off a chain of events that will bring a love into both their lives, unlike anything they ever imagined.",
                            "Comedy Drama Romance",
                            "The story is as straight and obvious as raw spaghetti."
                        ]
                    ),
                    get_text_features(
                        &[
                            "The brief, illicit love affair between an Iowa housewife and a post-middle-age free-lance photographer is chronicled in this powerful romance based on the best-selling novella by Robert James Waller. The story begins as globetrotting National Geographic photographer Robert Kincaid journeys to Madison County in 1965 to film its lovely covered bridges. Upon his arrival, he stops by an old farmhouse to ask directions. There he encounters housewife, Francesca Johnson, whose spouse and two children are out of town. Thus begins their four-day affair, a liaison that fundamentally changes them both. Later Francesca chronicles the affair in a diary which her flabbergasted grown children read; never would they have expected their mother to be capable of the passion she experienced with Kincaid.",
                            "Drama Romance",
                            "Streep and Eastwood's chemistry makes the film."
                        ]
                    ),
                    get_text_features(
                        &[
                            "FROGS FOR SNAKES is a neo-noir comic thriller centering on the hopes and ambitions of a motley ensemble of New York theater actors, who moonlight as illegal money collectors. Eva, the only group member with genuine acting ability and common sense, is fed up with the pretense of this twisted underworld and dreams of leaving town with her five-year-old son, Augie. She yearns to move to the suburbs and simply \"be.\" Eva's soul mate in this strange, incestuous community is Quint, who owns the diner where she waitresses. In the meantime, Eva supports herself and her son by making collections for a loan shark, who just happens to be her ex-husband, Al Santana. Al, the gang's \"boss\", also fancies himself a theatrical impresario. He plans to stage a production of David Mamet's American Buffalo and has already cast two of the three male parts. Naturally, everyone wants a part, particularly the coveted role of Teach, in this classic production. Al is confronted by his driver/hitman/actor UB who is desperate for a role. He tells Al, \"I f",
                            "Action and Adventure Comedy Mystery and Suspense",
                            ""
                        ]
                    ),
                    ]
                )
                .with_embedding_features(
                    &[
                        [
                            get_embeddings(
                                "0.06591551;0.17545816;-0.106429726;-0.025793165;0.11267845;0.022515077;0.020125767;-0.065815635;-0.018560313;2.2736514;-0.26001078;-0.0025890404;0.040776633;-0.07226429;-0.1336614;-0.067014955;-0.065268524;0.9316771;-0.10883043;-0.077764936;0.06849822;-0.064046755;-0.075767525;-0.028425833;0.092960514;0.067684226;-0.05416466;-0.024933387;0.026832199;0.07917403;0.0019155883;0.111263886;-0.11691434;0.08344514;0.16653705;-0.0783515;-0.0061788294;-0.027936835;-0.08295989;-0.020650104;0.0058098375;0.09582928;0.029174725;-0.13126816;0.06696881;-0.02967926;-0.066066824;0.043362536;0.109918796;-0.015266707;-0.091228686;-0.02355179;-0.04745555;-0.08583387;0.032411262;0.06411334;-0.030928567;-0.046571985;-0.0114634605;-0.05048448;0.08710633;0.021309132;-0.017952695;0.107670024;-0.008597072;-0.06615355;-0.0596599;0.12259922;0.011119431;0.15667018;0.010504215;0.017416079;0.06024107;-0.006429896;0.079966135;0.050660696;-0.009497022;-0.04471796;0.0047789547;0.08119506;-0.05115366;0.10762943;-0.11372074;-0.04814061;0.0095739635;-0.20092943;-0.03896641;-0.054524384;0.18792589;0.0006371836;-0.13654152;-0.00083834253;-0.07435005;-0.02168448;0.05654651;-0.048289265;0.006338888;-0.030263403;-0.06920877;-0.07986395;0.03187675;0.0015107153;-0.13532665;-0.073731944;0.020082474;-0.6047669;0.0819619;0.1200599;0.029826947;-0.045120038;0.09753709;-0.05617631;0.081729375;-0.10229691;-0.055884182;-0.043689184;0.037175566;0.102649346;0.016136212;0.010732608;0.059425842;-0.061326124;0.04121365;0.05341205;0.09663478;0.185338;0.00502902;-0.019056767;-0.0012696558;0.0076429746;-0.04030703;-0.033093814;-0.03125129;-0.020959219;0.06303154;-0.023178827;-0.108385734;-0.005955986;-0.03989085;-3.72486e-05;-1.2439638;-0.04969005;0.10552381;-0.0027277071;-0.09889552;-0.048225507;-0.041801836;0.038198967;0.013245535;-0.06322363;0.026813187;0.0072241086;0.050494675;0.027650762;-0.06651777;-0.04400423;-0.021533221;-0.11227327;-0.07107381;-0.046471596;0.020600053;-0.046495587;-0.04689671;-0.02566152;0.011430239;-0.09255377;0.018132852;-0.059643842;0.14112723;0.08177702;-0.011416223;-0.050513394;0.15707156;-0.023152418;-0.02714736;0.057965823;-0.010066822;-0.010735444;-0.01310905;-0.047815386;0.055800572;-0.055477634;-0.0006351865;-0.03357044;-0.019522265;0.013823117;-0.05854959;-0.050311014;-0.019354321;-0.039295778;-0.074529074;-0.033093154;-0.02230363;0.036000468;-0.033117108;0.057956908;-0.016487889;-0.10350574;0.0010023569;0.0826276;0.010912595;-0.11949145;-0.09971496;0.061876196;0.15841126;0.04041672;0.1330669;-0.0800515;-0.015156314;-0.012627003;-0.13996075;-0.03711124;-0.13638932;-0.07281614;0.007227377;0.17317638;0.029044013;-0.027613569;-0.04311167;-0.01406775;0.046321206;-0.036096286;-0.00924572;0.011127569;-0.02872563;0.05066628;0.032044027;0.06936376;0.064359486;0.03060681;-0.11672327;0.040661983;0.017622264;0.077183;-0.119398676;-0.08208353;-0.062549785;0.021867217;-0.09753934;0.08624468;0.03159285;-0.0050177216;0.050408423;-0.046376545;0.14752552;-0.10792168;-0.07453012;-0.06901426;-0.09950946;-0.016529616;0.037157185;0.018419538;-0.0409082;-0.047662847;0.10057549;0.13409832;0.060044724;-0.047447182;-0.067210175;0.07698244;0.0495881;0.17351109;-0.038422696;0.14981811;0.10258404;-0.08595968;0.033494737;0.058839254;0.13133706;0.050934467;-0.11414686;-0.037970062;-0.15440767;-0.023129601;-0.02902595;0.00076296745;-0.007429118;0.06302276;0.028770624;0.12528515;0.11938626;0.09456591;0.0560529;-0.055775337;-0.02652802;-0.028954517;0.20901832;-0.013204003;0.08604067;-0.06178137;-0.1381025;0.064480096;0.017970502;-0.0135319475;9.6959535e-05;-0.032560285;-0.017069733;-0.017116109;0.008258656;-0.0073263445"
                            ),
                            get_embeddings(
                                "0.115443;0.18615668;0.257301;-0.27859935;0.016661337;0.162918;0.005268;-0.45616338;0.33447334;2.0720665;-0.56869;0.012935668;0.233748;0.1354372;0.13036834;-0.24258669;0.59786004;-0.041401;0.21526967;0.09562733;-0.21670265;-0.29020333;0.20208664;-0.019865334;0.44040334;0.07361499;-0.046911996;0.14272706;-0.34197998;0.028146664;-0.07017667;-0.43357667;-0.48484334;0.020963332;0.44045332;0.019299;-0.134046;0.18605;-0.232583;-0.046464;0.21336533;0.34823334;0.21775933;0.18544333;0.5555733;0.116502665;0.23225367;0.05980933;0.12074527;-0.19417;0.5639967;-0.4711;0.24301334;-0.2651849;-0.30208668;0.36584666;0.198654;-0.52811;-0.28200665;-0.23008734;0.17049865;0.013570667;0.29143333;-0.19636135;0.0032533358;0.004193326;-0.16137767;-0.07548667;-0.16691633;-0.25867334;0.5163133;-0.31119022;-0.388904;0.1489591;-0.270648;0.06155334;-0.35720333;0.18805666;0.4542433;0.23309666;-0.12938733;-0.23644567;-0.39086;-0.19412434;-0.32292333;0.15210433;-0.60823333;-0.13538001;0.24382667;-0.20050567;-0.47512665;0.09012666;0.119396664;0.113464326;-0.030346671;-0.18919833;-0.23751666;0.44580665;-0.66138667;-0.32806;-0.014961004;-0.22137666;-0.3623233;-0.158527;0.28357002;-0.18509634;-0.21747;0.08752633;0.015625;-0.6329067;-0.38132;0.48033;0.22830665;0.054329008;-0.45397332;-0.08740199;-0.06642934;0.43486;0.25311998;-0.042959;-0.43325996;-0.22986;-0.24664946;-0.16646332;0.6812634;0.5877;0.010021667;0.39870667;-0.17135666;-0.011617;-0.08948034;-0.17992043;0.33441535;0.11545933;0.22462332;0.46427;-0.27839896;-0.69798666;-0.3010933;-0.2879233;-2.5496333;-0.37859666;-0.01042967;0.12966147;-0.59054;-0.0473099;0.18907236;-0.21927416;0.34572697;-0.224551;0.3599267;0.10184667;0.59665674;0.22803666;0.16172333;-0.095678665;0.038023334;-0.44514665;-0.16726667;0.41438666;-0.16693668;-0.5701567;0.19453867;-0.09598366;0.047186363;0.010952003;-0.26118267;-0.45356336;0.29952666;-0.25105166;0.02354;0.04095899;0.047626138;0.12313067;0.24226968;-0.32801276;-0.25354466;-0.5907433;0.33411202;-0.41684332;-0.19527866;-0.14092089;0.18385667;0.39437735;0.09009433;0.22027;-0.26934037;0.04419;0.07419067;-0.27107;-0.14613667;-0.36665335;-0.08106767;0.035543326;-0.33012;-0.62042665;-0.48168668;-0.58351;0.38191333;0.284423;0.07677467;-0.0015253326;0.084857665;0.18232124;-0.3303572;-0.06546966;0.09058001;0.130526;0.16064768;0.14846167;-0.51283664;0.31997535;0.21533;-0.68513995;-0.109845005;0.422819;0.46876;-0.04931733;0.35712668;-0.213462;0.21070333;0.11086347;0.31193998;0.010137673;-0.16098334;0.05751234;-0.22485667;-0.302317;0.50173336;0.17816;-0.10785333;0.6158633;0.20871401;-0.087708674;-0.5127967;-0.09397;-0.19074;0.30674335;-0.10575517;-0.140724;0.036133673;-0.186601;0.04354279;-0.46409;-0.013331999;0.19234668;-0.228458;0.39059234;0.5439021;-0.31506002;-0.05427767;0.058424007;0.08818;-0.23544665;-0.18453999;0.034691334;0.38865003;0.2112;-0.46842003;-0.35366333;-0.078990005;0.37915;0.0063580014;0.15887333;0.14138;-0.257405;0.19296502;0.42451668;0.15440334;0.0036229987;-1.0090866;-0.0117466645;-0.62091;0.28818932;-0.23984002;0.13105467;-0.22032;-0.60160667;-0.018166667;0.78189;0.120676674;-0.119366676;0.31777233;-0.3153533;-0.22421001;0.82226664;-0.008573338;-0.025182;-0.14374666;-0.28013733;0.13360633;0.028181667;-0.106188;0.30549332;-0.055417;0.039854333;0.094189994;-0.16149767;-0.23305333;-0.27587333"
                            ),
                            get_embeddings(
                                "-0.11861181;0.10208699;-0.09064615;0.0030908205;0.15578236;0.2800559;0.008929364;-0.056302723;-0.022617541;2.1170998;-0.14118661;0.20516019;0.08177291;0.01237818;-0.021924034;-0.11031037;-0.024725724;1.1072109;-0.17114808;-0.06725854;-0.12114826;-0.16666855;-0.050177094;-0.023168517;0.10510247;0.09800846;-0.012411633;-0.02850947;-0.08219506;-0.1663029;0.034275483;0.112007014;-0.027568275;-0.018971724;0.12854664;-0.038954005;0.012637182;0.12182803;0.0014515248;0.00072045013;-0.033007704;0.10799701;0.021518998;-0.18174678;0.018131994;0.02865709;-0.12664737;0.04707491;-0.07127209;0.009248003;0.026362639;0.043776464;-0.016813902;-0.056362186;0.037213366;0.07518468;0.07174036;-0.030466529;-0.043786857;-0.08426172;-0.14239852;-0.0846348;0.009529669;0.091467805;-0.030228907;-0.22365882;-0.032825883;-0.02552288;-0.05582106;0.24362947;0.02339472;0.106028534;0.21113001;0.067438185;-0.16025245;0.0077502728;-0.016267546;0.02683082;-0.088458166;0.16199993;0.0061795493;-0.027893819;-0.048484813;-0.02261161;0.037470456;-0.30861714;0.028569818;-0.038686093;0.119140096;0.037060454;-0.24143404;-0.008858091;0.013853357;0.011000183;0.20293492;-0.046176173;-0.049306985;-0.059952185;0.06468464;-0.1767501;0.017146377;0.054519545;-0.12096573;-0.09168064;-0.041444544;-0.7214285;-0.029097183;-0.011051455;0.0019379948;-0.054857634;-0.05616264;-0.18174563;0.013061907;-0.18449229;-0.012122;0.045589745;-0.15703236;0.07535935;-0.02972691;-0.0041596945;-0.091635995;0.0020175115;-0.05744964;0.032719217;-0.00419472;0.096174724;0.036698278;-0.17061219;-0.11289947;-0.08129109;-0.14920954;-0.07165873;0.0129058175;0.16543332;0.056674812;0.17286329;0.07077554;-0.08672028;0.08457392;-0.057476554;-1.4389755;0.04217363;0.16232747;0.077119365;-0.064567275;-0.16248727;-0.04684555;-0.058304463;0.18954;-0.069406815;0.007332359;-0.028866671;0.22749637;0.028148301;-0.11840412;-0.09994951;-0.14274727;-0.07100209;-0.01597655;0.009389376;-0.05614309;0.014844538;0.008559361;-0.013909364;-0.12403469;-0.27775392;-0.04789927;0.006314729;0.11497128;-0.029353822;0.020862088;-0.04956573;0.020289091;-0.17140882;-0.22628435;-0.01704178;0.0026768162;-0.038820542;-0.1709182;-0.086452;-0.039395455;-0.14734882;-0.13595964;-0.12437671;-0.053216126;0.021425365;-0.06557015;-0.14205973;0.07937009;-0.030674636;0.017339354;0.04663509;-0.00074953923;0.10384119;-0.09633306;0.020415364;0.025861366;-0.032851547;-0.13225283;0.23478208;0.009305543;0.0067198114;-0.15227956;0.08062554;0.091656;0.053407546;0.070357636;0.07482373;-0.057558082;0.12291945;-0.1684519;-0.03582764;0.052697;-0.22739267;-0.059381906;0.20128903;0.056607004;0.14613952;-0.15356354;-0.06857645;-0.10855054;0.12300964;-0.05734223;0.14158337;0.066797905;-0.086357206;-0.077720635;0.08315729;0.026963366;0.22619763;-0.20044845;-0.011382458;0.056805633;0.1473629;-0.14923245;-0.12962455;-0.0052188206;-0.07353345;-0.025415001;0.025547545;-0.059011307;0.0005303625;-0.19525756;-0.021717237;0.18460391;-0.08960845;-0.21807618;-0.17565791;0.107842274;-0.002158089;0.111908935;0.124908;0.062135484;-0.07992591;0.04806118;0.11575627;0.18598899;-0.006314;-0.011822909;-0.01377573;0.08289438;0.120640635;-0.025034618;-0.0099102715;0.064467974;0.06782394;-0.016152821;-0.11391967;0.36016458;-0.10674489;0.054474376;-0.102236815;-0.14596628;-0.18249798;-0.19920883;0.02331809;0.025977325;0.07938398;-0.0029983642;0.22813128;0.33147547;0.12081636;0.032840375;-0.16814037;-0.03245139;0.011451587;-0.023543727;-0.14084154;0.18182969;-0.13747792;-0.12809028;0.082569644;-0.07844155;-0.020777917;-0.18750833;0.029338451;-0.023957092;-0.14637455;-0.06628101;0.086792976"
                            )
                        ],
                        [
                            get_embeddings(
                                "0.00052754587;0.15159017;-0.059436813;-0.008451907;0.12592517;-0.015462559;0.02071232;-0.08119693;-0.015513279;2.2218988;-0.22101825;0.066773966;-0.00025419923;-0.09205102;-0.111598246;-0.04335176;-0.024376385;0.9570249;-0.069704935;-0.03569548;0.02739131;-0.114869095;-0.05359181;-0.007795872;0.097719096;0.059659157;-0.06554905;0.012588591;-0.04329238;0.016544517;-0.014545384;0.107136436;-0.06922973;0.05642346;0.076730855;-0.06228847;0.010533257;-0.004083691;-0.046684828;-0.055854052;0.06263673;0.02744734;0.046350785;-0.09710819;-0.00090791564;0.013527757;-0.077348076;-0.029234849;0.10503626;0.002065043;-0.01825544;-0.01211692;-0.08194248;-0.035267916;0.005656948;0.044336107;0.019705903;-0.043487392;-0.0031869346;-0.06675911;0.028970364;0.017000005;0.014709898;0.113303445;-0.022641167;-0.057850916;-0.027154604;0.02414771;0.015966892;0.11206986;-0.03424199;-0.034776244;0.054690696;-0.01173129;0.1299347;-0.03084038;-0.025815502;-0.011259608;-0.02644493;0.09326155;0.038756404;0.0776226;-0.07695409;0.02229276;-0.0050711706;-0.22227368;0.058394045;-0.06683393;0.11305024;0.048999198;-0.14555585;0.030347334;-0.046905376;-0.061936505;0.0866795;-0.04575085;0.014519191;-0.03618421;-0.06539481;-0.042031452;0.08223525;0.016211612;-0.07431791;-0.07876965;-0.020923095;-0.5813542;0.011241491;0.026119832;0.07593613;-0.06370052;0.08652204;-0.026784776;0.08117211;-0.10422726;-0.02109447;-0.088904455;0.018700631;0.040505085;0.012266292;0.03911654;0.013951495;-0.030636637;-0.016084144;-0.011472928;0.0042781904;0.109482184;-0.020473758;-0.06147418;-0.01713819;-0.0029989784;-0.015271557;-0.07047347;-0.010563407;-0.0371089;0.041680288;0.03580967;-0.05556615;0.020680524;0.01410151;-0.029953277;-1.1644329;-0.011711964;0.10690663;0.009042739;-0.060236357;-0.044285722;-0.029872695;-0.0049162395;0.01391702;-0.12025733;0.03359515;0.009752471;0.059814885;0.022440461;-0.047743358;-0.02001349;-0.06497968;-0.091853626;-0.05544196;-0.03570086;0.003592832;-0.004058264;0.025713647;-0.048698314;-0.040175494;-0.096957654;0.048520964;-0.06682626;0.18528962;0.034132313;-0.0052377335;0.0043804836;0.061311383;0.01575578;-0.037685648;0.036736578;-0.02871588;-0.04955481;-0.015689485;0.0020220203;0.007566432;-0.030613953;-0.046534527;-0.061471686;-0.02009188;0.068659954;-0.07263881;0.024428552;-0.009234041;-0.0016844773;-0.070724264;0.03152215;-0.0035280364;0.082845524;-0.0657735;0.09727368;-0.05419846;-0.027256818;-0.017892146;0.09090186;-0.0156208165;-0.0801865;-0.0159404;-0.009689137;0.06649907;-0.0028455027;0.046259895;-0.054203358;-0.053101458;0.023584193;-0.12164176;-0.00633703;-0.044046283;-0.094796956;0.013732989;0.10506235;0.014453219;0.035678245;0.0005872976;0.009956597;0.034610823;0.0040985094;-0.03373606;0.061595984;-0.025045715;0.01327654;-0.0014691307;0.09258599;0.026258314;0.043514647;-0.12391727;0.045238607;0.03727933;0.098124765;-0.083314575;-0.05753141;-0.05619724;-0.012002353;-0.0849707;0.07353914;-0.038790062;-0.0013800944;0.0475397;0.0033898843;0.08602842;-0.054665983;-0.10586195;-0.01856893;-0.06702105;-0.057078566;0.0470636;0.01446033;-0.018779023;-0.06048281;0.037093244;0.15044264;0.04288033;-0.096542835;-0.0387777;0.025354981;0.083559595;0.09339045;-0.010797628;0.099817365;0.05727179;-0.11967161;0.007129679;0.027626887;0.08454636;-0.018544402;-0.113928534;-0.027691472;-0.08664141;-0.042434398;-0.10974245;-0.004085783;-0.011912638;0.006900203;0.040022798;0.1709864;0.19394614;0.06540526;-0.008859256;-0.05680794;0.01195252;-0.037797507;0.12196461;-0.048726793;0.07731422;-0.07847224;-0.12349161;0.06821511;0.014746879;0.020868309;0.012237269;-0.019841932;0.02837918;-0.10204872;-0.022502214;-0.046597037"
                            ),
                            get_embeddings(
                                "0.12662199;0.194815;0.1789515;-0.189344;0.017215006;0.107227;0.101292;-0.363615;0.25693;2.1451;-0.46374;0.02707;0.284977;-0.079634205;0.112307504;-0.50945;0.540525;0.1389185;0.19767949;0.235511;-0.198214;-0.29878;0.17462;-0.0020200014;0.368155;0.1103695;0.124337;0.218594;-0.26635;0.22490999;0.056229994;-0.334305;-0.509025;-0.15175;0.438845;0.075072005;-0.010499001;0.20099;-0.0988195;-0.116042;0.27055;0.226325;0.28977;0.18600999;0.60487497;0.024219003;0.0966655;0.1046935;0.1230079;-0.21213;0.348105;-0.556085;0.11197999;-0.14963736;-0.32200497;0.427135;0.206561;-0.46052498;-0.296695;-0.132676;0.22159499;0.080131;0.147205;-0.300125;-0.123065;0.09769;-0.222754;-0.042535003;-0.1125945;-0.145585;0.430615;-0.15545535;-0.539165;0.22494501;-0.265522;0.185925;-0.38420498;0.17808;0.35303;0.137665;-0.10671101;-0.346665;-0.36249;-0.1215065;-0.38496;0.1558615;-0.55141;-0.078995;0.27906;-0.055473503;-0.515935;-0.042335;0.028155003;0.0144765;-0.13368;-0.0862325;-0.231965;0.37178;-0.807715;-0.25516;0.0877185;-0.13328;-0.22059;-0.18643549;0.31719;-0.320805;-0.412485;0.1557255;0.0058695003;-0.61689;-0.267405;0.45969;0.28411;0.06666501;-0.382175;-0.076778;-0.135875;0.390355;0.27829;-0.1083145;-0.387345;-0.22604;-0.15467419;-0.074255;0.940215;0.721645;-0.2253575;0.34421998;0.001975;0.0432795;-0.15211001;-0.21570565;0.287513;0.020844001;0.40269;0.39994;-0.416265;-0.6225;-0.283895;-0.32377;-2.3781;-0.26564002;0.1195855;0.0488672;-0.636485;-0.07402;-0.117886506;0.02253375;0.2150255;-0.317485;0.42596;0.25861502;0.76065505;0.45179498;0.18831;0.0035619996;-0.009750001;-0.591725;0.052755;0.529;-0.338985;-0.41637;-0.083607;0.0863445;0.0713885;0.150533;-0.117039;-0.42689502;0.386245;-0.2531475;0.0061049983;0.07322499;0.2236842;0.14636551;0.388145;-0.10000915;-0.121686995;-0.71659;0.15050301;-0.30448002;-0.121933;-0.28570133;0.114445;0.30609602;-0.0114935;0.190465;-0.44302002;-0.064624995;0.14302;-0.21991;0.012414992;-0.43575;-0.111605994;0.20688;-0.19337;-0.65;-0.421895;-0.49664998;0.312185;0.45157498;0.094989;-0.041724995;0.17490749;0.16777685;-0.49201;-0.08000401;0.215415;0.22986501;0.026506498;0.0985575;-0.373425;0.220818;0.15591;-0.78867;0.0569925;0.26573852;0.403665;-0.025099501;0.368715;-0.099402994;0.300785;0.0118152;0.26258;0.1565415;-0.229035;0.1329225;-0.21478;-0.439305;0.563155;0.09109;-0.25289002;0.59199;0.099556;-0.029243;-0.565945;-0.24259499;-0.15314001;0.30285498;-0.07163275;-0.170714;-0.1132795;-0.16364649;-0.08023082;-0.39605498;-0.014787;0.157345;-0.38957;0.549;0.812855;-0.36905003;-0.1524065;0.056685;0.044139996;-0.23116499;-0.144715;0.108432;0.308665;0.106215;-0.60766006;-0.38910002;-0.06188;0.247855;0.111587;0.34188;0.268795;-0.1952875;0.23257251;0.50689;0.0017250031;-0.021428;-1.01288;-0.082384996;-0.65393996;0.42471498;-0.18193999;0.140027;-0.19679502;-0.460725;-0.148815;0.72951;0.007665001;-0.34631503;0.2571185;-0.41372;-0.047769994;0.839915;-0.25128502;0.00092200004;-0.332085;-0.458135;0.15206501;0.015372001;-0.061462;0.209045;0.0258345;0.1236665;-0.074345;-0.1241765;-0.244185;-0.30306"
                            ),
                            get_embeddings(
                                "-0.08898213;0.20778978;0.031081995;-0.07832522;0.12986743;0.20057625;0.0073040063;-0.30579433;0.011077334;1.7422056;-0.22142495;0.114298;0.16283894;-0.076121666;-0.1415822;-0.001111235;0.13835333;0.9778001;-0.20750779;-0.13754822;-0.002817667;-0.16695355;0.015704332;-0.061691005;0.11773679;0.06990455;-0.14102866;-0.05943789;-0.03434285;-0.018546997;-0.071330875;0.051883016;-0.059427448;0.15768099;0.10044789;-0.13476345;0.103178;0.105727166;0.0067131254;0.0041513043;0.21595192;-0.09615022;0.06354895;-0.10012018;0.15811367;-0.015807888;-0.20384043;-0.02343712;-0.06573285;-0.021815555;0.22364844;-0.0693009;0.11424968;-0.20831756;-0.044495333;-0.07561999;-0.06634233;-0.16477323;-0.060110763;-0.17665066;-0.05549899;0.043891888;0.012118005;0.1608589;-0.01612364;-0.18124782;-0.077353336;0.11514334;-0.06478;-0.16505045;0.052335218;-0.24140933;0.014636884;0.12588555;-0.1296852;0.04888956;-0.04974145;0.07152888;0.1679036;0.16232234;0.0051428876;0.13849834;0.096178755;-0.048829723;-0.09595067;-0.36221778;-0.07908554;-0.108492;-0.052740328;0.029443668;-0.3310789;0.032653667;-0.14322856;-0.11729912;0.34989893;-0.0911253;0.09477003;-0.046793554;-0.18338577;0.13202468;0.10580656;-0.003151468;-0.16522954;-0.04646986;0.11902445;-0.47544444;0.05095711;-0.10302023;-0.0048160013;-0.2663566;0.14251721;0.13248378;0.091447555;-0.1779432;-0.07094611;-0.016414588;0.14097653;0.028556678;0.11619255;0.08227089;-0.06750099;-0.0094047785;-0.08412056;0.053355936;-0.06501255;0.22509898;-0.03765466;-0.008868668;-0.0038476652;-0.036441788;-0.17929678;-0.13613349;0.2218333;-0.09261635;0.116089106;0.1487979;-0.026520565;0.10224977;-0.008945782;-0.12423767;-1.4624289;-0.00018956263;0.12506367;0.17224269;-0.13770035;-0.0029974447;0.036455095;0.006606448;0.09340331;0.079595335;0.014961977;-0.08314267;0.16955201;0.009443809;-0.046841953;-0.05100234;-0.14786434;-0.19623388;-0.12644155;0.059434216;0.22356446;-0.124634;-0.17694922;0.09560631;0.073598;-0.29340702;0.07280178;-0.08237933;0.18877755;0.08425322;0.08777066;-0.11967356;0.09028522;-0.018266333;-0.09472582;-0.09976267;0.14878933;-0.025807552;0.037583925;-0.015163875;0.024262827;-0.059088554;-0.15648851;0.062491007;0.15119833;0.092487;0.0020253328;-0.008235674;-0.08124956;-0.039808445;-0.15360999;-0.0076008905;-0.08912277;0.09809855;0.11078399;0.13437788;0.03655078;-0.151319;0.15313834;0.014900655;-0.10238278;0.05558067;0.1053811;0.07159277;0.06654379;-0.009737669;0.03490668;0.079545334;-0.105555;0.110387824;-0.18054277;0.15626222;0.102483;-0.19522208;-0.068933114;0.22831532;0.011209551;-0.015995553;0.034434114;-0.03451822;-0.11381754;0.17957444;-0.06753049;-0.04218433;0.024216;0.02373796;-0.15514623;-0.089026004;0.18912527;0.07186522;-0.15656713;-0.11597378;-0.047680113;0.019602332;-0.3895327;-0.056831226;-0.064221226;0.018221557;-0.07484266;0.042173;0.07744141;0.0065578963;-0.08282784;-0.0069444445;0.17592523;0.008424442;-0.072848454;-0.062363327;-0.068382345;-0.09497982;0.09042707;0.087672114;0.04803511;-0.19442245;0.089798555;-0.035419658;0.11531387;0.041344743;-0.09109756;-0.14606811;0.07670966;0.012621773;-0.039050896;0.008579993;0.06880178;-0.21616022;-0.06400922;-0.14693037;0.37707877;0.13047102;-0.18849856;-0.02759589;0.06303497;-0.10231756;-0.070172;0.14954366;0.03210567;-0.085903;-0.087266326;0.275296;0.044648666;0.09335666;-0.10949144;-0.018416999;-0.017863223;0.00579039;0.08386078;-0.08962167;0.04974578;-0.17296335;-0.016242988;0.062485225;0.06671666;0.07956987;0.11339715;0.054604;0.010901777;-0.07385555;-0.10720089;0.075812995"
                            )
                        ],
                        [
                            get_embeddings(
                                "-0.021374183;0.13604495;-0.07831777;-0.028588101;0.097610265;0.02715092;0.031180834;-0.14498885;-0.033534806;2.139625;-0.18483542;0.04234712;-0.010559549;-0.07367076;-0.11820254;-0.053550333;-0.05767906;0.87241197;-0.11727233;-0.02188972;-0.022435047;-0.0818205;-0.10393037;-0.05522472;0.045302607;0.034768675;-0.053919572;-0.00028033362;0.04503322;-0.015877329;-0.023164323;0.05722583;-0.068791695;0.087432794;0.13157813;-0.08604374;0.018245969;0.00041015813;-0.042700514;-0.051781226;0.03804254;0.060724434;0.06045948;-0.05763662;0.037700124;0.021165535;-0.07024736;-0.0056724637;0.018310698;0.02346162;-0.018088235;0.016745957;-0.04781882;-0.0019236321;-0.010178297;0.043636918;0.00792577;-0.031339165;-0.006520706;-0.07570847;-0.0184121;0.0038982425;-0.0022324105;0.11650149;0.032083247;-0.113695085;0.0014107154;0.019720437;0.005597189;0.060068052;-0.016877972;-0.013608661;0.013123022;-0.02038405;-0.043653768;0.00639952;0.007916415;-0.026462361;0.016100623;0.11918226;-0.064776234;0.049545985;-0.099236816;-0.025986142;-0.009309912;-0.18366472;-0.0034029384;-0.09605963;0.11454358;0.030340673;-0.1113017;0.019476688;-0.02370956;-0.010302984;0.11289552;-0.076471135;-0.0045078914;-0.021720896;-0.11212702;-0.0089615835;-0.0065533808;0.057157446;-0.11074647;-0.056161944;0.0426859;-0.55500615;0.0643801;0.047150582;0.07843223;-0.036383852;0.05630407;-0.03777022;0.024027707;-0.095818095;-0.01560899;0.0077932444;0.062231127;0.0064143627;0.04870928;-0.030854953;0.013124876;-0.01551484;0.031774696;6.6422755e-05;0.00913846;0.09683793;0.002501166;-0.068780266;0.0022875366;-0.010486983;-0.036666926;0.0038305968;-0.030221032;0.0034285039;0.09647697;0.07019872;-0.07721249;-0.021599507;-0.017541796;-0.027188351;-1.2000506;-0.016899714;0.10088129;0.02843953;-0.0733487;-0.008066414;0.030955724;-0.033210594;-0.013397509;-0.07700101;-0.05541347;-0.008057746;0.093203604;0.0002021413;0.017385377;0.036859233;-0.106635675;-0.067039505;-0.05470714;-0.041578513;0.01418161;-0.008075921;-0.024128448;-0.018330546;0.0077610654;-0.16701241;0.03629566;-0.06576485;0.1355264;-0.003589662;-0.010559309;0.008757868;0.037939433;0.023928013;-0.0881972;0.01879523;-0.056598775;-0.010929245;0.006915337;-0.042900927;0.07769384;-0.026376093;-0.06522478;-0.071479015;-0.032350883;0.033463914;-0.015307132;0.045065545;-0.020253776;-0.05913953;-0.08388686;0.031915996;-0.09966745;0.07329251;-0.011087756;0.0026365344;-0.031003222;-0.055430263;-0.04256649;0.095978074;-0.05771918;-0.028882883;-0.06811448;-0.0027322527;0.11566737;0.009651601;0.07259161;0.0071317814;-0.024443533;-0.04152673;-0.10733361;-0.003318677;-0.071406774;-0.067392126;0.080331236;0.17105944;-0.019322287;0.032347377;-0.023550682;-0.0042366525;-0.048819903;0.0051855007;-0.06094034;0.043695766;-0.004099402;0.030717714;0.03931974;0.14040358;0.06465772;0.0470551;-0.12764946;0.031808138;0.022845916;0.081684195;-0.1157374;-0.05806818;-0.04015337;0.008232837;-0.05691095;0.08185748;-0.00051656517;0.0652176;0.035065;0.02023149;0.10494922;-0.062179018;-0.043732766;-0.06488521;-0.06298614;0.010050197;0.03710239;0.018351682;0.03150051;-0.0019700353;0.032400947;0.10801059;0.07666337;-0.0585964;-0.037321497;0.017290004;0.043632016;0.15329568;-0.016647479;0.093248926;0.029508255;-0.079430066;-0.0004833222;-0.05972807;0.17092262;-0.019503243;-0.05377883;-0.04216179;-0.12282179;-0.07557891;-0.07354912;-0.023173105;-0.02230374;0.034789;0.027705409;0.16726267;0.2018248;0.0900404;0.049107842;-0.07647271;-0.045008525;-0.011693861;0.12460521;-0.057605885;0.0907234;-0.050508097;-0.12336316;0.002875855;-0.009720708;-0.018331982;0.009715552;0.05594824;0.018057555;-0.054397326;-0.07967438;0.009102917"
                            ),
                            get_embeddings(
                                "0.12540357;0.05023324;0.019430429;-0.34626144;0.09135801;0.15918627;-0.08486215;-0.028912576;0.22467314;2.1376858;-0.34969625;-0.068082705;0.120021716;0.1579976;0.044480193;-0.27920848;0.13528457;0.5350028;-0.12082185;-0.07735628;-0.036709998;-0.18838915;0.22706428;-0.24492742;0.263142;-0.10689501;0.10070999;0.16429816;-0.16713677;-0.06748714;0.0038101417;-0.23170029;-0.28337303;0.12168258;0.3188693;-0.26650172;-0.056644287;0.10291828;-0.16071239;-0.038396288;0.26093194;0.015384297;0.09313171;-0.22611447;0.33143112;0.09034288;0.024651447;0.29274872;0.1420693;-0.19049035;0.52582425;-0.16607013;-0.016768575;-0.12199428;-0.20247827;0.21417172;-0.019457715;-0.32397342;-0.17701642;-0.051157434;0.059224002;0.17937043;0.07205428;-0.14859143;0.054663997;-0.1963257;0.033152573;-0.064246;-0.07062642;-0.07968185;0.45022625;-0.04475714;-0.06345886;0.12931404;0.048141714;-0.22897713;-0.10436;-0.023369428;0.10725914;0.25523204;-0.10719943;-0.37756523;-0.34662288;-0.018363858;0.117215276;-0.09013099;-0.2651453;0.07895628;0.3576243;0.17411713;-0.06273186;0.19059572;-0.08914571;0.042361714;0.10474194;-0.2247703;-0.25506395;0.21505387;-0.259422;-0.011308861;0.1489813;-0.28288057;-0.38569328;-0.15261957;0.16443714;-0.5399042;0.028949857;0.24440728;0.015592277;-0.33444643;0.047758568;0.19238427;0.03691142;-0.13383071;-0.1146245;-0.10601642;-0.08836872;0.32061774;0.20886286;0.07522428;-0.15133314;-0.1780357;-0.061615713;-0.04350158;0.084301434;0.4302557;-0.06719943;0.129069;-0.22259258;-0.010258571;0.056952138;-0.18502684;0.039118577;0.086295605;0.14442256;0.27483287;-0.1110624;-0.43793574;-0.12513958;-0.08347942;-1.6710027;-0.29941827;0.15517657;-0.08323757;-0.115692295;0.107359044;0.15678544;-0.19871572;0.25833815;-0.20699473;0.114133716;-0.13839556;0.46015057;0.105349235;0.0063128597;-0.16616745;-0.031715475;-0.16649114;-0.11121256;0.19011715;0.045575712;-0.213817;0.13346757;0.12311571;0.0682463;-0.1942953;-0.07347;-0.27215222;0.3366139;0.03815456;0.043078147;0.13968071;0.10765714;0.15222254;0.082805574;-0.08972728;-0.09348799;-0.45411173;0.15075244;-0.2703623;-0.09217485;0.06053286;0.046843328;0.124230854;0.16543286;0.013319002;-0.02826257;0.10093142;-0.009263714;-0.011855996;-0.3101832;-0.062449146;0.080253564;0.15087543;-0.28456113;-0.14098929;-0.12520814;-0.15383427;0.12280142;0.14633456;-0.034117866;-0.050518718;-0.018757427;0.2111926;-0.107305944;0.10846514;0.07598371;0.29798642;0.06895805;0.061188366;-0.23686416;0.09348857;0.19550072;-0.27328792;-0.12167256;0.24101259;0.20254001;0.06850101;0.12463428;0.009161;0.08004214;0.10304185;0.08728457;0.1384053;-0.15779886;0.053186886;-0.26158717;0.18344843;0.24553715;0.18736286;-0.14181744;0.2535071;-0.064134575;0.09085457;-0.29823402;0.05582842;-0.3215714;0.1373843;-0.114904284;0.05430428;0.109806046;-0.057697497;-0.26435712;-0.17899884;0.03566143;0.020383285;-0.43729487;0.09677957;0.3083566;-0.28486973;0.09915837;-0.015263147;0.04086529;0.045331713;-0.18926944;0.10318385;0.32769427;-0.09215357;-0.17990856;-0.026245283;-0.012829428;0.031441715;-0.14569;0.10964143;0.3304957;0.030272005;-0.08017172;0.07390506;0.49940613;-0.14122972;-0.3276594;-0.064070866;-0.3514183;0.020623142;-0.19045284;0.050992;-0.14555529;-0.042481147;0.03802914;0.32107285;0.21153045;-0.09008515;0.15432476;-0.13025187;-0.023384286;0.20153728;0.20306857;-0.047495;0.02452043;-0.25084314;-0.26655442;-0.028209714;-0.16302684;0.13224141;-0.0040907436;-0.16834728;0.22938286;-0.12449057;-0.14203243;-0.20452043"
                            ),
                            get_embeddings(
                                "0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0"
                            )
                        ]
                    ]
                )
        ).unwrap();

        let expected_prediction = [
            1.3269404077325404,
            -1.7375772811362642,
            -2.0048052240456595
        ];
        assert!( std::iter::zip(expected_prediction, prediction).all(|(l, r)| abs_diff_eq!(l, r, epsilon=1.0e-6)) )

    }

    #[test]
    fn test_predict_model_with_num_features_on_cpu() {
        test_predict_model_with_num_features(false);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_predict_model_with_num_features_on_gpu() {
        test_predict_model_with_num_features(true);
    }

    #[test]
    fn test_predict_model_with_num_cat_features_on_cpu() {
        test_predict_model_with_num_cat_features(false);
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[should_panic]
    fn test_predict_model_with_num_cat_features_on_gpu() {
        test_predict_model_with_num_cat_features(true);
    }

    #[test]
    fn test_predict_model_with_num_cat_text_features_on_cpu() {
        test_predict_model_with_num_cat_text_features(false);
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[should_panic]
    fn test_predict_model_with_num_cat_text_features_on_gpu() {
        test_predict_model_with_num_cat_text_features(true);
    }

    #[test]
    fn test_predict_model_with_num_cat_text_embedding_features_on_cpu() {
        test_predict_model_with_num_cat_text_embedding_features(false);
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[should_panic]
    fn test_predict_model_with_num_cat_text_embedding_features_on_gpu() {
        test_predict_model_with_num_cat_text_embedding_features(true);
    }

    #[test]
    fn get_model_stats() {
        let model = Model::load("tmp/model.bin").unwrap();

        assert_eq!(model.get_cat_features_count(), 1);
        assert_eq!(model.get_float_features_count(), 3);
        assert_eq!(model.get_tree_count(), 1000);
        assert_eq!(model.get_dimensions_count(), 1);
    }

    use std::io::Read;
    fn read_fast<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Vec<u8>> {
        let mut file = std::fs::File::open(path)?;
        let meta = file.metadata()?;
        let size = meta.len() as usize;
        let mut data = Vec::with_capacity(size);
        data.resize(size, 0);
        file.read_exact(&mut data)?;
        Ok(data)
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn calc_prediction_on_gpu() {
        let model = Model::load("tmp/model.bin").unwrap();
        assert!(model.enable_gpu_evaluation());
        let prediction = model
            .calc_model_prediction(
                vec![
                    vec![-10.0, 5.0, 753.0],
                    vec![30.0, 1.0, 760.0],
                    vec![40.0, 0.1, 705.0],
                ],
                vec![
                    vec![String::from("north")],
                    vec![String::from("south")],
                    vec![String::from("south")],
                ],
            )
            .unwrap();

        assert_eq!(prediction[0], 0.9980003729960197);
        assert_eq!(prediction[1], 0.00249414628534181);
        assert_eq!(prediction[2], -0.0013677527881450977);
    }
}
