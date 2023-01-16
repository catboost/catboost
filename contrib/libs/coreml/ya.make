PROTO_LIBRARY()

LICENSE(
    BSD-3-Clause
    MIT
)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)



SRCS(
    ArrayFeatureExtractor.proto
    CategoricalMapping.proto
    DataStructures.proto
    DictVectorizer.proto
    FeatureTypes.proto
    FeatureVectorizer.proto
    GLMClassifier.proto
    GLMRegressor.proto
    Identity.proto
    Imputer.proto
    Model.proto
    NeuralNetwork.proto
    Normalizer.proto
    OneHotEncoder.proto
    SVM.proto
    Scaler.proto
    TreeEnsemble.proto
)

# TODO: remove (DEVTOOLS-3496)
EXCLUDE_TAGS(
    GO_PROTO
    JAVA_PROTO
)

END()
