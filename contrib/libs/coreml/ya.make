PROTO_LIBRARY()

LICENSE(
    BSD
)



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
EXCLUDE_TAGS(JAVA_PROTO)

END()
