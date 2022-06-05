def binarize_float_features(model, binary_features, float_features):
    binary_feature_index = 0
    for i in range(len(model.float_feature_borders)):
        for border in model.float_feature_borders[i]:
            binary_features[binary_feature_index] += 1 if ((float_features[model.float_features_index[i]] > border) or math.isnan(float_features[model.float_features_index[i]])) else 0
        binary_feature_index += 1
    return binary_feature_index
