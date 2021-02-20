package ai.catboost;

import javax.annotation.Nullable;
import javax.validation.constraints.NotNull;

class CatBoostJNI {
    final void catBoostHashCatFeature(
            final @NotNull String catFeature,
            final @NotNull int[] hash) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostHashCatFeature(catFeature, hash));
    }

    final void catBoostHashCatFeatures(
            final @NotNull String[] catFeatures,
            final @NotNull int[] hashes) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostHashCatFeatures(catFeatures, hashes));
    }

    final void catBoostLoadModelFromFile(
            final @NotNull String fname,
            final @NotNull long[] handle,
            final @Nullable String formatName) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostLoadModelFromFile(fname, handle, formatName));
    }

    final void catBoostLoadModelFromArray(
            final @NotNull byte[] data,
            final @NotNull long[] handle) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostLoadModelFromArray(data, handle));
    }

    final void catBoostFreeModel(final long handle) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostFreeModel(handle));
    }

    final void catBoostModelGetPredictionDimension(
            final long handle,
            final @NotNull int[] classesCount) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetPredictionDimension(handle, classesCount));
    }

    final void catBoostModelGetUsedNumericFeatureCount(
            final long handle,
            final @NotNull int[] numericFeatureCount) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetUsedNumericFeatureCount(handle, numericFeatureCount));
    }

    final void catBoostModelGetUsedCategoricalFeatureCount(
            final long handle,
            final @NotNull int[] catFeatureCount) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetUsedCategoricalFeatureCount(handle, catFeatureCount));
    }

    final void catBoostModelGetFlatFeatureVectorExpectedSize(
            final long handle,
            final @NotNull int[] featureVectorExpectedSize) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetFlatFeatureVectorExpectedSize(handle, featureVectorExpectedSize));
    }

    final void catBoostModelGetMetadata(
            final long handle,
            final @NotNull String[][] keys,
            final @NotNull String[][] values) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetMetadata(handle, keys, values));
    }

    final void catBoostModelGetFloatFeatures(
            final long handle,
            final @NotNull String[][] names,
            final @NotNull int[][] flat_feature_index,
            final @NotNull int[][] feature_index,
            final @NotNull int[][] has_nans,
            final @NotNull String[][] nan_value_treatment) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetFloatFeatures(handle, names, flat_feature_index, feature_index, has_nans, nan_value_treatment));
    }

    final void catBoostModelGetCatFeatures(
            final long handle,
            final @NotNull String[][] names,
            final @NotNull int[][] flat_feature_index,
            final @NotNull int[][] feature_index) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetCatFeatures(handle, names, flat_feature_index, feature_index));
    }

    final void catBoostModelGetTextFeatures(
            final long handle,
            final @NotNull String[][] names,
            final @NotNull int[][] flat_feature_index,
            final @NotNull int[][] feature_index) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetTextFeatures(handle, names, flat_feature_index, feature_index));
    }

    final void catBoostModelGetUsedFeatureIndices(
            final long handle,
            final @NotNull int[][] featureIndices) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetUsedFeatureIndices(handle, featureIndices));
    }

    final void catBoostModelGetTreeCount(
            final long handle,
            final @NotNull int[] treeCount) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetTreeCount(handle, treeCount));
    }

    final void catBoostModelPredict(
            final long handle,
            final @Nullable float[] numericFeatures,
            final @Nullable String[] catFeatures,
            final @NotNull double[] predictions) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelPredict(handle, numericFeatures, catFeatures, predictions));
    }

    final void catBoostModelPredict(
            final long handle,
            final @Nullable float[] numericFeatures,
            final @Nullable int[] catFeatureHashes,
            final @NotNull double[] predictions) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelPredict(handle, numericFeatures, catFeatureHashes, predictions));
    }

    final void catBoostModelPredict(
            final long handle,
            final @Nullable float[][] numericFeatures,
            final @Nullable String[][] catFeatures,
            final @NotNull double[] predictions) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelPredict(handle, numericFeatures, catFeatures, predictions));
    }

    final void catBoostModelPredict(
            final long handle,
            final @Nullable float[][] numericFeatures,
            final @Nullable int[][] catFeatureHashes,
            final @NotNull double[] predictions) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelPredict(handle, numericFeatures, catFeatureHashes, predictions));
    }
}
