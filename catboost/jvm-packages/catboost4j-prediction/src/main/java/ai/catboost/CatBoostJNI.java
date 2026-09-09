package ai.catboost;

import org.jspecify.annotations.Nullable;

class CatBoostJNI {
    final void catBoostHashCatFeature(
            final String catFeature,
            final int[] hash) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostHashCatFeature(catFeature, hash));
    }

    final void catBoostHashCatFeatures(
            final String[] catFeatures,
            final int[] hashes) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostHashCatFeatures(catFeatures, hashes));
    }

    final void catBoostLoadModelFromFile(
            final String fname,
            final long[] handle,
            final @Nullable String formatName) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostLoadModelFromFile(fname, handle, formatName));
    }

    final void catBoostLoadModelFromArray(
            final byte[] data,
            final long[] handle,
            final @Nullable String formatName) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostLoadModelFromArray(data, handle, formatName));
    }

    final void catBoostFreeModel(final long handle) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostFreeModel(handle));
    }

    final void catBoostModelGetSupportedEvaluatorTypes(
            final long handle,
            final String[][] evaluatorTypes) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetSupportedEvaluatorTypes(handle, evaluatorTypes));
    }

    final void catBoostModelSetEvaluatorType(
            final long handle,
            final String evaluatorType) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelSetEvaluatorType(handle, evaluatorType));
    }

    final void catBoostModelGetEvaluatorType(
            final long handle,
            final String[] evaluatorType) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetEvaluatorType(handle, evaluatorType));
    }

    final void catBoostModelGetPredictionDimension(
            final long handle,
            final int[] classesCount) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetPredictionDimension(handle, classesCount));
    }

    final void catBoostModelGetUsedNumericFeatureCount(
            final long handle,
            final int[] numericFeatureCount) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetUsedNumericFeatureCount(handle, numericFeatureCount));
    }

    final void catBoostModelGetUsedCategoricalFeatureCount(
            final long handle,
            final int[] catFeatureCount) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetUsedCategoricalFeatureCount(handle, catFeatureCount));
    }

    final void catBoostModelGetUsedTextFeatureCount(
            final long handle,
            final int[] textFeatureCount) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetUsedTextFeatureCount(handle, textFeatureCount));
    }

    final void catBoostModelGetUsedEmbeddingFeatureCount(
            final long handle,
            final int[] embeddingFeatureCount) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetUsedEmbeddingFeatureCount(handle, embeddingFeatureCount));
    }

    final void catBoostModelGetFlatFeatureVectorExpectedSize(
            final long handle,
            final int[] featureVectorExpectedSize) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetFlatFeatureVectorExpectedSize(handle, featureVectorExpectedSize));
    }

    final void catBoostModelGetMetadata(
            final long handle,
            final String[][] keys,
            final String[][] values) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetMetadata(handle, keys, values));
    }

    final void catBoostModelGetFloatFeatures(
            final long handle,
            final String[][] names,
            final int[][] flat_feature_index,
            final int[][] feature_index,
            final int[][] has_nans,
            final String[][] nan_value_treatment) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetFloatFeatures(handle, names, flat_feature_index, feature_index, has_nans, nan_value_treatment));
    }

    final void catBoostModelGetCatFeatures(
            final long handle,
            final String[][] names,
            final int[][] flat_feature_index,
            final int[][] feature_index) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetCatFeatures(handle, names, flat_feature_index, feature_index));
    }

    final void catBoostModelGetTextFeatures(
            final long handle,
            final String[][] names,
            final int[][] flat_feature_index,
            final int[][] feature_index) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetTextFeatures(handle, names, flat_feature_index, feature_index));
    }

    final void catBoostModelGetEmbeddingFeatures(
            final long handle,
            final String[][] names,
            final int[][] flat_feature_index,
            final int[][] feature_index) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetEmbeddingFeatures(handle, names, flat_feature_index, feature_index));
    }

    final void catBoostModelGetUsedFeatureIndices(
            final long handle,
            final int[][] featureIndices) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetUsedFeatureIndices(handle, featureIndices));
    }

    final void catBoostModelGetTreeCount(
            final long handle,
            final int[] treeCount) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetTreeCount(handle, treeCount));
    }

    final void catBoostModelPredict(
            final long handle,
            final float @Nullable [] numericFeatures,
            final String @Nullable [] catFeatures,
            final String @Nullable [] textFeatures,
            final float @Nullable [][] embeddingFeatures,
            final double[] predictions) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelPredict(handle, numericFeatures, catFeatures, textFeatures, embeddingFeatures, predictions));
    }

    final void catBoostModelPredict(
            final long handle,
            final float @Nullable [] numericFeatures,
            final int @Nullable [] catFeatureHashes,
            final String @Nullable [] textFeatures,
            final float @Nullable [][] embeddingFeatures,
            final double[] predictions) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelPredict(handle, numericFeatures, catFeatureHashes, textFeatures, embeddingFeatures, predictions));
    }

    final void catBoostModelPredict(
            final long handle,
            final float @Nullable [][] numericFeatures,
            final String @Nullable [][] catFeatures,
            final String @Nullable [][] textFeatures,
            final float @Nullable [][][] embeddingFeatures,
            final double[] predictions) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelPredict(handle, numericFeatures, catFeatures, textFeatures, embeddingFeatures, predictions));
    }

    final void catBoostModelPredict(
            final long handle,
            final float @Nullable [][] numericFeatures,
            final int @Nullable [][] catFeatureHashes,
            final String @Nullable [][] textFeatures,
            final float @Nullable [][][] embeddingFeatures,
            final double[] predictions) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelPredict(handle, numericFeatures, catFeatureHashes, textFeatures, embeddingFeatures, predictions));
    }

    final void catBoostModelPredictTransposed(
            final long handle,
            final float[][] numericFeatures,
            final double[] predictions) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelPredictTransposed(handle, numericFeatures, predictions));
    }
}
