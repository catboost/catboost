package ai.catboost;

import org.jetbrains.annotations.Contract;
import org.jspecify.annotations.Nullable;

class CatBoostJNIImpl {
    @Contract("!null -> fail")
    static void checkCall(@Nullable String message) throws CatBoostError {
        if (message != null) {
            throw new CatBoostError(message);
        }
    }

    static native @Nullable String catBoostHashCatFeature(
            String catFeature,
            int[] hash);

    static native @Nullable String catBoostHashCatFeatures(
            String[] catFeatures,
            int[] hashes);

    static native @Nullable String catBoostLoadModelFromFile(
            String fname,
            long[] handle,
            @Nullable String formatName);

    static native @Nullable String catBoostLoadModelFromArray(
            byte[] data,
            long[] handle,
            @Nullable String formatName);

    static native @Nullable String catBoostFreeModel(long handle);

    static native @Nullable String catBoostModelGetSupportedEvaluatorTypes(
            long handle,
            String[][] evaluatorTypes);

    static native @Nullable String catBoostModelSetEvaluatorType(
            long handle,
            String evaluatorType);

    static native @Nullable String catBoostModelGetEvaluatorType(
            long handle,
            String[] evaluatorType);

    static native @Nullable String catBoostModelGetPredictionDimension(
            long handle,
            int[] classesCount);

    static native @Nullable String catBoostModelGetUsedNumericFeatureCount(
            long handle,
            int[] numericFeatureCount);

    static native @Nullable String catBoostModelGetUsedCategoricalFeatureCount(
            long handle,
            int[] catFeatureCount);

    static native @Nullable String catBoostModelGetUsedTextFeatureCount(
            long handle,
            int[] textFeatureCount);

    static native @Nullable String catBoostModelGetUsedEmbeddingFeatureCount(
            long handle,
            int[] embeddingFeatureCount);

    static native @Nullable String catBoostModelGetFlatFeatureVectorExpectedSize(
            long handle,
            int[] featureVectorExpectedSize);

    static native @Nullable String catBoostModelGetMetadata(
            long handle,
            String[][] keys,
            String[][] values);

    static native @Nullable String catBoostModelGetFloatFeatures(
            long handle,
            String[][] names,
            int[][] flat_feature_index,
            int[][] feature_index,
            int[][] has_nans,
            String[][] nan_value_treatment);

    static native @Nullable String catBoostModelGetCatFeatures(
            long handle,
            String[][] names,
            int[][] flat_feature_index,
            int[][] feature_index);

    static native @Nullable String catBoostModelGetTextFeatures(
            long handle,
            String[][] names,
            int[][] flat_feature_index,
            int[][] feature_index);

    static native @Nullable String catBoostModelGetEmbeddingFeatures(
            long handle,
            String[][] names,
            int[][] flat_feature_index,
            int[][] feature_index);

    static native @Nullable String catBoostModelGetUsedFeatureIndices(
            long handle,
            int[][] featureIndices);

    static native @Nullable String catBoostModelGetTreeCount(
            long handle,
            int[] treeCount);

    static native @Nullable String catBoostModelPredict(
            long handle,
            float @Nullable [] numericFeatures,
            String @Nullable [] catFeatures,
            String @Nullable [] textFeatures,
            float @Nullable [][] embeddingFeatures,
            double[] predictions);

    static native @Nullable String catBoostModelPredict(
            long handle,
            float @Nullable [] numericFeatures,
            int @Nullable [] catFeatureHashes,
            String @Nullable [] textFeatures,
            float @Nullable [][] embeddingFeatures,
            double[] predictions);

    static native @Nullable String catBoostModelPredict(
            long handle,
            float @Nullable [][] numericFeatures,
            String @Nullable [][] catFeatures,
            String @Nullable [][] textFeatures,
            float @Nullable [][][] embeddingFeatures,
            double[] predictions);

    static native @Nullable String catBoostModelPredict(
            long handle,
            float @Nullable [][] numericFeatures,
            int @Nullable [][] catFeatureHashes,
            String @Nullable [][] textFeatures,
            float @Nullable [][][] embeddingFeatures,
            double[] predictions);

    static native @Nullable String catBoostModelPredictTransposed(
            long handle,
            float[][] numericFeatures,
            double[] predictions);
}
