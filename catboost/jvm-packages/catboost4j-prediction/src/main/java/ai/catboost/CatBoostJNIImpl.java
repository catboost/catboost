package ai.catboost;

import javax.annotation.Nullable;
import javax.validation.constraints.NotNull;

class CatBoostJNIImpl {
    final static void checkCall(@Nullable String message) throws CatBoostError {
        if (message != null) {
            throw new CatBoostError(message);
        }
    }

    @Nullable
    final static native String catBoostHashCatFeature(
            @NotNull String catFeature,
            @NotNull int[] hash);

    @Nullable
    final static native String catBoostHashCatFeatures(
            @NotNull String[] catFeatures,
            @NotNull int[] hashes);

    @Nullable
    final static native String catBoostLoadModelFromFile(
            @NotNull String fname,
            @NotNull long[] handle);

    @Nullable
    final static native String catBoostLoadModelFromArray(
            @NotNull byte[] data,
            @NotNull long[] handle);

    @Nullable
    final static native String catBoostFreeModel(long handle);

    @Nullable
    final static native String catBoostModelGetPredictionDimension(
            long handle,
            @NotNull int[] classesCount);

    @Nullable
    final static native String catBoostModelGetUsedNumericFeatureCount(
            long handle,
            @NotNull int[] numericFeatureCount);

    @Nullable
    final static native String catBoostModelGetUsedCategoricalFeatureCount(
            long handle,
            @NotNull int[] catFeatureCount);

    @Nullable
    final static native String catBoostModelGetFlatFeatureVectorExpectedSize(
            long handle,
            @NotNull int[] featureVectorExpectedSize);

    @Nullable
    final static native String catBoostModelGetTreeCount(
            long handle,
            @NotNull int[] treeCount);

    @Nullable
    final static native String catBoostModelGetFeatureNames(
            long handle,
            @NotNull String[] featureNames);

    @Nullable
    final static native String catBoostModelPredict(
            long handle,
            @Nullable float[] numericFeatures,
            @Nullable String[] catFeatures,
            @NotNull double[] predictions);

    @Nullable
    final static native String catBoostModelPredict(
            long handle,
            @Nullable float[] numericFeatures,
            @Nullable int[] catFeatureHashes,
            @NotNull double[] predictions);

    @Nullable
    final static native String catBoostModelPredict(
            long handle,
            @Nullable float[][] numericFeatures,
            @Nullable String[][] catFeatures,
            @NotNull double[] predictions);

    @Nullable
    final static native String catBoostModelPredict(
            long handle,
            @Nullable float[][] numericFeatures,
            @Nullable int[][] catFeatureHashes,
            @NotNull double[] predictions);
}