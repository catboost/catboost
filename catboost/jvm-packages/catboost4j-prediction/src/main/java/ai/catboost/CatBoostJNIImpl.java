package ai.catboost;

import javax.annotation.Nullable;
import javax.validation.constraints.NotNull;

class CatBoostJNIImpl {
    final static void checkCall(int ret) throws CatBoostException {
        if (ret != 0) {
            throw new CatBoostException(catBoostGetLastError());
        }
    }

    final static native String catBoostGetLastError();

    final static native int catBoostHashCatFeature(
            @NotNull String catFeature,
            @NotNull int[] hash);

    final static native int catBoostHashCatFeatures(
            @NotNull String[] catFeatures,
            @NotNull int[] hashes);

    final static native int catBoostLoadModelFromFile(
            @NotNull String fname,
            @NotNull long[] handle);

    final static native int catBoostLoadModelFromArray(
            @NotNull byte[] data,
            @NotNull long[] handle);

    final static native int catBoostFreeModel(long handle);

    final static native int catBoostModelGetPredictionDimension(
            long handle,
            @NotNull int[] classesCount);

    final static native int catBoostModelGetNumericFeatureCount(
            long handle,
            @NotNull int[] numericFeatureCount);

    final static native int catBoostModelGetCategoricalFeatureCount(
            long handle,
            @NotNull int[] catFeatureCount);

    final static native int catBoostModelGetTreeCount(
            long handle,
            @NotNull int[] treeCount);

    final static native int catBoostModelPredict(
            long handle,
            @Nullable float[] numericFeatures,
            @Nullable String[] catFeatures,
            @NotNull double[] predictions);

    final static native int catBoostModelPredict(
            long handle,
            @Nullable float[] numericFeatures,
            @Nullable int[] catFeatureHashes,
            @NotNull double[] predictions);

    final static native int catBoostModelPredict(
            long handle,
            @Nullable float[][] numericFeatures,
            @Nullable String[][] catFeatures,
            @NotNull double[] predictions);

    final static native int catBoostModelPredict(
            long handle,
            @Nullable float[][] numericFeatures,
            @Nullable int[][] catFeatureHashes,
            @NotNull double[] predictions);
}