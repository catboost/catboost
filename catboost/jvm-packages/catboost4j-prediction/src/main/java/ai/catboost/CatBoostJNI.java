package ai.catboost;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import javax.validation.constraints.NotNull;

class CatBoostJNI {
    private static final Logger logger = LoggerFactory.getLogger(CatBoostJNI.class);

    static {
        try {
            NativeLibLoader.initCatBoost();
        } catch (Exception ex) {
            logger.error("Failed to load native library", ex);
            throw new RuntimeException(ex);
        }
    }

    static void checkCall(int ret) throws CatBoostException {
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
            @NotNull double[] result);

    final static native int catBoostModelPredict(
            long handle,
            @Nullable float[] numericFeatures,
            @Nullable int[] catFeatureHashes,
            @NotNull double[] result);

    final static native int catBoostModelPredict(
            long handle,
            @Nullable float[][] numericFeatures,
            @Nullable String[][] catFeatures,
            @NotNull double[] results);

    final static native int catBoostModelPredict(
            long handle,
            @Nullable float[][] numericFeatures,
            @Nullable int[][] catFeatureHashes,
            @NotNull double[] results);
}