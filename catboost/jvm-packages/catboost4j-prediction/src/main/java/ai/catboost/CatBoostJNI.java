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
            final @NotNull long[] handle) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostLoadModelFromFile(fname, handle));
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

    final void catBoostModelGetTreeCount(
            final long handle,
            final @NotNull int[] treeCount) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetTreeCount(handle, treeCount));
    }

    final void catBoostModelGetFeatureNames(
            final long handle,
            @NotNull String[] featureNames) throws CatBoostError {
        CatBoostJNIImpl.checkCall(CatBoostJNIImpl.catBoostModelGetFeatureNames(handle, featureNames));
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
