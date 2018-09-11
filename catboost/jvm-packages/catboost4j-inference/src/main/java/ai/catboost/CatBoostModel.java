package ai.catboost;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import javax.annotation.Nullable;
import javax.validation.constraints.NotNull;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * CatBoost model, supports basic model inference.
 */
public class CatBoostModel implements AutoCloseable {
    private static final Log logger = LogFactory.getLog(CatBoostModel.class);

    // handle to native C++ model
    private long handle = 0;
    private int predictionDimension = 0;
    private int treeCount = 0;
    private int numericFeatureCount = 0;
    private int categoricFeatureCount = 0;

    /**
     * Load CatBoost model from file modelPath.
     *
     * @param modelPath Path to the model.
     * @return          CatBoost model.
     * @throws CatBoostException
     */
    @NotNull
    public static CatBoostModel loadModel(@NotNull String modelPath) throws CatBoostException {
        final long[] handles = new long[1];
        final int[] predictionDimension = new int[1];
        final int[] treeCount = new int[1];
        final int[] numericFeatureCount = new int[1];
        final int[] catFeatureCount = new int[1];

        final CatBoostModel model = new CatBoostModel();
        CatBoostJNI.checkCall(CatBoostJNI.catBoostLoadModelFromFile(modelPath, handles));
        model.handle = handles[0];

        try {
            CatBoostJNI.checkCall(CatBoostJNI.catBoostModelGetPredictionDimension(model.handle, predictionDimension));
            CatBoostJNI.checkCall(CatBoostJNI.catBoostModelGetTreeCount(model.handle, treeCount));
            CatBoostJNI.checkCall(CatBoostJNI.catBoostModelGetNumericFeatureCount(model.handle, numericFeatureCount));
            CatBoostJNI.checkCall(CatBoostJNI.catBoostModelGetCategoricalFeatureCount(model.handle, catFeatureCount));
        } catch (CatBoostException e) {
            model.close();
            throw e;
        }

        model.predictionDimension = predictionDimension[0];
        model.treeCount = treeCount[0];
        model.numericFeatureCount = numericFeatureCount[0];
        model.categoricFeatureCount = catFeatureCount[0];

        return model;
    }

    /**
     * Load CatBoost model from stream.
     *
     * @param in Input stream containing model.
     * @return   CatBoost model.
     * @throws CatBoostException
     * @throws IOException
     */
    @NotNull
    public static CatBoostModel loadModel(InputStream in) throws CatBoostException, IOException {
        final long[] handles = new long[1];
        final int[] predictionDimension = new int[1];
        final int[] treeCount = new int[1];
        final int[] numericFeatureCount = new int[1];
        final int[] catFeatureCount = new int[1];
        final byte[] copyBuffer = new byte[4 * 1024];

        int bytesRead;
        final ByteArrayOutputStream out = new ByteArrayOutputStream();

        while ((bytesRead = in.read(copyBuffer)) != -1) {
            out.write(copyBuffer, 0, bytesRead);
        }

        final CatBoostModel model = new CatBoostModel();
        CatBoostJNI.checkCall(CatBoostJNI.catBoostLoadModelFromArray(out.toByteArray(), handles));
        model.handle = handles[0];

        try {
            CatBoostJNI.checkCall(CatBoostJNI.catBoostModelGetPredictionDimension(model.handle, predictionDimension));
            CatBoostJNI.checkCall(CatBoostJNI.catBoostModelGetTreeCount(model.handle, treeCount));
            CatBoostJNI.checkCall(CatBoostJNI.catBoostModelGetNumericFeatureCount(model.handle, numericFeatureCount));
            CatBoostJNI.checkCall(CatBoostJNI.catBoostModelGetCategoricalFeatureCount(model.handle, catFeatureCount));
        } catch (CatBoostException e) {
            model.close();
            throw e;
        }

        model.predictionDimension = predictionDimension[0];
        model.treeCount = treeCount[0];
        model.numericFeatureCount = numericFeatureCount[0];
        model.categoricFeatureCount = catFeatureCount[0];

        return model;
    }

    /**
     * Hash categorical feature.
     *
     * @param catFeature String representation of categorical feature.
     * @return           Hash for categorical feature.
     * @throws CatBoostException
     */
    static int hashCategoricalFeature(@NotNull String catFeature) throws CatBoostException {
        int hash[] = new int[1];
        CatBoostJNI.checkCall(CatBoostJNI.catBoostHashCatFeature(catFeature, hash));
        return hash[0];
    }

    /**
     * Hash array of categorical features.
     *
     * May be cheaper to call this function once instead of calling {@link #hashCategoricalFeature(String)} for each
     * categorical feature in array.
     *
     * @param catFeatures Array of categorical features.
     * @param hashes      Array of hashes of categorical features.
     * @throws CatBoostException
     */
    static void hashCategoricalFeatures(
            @NotNull String[] catFeatures,
            @NotNull int[] hashes) throws CatBoostException {
        CatBoostJNI.checkCall(CatBoostJNI.catBoostHashCatFeatures(catFeatures, hashes));
    }

    /**
     * Hash array of categorical features.
     *
     * @param catFeatures Array of categorical features.
     * @return            Array of hashes of categorical features.
     * @throws CatBoostException
     */
    @NotNull
    static int[] hashCategoricalFeatures(@NotNull String[] catFeatures) throws CatBoostException {
        final int[] hashes = new int[catFeatures.length];
        hashCategoricalFeatures(catFeatures, hashes);
        return hashes;
    }

    /**
     * @return Dimension of model prediction.
     */
    public int getPredictionDimension() {
        return predictionDimension;
    }

    /**
     * @return Number of tees in model.
     */
    public int getTreeCount() {
        return treeCount;
    }

    /**
     * @return Number of numeric features used by the model.
     */
    public int getNumericFeatureCount() {
        return numericFeatureCount;
    }

    /**
     * @return Number of categorical features used by the model.
     */
    public int getCategoricFeatureCount() {
        return categoricFeatureCount;
    }

    /**
     * Apply model to object defined by features.
     *
     * @param numericFeatures Numeric features.
     * @param catFeatures     Categoric features.
     * @param prediction      Model predictions.
     * @throws CatBoostException
     */
    public void predict(
            @Nullable float[] numericFeatures,
            @Nullable String[] catFeatures,
            @NotNull CatBoostPredictions prediction) throws CatBoostException {
        CatBoostJNI.checkCall(CatBoostJNI.catBoostModelPredict(
                handle,
                numericFeatures,
                catFeatures,
                prediction.getRawData()));
    }

    /**
     * Same as {@link #predict(float[], String[], CatBoostPredictions)}, but returns prediction instead of taking it
     * as third parameter.
     *
     * @param numericFeatures Numeric features.
     * @param catFeatures     Categoric features.
     * @return                Model predictions.
     * @throws CatBoostException
     */
    @NotNull
    public CatBoostPredictions predict(
            @Nullable float[] numericFeatures,
            @Nullable String[] catFeatures) throws CatBoostException {
        final CatBoostPredictions prediction = new CatBoostPredictions(1, getPredictionDimension());
        predict(numericFeatures, catFeatures, prediction);
        return prediction;
    }

    /**
     * Same as {@link #predict(float[], String[], CatBoostPredictions)}, but accept categoric features as hashes
     * computed by {@link #hashCategoricalFeature(String)}.
     *
     * @param numericFeatures  Numeric features.
     * @param catFeatureHashes Categoric feature hashes.
     * @param prediction       Model predictions.
     * @throws CatBoostException
     */
    public void predict(
            @Nullable float[] numericFeatures,
            @Nullable int[] catFeatureHashes,
            @NotNull CatBoostPredictions prediction) throws CatBoostException {
        CatBoostJNI.checkCall(CatBoostJNI.catBoostModelPredict(
                handle,
                numericFeatures,
                catFeatureHashes,
                prediction.getRawData()));
    }

    /**
     * Same as {@link #predict(float[], int[], CatBoostPredictions)}, but returns prediction instead of taking it as
     * third parameter.
     *
     * @param numericFeatures  Numeric features.
     * @param catFeatureHashes Categoric feature hashes.
     * @return                 Model predictions.
     * @throws CatBoostException
     */
    @NotNull
    public CatBoostPredictions predict(
            @Nullable float[] numericFeatures,
            @Nullable int[] catFeatureHashes) throws CatBoostException {
        final CatBoostPredictions prediction = new CatBoostPredictions(1, getPredictionDimension());
        predict(numericFeatures, catFeatureHashes, prediction);
        return prediction;
    }

    /**
     * Apply model to a batch of objects.
     *
     * @param numericFeatures Numeric features matrix.
     * @param catFeatures     Categoric features matrix.
     * @param prediction      Model predictions.
     * @throws CatBoostException
     */
    public void predict(
            @Nullable float[][] numericFeatures,
            @Nullable String[][] catFeatures,
            @NotNull CatBoostPredictions prediction) throws CatBoostException {
        CatBoostJNI.checkCall(CatBoostJNI.catBoostModelPredict(
                handle,
                numericFeatures,
                catFeatures,
                prediction.getRawData()));
    }

    /**
     * Same as {@link #predict(float[][], String[][], CatBoostPredictions)}, but returns prediction instead of taking
     * it as third parameter.
     *
     * @param numericFeatures Numeric features.
     * @param catFeatures     Categoric features.
     * @return                Model predictions.
     * @throws CatBoostException
     */
    @NotNull
    public CatBoostPredictions predict(
            @Nullable float[][] numericFeatures,
            @Nullable String[][] catFeatures) throws CatBoostException {
        if (numericFeatures == null && catFeatures == null) {
            throw new CatBoostException("both arguments are null");
        }

        final CatBoostPredictions prediction = new CatBoostPredictions(
            numericFeatures == null ? catFeatures.length : numericFeatures.length,
            getPredictionDimension());
        predict(numericFeatures, catFeatures, prediction);
        return prediction;
    }

    /**
     * Same as {@link #predict(float[][], String[][], CatBoostPredictions)}, but accept categoric features as hashes
     * computed by {@link #hashCategoricalFeature(String)}.
     *
     * @param numericFeatures  Numeric features.
     * @param catFeatureHashes Categoric feature hashes.
     * @param prediction       Model predictions.
     * @throws CatBoostException
     */
    public void predict(
            @Nullable float[][] numericFeatures,
            @Nullable int[][] catFeatureHashes,
            @NotNull CatBoostPredictions prediction) throws CatBoostException {
        CatBoostJNI.checkCall(CatBoostJNI.catBoostModelPredict(
            handle,
            numericFeatures,
            catFeatureHashes,
            prediction.getRawData()));
    }

    /**
     * Same as {@link #predict(float[][], String[][], CatBoostPredictions)}, but returns predictions instead of taking
     * it as third parameter.
     *
     * @param numericFeatures  Numeric features.
     * @param catFeatureHashes Categoric feature hashes.
     * @return                 Model predictions.
     * @throws CatBoostException
     */
    @NotNull
    public CatBoostPredictions predict(
            @Nullable float[][] numericFeatures,
            @Nullable int[][] catFeatureHashes) throws CatBoostException {
        if (numericFeatures == null && catFeatureHashes == null) {
            throw new CatBoostException("both arguments are null");
        }

        final CatBoostPredictions prediction = new CatBoostPredictions(
                numericFeatures == null ? catFeatureHashes.length : numericFeatures.length,
                getPredictionDimension());
        predict(numericFeatures, catFeatureHashes, prediction);
        return prediction;
    }

    @Override
    protected void finalize() throws Throwable {
        try {
            dispose();
        } finally {
            super.finalize();
        }
    }

    private synchronized void dispose() {
        if (handle != 0) {
            CatBoostJNI.catBoostFreeModel(handle);
            handle = 0;
        }
    }

    @Override
    public void close() {
        dispose();
    }
}