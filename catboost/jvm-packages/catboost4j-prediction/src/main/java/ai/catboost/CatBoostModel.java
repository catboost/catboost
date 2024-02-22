package ai.catboost;

import javax.annotation.Nullable;
import javax.validation.constraints.NotNull;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

import ai.catboost.common.NativeLib;

/**
 * CatBoost model, supports basic model application.
 */
public class CatBoostModel implements AutoCloseable {
    // handle to native C++ model
    private long handle = 0;
    private int predictionDimension = 0;
    private int treeCount = 0;
    private int usedNumericFeatureCount = 0;
    private int usedCategoricFeatureCount = 0;
    private int usedTextFeatureCount = 0;
    private int usedEmbeddingFeatureCount = 0;
    private String[] featureNames;
    private Map<String, String> metadata = new HashMap<String, String>();
    private List<Feature> features = new ArrayList<Feature>();
    private static CatBoostJNI implLibrary = null;

    public enum FormulaEvaluatorType {
        CPU,
        GPU
    }

    public static abstract class Feature {
        private String name;
        private int featureIndex;
        private int flatFeatureIndex;
        private boolean usedInModel;

        protected Feature(String name, int featureIndex, int flatFeatureIndex, boolean usedInModel) {
            this.name = name;
            this.featureIndex = featureIndex;
            this.flatFeatureIndex = flatFeatureIndex;
            this.usedInModel = usedInModel;
        }

        public String getName() {
            return name;
        }

        public int getFeatureIndex() {
            return featureIndex;
        }

        public int getFlatFeatureIndex() {
            return flatFeatureIndex;
        }

        public boolean isUsedInModel() {
            return usedInModel;
        }
    }

    public static final class TextFeature extends Feature {
        protected TextFeature(String name, int featureIndex, int flatFeatureIndex, boolean usedInModel) {
            super(name, featureIndex, flatFeatureIndex, usedInModel);
        }
    }

    public static final class FloatFeature extends Feature {
        public enum NanValueTreatment {
            AsIs,
            AsTrue,
            AsFalse,
        }

        private NanValueTreatment nanValueTreatment;
        private boolean hasNans;

        protected FloatFeature(String name, int featureIndex, int flatFeatureIndex, boolean usedInModel, int hasNans, String nanValueTreatment) {
            super(name, featureIndex, flatFeatureIndex, usedInModel);
            this.hasNans = hasNans > 0;
            this.nanValueTreatment = NanValueTreatment.valueOf(nanValueTreatment);
        }

        public boolean hasNans() {
            return hasNans;
        }

        public NanValueTreatment getNanValueTreatment() {
            return nanValueTreatment;
        }
    }

    public static final class CatFeature extends Feature {
        protected CatFeature(String name, int featureIndex, int flatFeatureIndex, boolean usedInModel) {
            super(name, featureIndex, flatFeatureIndex, usedInModel);
        }
    }

    public static final class EmbeddingFeature extends Feature {
        protected EmbeddingFeature(String name, int featureIndex, int flatFeatureIndex, boolean usedInModel) {
            super(name, featureIndex, flatFeatureIndex, usedInModel);
        }
    }

    static {
        try {
            NativeLib.smartLoad("catboost4j-prediction");
        } catch (Exception ex) {
            throw new RuntimeException("Failed to load catboost4j-prediction native library", ex);
        }
        implLibrary = new CatBoostJNI();
    }


    private CatBoostModel(long handle) throws CatBoostError {
        this.handle = handle;
        final int[] predictionDimension = new int[1];
        final int[] treeCount = new int[1];
        final int[] usedNumericFeatureCount = new int[1];
        final int[] usedCatFeatureCount = new int[1];
        final int[] usedTextFeatureCount = new int[1];
        final int[] usedEmbeddingFeatureCount = new int[1];
        final int[] featureVectorExpectedSize = new int[1];
        final String[][] modelMetadataKeys = new String[1][];
        final String[][] modelMetadataValues = new String[1][];
        final String[][] floatFeatureNames = new String[1][];
        final int[][] floatFlatFeatureIndex = new int[1][];
        final int[][] floatFeatureIndex = new int[1][];
        final int[][] floatHasNans = new int[1][];
        final String[][] floatNanValueTreatment = new String[1][];

        final String[][] catFeatureNames = new String[1][];
        final int[][] catFlatFeatureIndex = new int[1][];
        final int[][] catFeatureIndex = new int[1][];

        final String[][] textFeatureNames = new String[1][];
        final int[][] textFlatFeatureIndex = new int[1][];
        final int[][] textFeatureIndex = new int[1][];

        final String[][] embeddingFeatureNames = new String[1][];
        final int[][] embeddingFlatFeatureIndex = new int[1][];
        final int[][] embeddingFeatureIndex = new int[1][];

        final int[][] usedFeatureIndicesArr = new int[1][];

        try {
            implLibrary.catBoostModelGetPredictionDimension(handle, predictionDimension);
            implLibrary.catBoostModelGetTreeCount(handle, treeCount);
            implLibrary.catBoostModelGetUsedNumericFeatureCount(handle, usedNumericFeatureCount);
            implLibrary.catBoostModelGetUsedCategoricalFeatureCount(handle, usedCatFeatureCount);
            implLibrary.catBoostModelGetUsedTextFeatureCount(handle, usedTextFeatureCount);
            implLibrary.catBoostModelGetUsedEmbeddingFeatureCount(handle, usedEmbeddingFeatureCount);
            implLibrary.catBoostModelGetFlatFeatureVectorExpectedSize(handle, featureVectorExpectedSize);
            implLibrary.catBoostModelGetMetadata(handle, modelMetadataKeys, modelMetadataValues);
            implLibrary.catBoostModelGetFloatFeatures(handle, floatFeatureNames, floatFlatFeatureIndex, floatFeatureIndex, floatHasNans, floatNanValueTreatment);
            implLibrary.catBoostModelGetCatFeatures(handle, catFeatureNames, catFlatFeatureIndex, catFeatureIndex);
            implLibrary.catBoostModelGetTextFeatures(handle, textFeatureNames, textFlatFeatureIndex, textFeatureIndex);
            implLibrary.catBoostModelGetEmbeddingFeatures(handle, embeddingFeatureNames, embeddingFlatFeatureIndex, embeddingFeatureIndex);
            implLibrary.catBoostModelGetUsedFeatureIndices(handle, usedFeatureIndicesArr);
        } catch (CatBoostError e) {
            this.close();
            throw e;
        }

        final HashSet<Integer> usedFeatureIndices = new HashSet<>();
        for (int i = 0; i < usedFeatureIndicesArr[0].length; i++) {
            usedFeatureIndices.add(usedFeatureIndicesArr[0][i]);
        }

        this.predictionDimension = predictionDimension[0];
        this.treeCount = treeCount[0];
        this.usedNumericFeatureCount = usedNumericFeatureCount[0];
        this.usedCategoricFeatureCount = usedCatFeatureCount[0];
        this.usedTextFeatureCount = usedTextFeatureCount[0];
        this.usedEmbeddingFeatureCount = usedEmbeddingFeatureCount[0];

        for (int i = 0; i < modelMetadataKeys[0].length; i++) {
            this.metadata.put(modelMetadataKeys[0][i], modelMetadataValues[0][i]);
        }

        for (int i = 0; i < floatFeatureNames[0].length; i++) {
            this.features.add(new FloatFeature(floatFeatureNames[0][i],
                                               floatFeatureIndex[0][i],
                                               floatFlatFeatureIndex[0][i],
                                               usedFeatureIndices.contains(floatFlatFeatureIndex[0][i]),
                                               floatHasNans[0][i],
                                               floatNanValueTreatment[0][i]));
        }
        for (int i = 0; i < catFeatureNames[0].length; i++) {
            this.features.add(new CatFeature(catFeatureNames[0][i],
                                             catFeatureIndex[0][i],
                                             catFlatFeatureIndex[0][i],
                                             usedFeatureIndices.contains(catFlatFeatureIndex[0][i])));
        }
        for (int i = 0; i < textFeatureNames[0].length; i++) {
            this.features.add(new TextFeature(textFeatureNames[0][i],
                                              textFeatureIndex[0][i],
                                              textFlatFeatureIndex[0][i],
                                              usedFeatureIndices.contains(textFlatFeatureIndex[0][i])));
        }
        for (int i = 0; i < embeddingFeatureNames[0].length; i++) {
            this.features.add(new EmbeddingFeature(embeddingFeatureNames[0][i],
                                                   embeddingFeatureIndex[0][i],
                                                   embeddingFlatFeatureIndex[0][i],
                                                   usedFeatureIndices.contains(embeddingFlatFeatureIndex[0][i])));
        }
        Collections.sort(this.features, new Comparator<Feature>() {
            public int compare(Feature v1, Feature v2) {
                return v1.getFlatFeatureIndex() - v2.getFlatFeatureIndex();
            }
        });
        this.featureNames = new String[this.features.size()];
        for (Feature f : this.features) {
            this.featureNames[f.getFlatFeatureIndex()] = f.getName();
        }
    }

    /**
     * Load CatBoost model from file modelPath.
     *
     * @param modelPath Path to the model.
     * @return          CatBoost model.
     * @throws CatBoostError When failed to load model.
     */
    @NotNull
    public static CatBoostModel loadModel(final @NotNull String modelPath) throws CatBoostError {
        return loadModel(modelPath, "bin");
    }

    /**
     * Load CatBoost model from file modelPath.
     *
     * @param modelPath   Path to the model.
     * @param modelFormat Model file format (bin or json)
     * @return            CatBoost model.
     * @throws CatBoostError When failed to load model.
     */
    @NotNull
    public static CatBoostModel loadModel(final @NotNull String modelPath, @NotNull String modelFormat) throws CatBoostError {
        final long[] handles = new long[1];

        implLibrary.catBoostLoadModelFromFile(modelPath, handles, modelFormat);
        return new CatBoostModel(handles[0]);
    }

    /**
     * Load CatBoost model serialized in an array.
     *
     * @param serializedModel   Byte array containing model.
     * @return                  CatBoost model.
     * @throws CatBoostError When failed to load model.
     */
    @NotNull
    public static CatBoostModel loadModel(final @NotNull byte[] serializedModel) throws CatBoostError {
        return loadModel(serializedModel, "bin");
    }

    /**
     * Load CatBoost model serialized in an array.
     *
     * @param serializedModel   Byte array containing model.
     * @param modelFormat       Model file format (bin or json)
     * @return                  CatBoost model.
     * @throws CatBoostError When failed to load model.
     */
    @NotNull
    public static CatBoostModel loadModel(final @NotNull byte[] serializedModel, @NotNull String modelFormat) throws CatBoostError {
        final long[] handles = new long[1];

        implLibrary.catBoostLoadModelFromArray(serializedModel, handles, modelFormat);
        return new CatBoostModel(handles[0]);
    }

    /**
     * Load CatBoost model from stream.
     *
     * @param in Input stream containing model.
     * @return   CatBoost model.
     * @throws CatBoostError When failed to load model.
     * @throws IOException When failed to read model from file.
     */
    @NotNull
    public static CatBoostModel loadModel(final InputStream in) throws CatBoostError, IOException {
        return loadModel(in, "bin");
    }

    /**
     * Load CatBoost model from stream.
     *
     * @param in Input stream containing model.
     * @param modelFormat Model file format (bin or json)
     * @return   CatBoost model.
     * @throws CatBoostError When failed to load model.
     * @throws IOException When failed to read model from file.
     */
    @NotNull
    public static CatBoostModel loadModel(final InputStream in, @NotNull String modelFormat) throws CatBoostError, IOException {
        final byte[] copyBuffer = new byte[4 * 1024];

        int bytesRead;
        final ByteArrayOutputStream out = new ByteArrayOutputStream();

        while ((bytesRead = in.read(copyBuffer)) != -1) {
            out.write(copyBuffer, 0, bytesRead);
        }

        return loadModel(out.toByteArray(), modelFormat);
    }

    /**
     * Hash categorical feature.
     *
     * @param catFeature String representation of categorical feature.
     * @return           Hash for categorical feature.
     * @throws CatBoostError In case of error within native library.
     */
    static int hashCategoricalFeature(final @NotNull String catFeature) throws CatBoostError {
        int hash[] = new int[1];
        implLibrary.catBoostHashCatFeature(catFeature, hash);
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
     * @throws CatBoostError In case of error within native library.
     */
    static void hashCategoricalFeatures(
            final @NotNull String[] catFeatures,
            final @NotNull int[] hashes) throws CatBoostError {
        implLibrary.catBoostHashCatFeatures(catFeatures, hashes);
    }

    /**
     * Hash array of categorical features.
     *
     * @param catFeatures Array of categorical features.
     * @return            Array of hashes of categorical features.
     * @throws CatBoostError In case of error within native library.
     */
    @NotNull
    static int[] hashCategoricalFeatures(final @NotNull String[] catFeatures) throws CatBoostError {
        final int[] hashes = new int[catFeatures.length];
        hashCategoricalFeatures(catFeatures, hashes);
        return hashes;
    }

    FormulaEvaluatorType[] getSupportedEvaluatorTypes() throws CatBoostError {
        final String[][] evaluatorTypesAsStrings = new String[1][];
        implLibrary.catBoostModelGetSupportedEvaluatorTypes(handle, evaluatorTypesAsStrings);
        int evaluatorTypesSize = evaluatorTypesAsStrings[0].length;
        final FormulaEvaluatorType[] evaluatorTypes = new FormulaEvaluatorType[evaluatorTypesSize];
        for (int i = 0; i < evaluatorTypesSize; ++i) {
            evaluatorTypes[i] = FormulaEvaluatorType.valueOf(evaluatorTypesAsStrings[0][i]);
        }
        return evaluatorTypes;
    }

    void setEvaluatorType(FormulaEvaluatorType evaluatorType) throws CatBoostError {
        implLibrary.catBoostModelSetEvaluatorType(handle, evaluatorType.toString());
    }

    FormulaEvaluatorType getEvaluatorType() throws CatBoostError, IllegalArgumentException {
        final String[] evaluatorTypeAsString = new String[1];
        implLibrary.catBoostModelGetEvaluatorType(handle, evaluatorTypeAsString);
        return FormulaEvaluatorType.valueOf(evaluatorTypeAsString[0]);
    }

    /**
     * @return Dimension of model prediction.
     */
    public int getPredictionDimension() {
        return predictionDimension;
    }

    /**
     * @return Number of trees in model.
     */
    public int getTreeCount() {
        return treeCount;
    }

    /**
     * @return Number of numeric features used by the model.
     */
    public int getUsedNumericFeatureCount() {
        return usedNumericFeatureCount;
    }

    /**
     * @return Number of categorical features used by the model.
     */
    public int getUsedCategoricFeatureCount() {
        return usedCategoricFeatureCount;
    }

    /**
     * @return Number of text features used by the model.
     */
    public int getUsedTextFeatureCount() {
        return usedTextFeatureCount;
    }

    /**
     * @return Number of embedding features used by the model.
     */
    public int getUsedEmbeddingFeatureCount() {
        return usedEmbeddingFeatureCount;
    }

    /**
     * @return Names of features used by the model.
     */
    public String[] getFeatureNames() { return featureNames; }

    /**
     * @return A map of metadata
     */
    public Map<String, String> getMetadata() {
        return metadata;
    }

    /**
     * @return A list of feature metadata, sorted by flat index.
     */
    public List<Feature> getFeatures() {
        return features;
    }

    /**
     * Apply model to object defined by features.
     *
     * @param numericFeatures Numeric features.
     * @param catFeatures     Categoric features.
     * @param prediction      Model predictions.
     * @throws CatBoostError In case of error within native library.
     */
    public void predict(
            final @Nullable float[] numericFeatures,
            final @Nullable String[] catFeatures,
            final @NotNull CatBoostPredictions prediction) throws CatBoostError {
        implLibrary.catBoostModelPredict(
                handle,
                numericFeatures,
                catFeatures,
                /*textFeatures*/ null,
                /*embeddingFeatures*/ null,
                prediction.getRawData());
    }

    /**
     * Apply model to object defined by features.
     *
     * @param numericFeatures   Numeric features.
     * @param catFeatures       Categoric features.
     * @param textFeatures      Text features.
     * @param embeddingFeatures Embedding features.
     * @param prediction        Model predictions.
     * @throws CatBoostError In case of error within native library.
     */
    public void predict(
            final @Nullable float[] numericFeatures,
            final @Nullable String[] catFeatures,
            final @Nullable String[] textFeatures,
            final @Nullable float[][] embeddingFeatures,
            final @NotNull CatBoostPredictions prediction) throws CatBoostError {
        implLibrary.catBoostModelPredict(
                handle,
                numericFeatures,
                catFeatures,
                textFeatures,
                embeddingFeatures,
                prediction.getRawData());
    }

    /**
     * Same as {@link #predict(float[], String[], CatBoostPredictions)}, but returns prediction instead of taking it
     * as the last parameter.
     *
     * @param numericFeatures Numeric features.
     * @param catFeatures     Categoric features.
     * @return                Model predictions.
     * @throws CatBoostError In case of error within native library.
     */
    @NotNull
    public CatBoostPredictions predict(
            final @Nullable float[] numericFeatures,
            final @Nullable String[] catFeatures) throws CatBoostError {
        final CatBoostPredictions prediction = new CatBoostPredictions(1, getPredictionDimension());
        predict(numericFeatures, catFeatures, prediction);
        return prediction;
    }

    /**
     * Same as {@link #predict(float[], String[], String[], float[][], CatBoostPredictions)}, but returns prediction instead of taking it
     * as the last parameter.
     *
     * @param numericFeatures   Numeric features.
     * @param catFeatures       Categoric features.
     * @param textFeatures      Text features.
     * @param embeddingFeatures Embedding features.
     * @return                  Model predictions.
     * @throws CatBoostError In case of error within native library.
     */
    @NotNull
    public CatBoostPredictions predict(
            final @Nullable float[] numericFeatures,
            final @Nullable String[] catFeatures,
            final @Nullable String[] textFeatures,
            final @Nullable float[][] embeddingFeatures) throws CatBoostError {
        final CatBoostPredictions prediction = new CatBoostPredictions(1, getPredictionDimension());
        predict(numericFeatures, catFeatures, textFeatures, embeddingFeatures, prediction);
        return prediction;
    }

    /**
     * Same as {@link #predict(float[], String[], CatBoostPredictions)}, but accept categoric features as hashes
     * computed by {@link #hashCategoricalFeature(String)}.
     *
     * @param numericFeatures  Numeric features.
     * @param catFeatureHashes Categoric feature hashes.
     * @param prediction       Model predictions.
     * @throws CatBoostError In case of error within native library.
     */
    public void predict(
            final @Nullable float[] numericFeatures,
            final @Nullable int[] catFeatureHashes,
            final @NotNull CatBoostPredictions prediction) throws CatBoostError {
        implLibrary.catBoostModelPredict(
                handle,
                numericFeatures,
                catFeatureHashes,
                /*textFeatures*/ null,
                /*embeddingFeatures*/ null,
                prediction.getRawData());
    }

   /**
     * Same as {@link #predict(float[], String[], String[], float[][], CatBoostPredictions)}, but accept categoric features as hashes
     * computed by {@link #hashCategoricalFeature(String)}.
     *
     * @param numericFeatures   Numeric features.
     * @param catFeatureHashes  Categoric feature hashes.
     * @param textFeatures      Text features.
     * @param embeddingFeatures Embedding features.
     * @param prediction        Model predictions.
     * @throws CatBoostError In case of error within native library.
     */
    public void predict(
            final @Nullable float[] numericFeatures,
            final @Nullable int[] catFeatureHashes,
            final @Nullable String[] textFeatures,
            final @Nullable float[][] embeddingFeatures,
            final @NotNull CatBoostPredictions prediction) throws CatBoostError {
        implLibrary.catBoostModelPredict(
                handle,
                numericFeatures,
                catFeatureHashes,
                textFeatures,
                embeddingFeatures,
                prediction.getRawData());
    }

    /**
     * Same as {@link #predict(float[], int[], CatBoostPredictions)}, but returns prediction instead of taking it as
     * third parameter.
     *
     * @param numericFeatures  Numeric features.
     * @param catFeatureHashes Categoric feature hashes.
     * @return                 Model predictions.
     * @throws CatBoostError In case of error within native library.
     */
    @NotNull
    public CatBoostPredictions predict(
            final @Nullable float[] numericFeatures,
            final @Nullable int[] catFeatureHashes) throws CatBoostError {
        final CatBoostPredictions prediction = new CatBoostPredictions(1, getPredictionDimension());
        predict(numericFeatures, catFeatureHashes, prediction);
        return prediction;
    }

   /**
     * Same as {@link #predict(float[], int[], String[], float[][], CatBoostPredictions)}, but returns prediction instead of taking it as
     * third parameter.
     *
     * @param numericFeatures   Numeric features.
     * @param catFeatureHashes  Categoric feature hashes.
     * @param textFeatures      Text features.
     * @param embeddingFeatures Embedding features.
     * @return                  Model predictions.
     * @throws CatBoostError In case of error within native library.
     */
    @NotNull
    public CatBoostPredictions predict(
            final @Nullable float[] numericFeatures,
            final @Nullable int[] catFeatureHashes,
            final @Nullable String[] textFeatures,
            final @Nullable float[][] embeddingFeatures) throws CatBoostError {
        final CatBoostPredictions prediction = new CatBoostPredictions(1, getPredictionDimension());
        predict(numericFeatures, catFeatureHashes, textFeatures, embeddingFeatures, prediction);
        return prediction;
    }

    /**
     * Apply model to a batch of objects.
     *
     * @param numericFeatures Numeric features matrix.
     * @param catFeatures     Categoric features matrix.
     * @param prediction      Model predictions.
     * @throws CatBoostError In case of error within native library.
     */
    public void predict(
            final @Nullable float[][] numericFeatures,
            final @Nullable String[][] catFeatures,
            final @NotNull CatBoostPredictions prediction) throws CatBoostError {
        implLibrary.catBoostModelPredict(
                handle,
                numericFeatures,
                catFeatures,
                /*textFeatures*/ null,
                /*embeddingFeatures*/ null,
                prediction.getRawData());
    }

    /**
     * Apply model to a batch of objects.
     *
     * @param numericFeatures   Numeric features matrix.
     * @param catFeatures       Categoric features matrix.
     * @param textFeatures      Text features matrix.
     * @param embeddingFeatures Embedding features matrix.
     * @param prediction        Model predictions.
     * @throws CatBoostError In case of error within native library.
     */
    public void predict(
            final @Nullable float[][] numericFeatures,
            final @Nullable String[][] catFeatures,
            final @Nullable String[][] textFeatures,
            final @Nullable float[][][] embeddingFeatures,
            final @NotNull CatBoostPredictions prediction) throws CatBoostError {
        implLibrary.catBoostModelPredict(
                handle,
                numericFeatures,
                catFeatures,
                textFeatures,
                embeddingFeatures,
                prediction.getRawData());
    }

    /**
     * Same as {@link #predict(float[][], String[][], CatBoostPredictions)}, but returns prediction instead of taking
     * it as the last parameter.
     *
     * @param numericFeatures Numeric features.
     * @param catFeatures     Categoric features.
     * @return                Model predictions.
     * @throws CatBoostError In case of error within native library.
     */
    @NotNull
    public CatBoostPredictions predict(
            final @Nullable float[][] numericFeatures,
            final @Nullable String[][] catFeatures) throws CatBoostError {
        return predict(numericFeatures, catFeatures, /*textFeatures*/ null, /*embeddingFeatures*/ null);
    }

    /**
     * Same as {@link #predict(float[][], String[][], String[][], float[][][], CatBoostPredictions)}, but returns prediction instead of taking
     * it as the last parameter.
     *
     * @param numericFeatures   Numeric features.
     * @param catFeatures       Categoric features.
     * @param textFeatures      Text features.
     * @param embeddingFeatures Embedding features.
     * @return                  Model predictions.
     * @throws CatBoostError In case of error within native library.
     */
    @NotNull
    public CatBoostPredictions predict(
            final @Nullable float[][] numericFeatures,
            final @Nullable String[][] catFeatures,
            final @Nullable String[][] textFeatures,
            final @Nullable float[][][] embeddingFeatures) throws CatBoostError {
        int resultSize = 0;
        if (numericFeatures != null) {
            resultSize = numericFeatures.length;
        } else if (catFeatures != null) {
            resultSize = catFeatures.length;
        } else if (textFeatures != null) {
            resultSize = textFeatures.length;
        } else if (embeddingFeatures != null) {
            resultSize = embeddingFeatures.length;
        } else {
            throw new CatBoostError("all arguments are null");
        }

        final CatBoostPredictions prediction = new CatBoostPredictions(resultSize, getPredictionDimension());
        predict(numericFeatures, catFeatures, textFeatures, embeddingFeatures, prediction);
        return prediction;
    }

    /**
     * Same as {@link #predict(float[][], String[][], CatBoostPredictions)}, but accept categoric features as hashes
     * computed by {@link #hashCategoricalFeature(String)}.
     *
     * @param numericFeatures  Numeric features.
     * @param catFeatureHashes Categoric feature hashes.
     * @param prediction       Model predictions.
     * @throws CatBoostError In case of error within native library.
     */
    public void predict(
            final @Nullable float[][] numericFeatures,
            final @Nullable int[][] catFeatureHashes,
            final @NotNull CatBoostPredictions prediction) throws CatBoostError {
        implLibrary.catBoostModelPredict(
            handle,
            numericFeatures,
            catFeatureHashes,
            /*textFeatures*/ null,
            /*embeddingFeatures*/ null,
            prediction.getRawData());
    }

    /**
     * Same as {@link #predict(float[][], String[][], String[][], float[][][], CatBoostPredictions)}, but accept categoric features as hashes
     * computed by {@link #hashCategoricalFeature(String)}.
     *
     * @param numericFeatures   Numeric features.
     * @param catFeatureHashes  Categoric feature hashes.
     * @param textFeatures      Text features.
     * @param embeddingFeatures Embedding features.
     * @param prediction        Model predictions.
     * @throws CatBoostError In case of error within native library.
     */
    public void predict(
            final @Nullable float[][] numericFeatures,
            final @Nullable int[][] catFeatureHashes,
            final @Nullable String[][] textFeatures,
            final @Nullable float[][][] embeddingFeatures,
            final @NotNull CatBoostPredictions prediction) throws CatBoostError {
        implLibrary.catBoostModelPredict(
            handle,
            numericFeatures,
            catFeatureHashes,
            textFeatures,
            embeddingFeatures,
            prediction.getRawData());
    }

    /**
     * Same as {@link #predict(float[][], String[][], CatBoostPredictions)}, but returns predictions instead of taking
     * it as the last parameter.
     *
     * @param numericFeatures  Numeric features.
     * @param catFeatureHashes Categoric feature hashes.
     * @return                 Model predictions.
     * @throws CatBoostError In case of error within native library.
     */
    @NotNull
    public CatBoostPredictions predict(
            final @Nullable float[][] numericFeatures,
            final @Nullable int[][] catFeatureHashes) throws CatBoostError {
        if (numericFeatures == null && catFeatureHashes == null) {
            throw new CatBoostError("both arguments are null");
        }

        final CatBoostPredictions prediction = new CatBoostPredictions(
                numericFeatures == null ? catFeatureHashes.length : numericFeatures.length,
                getPredictionDimension());
        predict(numericFeatures, catFeatureHashes, prediction);
        return prediction;
    }

    /**
     * Same as {@link #predict(float[][], String[][], String[][], float[][][], CatBoostPredictions)}, but returns predictions instead of taking
     * it as the last parameter.
     *
     * @param numericFeatures   Numeric features.
     * @param catFeatureHashes  Categoric feature hashes.
     * @param textFeatures      Text features.
     * @param embeddingFeatures Embedding features.
     * @return                  Model predictions.
     * @throws CatBoostError In case of error within native library.
     */
    @NotNull
    public CatBoostPredictions predict(
            final @Nullable float[][] numericFeatures,
            final @Nullable int[][] catFeatureHashes,
            final @Nullable String[][] textFeatures,
            final @Nullable float[][][] embeddingFeatures) throws CatBoostError {
        int resultSize = 0;
        if (numericFeatures != null) {
            resultSize = numericFeatures.length;
        } else if (catFeatureHashes != null) {
            resultSize = catFeatureHashes.length;
        } else if (textFeatures != null) {
            resultSize = textFeatures.length;
        } else if (embeddingFeatures != null) {
            resultSize = embeddingFeatures.length;
        } else {
            throw new CatBoostError("all arguments are null");
        }
        final CatBoostPredictions prediction = new CatBoostPredictions(resultSize, getPredictionDimension());
        predict(numericFeatures, catFeatureHashes, textFeatures, embeddingFeatures, prediction);
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

    private synchronized void dispose() throws CatBoostError {
        if (handle != 0) {
            implLibrary.catBoostFreeModel(handle);
            handle = 0;
        }
    }

    @Override
    public void close() throws CatBoostError {
        dispose();
    }
}
