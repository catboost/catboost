package ai.catboost;

import javax.validation.constraints.NotNull;

// TODO(yazevnul): add CatBoostClassificationPrediction

/**
 * CatBoost model prediction.
 */
public class CatBoostPredictions {
    @NotNull
    final private double[] data;
    final private int objectCount;
    final private int predictionDimension;

    /**
     *  Constructs CatBoost model prediction from preallocated array.
     *
     * @param objectCount         Number of objects
     * @param predictionDimension CatBoost model prediction dimension.
     * @param data                Array containing flattened prediction matrix.
     */
    CatBoostPredictions(final int objectCount, final int predictionDimension, final @NotNull double[] data) {
        if (data.length != objectCount * predictionDimension) {
            final String message = "data size is incorrect, must be objectCount * predictionDimension = "
                    + String.valueOf(objectCount * predictionDimension)
                    + "(objectCount=" + String.valueOf(objectCount) + ", "
                    + " predictionDimension=" + String.valueOf(predictionDimension) + ")"
                    + " but got " + String.valueOf(data.length);
            throw new IllegalArgumentException(message);
        }

        this.objectCount = objectCount;
        this.predictionDimension = predictionDimension;
        this.data = data;
    }

    /**
     * Construct CatBoost model prediction based on number of objects and model prediction dimension.
     *
     * @param objectCount         Object count.
     * @param predictionDimension CatBoost model prediction dimension.
     */
    public CatBoostPredictions(final int objectCount, final int predictionDimension) {
        this.objectCount = objectCount;
        this.predictionDimension = predictionDimension;
        this.data = new double[objectCount * predictionDimension];
    }

    /**
     * @return Number of objects in prediction.
     */
    public int getObjectCount() {
        return objectCount;
    }

    /**
     * @return Model prediction dimension.
     */
    public int getPredictionDimension() {
        return predictionDimension;
    }

    /**
     * Get model prediction for particular object and particular dimension.
     *
     * @param objectIndex     Object index.
     * @param predictionIndex Prediction dimension index.
     * @return                Model prediction value.
     */
    public double get(final int objectIndex, final int predictionIndex) {
        return data[objectIndex * getPredictionDimension() + predictionIndex];
    }

    /**
     * Copy object prediction to a specified array.
     *
     * @param objectIndex Object index.
     * @param predictions Array to copy predictions to.
     */
    public void copyObjectPredictions(final int objectIndex, final @NotNull double[] predictions) {
        if (predictions.length < getPredictionDimension()) {
            throw new IllegalArgumentException("`predictions` size is insufficient, got " + String.valueOf(predictions.length) + "but must be at least " + String.valueOf(getPredictionDimension()));
        }

        System.arraycopy(data, objectIndex * getPredictionDimension(), predictions, 0, getPredictionDimension());
    }

    /**
     * Copy object prediction to a separate array.
     *
     * @see #copyObjectPredictions(int, double[])
     *
     * @param objectIndex Object index.
     * @return            Array with object predictions.
     */
    @NotNull
    public double[] copyObjectPredictions(final int objectIndex) {
        final double[] predictions = new double[getPredictionDimension()];
        copyObjectPredictions(objectIndex, predictions);
        return predictions;
    }

    /**
     * Return row-major copy of prediction matrix. Prediction for object with index `i` in dimension `j` will be at
     * `i*getPredictionDimension() + j`.
     *
     * @return Row-major copy of prediction matrix.
     */
    @NotNull
    public double[] copyRowMajorPredictions() {
        return data;
    }

    @NotNull
    double[] getRawData() {
        return data;
    }
}
