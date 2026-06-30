/** CatBoost numeric features for multiple samples. */
export type CatBoostFloatFeatures = Array<number[]>;
/**
 * CatBoost categorical features for multiple samples - either integer hashes
 * or string values.
 */
export type CatBoostCategoryFeatures = Array<number[]>|Array<string[]>;
/** CatBoost text features for multiple samples. */
export type CatBoostTextFeatures = Array<string[]>;
/** CatBoost embedding features for multiple samples. */
export type CatBoostEmbeddingFeatures = Array<Array<number[]>>;

/** CatBoost model instance. */
export class Model {
    constructor(path?: string);

	/** Load a model from the file. */
    loadModel(path: string): void;
    /** Set model prediction postprocessing type. Possible value are:
     * RawFormulaVal - raw sum of leaf values for each dimension, this is the default
     * Exponent - exp(sum(leaf values)),
     * RMSEWithUncertainty - pair (prediction, uncertainty),
     * Probability - (probablity for class_0, ..., probablity for class_i,...)
     * MultiProbability - probability for each label (used for multilabel classification)
     * Class - index of a class with the maximum predicted probability
     * */
    setPredictionType(predictionType: string): void;
    /**
     * Calculate the prediction for multiple samples.
     * All defined feature arguments must have the same length.
     *
     * The returned value contains [sampleCount x predictionDimensions] elements
     *  (should be accessed using [sampleIndex * predictionDimensions + predictionDimensionIdx],
     *   for simple cases when predictionDimensions = 1 it is just [sampleIndex])
     *  and its interpretation depends on prediction type (can be set with 'setPredictionType'):
     *  - RawFormulaVal (this is the default):
     *    array of raw sum of leaf values for each dimension
     *  - Exponent:
     *    array of exp(sum(leaf values)) for each dimension
     *  - RMSEWithUncertainty:
     *    array of pairs (prediction, uncertainty)
     *  - Probability:
     *    - for binary classification models:
     *      array of probabilities for positive class (calculated as sigmoid(rawFormulaVal))
     *    - for multiclassification models:
     *      array of array of probabilities for each class (calculated as softmax(rawFormulaVal))
     *  - MultiProbability:
     *    array of probabilities for each label (calculated as sigmoid(rawFormulaVal))
     *      (used for multilabel classification)
     *  - Class:
     *    array of predicted class indices.
     *
     * predictionDimensions can be obtained using 'getPredictionDimensionsCount' method.
     */
    predict(floatFeatures: CatBoostFloatFeatures,
        catFeatures?: CatBoostCategoryFeatures,
        textFeatures?: CatBoostTextFeatures,
        embeddingFeatures?: CatBoostEmbeddingFeatures): number[];
    /** Enable evaluation on GPU device. */
    enableGPUEvaluation(deviceId: number): void;
    /** The number of numeric features. */
    getFloatFeaturesCount(): number;
    /** The number of categorical features. */
    getCatFeaturesCount(): number;
    /** The number of text features. */
    getTextFeaturesCount(): number;
    /** The number of embedding features. */
    getEmbeddingFeaturesCount(): number;
    /** The number of trees in the model. */
    getTreeCount(): number;
    /** The number of dimensions in the model. */
    getDimensionsCount(): number;
    /** The number of dimensions in the prediction. */
    getPredictionDimensionsCount(): number;
}
