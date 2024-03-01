/** CatBoost numeric features for multiple samples. */
export type CatBoostFloatFeatures = Array<number[]>;
/**
 * CatBoost categorial features for multiple samples - either integer hashes
 * or string values.
 */
export type CatBoostCategoryFeatures = Array<number[]>|Array<string[]>;
/** CatBoost text features for multiple documents. */
export type CatBoostTextFeatures = Array<string[]>;
/** CatBoost embedding features for multiple documents. */
export type CatBoostEmbeddingFeatures = Array<Array<number[]>>;

/** CatBoost model instance. */
export class Model {
	constructor(path?: string);

	/** Loads a model from the file. */
	loadModel(path: string): void;
	/** Sets model prediction postprocessing type. Possible value are:
	 * RawFormulaVal - raw sum of leaf values for each dimension,
     * Exponent - exp(sum(leaf values)),
	 * RMSEWithUncertainty - pair (prediction, uncertainty),
	 * Probability - (probablity for class_0, ..., probablity for class_i,...)
	 * Class - id of class with maximal predicted probability
	 * */
	setPredictionType(predictionType: string): void;
	/**
	 * Calculate the prediction for multiple samples.
	 * All defined feature arguments must have the same length.
	 */
	predict(floatFeatures: CatBoostFloatFeatures,
		catFeatures: CatBoostCategoryFeatures,
		textFeatures?: CatBoostTextFeatures,
		embeddingFeatures?: CatBoostEmbeddingFeatures): number[];
	/** Enable evaluation on GPU device. */
	enableGPUEvaluation(deviceId: number): void;
	/** The number of numeric features. */
	getFloatFeaturesCount(): number;
	/** The number of categorial features. */
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
