/** CatBoost numeric features for multiple documents. */
export type CatBoostFloatFeatures = Array<number[]>;
/**
 * CatBoost categorial features for multiple documents - either integer hashes
 * or string values.
 */
export type CatBoostCategoryFeatures = Array<number[]>|Array<string[]>;

/** CatBoost model instance. */
export class Model {
	constructor(path?: string);

	/** Loads model from file. */
	loadModel(path: string): void;
	/**
	 * Calculate prediction for multiple documents.
	 * The same number of numeric and categorial features is expected.
	 */
	predict(floatFeatures: CatBoostFloatFeatures,
		catFeatures: CatBoostCategoryFeatures): number[];
	/** Enable evaluation on GPU device. */
	enableGPUEvaluation(deviceId: number): void;
	/** The number of numeric features. */
	getFloatFeaturesCount(): number;
	/** The number of categorial features. */
	getCatFeaturesCount(): number;
	/** The number of trees in the model. */
	getTreeCount(): number;
	/** The number of dimensions in the model. */
	getDimensionsCount(): number;
}
