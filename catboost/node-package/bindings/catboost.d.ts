/** CatBoost numeric features for multiple documents. */
export type CatBoostFloatFeatures = Array<number[]>;
/** 
 * CatBoost categorial features for multiple documents - either integer hashes
 * or string values. 
 */
export type CatBoostCategoryFeatures = Array<number[]>|Array<string[]>;

/** CatBoost model instance. */
export class Model {
	/** Loads model from file. */
	loadFullFromFile(path: string): void;
	/** 
	 * Calculate prediction for multiple documents. 
	 * The same number of numeric and categorial features is expected.
	 */
	calcPrediction(floatFeatures: CatBoostFloatFeatures, 
		catFeatures: CatBoostCategoryFeatures): number[];
	/** The number of numeric features. */
	getFloatFeaturesCount(): number;
	/** The number of categorial features. */
	getCatFeaturesCount(): number;
	/** The number of trees in the model. */
	getTreeCount(): number;
	/** The number of dimensions in the model. */
	getDimensionsCount(): number;
}
