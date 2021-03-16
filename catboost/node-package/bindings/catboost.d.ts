export const DESCRIPTION: string;
export function CreateHandle(): string;

export type CatBoostFloatFeatures = Array<number[]>;
export type CatBoostCategoryFeatures = Array<number[]>|Array<string[]>;

export class Model {
	loadFullFromFile(path: string): void;
	calcPrediction(floatFeatures: number[], 
		catFeatures: number[]|string[]): number[];
}
