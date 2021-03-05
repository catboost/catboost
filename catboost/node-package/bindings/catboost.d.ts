export const DESCRIPTION: string;
export function CreateHandle(): string;
export class Model {
	loadFullFromFile(path: string): void;
	calcPrediction(floatFeatures: number[], catFeatures: number[]|string[]): number[];
}
