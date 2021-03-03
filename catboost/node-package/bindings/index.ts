const catboostAddon = require('../build/Release/catboost-node');

interface CatBoostModule {
	DESCRIPTION: string;
	CreateHandle(): string;
}

const catboost: CatBoostModule = catboostAddon;

export = catboostAddon;

