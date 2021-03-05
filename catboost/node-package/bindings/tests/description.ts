import * as catboost from '../index';
import * as assert from 'assert';

function testDescription() {
    const message = catboost.DESCRIPTION;
    assert.strictEqual(message, "CatBoost is a machine learning method based on gradient boosting " + 
		       "over decision trees.", "Unexpected description value");
}

assert.doesNotThrow(testDescription);

function testCreateHandle() {
    const message = catboost.CreateHandle();
    assert.strictEqual(message, "CatBoost model got successfully created and destroyed");
}

assert.doesNotThrow(testCreateHandle);

console.log("All passed")
