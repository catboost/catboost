import {Model} from '../index';
import * as assert from 'assert';

function testInitEmptyModel() {
    const modelA = new Model();
    const modelB = new Model();
}
assert.doesNotThrow(testInitEmptyModel);

function testLoadFromFile() {
    const model = new Model();
    model.loadFullFromFile('./test_data/adult.cbm');
}
assert.doesNotThrow(testLoadFromFile);

function testLoadFromNonExistantFile() {
    const model = new Model();
    model.loadFullFromFile('./test_data/non-existant.cbm');
}
assert.throws(testLoadFromNonExistantFile);

function testCalculateSingle() {
    const model = new Model();
    model.loadFullFromFile('./test_data/adult.cbm');

    const predictions = model.calcPrediction([[40., 85019., 16., 0., 0., 45.]], 
        [["Private", "Doctorate", "Married-civ-spouce", "Prof-specialty", "Husband", "Asian-Pac-Islander", "Male", "nan"]]);
    assert.strictEqual(predictions[0].toFixed(2), '1.54', `Expected [1.54], got ${predictions}`);
}
assert.doesNotThrow(testCalculateSingle);

function testCalculateMany() {
    const model = new Model();
    model.loadFullFromFile('./test_data/adult.cbm');

    const predictions = model.calcPrediction([
            [40., 85019., 16., 0., 0., 45.],
            [28., 85019., 13., 0., 0., 13.],
        ], 
        [
            ["Private", "Doctorate", "Married-civ-spouce", "Prof-specialty", "Husband", "Asian-Pac-Islander", "Male", "nan"],
            ["Self-emp-not-inc", "Bachelors", "Married-civ-spouce", "Exec-managerial", "Husband", "White", "Male", "United-States"],
        ]);
        assert.strictEqual(predictions.length, 2, `Expected 2 elements, got ${predictions}`);
    assert.strictEqual(predictions[0].toFixed(2), '1.54', `Expected [1.54], got ${predictions}`);
    assert.strictEqual(predictions[1].toFixed(2), '-1.17', `Expected [-1.17], got ${predictions}`);
}
assert.doesNotThrow(testCalculateMany);

function testFloatFeaturesCount() {
    const model = new Model();
    model.loadFullFromFile('./test_data/adult.cbm');

    const count = model.getFloatFeaturesCount();
    assert.strictEqual(count, 6, `Expected [6], got ${count}`);
}
assert.doesNotThrow(testFloatFeaturesCount);

function testCatFeaturesCount() {
    const model = new Model();
    model.loadFullFromFile('./test_data/adult.cbm');

    const count = model.getCatFeaturesCount();
    assert.strictEqual(count, 8, `Expected [8], got ${count}`);
}
assert.doesNotThrow(testCatFeaturesCount);

function testTreeCount() {
    const model = new Model();
    model.loadFullFromFile('./test_data/adult.cbm');

    const count = model.getTreeCount();
    assert.strictEqual(count, 100, `Expected [100], got ${count}`);
}
assert.doesNotThrow(testTreeCount);

function testDimensionsCount() {
    const model = new Model();
    model.loadFullFromFile('./test_data/adult.cbm');

    const count = model.getDimensionsCount();
    assert.strictEqual(count, 1, `Expected [1], got ${count}`);
}
assert.doesNotThrow(testDimensionsCount);

console.log("Model tests passed")
