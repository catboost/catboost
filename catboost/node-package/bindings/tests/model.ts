import {Model} from '../index';
import * as assert from 'assert';
import * as fs from 'fs';

function testInitEmptyModel() {
    const modelA = new Model();
    const modelB = new Model();
}
assert.doesNotThrow(testInitEmptyModel);

function testLoadFromFile() {
    const model = new Model();
    model.loadModel('./test_data/adult.cbm');
}
assert.doesNotThrow(testLoadFromFile);

function testLoadFromNonExistentFile() {
    const model = new Model();
    model.loadModel('./test_data/non-existent.cbm');
}
assert.throws(testLoadFromNonExistentFile);

function testCalculateSingle() {
    const model = new Model();
    model.loadModel('./test_data/adult.cbm');

    const predictions = model.predict([[40., 85019., 16., 0., 0., 45.]],
        [["Private", "Doctorate", "Married-civ-spouce", "Prof-specialty", "Husband", "Asian-Pac-Islander", "Male", "nan"]]);
    assert.strictEqual(predictions[0].toFixed(2), '1.54', `Expected [1.54], got ${predictions}`);
}
assert.doesNotThrow(testCalculateSingle);

function testLoadOnCreation() {
    const model = new Model('./test_data/adult.cbm');

    const predictions = model.predict([[40., 85019., 16., 0., 0., 45.]],
        [["Private", "Doctorate", "Married-civ-spouce", "Prof-specialty", "Husband", "Asian-Pac-Islander", "Male", "nan"]]);
    assert.strictEqual(predictions[0].toFixed(2), '1.54', `Expected [1.54], got ${predictions}`);
}
assert.doesNotThrow(testLoadOnCreation);

function testCalculateMany() {
    const model = new Model();
    model.loadModel('./test_data/adult.cbm');

    const predictions = model.predict([
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
    model.loadModel('./test_data/adult.cbm');

    const count = model.getFloatFeaturesCount();
    assert.strictEqual(count, 6, `Expected [6], got ${count}`);
}
assert.doesNotThrow(testFloatFeaturesCount);

function testCatFeaturesCount() {
    const model = new Model();
    model.loadModel('./test_data/adult.cbm');

    const count = model.getCatFeaturesCount();
    assert.strictEqual(count, 8, `Expected [8], got ${count}`);
}
assert.doesNotThrow(testCatFeaturesCount);

function testTreeCount() {
    const model = new Model();
    model.loadModel('./test_data/adult.cbm');

    const count = model.getTreeCount();
    assert.strictEqual(count, 100, `Expected [100], got ${count}`);
}
assert.doesNotThrow(testTreeCount);

function testDimensionsCount() {
    const model = new Model();
    model.loadModel('./test_data/adult.cbm');

    const count = model.getDimensionsCount();
    const predictionDimCount = model.getPredictionDimensionsCount();
    assert.strictEqual(
        count, predictionDimCount,
        `Expected count == predictionDimCount, got ${count} != ${predictionDimCount}`
    );
    assert.strictEqual(count, 1, `Expected [1], got ${count}`);
}
assert.doesNotThrow(testDimensionsCount);

function testLoadOnConstruction() {
    const model = new Model('./test_data/adult.cbm');

    const predictions = model.predict([[40., 85019., 16., 0., 0., 45.]],
        [["Private", "Doctorate", "Married-civ-spouce", "Prof-specialty", "Husband", "Asian-Pac-Islander", "Male", "nan"]]);
    assert.strictEqual(predictions[0].toFixed(2), '1.54', `Expected [1.54], got ${predictions}`);
}
assert.doesNotThrow(testLoadOnConstruction);

function testLoadNonExistentOnConstruction() {
    new Model('./test_data/non-existent.cbm');
}
assert.throws(testLoadNonExistentOnConstruction);

function testPredictOnEmptyModel() {
    const model = new Model();
    model.predict([[10.0]], [["a"]]);
}
assert.throws(testPredictOnEmptyModel);


function testMulticlassCloudnessSmall() {
    const model = new Model('./test_data/cloudness_small.cbm');
    const test_small = fs.readFileSync(
        '../pytest/data/cloudness_small/test_small','utf8')
        .split('\n')
        .filter(x => x.length > 0);
    const floatFeatures = test_small.map(x => x.split('\t').slice(1, 103).map(x => +x));


    const catFeatures = test_small.map(x => x.split('\t').slice(103, 145));
    model.setPredictionType('RawFormulaVal');
    var predictionDimensionsCount = model.getPredictionDimensionsCount();
    assert.strictEqual(predictionDimensionsCount, 3, `Expected 3, got ${predictionDimensionsCount}`);
    var predictions = model.predict(
        floatFeatures,
        catFeatures
    );
    assert.strictEqual(
        predictions.length,
        predictionDimensionsCount * floatFeatures.length,
        `Expected ${predictionDimensionsCount * floatFeatures.length}, got ${predictions.length}`
    );
    model.setPredictionType('Class');
    predictionDimensionsCount = model.getPredictionDimensionsCount();
    assert.strictEqual(predictionDimensionsCount, 1, `Expected 1, got ${predictionDimensionsCount}`);
    assert.strictEqual(model.getDimensionsCount(), 3, `Expected 3, got ${model.getDimensionsCount()}`);
    predictions = model.predict(
        floatFeatures,
        catFeatures
    );
    assert.strictEqual(
        predictions.length,
        predictionDimensionsCount * floatFeatures.length,
        `Expected ${predictionDimensionsCount * floatFeatures.length}, got ${predictions.length}`
    );
}
assert.doesNotThrow(testMulticlassCloudnessSmall);

console.log("Model tests passed")
