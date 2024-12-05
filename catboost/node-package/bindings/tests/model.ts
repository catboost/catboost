import {Model} from '../index';
import * as assert from 'assert';
import * as fs from 'fs';

function testInitEmptyModel() {
    const modelA = new Model();
    const modelB = new Model();
}

function testLoadFromFile() {
    const model = new Model();
    model.loadModel('./test_data/adult.cbm');
}

function testLoadFromNonExistentFile() {
    const model = new Model();
    model.loadModel('./test_data/non-existent.cbm');
}

function testPredictModelWithNumFeaturesSingle() {
    const model = new Model();
    model.loadModel('../pytest/data/models/features_num__dataset_querywise.cbm');

    const numFeatures = "0.257727	0.0215909	0.171299	1	1	1	1	1	0	0	0	0	0	0	0.431373	0.935065	0.0208333	0.070824	1	0	0.313726	1	1	0	0.937724	0	1	0	0	0	0	0.0566038	0	0	1	0.73929	1	0.000505391	0.885819	0.000172727	0	0	0	0	0	0	0.153262	0.578118	0.222098	1".split('\t').map(x => parseFloat(x))

    const predictions = model.predict([numFeatures])
    assert.strictEqual(predictions[0].toFixed(4), '0.0882', `Expected [0.0882], got ${predictions}`);
}

function testPredictModelWithNumFeaturesMany() {
    const model = new Model();
    model.loadModel('../pytest/data/models/features_num__dataset_querywise.cbm');

    const numFeatures = [
        "0.257727	0.0215909	0.171299	1	1	1	1	1	0	0	0	0	0	0	0.431373	0.935065	0.0208333	0.070824	1	0	0.313726	1	1	0	0.937724	0	1	0	0	0	0	0.0566038	0	0	1	0.73929	1	0.000505391	0.885819	0.000172727	0	0	0	0	0	0	0.153262	0.578118	0.222098	1",
        "0.424438	0.164384	0.572649	1	1	0	1	0	0	1	0	0	0	0	0.360784	0.512195	0.0447049	0.0717587	1	1	0.321569	1	1	0	0.941214	0	1	0	0	0	0	0.275362	1	0.209302	1	0.391239	1	0.0143194	0.885819	0.00140996	0	0	0	0	0	0	0.357143	0.883721	0.820312	1",
        "0.345548	0.248034	0.0853067	1	1	0	1	0	0	1	0	0	0	0	0.6	0.5	0.114292	0.071371	1	0	0.396078	1	1	0	0.939218	0	1	0	0	0	0	0.0384615	0	0	1	0	0	0	0.885819	0.00224069	0	0	0	0	0	0	0.588235	0.444444	0.608696	1",
        "0.946305	0.139752	0.429885	1	1	1	1	1	0	1	0	0	0	0	0.811765	0.75	0.119755	0.071636	0	1	0.541176	1	1	0	0.993828	0	1	0	0	0	0	0.509804	1	0.000398	1	1.5	1	0.417659	0.885819	0.00117792	0	0	0	0	0	0	0.52962	0.958103	0.885843	1"
    ].map(arr => arr.split('\t').map(x => parseFloat(x)))

    const predictions = model.predict(numFeatures)

    const expectedPredictions = [
        0.08819508860736715,
        0.043193651033534904,
        -0.0019333444540111586,
        0.0836685835428004
    ]

    assert.deepStrictEqual(
        predictions.map(x => x.toFixed(4)),
        expectedPredictions.map(x => x.toFixed(4)),
        `Expected ${expectedPredictions}, got ${predictions}`
    );
}

function testCalculateSingle() {
    const model = new Model();
    model.loadModel('./test_data/adult.cbm');

    const predictions = model.predict([[40., 85019., 16., 0., 0., 45.]],
        [["Private", "Doctorate", "Married-civ-spouce", "Prof-specialty", "Husband", "Asian-Pac-Islander", "Male", "nan"]]);
    assert.strictEqual(predictions[0].toFixed(2), '1.54', `Expected [1.54], got ${predictions}`);
}

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

function testFloatFeaturesCount() {
    const model = new Model();
    model.loadModel('./test_data/adult.cbm');

    const count = model.getFloatFeaturesCount();
    assert.strictEqual(count, 6, `Expected [6], got ${count}`);
}

function testCatFeaturesCount() {
    const model = new Model();
    model.loadModel('./test_data/adult.cbm');

    const count = model.getCatFeaturesCount();
    assert.strictEqual(count, 8, `Expected [8], got ${count}`);
}

function testTreeCount() {
    const model = new Model();
    model.loadModel('./test_data/adult.cbm');

    const count = model.getTreeCount();
    assert.strictEqual(count, 100, `Expected [100], got ${count}`);
}

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

function testLoadOnConstruction() {
    const model = new Model('./test_data/adult.cbm');

    const predictions = model.predict([[40., 85019., 16., 0., 0., 45.]],
        [["Private", "Doctorate", "Married-civ-spouce", "Prof-specialty", "Husband", "Asian-Pac-Islander", "Male", "nan"]]);
    assert.strictEqual(predictions[0].toFixed(2), '1.54', `Expected [1.54], got ${predictions}`);
}

function testLoadNonExistentOnConstruction() {
    new Model('./test_data/non-existent.cbm');
}

function testPredictOnEmptyModel() {
    const model = new Model();
    model.predict([[10.0]], [["a"]]);
}


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


function runAll() {
    assert.doesNotThrow(testInitEmptyModel);
    assert.doesNotThrow(testLoadFromFile);
    assert.throws(testLoadFromNonExistentFile);
    assert.doesNotThrow(testPredictModelWithNumFeaturesSingle);
    assert.doesNotThrow(testPredictModelWithNumFeaturesMany);
    assert.doesNotThrow(testCalculateSingle);
    assert.doesNotThrow(testCalculateMany);
    assert.doesNotThrow(testFloatFeaturesCount);
    assert.doesNotThrow(testCatFeaturesCount);
    assert.doesNotThrow(testTreeCount);
    assert.doesNotThrow(testDimensionsCount);
    assert.doesNotThrow(testLoadOnConstruction);
    assert.throws(testLoadNonExistentOnConstruction);
    assert.throws(testPredictOnEmptyModel);
    assert.doesNotThrow(testMulticlassCloudnessSmall);
}

runAll()
console.log("Model tests passed")
