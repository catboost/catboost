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

    const predictions = model.calcPrediction([40., 85019., 16., 0., 0., 45.], 
        ["Private", "Doctorate", "Married-civ-spouce", "Prof-specialty", "Husband", "Asian-Pac-Islander", "Male", "nan"]);
    assert.strictEqual(predictions[0].toFixed(2), '1.54', `Expected [1.54], got ${predictions}`);
}
assert.doesNotThrow(testCalculateSingle);

console.log("Model tests passed")
