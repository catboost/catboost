import {execProcess} from './common';
import {readFileSync, writeFileSync} from 'fs'

async function compileTypeScript() {
    const result = await execProcess('tsc');
    if (result.code !== 0) {
        console.error(`Building ts library failed: 
            ${result.code} ${result.signal} ${result.err?.message}`);
        process.exit(1);
    }
}

function copyBindings() {
    writeFileSync('./lib/catboost.d.ts', readFileSync('./bindings/catboost.d.ts'));
    writeFileSync('./lib/catboost.js', readFileSync('./bindings/catboost.js'));
}

async function compileNativeAddon() {
    const result = await execProcess('node-gyp build');
    if (result.code !== 0) {
        console.error(`Building native-addon library failed: 
            ${result.code} ${result.signal} ${result.err?.message}`);
        process.exit(1);
    }
}

export async function compileBindings() {
    await compileTypeScript();
    copyBindings();
    await compileNativeAddon();
}

async function buildModelInterfaceLibrary() {
    const srcPath = process.env['CATBOOST_SRC_PATH'] || '../..';
    const result = await execProcess(`${srcPath}/ya make -r ${srcPath}/catboost/libs/model_interface -o ./build`);
    if (result.code !== 0) {
        console.error(`Building catboostmodel library failed: 
            ${result.code} ${result.signal} ${result.err?.message}`);
        process.exit(1);
    }
}

export async function buildModel() {
    const result = await execProcess('node-gyp configure');
    if (result.code !== 0) {
        console.error(`Node-gyp configuration failed: 
${result.code} ${result.signal} ${result.err?.message}`);
        process.exit(1);
    }

    await buildModelInterfaceLibrary();
    // TODO
}

