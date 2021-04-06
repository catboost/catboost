import {execProcess} from './common';

async function buildModelInterfaceLibrary() {
    const srcPath = process.env['CATBOOST_SRC_PATH'] || '../..';
    const result = await execProcess(`${srcPath}/ya make -r ${srcPath}/catboost/libs/model_interface -o ./build`);
    if (result.code !== 0) {
        console.error(`Building catboostmodel library failed: 
            ${result.code} ${result.signal} ${result.err?.message}`);
        process.exit(1);
    }
}

async function buildModel() {
    const result = await execProcess('node-gyp configure');
    if (result.code !== 0) {
        console.error(`Node-gyp configuration failed: 
${result.code} ${result.signal} ${result.err?.message}`);
        process.exit(1);
    }

    await buildModelInterfaceLibrary();
    // TODO
}

buildModel();
