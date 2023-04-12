import {execProcess} from './common';
import {linkSync, readFileSync, writeFileSync} from 'fs'
import {downloadBinaryFile} from './download';
import {readConfig} from './config';

async function compileTypeScript() {
    const result = await execProcess('npm run tsc');
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

async function compileNativeAddon(srcPath = '../..') {
    process.env['CATBOOST_SRC_PATH'] = srcPath;
    const result = await execProcess('node-gyp build');
    if (result.code !== 0) {
        console.error(`Building native-addon library failed:
            ${result.code} ${result.signal} ${result.err?.message}`);
        process.exit(1);
    }
}

export async function compileJs() {
    await compileTypeScript();
    copyBindings();
}

export async function compileBindings() {
    await compileJs();
    await compileNativeAddon();
}

async function buildModelInterfaceLibrary() {
    const srcPath = process.env['CATBOOST_SRC_PATH'] || '../..';
    const result = await execProcess(`${srcPath}/build/build_native.py --targets catboostmodel --build-root-dir ./build`);
    if (result.code !== 0) {
        console.error(`Building catboostmodel library failed:
            ${result.code} ${result.signal} ${result.err?.message}`);
        process.exit(1);
    }
}

async function configureGyp(srcPath = '../..') {
    process.env['CATBOOST_SRC_PATH'] = srcPath;
    const result = await execProcess('node-gyp configure');
    if (result.code !== 0) {
        console.error(`Node-gyp configuration failed:
            ${result.code} ${result.signal} ${result.err?.message}`);
        process.exit(1);
    }
}

export async function buildModel() {
    await configureGyp();
    await buildModelInterfaceLibrary();
}

/** Build binary from repository. */
export async function buildNative() {
    await configureGyp();
    await buildModelInterfaceLibrary();
    await compileBindings();
}

async function preparePlatformBinary(platform: string) {
    const config = readConfig();
    switch (platform) {
        case 'linux':
            for (const binary of config.binaries['linux']) {
                await downloadBinaryFile('./build/catboost/libs/model_interface',
                    binary);
            }
            linkSync('./build/catboost/libs/model_interface/libcatboostmodel.so',
                './build/catboost/libs/model_interface/libcatboostmodel.so.1');
            return;
        case 'darwin':
            for (const binary of config.binaries['mac']) {
                await downloadBinaryFile('./build/catboost/libs/model_interface',
                    binary);
            }
            return;
        case 'win32':
            for (const binary of config.binaries['win']) {
                await downloadBinaryFile('./build/catboost/libs/model_interface',
                    binary);
            }
            return;
        default:
            throw new Error(`Platform ${platform} is not supported`);
    }

}

export async function buildLocal(platform: string) {
    await preparePlatformBinary(platform);
    await configureGyp('./inc');
    await compileNativeAddon('./inc');
}
