import {mkdirSync, existsSync, readFileSync, writeFileSync} from 'fs';
import {copyFile} from './download';
import {join} from 'path';
import {createConfigForVersion, writeConfig} from './config';

export function prepareHeaders() {
    const incDir = './inc';

    if (!existsSync(incDir)) {
        mkdirSync(incDir);
    }

    const rootPath = '../..';

    const cApiPath = 'catboost/libs/model_interface/c_api.h';
    copyFile(join(rootPath, cApiPath), join(incDir, cApiPath));
}

export async function generateConfigForVersion(version: string) {

    console.log(`Preparing config for version ${version}`);
    const [config, error] = await createConfigForVersion(version);
    writeConfig(config);
    if (error !== undefined) {
        throw error;
    }
}

export function patchPackageJSONWithVersion(version: string) {
    const pkg = readFileSync('./package.json', 'utf8');
    const packageObj = JSON.parse(pkg);
    packageObj['version'] = version;
    writeFileSync('./package.json', JSON.stringify(packageObj, null, 2));
}