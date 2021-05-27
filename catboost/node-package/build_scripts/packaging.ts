import {mkdirSync, existsSync, readdirSync, lstatSync, readFileSync, writeFileSync} from 'fs';
import {copyFile} from './download';
import {extname, join} from 'path';
import {createConfigForVersion, writeConfig} from './config';

function copyHeadersRecursively(srcPath: string, targetPath: string) {
    for (const part of readdirSync(srcPath)) {
        const path = join(srcPath, part);
        const target = join(targetPath, part);
        if (lstatSync(path).isDirectory()) {
            copyHeadersRecursively(path, target);
            continue;
        }

        if (extname(path) === '.h') {
            copyFile(path, target);
        }
    }
}

export function prepareHeaders() {
    const incDir = './inc';

    if (!existsSync(incDir)) {
        mkdirSync(incDir);
    }

    const rootPath = '../..';

    const cApiPath = 'catboost/libs/model_interface/c_api.h';
    copyFile(join(rootPath, cApiPath), join(incDir, cApiPath));

    const stlfwdPath = 'contrib/libs/cxxsupp/system_stl/include/stlfwd';
    copyFile(join(rootPath, stlfwdPath), join(incDir, stlfwdPath));

    const utilPath = 'util';
    copyHeadersRecursively(join(rootPath, utilPath), join(incDir, utilPath));
}

export async function generateConfigForVersion(version: string) {
    
    console.log(`Preparing config for verion ${version}`);
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