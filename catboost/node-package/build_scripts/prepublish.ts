import {mkdirSync, existsSync, readdirSync, lstatSync} from 'fs';
import {copyFile} from './download';
import {extname, join} from 'path';
import { compileJs } from './build';

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

function prepareHeaders() {
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

async function prepareAndBuildPackage() {
    prepareHeaders();
    compileJs();
}

prepareAndBuildPackage();