import {readFileSync, writeFileSync} from 'fs';
import { join } from 'path';
import { calculateFileHash, downloadBinaryFile, downloadFile } from './download';

const GITHUB_PATH = 'https://github.com/catboost/catboost/releases/download/';

const PLATFORM_TO_BINARY: {[name: string]: string} = {
    'linux': 'libcatboostmodel.so',
    'mac': 'libcatboostmodel.dylib',
    'win': 'catboostmodel.dll',
};

export interface BinaryFileData {
    readonly url: string;
    readonly targetFile: string;
    readonly sha256: string;
}

export interface BinaryConfig {
    readonly binaries: {
        [platform: string]: BinaryFileData,
    };
}

function parseBinaryFileData(mapEntry: {[name: string]: string|undefined}): BinaryFileData {
    if (mapEntry['url'] === undefined || mapEntry['sha256'] === undefined || mapEntry['targetFile'] === undefined) {
        throw new Error(`Not a correct binary file entry in the config: ${JSON.stringify(mapEntry)}`);
    }

    return {
        url: mapEntry['url'],
        targetFile: mapEntry['targetFile'],
        sha256: mapEntry['sha256'],
    };
}

function readConfigFile(path: string) {
    const data = JSON.parse(readFileSync(path).toString());
    console.log(typeof(data));
    const entries: {
        [platform: string]: BinaryFileData,
    } = {};
    for (const platform of Object.keys(data)) {
        entries[platform] = parseBinaryFileData(data[platform]);
    }

    return {
        binaries: entries,
    };
}

export function readConfig(): BinaryConfig {
    return readConfigFile('./config.json');
}

export function writeConfig(config: BinaryConfig) {
    writeFileSync('./config.json', JSON.stringify(config.binaries));
}

export async function createConfigForVersion(version: string): Promise<BinaryConfig> {
    const config: BinaryConfig = {
        binaries: {},
    };
    for (const platform of Object.keys(PLATFORM_TO_BINARY)) {
        const targetFile = PLATFORM_TO_BINARY[platform];
        const url = join(GITHUB_PATH, version, targetFile);
        const tmpPath = join('./build/tmp/', targetFile);
        await downloadFile(url, tmpPath);
        const sha256 = calculateFileHash(tmpPath);

        config.binaries[platform] = {url, sha256, targetFile};
    }

    return config;
}
