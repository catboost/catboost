import {readFileSync, writeFileSync} from 'fs';
import {join} from 'path';
import {calculateFileHash, downloadFile} from './download';

const GITHUB_PATH = 'https://github.com/catboost/catboost/releases/download/';

const PLATFORM_TO_BINARY: {[name: string]: string[]} = {
    'linux': ['libcatboostmodel.so'],
    'mac': ['libcatboostmodel.dylib'],
    'win': ['catboostmodel.dll', 'catboostmodel.lib'],
};

export interface BinaryFileData {
    readonly url: string;
    readonly targetFile: string;
    readonly sha256: string;
}

export interface BinaryConfig {
    readonly binaries: {
        [platform: string]: BinaryFileData[],
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
    const entries: {
        [platform: string]: BinaryFileData[],
    } = {};
    for (const platform of Object.keys(data)) {
        entries[platform] = data[platform].map(parseBinaryFileData)
    }

    return {
        binaries: entries,
    };
}

export function readConfig(): BinaryConfig {
    return readConfigFile('./config.json');
}

export function writeConfig(config: BinaryConfig) {
    writeFileSync('./config.json', JSON.stringify(config.binaries, null, 1));
}

export async function createConfigForVersion(version: string): Promise<[BinaryConfig, Error|undefined]> {
    const config: BinaryConfig = {
        binaries: {},
    };
    const failedToDownload: string[] = [];
    for (const platform of Object.keys(PLATFORM_TO_BINARY)) {
        config.binaries[platform] = [];
        for (const binary of PLATFORM_TO_BINARY[platform]) {
            const targetFile = binary;
            const url = join(GITHUB_PATH, version, targetFile);
            const tmpPath = join('./build/tmp/', targetFile);
            try {
                await downloadFile(url, tmpPath);
            } catch(error) {
                failedToDownload.push(targetFile);
                config.binaries[platform].push({
                    url: `Not found: ${targetFile}`, 
                    sha256: '', 
                    targetFile,
                });
                continue;
            }
            const sha256 = calculateFileHash(tmpPath);

            config.binaries[platform].push({url, sha256, targetFile});
        }
    }

    let error: Error|undefined;
    if (failedToDownload.length > 0) {
        error = new Error(`\n==========================================
The following release files failed to get auto-detected:\n${
            failedToDownload.map(file => `\t- ${file}`).join('\n')
        }\nPlease open 'config.json' and adjust the links and checksums manually.
==========================================\n`);
    }

    return [config, error];
}
