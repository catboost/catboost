import {readFileSync, writeFileSync} from 'fs';
import {join} from 'path';
import {calculateFileHash, downloadFile} from './download';

const GITHUB_PATH = 'https://github.com/catboost/catboost/releases/download/';


function getPlatformArchToBinary(version: string) : any {
    return {
        'linux-x64': [
            [`libcatboostmodel-linux-x86_64-${version}.so`, 'libcatboostmodel.so']
        ],
        'linux-arm64': [
            [`libcatboostmodel-linux-aarch64-${version}.so`, 'libcatboostmodel.so']
        ],
        'darwin-universal2': [
            [`libcatboostmodel-darwin-universal2-${version}.dylib`, 'libcatboostmodel.dylib']
        ],
        'win32-x64': [
            [`catboostmodel-windows-x86_64-${version}.dll`, 'catboostmodel.dll'],
            [`catboostmodel-windows-x86_64-${version}.lib`, 'catboostmodel.lib']
        ],
    }
}


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
    const platform_arch_to_binaries = getPlatformArchToBinary(version);
    for (const platform_arch of Object.keys(platform_arch_to_binaries)) {
        config.binaries[platform_arch] = [];
        for (const src_and_target_file of platform_arch_to_binaries[platform_arch]) {
            const srcFile = src_and_target_file[0];
            const targetFile = src_and_target_file[1];
            const url = join(GITHUB_PATH, 'v' + version, srcFile);
            const tmpPath = join('./build/tmp/', srcFile);
            try {
                await downloadFile(url, tmpPath);
            } catch(error) {
                failedToDownload.push(srcFile);
                config.binaries[platform_arch].push({
                    url: `Not found: ${srcFile}`,
                    sha256: '',
                    targetFile,
                });
                continue;
            }
            const sha256 = calculateFileHash(tmpPath);

            config.binaries[platform_arch].push({url, sha256, targetFile});
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
