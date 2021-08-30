import {get, RequestOptions} from 'https';
import {createReadStream, createWriteStream, existsSync, mkdirSync, readFileSync, writeFileSync} from 'fs';
import {createHash} from 'crypto';
import {dirname, join, resolve, sep} from 'path';

import {BinaryFileData} from './config';

export async function downloadFile(url: string, targetPath: string): Promise<void> {
    checkMkdir(dirname(targetPath), '.');
    return new Promise((resolve, reject) => {
        
        try {
            get(url as RequestOptions, async rsp => {
                if (rsp.statusCode && rsp.statusCode > 300 && rsp.statusCode < 400 && rsp.headers.location) {
                    await downloadFile(rsp.headers.location, targetPath);
                    return resolve();
                }
                if (rsp.statusCode && rsp.statusCode >= 400) {
                    return reject(new Error(`Status code: ${rsp.statusCode}`));
                }
                const chunks: Buffer[] = [];
                rsp.on('data', (data) => {
                    chunks.push(data as Buffer);
                })
                rsp.on('end', () => {
                    writeFileSync(targetPath, Buffer.concat(chunks));
                    resolve();
                });
            });
        } catch (err) {
            reject(err);
        }
    });
}

function checkMkdir(dirPath: string, basePath: string) {
    const parts = dirPath.split(sep);

    parts.reduce((base: string, part: string) => {
        const path = resolve(base, part);
        if (!existsSync(path)) {
            mkdirSync(path)
        }

        return path;
    }, basePath);
}

/** Copy file to local path ./{targetPath}. */
export function copyFile(srcPath: string, targetPath: string) {
    const targetDir = dirname(targetPath);
    if (!existsSync(targetDir)) {
        checkMkdir(targetDir, '.');
    }
    const targetFile = createWriteStream(targetPath);
    createReadStream(srcPath).pipe(targetFile);
}

export function calculateFileHash(path: string): string {
    return createHash('sha256').update(readFileSync(path)).digest('hex');
}

export async function downloadBinaryFile(targetDirPath: string, binary: BinaryFileData): Promise<string> {
    const targetFile = join(targetDirPath, binary.targetFile);
    await downloadFile(binary.url, targetFile);
    const actualHash = calculateFileHash(targetFile);
    if (actualHash !== binary.sha256) {
        throw new Error(`Expected hash ${binary.sha256}, but actual hash value is ${actualHash}`);
    }

    return targetFile;
}