import {buildLocal, buildNative} from './build';
import {existsSync} from 'fs';

async function install(): Promise<void> {
    if (existsSync('../../build/build_native.py')) {
        const buildNativeExtraArgs = process.argv.slice(5)
        await buildNative(buildNativeExtraArgs);
        return;
    }

    await buildLocal(process.platform, process.arch);
}

install().catch(err => {
    console.error(err);
    process.exit(1);
});