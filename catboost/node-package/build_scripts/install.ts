import {buildLocal, buildNative} from './build';
import {existsSync} from 'fs';

async function install(): Promise<void> {
    if (existsSync('../../build/build_native.py')) {
        await buildNative();
        return;
    }

    await buildLocal(process.platform);
}

install().catch(err => {
    console.error(err);
    process.exit(1);
});