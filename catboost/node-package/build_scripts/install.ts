import {buildLocal, buildYa} from './build';
import {existsSync} from 'fs';

async function install(): Promise<void> {
    if (existsSync('../../ya')) {
        await buildYa();
        return;
    }
    await buildLocal(process.platform);
}

install();