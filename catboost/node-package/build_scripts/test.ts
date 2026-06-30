import {execProcess} from './common';

export async function test(haveCuda: boolean) {
    var cmd = 'node ./lib/tests/model.js'
    if (haveCuda) {
        cmd += ' --have-cuda'
    }
    const result = await execProcess(cmd);
    if (result.code !== 0) {
        throw(new Error(`Unit tests failed :${
            result.code} ${result.signal} ${result.err?.message}`))
    }
}
