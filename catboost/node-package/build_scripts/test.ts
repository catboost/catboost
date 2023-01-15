import {execProcess} from './common';

export async function test() {
    const result = await execProcess('node ./lib/tests/model.js');
    if (result.code !== 0) {
        throw(new Error(`Unit tests failed :${
            result.code} ${result.signal} ${result.err?.message}`))
    }
}
