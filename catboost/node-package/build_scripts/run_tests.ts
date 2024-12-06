import {compileJs, compileBindings} from './build';
import {test} from './test';

async function allTests() {
    await compileBindings();
    await test(process.argv.indexOf('--have-cuda') > -1);
}

allTests().catch(err => {
    console.error(err);
    process.exit(1);
});