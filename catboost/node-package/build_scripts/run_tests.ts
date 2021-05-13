import {compileJs, compileBindings} from './build';
import {test} from './test';

async function allTests() {
    await compileBindings();
    await test();
}

allTests().catch(err => {
    console.error(err);
    process.exit(1);
});