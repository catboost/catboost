import {buildModel, compileBindings} from './build';

async function buildYa() {
    await buildModel();
    await compileBindings();
}

buildYa();