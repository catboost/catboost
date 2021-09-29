import {generateConfigForVersion, prepareHeaders} from './packaging';
import {compileJs} from './build';

async function prepareAndBuildPackage() {
    const version = process.argv[process.argv.length - 1];
    if (!/v[0-9\.]*/.exec(version)) {
        console.error(`Version "${version}" is not valid. Please add version as the last argument`);
        process.exit(1);
    }

    prepareHeaders();
    await compileJs();
    await generateConfigForVersion(version);
}

prepareAndBuildPackage();