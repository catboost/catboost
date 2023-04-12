import {buildNative} from './build';
import {generateConfigForVersion, patchPackageJSONWithVersion, prepareHeaders} from './packaging';
import {test} from './test';
import {E2EDeploymentTest} from './docker';

async function ci() {
    if (process.argv.length !== 7) {
        throw new Error('ci script --- Usage: npm run ci <catboost-release version> <catboost-node-package version>');
    }
    const catboostVersion = process.argv[5];
    if (!/v[0-9\.]*/.exec(catboostVersion)) {
        throw new Error(`Version '${catboostVersion}' is not a valid catboost version.`);
    }

    const packageVersion = process.argv[6];
    if (!/[0-9\.]*/.exec(packageVersion)) {
        throw new Error(`Version '${packageVersion}' is not a valid node-package version.`);
    }

    console.log(`Patching "package.json" with version "${packageVersion}"...`);
    patchPackageJSONWithVersion(packageVersion);
    console.log('Building catboost package against repository sources...');
    await buildNative();
    console.log('Running local unit tests...');
    await test();
    console.log('Preparing package...');
    prepareHeaders();
    await generateConfigForVersion(catboostVersion);
    console.log('Running e2e-deployment test...');
    await E2EDeploymentTest('./e2e_tests/e2eTest.Dockerfile');
    console.log(`Looks like you are good! Run "npm publish" from "catboost/catboost/node-package/" directory to publish the version "${
        packageVersion
    }"`);
}

ci().catch(err => {
    console.error(err);
    process.exit(1);
});
