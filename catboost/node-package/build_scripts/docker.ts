import {execProcess} from './common';

async function buildE2ETestImage(imageName: string, dockerfile: string) {
    return await execProcess(`docker build -f ${dockerfile} . -t ${
        imageName
    }:latest`);
}

async function runE2EContainerTest(imageName: string) {
    return await execProcess(`docker run ${imageName}:latest`).then(result => {
        if (result.code !== 0) {
            throw new Error('E2E test have failed! Check logs for execution details.');
        }
    });
}

export async function E2EDeploymentTest(dockerfile: string) {
    const imageName = 'node-catboost-e2e-deploy-test';
    const imageBuildExecution = await buildE2ETestImage(imageName, dockerfile);
    if (imageBuildExecution.code !== 0) {
        throw imageBuildExecution.err ?? new Error('Failed to build E2E test image!' + 
            'Check execution logs for details.');
    }

    return runE2EContainerTest(imageName);
}
