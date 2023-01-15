import {exec} from 'child_process';

export interface ExecutionResult {
    code?: number;
    signal?: string;
    err?: Error;
};

export function execProcess(command: string): Promise<ExecutionResult> {   
    const child = exec(command);
    return new Promise(resolve => {
        child.stdout.on('data', chunk => {
            console.log(chunk.toString());
        });
        child.stderr.on('data', chunk => {
            console.error(chunk.toString());
        });
        child.on('error', err => resolve({err}));
        child.on('exit', (code, signal) => resolve({code, signal}));
    });
}