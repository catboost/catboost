import {spawn} from 'child_process';

export interface ExecutionResult {
    code?: number;
    signal?: string;
    err?: Error;
};

export function execProcess(command: string): Promise<ExecutionResult> {   
    const child = spawn(command, [], { shell: true });
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