const childProcess = require('child_process');
const fs = require('fs');
var path = require('path');

process.chdir('./build_scripts');

function compileBuildScripts() {
    const buildRun = childProcess.exec(path.join('..','node_modules', '.bin', 'tsc'));
    buildRun.stdout.on('data', chunk => {
        console.log(chunk);
    });
    buildRun.stderr.on('data', chunk => {
        console.error(chunk);
    });
    
    return new Promise(resolve => {
        buildRun.on('error', err => resolve({err}));
        buildRun.on('exit', (code, signal) => resolve({code, signal}));
    });
}

function runScript(script) {
    process.chdir('..');

    const child = childProcess.fork('./build_scripts/out/' + script + '.js', process.argv);
    return new Promise((resolve, reject) => {
        child.on('error', err => reject({err}));
        child.on('exit', (code, signal) => {
            if (code !== 0) {
                console.error(`Script failed with exit code: ${code}`);
                reject(code);
                return;
            } 
            resolve({code, signal});
        });
    });
}

function runCommand(command) {
    if (command === 'bootstrap') {
        compileBuildScripts().then(console.log);
        return;
    }

    runScript(command);
} 

const command = process.argv[2];
const always_compile_scripts = ['ci', 'package_prepublish'];

if (!fs.existsSync('./out') || always_compile_scripts.indexOf(command) !== -1) {
    compileBuildScripts().then(result => {
        if (result.code === 0) {
            runCommand(command);
            return;
        }
        console.error(result);
    });
} else {
    runCommand(command);
}

