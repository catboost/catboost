"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.execProcess = void 0;
var child_process_1 = require("child_process");
;
function execProcess(command) {
    var child = child_process_1.exec(command);
    return new Promise(function (resolve) {
        child.stdout.on('data', function (chunk) {
            console.log(chunk.toString());
        });
        child.stderr.on('data', function (chunk) {
            console.error(chunk.toString());
        });
        child.on('error', function (err) { return resolve({ err: err }); });
        child.on('exit', function (code, signal) { return resolve({ code: code, signal: signal }); });
    });
}
exports.execProcess = execProcess;
