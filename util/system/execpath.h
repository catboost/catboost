#pragma once

class TString;
class TMappedFile;

// NOTE: This function has rare sporadic failures (throws exceptions) on FreeBSD. See REVIEW:54297
const TString& GetExecPath();

// Opens the program's binary for reading, if it still exists. Throws an exception on failure.
TMappedFile* OpenExecFile();
