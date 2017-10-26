#pragma once

class TString;

// NOTE: This function has rare sporadic failures (throws exceptions) on FreeBSD. See REVIEW:54297
const TString& GetExecPath();
