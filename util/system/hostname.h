#pragma once

class TString;

const char* GetHostName();
const TString& HostName();

const char* GetFQDNHostName();
const TString& FQDNHostName();
bool IsFQDN(const TString& name);
