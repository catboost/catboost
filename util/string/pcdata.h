#pragma once

#include <util/generic/fwd.h>

/// Converts a text into HTML-code. Special characters of HTML («<», «>», ...) replaced with entities.
TString EncodeHtmlPcdata(const TStringBuf str, bool qAmp = true);
void EncodeHtmlPcdataAppend(const TStringBuf str, TString& strout);

/// Reverse of EncodeHtmlPcdata()
TString DecodeHtmlPcdata(const TString& sz);
