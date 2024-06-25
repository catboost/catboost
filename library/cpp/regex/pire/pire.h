#pragma once

#ifndef PIRE_NO_CONFIG
#define PIRE_NO_CONFIG
#endif

#include <contrib/libs/pire/pire/pire.h>
#include <contrib/libs/pire/pire/extra.h>

#include <library/cpp/charset/doccodes.h>

namespace NPire {
    using TChar = Pire::Char;
    using Pire::MaxChar;

    // Scanner classes
    using TScanner = Pire::Scanner;
    using TNonrelocScanner = Pire::NonrelocScanner;
    using TScannerNoMask = Pire::ScannerNoMask;
    using TNonrelocScannerNoMask = Pire::NonrelocScannerNoMask;
    using THalfFinalScanner = Pire::HalfFinalScanner;
    using TNonrelocHalfFinalScanner = Pire::NonrelocHalfFinalScanner;
    using THalfFinalScannerNoMask = Pire::HalfFinalScannerNoMask;
    using TNonrelocHalfFinalScannerNoMask = Pire::NonrelocHalfFinalScannerNoMask;
    using TSimpleScanner = Pire::SimpleScanner;
    using TSlowScanner = Pire::SlowScanner;
    using TCapturingScanner = Pire::CapturingScanner;
    using TSlowCapturingScanner = Pire::SlowCapturingScanner;
    using TCountingScanner = Pire::CountingScanner;

    template <typename T1, typename T2>
    using TScannerPair = Pire::ScannerPair<T1, T2>;

    // Helper classes
    using TFsm = Pire::Fsm;
    using TLexer = Pire::Lexer;
    using TTerm = Pire::Term;
    using TEncoding = Pire::Encoding;
    using TFeature = Pire::Feature;
    using TFeaturePtr = Pire::Feature::Ptr;
    using TError = Pire::Error;

    // Helper functions
    using Pire::LongestPrefix;
    using Pire::LongestSuffix;
    using Pire::Matches;
    using Pire::MmappedScanner;
    using Pire::Run;
    using Pire::Runner;
    using Pire::ShortestPrefix;
    using Pire::ShortestSuffix;
    using Pire::Step;

    using namespace Pire::SpecialChar;
    using namespace Pire::Consts;

    namespace NFeatures {
        using Pire::Features::AndNotSupport;
        using Pire::Features::Capture;
        using Pire::Features::CaseInsensitive;
        using Pire::Features::GlueSimilarGlyphs;
    }

    namespace NEncodings {
        using Pire::Encodings::Latin1;
        using Pire::Encodings::Utf8;

        const NPire::TEncoding& Koi8r();
        const NPire::TEncoding& Cp1251();
        const NPire::TEncoding& Get(ECharset encoding);
    }

    namespace NTokenTypes {
        using namespace Pire::TokenTypes;
    }
}
