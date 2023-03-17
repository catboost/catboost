#include "junit.h"

#include <libxml/xmlwriter.h>
#include <util/system/fs.h>
#include <util/system/file.h>

namespace NUnitTest {

#define CHECK_CALL(expr) if ((expr) < 0) { \
    Cerr << "Faield to write to xml" << Endl; \
    return; \
}

#define XML_STR(s) ((const xmlChar*)(s))

void TJUnitProcessor::Save() {
    TString path = FileName;
    auto sz = path.size();
    TFile lockFile;
    TFile reportFile;
    TString lockFileName;
    const int MaxReps = 200;
#if defined(_win_)
    const char dirSeparator = '\\';
#else
    const char dirSeparator = '/';
#endif
    if ((sz == 0) or (path[sz - 1] == dirSeparator)) {
        if (sz > 0) {
            NFs::MakeDirectoryRecursive(path);
        }
        TString reportFileName;
        for (int i = 0; i < MaxReps; i++) {
            TString suffix = (i > 0) ? ("-" + std::to_string(i)) : "";
            lockFileName = path + ExecName + suffix + ".lock";
            try {
                lockFile = TFile(lockFileName, EOpenModeFlag::CreateNew);
            } catch (const TFileError&) {}
            if (lockFile.IsOpen()) {
                // Inside a lock, ensure the .xml file does not exist
                reportFileName = path + ExecName + suffix + ".xml";
                try {
                    reportFile = TFile(reportFileName, EOpenModeFlag::OpenExisting | EOpenModeFlag::RdOnly);
                } catch (const TFileError&) {
                    break;
                }
                reportFile.Close();
                lockFile.Close();
                NFs::Remove(lockFileName);
            }
        }
        if (!lockFile.IsOpen()) {
            Cerr << "Could not find a vacant file name to write report for path " << path << ", maximum number of reports: " << MaxReps << Endl;
            Y_FAIL("Cannot write report");
        }
        path = reportFileName;
    }
    auto file = xmlNewTextWriterFilename(path.c_str(), 0);
    if (!file) {
        Cerr << "Failed to open xml file for writing: " << path.c_str() << Endl;
        return;
    }

    CHECK_CALL(xmlTextWriterStartDocument(file, nullptr, "UTF-8", nullptr));
    CHECK_CALL(xmlTextWriterStartElement(file, XML_STR("testsuites")));
    CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("tests"), XML_STR(ToString(GetTestsCount()).c_str())));
    CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("failures"), XML_STR(ToString(GetFailuresCount()).c_str())));

    for (const auto& [suiteName, suite] : Suites) {
        CHECK_CALL(xmlTextWriterStartElement(file, XML_STR("testsuite")));
        CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("name"), XML_STR(suiteName.c_str())));
        CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("id"), XML_STR(suiteName.c_str())));
        CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("tests"), XML_STR(ToString(suite.GetTestsCount()).c_str())));
        CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("failures"), XML_STR(ToString(suite.GetFailuresCount()).c_str())));

        for (const auto& [testName, test] : suite.Cases) {
            CHECK_CALL(xmlTextWriterStartElement(file, XML_STR("testcase")));
            CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("name"), XML_STR(testName.c_str())));
            CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("id"), XML_STR(testName.c_str())));

            for (const auto& failure : test.Failures) {
                CHECK_CALL(xmlTextWriterStartElement(file, XML_STR("failure")));
                CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("message"), XML_STR(failure.c_str())));
                CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("type"), XML_STR("ERROR")));
                CHECK_CALL(xmlTextWriterEndElement(file));
            }

            CHECK_CALL(xmlTextWriterEndElement(file));
        }

        CHECK_CALL(xmlTextWriterEndElement(file));
    }

    CHECK_CALL(xmlTextWriterEndElement(file));
    CHECK_CALL(xmlTextWriterEndDocument(file));
    xmlFreeTextWriter(file);

    if (lockFile.IsOpen()) {
        lockFile.Close();
        NFs::Remove(lockFileName.c_str());
    }
}

} // namespace NUnitTest
