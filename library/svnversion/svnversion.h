#pragma once

#if !defined(FROM_IMPL)
#define PROGRAM_VERSION GetProgramSvnVersion()
#define ARCADIA_SOURCE_PATH GetArcadiaSourcePath()
#define PRINT_VERSION PrintSvnVersionAndExit(argc, (char**)argv)
#define PRINT_VERSION_EX(opts) PrintSvnVersionAndExitEx(argc, (char**)argv, opts)
#endif

#if defined(__cplusplus)
extern "C" {
#endif
    const char* GetProgramSvnVersion();  // verbose multiline message
    void PrintProgramSvnVersion();
    const char* GetArcadiaSourcePath();  // "/home/myltsev/arcadia"
    const char* GetArcadiaSourceUrl();   // "svn+ssh://arcadia.yandex.ru/arc/trunk/arcadia"
    const char* GetArcadiaLastChange();  // "2902074"
    const char* GetArcadiaLastAuthor();  // "dieash"
    int GetProgramSvnRevision();         // 2902074
    void PrintSvnVersionAndExit(int argc, char *argv[]);
    void PrintSvnVersionAndExitEx(int argc, char *argv[], const char *opts);
    void PrintSvnVersionAndExit0();
    const char* GetProgramScmData();     // verbose multiline message
    const char* GetProgramBuildUser();
    const char* GetProgramBuildHost();
    const char* GetProgramBuildDate();
#if defined(__cplusplus)
}
#endif
