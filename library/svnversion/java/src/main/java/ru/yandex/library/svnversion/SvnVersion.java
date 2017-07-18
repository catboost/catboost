package ru.yandex.library.svnversion;

public class SvnVersion {
    private SvnVersion() {}

    public static void main(String[] args) {
        printSvnVersionAndExit0();
    }

    public static String getProgramSvnVersion() {
        return SvnConstants.GetProperty("PROGRAM_VERSION", "No program version found");
    }

    public static void printProgramSvnVersion() {
        System.out.println(getProgramSvnVersion());
    }

    public static void printSvnVersionAndExit0() {
        printProgramSvnVersion();
        System.exit(0);
    }

    public static String getArcadiaSourcePath() {
        return SvnConstants.GetProperty("ARCADIA_SOURCE_PATH", "");
    }

    public static String getArcadiaSourceUrl() {
        return SvnConstants.GetProperty("ARCADIA_SOURCE_URL", "");
    }

    public static String getArcadiaLastChange() {
        return SvnConstants.GetProperty("ARCADIA_SOURCE_LAST_CHANGE", "");
    }

    public static String getArcadiaLastAuthor() {
        return SvnConstants.GetProperty("ARCADIA_SOURCE_LAST_AUTHOR", "");
    }

    public static int getProgramSvnRevision() {
        return Integer.parseInt(SvnConstants.GetProperty("ARCADIA_SOURCE_REVISION", "0").trim());
    }

    public static String getProgramScmData() {
        return SvnConstants.GetProperty("SCM_DATA", "");
    }

    public static String getProgramBuildUser() {
        return SvnConstants.GetProperty("BUILD_USER", "");
    }

    public static String getProgramBuildHost() {
        return SvnConstants.GetProperty("BUILD_HOST", "");
    }

    public static String getProgramBuildDate() {
        return SvnConstants.GetProperty("BUILD_DATE", "");
    }
}
