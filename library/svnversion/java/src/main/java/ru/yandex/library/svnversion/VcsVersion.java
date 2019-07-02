package ru.yandex.library.svnversion;

import java.net.URL;
import java.net.URLClassLoader;
import java.util.jar.Manifest;
import java.util.jar.Attributes;
import java.util.Enumeration;
import java.util.Base64;
import java.util.Base64.Decoder;
import java.util.jar.JarFile;
import java.io.InputStream;

public class VcsVersion {
    public VcsVersion(Class cls) {
        if (cls == null) {
            return;
        }

        URL clsUrl = cls.getResource(cls.getSimpleName() + ".class");
        if (!clsUrl.getProtocol().equals("jar")) {
            return;
        }

        ClassLoader cld = cls.getClassLoader();
        String clsPath = clsUrl.toString();
        String manifestPath = clsPath.substring(0, clsPath.lastIndexOf("!") + 2) + JarFile.MANIFEST_NAME;
        try {
            Enumeration manEnum = cld.getResources(JarFile.MANIFEST_NAME);
            while (manEnum.hasMoreElements()) {
                URL url = (URL)manEnum.nextElement();
                if (!url.toString().equals(manifestPath)) {
                    continue;
                }
                InputStream is = url.openStream();
                if (is != null) {
                    Manifest = new Manifest(is);
                }
            }
        }
        catch (Exception e) {
        }
    }

    public String getProgramSvnVersion() {
        return getLongBase64StringSafe("Program-Version-String");
    }

    public void printProgramSvnVersion() {
        System.out.println(getProgramSvnVersion());
    }

    public void printSvnVersionAndExit0() {
        printProgramSvnVersion();
        System.exit(0);
    }

    public String getArcadiaSourcePath() {
        return getStringParamSafe("Arcadia-Source-Path");
    }

    public String getArcadiaSourceUrl() {
        return getStringParamSafe("Arcadia-Source-URL");
    }

    public String getArcadiaLastChange() {
        return getStringParamSafe("Arcadia-Source-Last-Change");
    }

    public int getArcadiaLastChangeNum() {
        return getIntParam("Arcadia-Source-Last-Change");
    }

    public String getArcadiaLastAuthor() {
        return getStringParamSafe("Arcadia-Source-Last-Author");
    }

    public int getProgramSvnRevision() {
        return getIntParam("Arcadia-Source-Revision");
    }

    public int getArcadiaGitPatchNumer() {
        return getIntParam("Arcadia-Patch-Number");
    }

    public String getProgramHash() {
        return getStringParamSafe("Arcadia-Source-Hash");
    }

    public String getProgramScmData() {
        return getLongBase64StringSafe("SCM-String");
    }

    public String getProgramBuildUser() {
        return getStringParamSafe("Build-User");
    }

    public String getProgramBuildHost() {
        return getStringParamSafe("Build-Host");
    }

    public String getProgramBuildDate() {
        return getStringParamSafe("Build-Date");
    }

    public String getVCS() {
        return getStringParamSafe("Version-Control-System");
    }

    public String getBranch() {
        return getStringParamSafe("Branch");
    }

    public String getTag() {
        return getStringParamSafe("Arcadia-Tag");
    }

    private int getIntParam(String Str) {
        String Val = getStringParam(Str);
        if (Val == null) {
            return -1;
        }
        return Integer.parseInt(Val);
    }

    private String getStringParam(String str) {
        if (Manifest == null) {
            return null;
        }
        Attributes MainSection = Manifest.getMainAttributes();
        return MainSection.getValue(str);

    }

    private String getStringParamSafe(String str) {
        String val = getStringParam(str);
        return val == null ? "" : val;
    }

    private String getLongBase64StringSafe(String str) {
        String Val = getStringParam(str);
        if (Val == null) {
            return "";
        }
        return new String(Base64.getDecoder().decode(Val));
    }

    public static void main(String[] args) {
        VcsVersion vcs = new VcsVersion(VcsVersion.class);
        vcs.printSvnVersionAndExit0();
    }

    private Manifest Manifest;
}
