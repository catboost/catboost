import org.junit.Test;

import ru.yandex.library.svnversion.VcsVersion;


public class TestOne {
    private static void outAll(String cls) throws Exception {
        VcsVersion VCS = cls.equals("") ? new VcsVersion() : new VcsVersion(cls);
        outAll(VCS);
    }

    private static void outAll(Class cls) throws Exception {
        outAll(new VcsVersion(cls));
    }

    private static void outAll(VcsVersion vcs) throws Exception {
        System.out.println(vcs.getProgramSvnVersion());
        System.out.println(vcs.getArcadiaSourcePath());
        System.out.println(vcs.getArcadiaSourceUrl());
        System.out.println(vcs.getArcadiaLastChange());
        System.out.println(vcs.getArcadiaLastChangeNum());
        System.out.println(vcs.getArcadiaLastAuthor());
        System.out.println(vcs.getProgramSvnRevision());
        System.out.println(vcs.getProgramHgHash());
        System.out.println(vcs.getProgramScmData());
        System.out.println(vcs.getProgramBuildUser());
        System.out.println(vcs.getProgramBuildHost());
        System.out.println(vcs.getProgramBuildDate());
        System.out.println(vcs.getVCS());
        System.out.println(vcs.getBranch());
    }

    @Test
    public void test1() throws Exception {
        outAll("");
    }

    @Test
    public void test2() throws Exception {
        outAll("ru.yandex.library.svnversion.VcsVersion");
    }

    @Test
    public void test3() throws Exception {
        outAll(VcsVersion.class);
    }
}
