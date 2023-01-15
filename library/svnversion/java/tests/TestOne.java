import junit.framework.Assert;
import org.junit.Test;

import ru.yandex.library.svnversion.VcsVersion;


public class TestOne {
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
        System.out.println(vcs.getProgramHash());
        System.out.println(vcs.getProgramScmData());
        System.out.println(vcs.getProgramBuildUser());
        System.out.println(vcs.getProgramBuildHost());
        System.out.println(vcs.getProgramBuildDate());
        System.out.println(vcs.getVCS());
        System.out.println(vcs.getBranch());
        System.out.println(vcs.getTag());
        System.out.println(vcs.getArcadiaGitPatchNumer());
    }

    @Test
    public void test3() throws Exception {
        outAll(VcsVersion.class);
    }

    @Test
    public void testSanity() throws Exception {
        VcsVersion vcs = new VcsVersion(VcsVersion.class);
        String vcsType = vcs.getVCS();

        Assert.assertTrue(
                vcsType.equals("arc") ||
                vcsType.equals("git") ||
                vcsType.equals("hg") ||
                vcsType.equals("svn"));

        Assert.assertFalse(vcs.getProgramBuildDate().equals(""));
        Assert.assertFalse(vcs.getProgramBuildHost().equals(""));
        Assert.assertFalse(vcs.getProgramBuildUser().equals(""));
        if (vcsType.equals("arc")) {
            Assert.assertFalse(vcs.getProgramHash().equals(""));
        }
        if (vcsType.equals("git")) {
            Assert.assertFalse(vcs.getProgramHash().equals(""));
        }
        if (vcsType.equals("hg")) {
            Assert.assertFalse(vcs.getProgramHash().equals(""));
        }
        if (vcsType.equals("svn")) {
            Assert.assertFalse(vcs.getArcadiaSourceUrl().equals(""));
            Assert.assertTrue(vcs.getProgramSvnRevision() > 0);
            Assert.assertTrue(vcs.getArcadiaLastChangeNum() > 0);
        }
    }

    // ya make -DFORCE_VCS_INFO_UPDATE --vcs-file=<vcs.json> library/svnversion
    // ya make -DFORCE_VCS_INFO_UPDATE --vcs-file=<vcs.json> library/svnversion/java/
    // ya tool java11 -cp library/svnversion/java/tests/java-tests.jar:library/svnversion/java/library-svnversion-java.jar TestOne
    public static void main(String[] args) {
        VcsVersion vcs = new VcsVersion(VcsVersion.class);
        System.out.println("getProgramSvnVersion(): " + vcs.getProgramSvnVersion());
        System.out.println("getArcadiaSourcePath(): " + vcs.getArcadiaSourcePath());
        System.out.println("getArcadiaSourceUrl(): " + vcs.getArcadiaSourceUrl());
        System.out.println("getArcadiaLastChange(): " + vcs.getArcadiaLastChange());
        System.out.println("getArcadiaLastChangeNum(): " + vcs.getArcadiaLastChangeNum());
        System.out.println("getArcadiaLastAuthor(): " + vcs.getArcadiaLastAuthor());
        System.out.println("getProgramSvnRevision(): " + vcs.getProgramSvnRevision());
        System.out.println("getProgramHash(): " + vcs.getProgramHash());
        System.out.println("getProgramScmData(): " + vcs.getProgramScmData());
        System.out.println("getProgramBuildUser(): " + vcs.getProgramBuildUser());
        System.out.println("getProgramBuildHost(): " + vcs.getProgramBuildHost());
        System.out.println("getProgramBuildDate(): " + vcs.getProgramBuildDate());
        System.out.println("getVCS(): " + vcs.getVCS());
        System.out.println("getBranch(): " + vcs.getBranch());
        System.out.println("getTag(): " + vcs.getTag());
        System.out.println("getArcadiaGitPatchNumer(): " + vcs.getArcadiaGitPatchNumer());
    }
}
