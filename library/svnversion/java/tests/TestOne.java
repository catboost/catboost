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
        System.out.println(vcs.getProgramHgHash());
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
            Assert.assertFalse(vcs.getProgramHgHash().equals(""));
        }
        if (vcsType.equals("git")) {
            Assert.assertFalse(vcs.getProgramHgHash().equals(""));
        }
        if (vcsType.equals("hg")) {
            Assert.assertFalse(vcs.getProgramHgHash().equals(""));
        }
        if (vcsType.equals("svn")) {
            Assert.assertFalse(vcs.getArcadiaSourceUrl().equals(""));
            Assert.assertTrue(vcs.getProgramSvnRevision() > 0);
            Assert.assertTrue(vcs.getArcadiaLastChangeNum() > 0);
        }
    }
}
