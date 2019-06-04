package ai.catboost;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.validation.constraints.NotNull;
import java.io.*;

/**
 * Shared library loader, this class is intended to be an extension point for users of this modele. If you want to
 * define a custom loader for shared library you may override this class via `-classpath`, only requirement on your
 * your class will be to have {@link #handle()} method that returns CatBoostJNI class.
 */
class NativeLib {
    private static final Logger logger = LoggerFactory.getLogger(NativeLib.class);

    static {
        try {
            smartLoad("catboost4j-prediction");
        } catch (Exception ex) {
            logger.error("Failed to load native library", ex);
            throw new RuntimeException(ex);
        }
    }

    /**
     * @return JNI handle for CatBoost model application.
     */
    @NotNull
    public static CatBoostJNI handle() {
        return SingletonHolder.catBoostJNI;
    }

    private static final class SingletonHolder {
        public static final CatBoostJNI catBoostJNI = new CatBoostJNI();
    }

    /**
     * Load libName, first will try try to load libName from default location then will try to load library from JAR.
     *
     * @param libName
     * @throws IOException
     */
    private static void smartLoad(final @NotNull String libName) throws IOException {
        try {
            loadNativeLibraryFromJar(libName);
        } catch (IOException ioe) {
            logger.error("failed to load native library from both default location and JAR");
            throw ioe;
        }
    }

    @NotNull
    private static String getCurrentMachineResourcesDir() {
        // NOTE: This is an incomplete list of all possible combinations! But CatBoost officially only supports x86_64
        // platform (on Mac, Linux and Windows). If you wont support for other platforms you'll have to build JNI from
        // sources by yourself for your target platform and probably write your own shared library loader for shared
        // library.

        String osArch = System.getProperty("os.arch").toLowerCase();
        // Java is inconsistent with Python, and returns `amd64` on my dev machine, while Python `platform.machine()`
        // returns `x86_64`, so we'll have to fix this
        if (osArch.equals("amd64")) {
            osArch = "x86_64";
        }

        String osName = System.getProperty("os.name").toLowerCase();

        // Java doesn't seem to have analog for python's `sys.platform` or `platform.platform`, so we have to do it by
        // hand.
        if (osName.contains("mac")) {
            osName = "darwin";
        } else if (osName.contains("win")) {
            osName = "win32";
        }

        // Will result in something like "linux-x86_64"
        return osName + "-" + osArch;
    }

    private static void loadNativeLibraryFromJar(final @NotNull String libName) throws IOException {
        final String pathWithinJar = "/" + getCurrentMachineResourcesDir() + "/lib/" + System.mapLibraryName(libName);
        final String tempLibPath = createTemporaryFileFromJar(pathWithinJar);
        System.load(tempLibPath);
    }

    private static void copyFileFromJar(final @NotNull String pathWithinJar, final @NotNull String pathOnDisk) throws IOException {
        byte[] copyBuffer = new byte[4 * 1024];
        int bytesRead;

        try(OutputStream out = new BufferedOutputStream(new FileOutputStream(pathOnDisk));
            InputStream in = NativeLib.class.getResourceAsStream(pathWithinJar)) {

            if (in == null) {
                throw new FileNotFoundException("File " + pathWithinJar + " was not found inside JAR.");
            }

            while ((bytesRead = in.read(copyBuffer)) != -1) {
                out.write(copyBuffer, 0, bytesRead);
            }
        }
    }

    @NotNull
    private static String createTemporaryFileFromJar(final @NotNull String pathWithinJar) throws IOException, IllegalArgumentException {
        if (!pathWithinJar.startsWith("/")) {
            throw new IllegalArgumentException("Path must be absolute (start with '/')");
        }

        if (pathWithinJar.endsWith("/")) {
            throw new IllegalArgumentException("Must be a path to file not directory (ends with '/')");
        }

        String[] parts = pathWithinJar.split("/");
        final String filename = parts[parts.length - 1];

        parts = filename.split("\\.", 2);
        final String prefix = parts[0];
        final String suffix = parts.length > 1 ? "." + parts[parts.length - 1] : null;

        final File libOnDisk = File.createTempFile(prefix, suffix);
        libOnDisk.deleteOnExit();

        copyFileFromJar(pathWithinJar, libOnDisk.getPath());

        return libOnDisk.getAbsolutePath();
    }
}
