package ai.catboost.common;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.validation.constraints.NotNull;
import java.io.*;


public class NativeLib {
    private static final Logger logger = LoggerFactory.getLogger(NativeLib.class);

    /**
     * Load libName, first will try try to load libName from default location then will try to load library from JAR.
     *
     * @param libName
     * @throws IOException
     */
    public static void smartLoad(final @NotNull String libName) throws IOException {
        try {
            loadNativeLibraryFromJar(libName);
        } catch (IOException ioe) {
            logger.error("failed to load native library from both default location and JAR");
            throw ioe;
        }
    }

    @NotNull
    private static String[] getCurrentMachineResourcesDirs() {
        // NOTE: This is an incomplete list of all possible combinations! But CatBoost officially only supports x86_64
        // and arm64 platfroms on Mac, and only x86_64 on Linux and Windows. If you wont support for other platforms 
        // you'll have to build JNI from sources by yourself for your target platform and probably write your own shared
        // library loader for shared library.
        // On macOS this function returns paths for both arch-specific and universal binaries paths.

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

            // For macOS they are the same
            if (osArch.equals("aarch64")) {
                osArch = "arm64";
            }
            return new String[] {osName + "-" + osArch, osName + "-universal2"};
        } else {
            if (osName.contains("win")) {
                osName = "win32";
            }
            // Will result in something like "linux-x86_64"
            return new String[] {osName + "-" + osArch};
        }
    }

    private static void loadNativeLibraryFromJar(final @NotNull String libName) throws IOException {
        for (String machineResourcesDir : getCurrentMachineResourcesDirs()) {
            final String pathWithinJar = "/" + machineResourcesDir + "/lib/" + System.mapLibraryName(libName);
            if (NativeLib.class.getResource(pathWithinJar) != null) {
                final String tempLibPath = createTemporaryFileFromJar(pathWithinJar);
                System.load(tempLibPath);
                return;
            }
        }
        throw new IOException("Native library '" + libName + "' not found");
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
