package ai.catboost;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.io.*;
import java.lang.reflect.Field;

// TODO(yazevnul): allow user to define it's custom way of loading a shared library

class NativeLibLoader {
    private static final Log logger = LogFactory.getLog(NativeLibLoader.class);

    private static boolean initialized = false;
    private static final String nativeLibDirectory = "../lib";
    private static final String nativeLibDirectoryInJar = "/lib";

    static synchronized void initCatBoost() throws IOException {
        if (!initialized) {
            smartLoad("catboost4j-inference");
            initialized = true;
        }
    }

    /**
     * Load libName, first will try try to load libName from default location then will try to load library from JAR.
     *
     * @param libName
     * @throws IOException
     */
    private static void smartLoad(String libName) throws IOException {
        addDirectoryToNativeLibSearchList(nativeLibDirectory);
        try {
            System.loadLibrary(libName);
        } catch (UnsatisfiedLinkError e) {
            logger.debug(e.getMessage());
            try {
                loadNativeLibraryFromJar(libName);
            } catch (IOException ioe) {
                logger.error("failed to load native library from both default location and JAR");
                throw ioe;
            }
        }
    }

    private static void loadNativeLibraryFromJar(String libName) throws IOException {
        final String pathWithinJar = nativeLibDirectoryInJar + "/" + System.mapLibraryName(libName);
        final String tempLibPath = createTemporaryFileFromJar(pathWithinJar);
        System.load(tempLibPath);
    }

    private static void copyFileFromJar(String pathWithinJar, String pathOnDisk) throws IOException {
        byte[] copyBuffer = new byte[4 * 1024];
        int bytesRead;

        try(OutputStream out = new BufferedOutputStream(new FileOutputStream(pathOnDisk));
            InputStream in = NativeLibLoader.class.getResourceAsStream(pathWithinJar)) {

            if (in == null) {
                throw new FileNotFoundException("File " + pathWithinJar + " was not found inside JAR.");
            }

            while ((bytesRead = in.read(copyBuffer)) != -1) {
                out.write(copyBuffer, 0, bytesRead);
            }
        }
    }

    private static String createTemporaryFileFromJar(String pathWithinJar) throws IOException, IllegalArgumentException {
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

    /**
     * Add dirToAdd to java.library.path so that native libraries will be loaded automatically from dirToAdd
     *
     * @param dirToAdd directory with native libraries
     * @throws IOException exception
     */
    private static void addDirectoryToNativeLibSearchList(String dirToAdd) throws IOException {
        try {
            Field userPathsField = ClassLoader.class.getDeclaredField("usr_paths");
            userPathsField.setAccessible(true);

            String[] paths = (String[])userPathsField.get(null);

            for (String path : paths) {
                if (path.equals(dirToAdd)) {
                    return;
                }
            }

            String[] newPaths = new String[paths.length + 1];
            System.arraycopy(paths, 0, newPaths, 0, paths.length);
            newPaths[newPaths.length - 1] = dirToAdd;
            userPathsField.set(null, newPaths);
        } catch (NoSuchFieldException e) {
            logger.error(e.getMessage());
            throw new IOException("Failed to get field handle for `usr_path`");
        } catch (IllegalAccessException e) {
            logger.error(e.getMessage());
            throw new IOException("failed to set field handle for `usr_path`");
        }
    }
}
