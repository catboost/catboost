package ai.catboost;

import junit.framework.TestCase;
import org.junit.Before;
import org.junit.Test;

import javax.validation.constraints.NotNull;
import java.io.*;

import static org.junit.Assert.fail;

public class CatBoostModelTest {
    @Before
    public void Init() {
        System.setProperty("java.util.logging.config.file", ClassLoader.getSystemResource("logging.properties").getPath());
    }

    static void assertEqualArrays(@NotNull int[] expected, @NotNull int[] actual) {
        TestCase.assertEquals(expected.length, actual.length);

        for (int i = 0; i < expected.length; ++i) {
            TestCase.assertEquals("at " + String.valueOf(i), expected[i], actual[i]);
        }
    }

    static void assertEqual(@NotNull CatBoostPredictions expected, @NotNull CatBoostPredictions actual) {
        TestCase.assertEquals(expected.getObjectCount(), actual.getObjectCount());
        TestCase.assertEquals(expected.getPredictionDimension(), actual.getPredictionDimension());

        for (int objectIndex = 0; objectIndex < expected.getObjectCount(); ++objectIndex) {
            for (int predictionIndex = 0; predictionIndex < expected.getPredictionDimension(); ++predictionIndex)  {
                TestCase.assertEquals(
                        "at objectIndex=" + String.valueOf(objectIndex) + " predictionIndex=" + String.valueOf(predictionIndex),
                        expected.get(objectIndex, predictionIndex),
                        actual.get(objectIndex, predictionIndex));
            }
        }
    }

    static CatBoostModel loadNumericOnlyTestModel() throws CatBoostError {
        try {
            return CatBoostModel.loadModel(ClassLoader.getSystemResourceAsStream("models/numeric_only_model.cbm"));
        } catch (IOException ioe) {
        }

        fail("failed to load numeric only model from resource, can't run tests without it");
        return null;
    }

    @Test
    public void testHashCategoricalFeature() throws CatBoostError {
        final int hash = CatBoostModel.hashCategoricalFeature("foo");
        TestCase.assertEquals(-553946371, hash);
    }

    @Test
    public void testHashCategoricalFeatures() throws CatBoostError {
        final String[] catFeatures = new String[]{"foo", "bar", "baz"};
        final int[] expectedHashes = new int[]{-553946371, 50123586, 825262476};

        final int[] hashes1 = CatBoostModel.hashCategoricalFeatures(catFeatures);
        assertEqualArrays(expectedHashes, hashes1);

        final int[] hashes2 = new int[3];
        CatBoostModel.hashCategoricalFeatures(catFeatures, hashes2);
        assertEqualArrays(expectedHashes, hashes2);

        // test insufficient `hashes` size
        try {
            final int[] hashes = new int[2];
            CatBoostModel.hashCategoricalFeatures(catFeatures, hashes);
            fail();
        } catch (CatBoostError e) {
        }
    }

    static void copyStream(InputStream in, OutputStream out) throws IOException {
        byte[] copyBuffer = new byte[4 * 1024];
        int bytesRead;

        while ((bytesRead = in.read(copyBuffer)) != -1) {
            out.write(copyBuffer, 0, bytesRead);
        }
    }

    @Test
    public void testSuccessfulLoadModelFromStream() throws CatBoostError, IOException {
        final CatBoostModel model = CatBoostModel.loadModel(ClassLoader.getSystemResourceAsStream("models/numeric_only_model.cbm"));
        model.close();
    }

    @Test
    public void testSuccessfulLoadModelFromFile() throws IOException, CatBoostError {
        final File file = File.createTempFile("numeric_only_model", "cbm");
        file.deleteOnExit();

        try(OutputStream out = new BufferedOutputStream(new FileOutputStream(file.getAbsoluteFile()))) {
            copyStream(
                    ClassLoader.getSystemResourceAsStream("models/numeric_only_model.cbm"),
                    out);
        }

        final CatBoostModel model = CatBoostModel.loadModel(file.getAbsolutePath());
        model.close();
    }

    @Test
    public void testFailLoadModelFromStream() throws IOException {
        try {
            final CatBoostModel model = CatBoostModel.loadModel(ClassLoader.getSystemResourceAsStream("models/not_a_model.cbm"));
            model.close();
            fail();
        } catch (CatBoostError e) {
        }
    }

    @Test
    public void testFailLoadModelFromFile() throws IOException {
        try {
            final File file = File.createTempFile("not_a_model", "cbm");
            file.deleteOnExit();

            try(OutputStream out = new BufferedOutputStream(new FileOutputStream(file.getAbsoluteFile()))) {
                copyStream(
                        ClassLoader.getSystemResourceAsStream("models/not_a_model.cbm"),
                        out);
            }
            final CatBoostModel model = CatBoostModel.loadModel(file.getAbsolutePath());
            model.close();
            fail();
        } catch (CatBoostError e) {
        }
    }

    @Test
    public void testModelAttributes() throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            TestCase.assertEquals(1, model.getPredictionDimension());
            TestCase.assertEquals(5, model.getTreeCount());
            TestCase.assertEquals(3, model.getNumericFeatureCount());
            TestCase.assertEquals(0, model.getCategoricFeatureCount());
        }
    }

    @Test
    public void testSuccessfulPredictSingleNumericOnly() throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            final float[] numericFeatuers = new float[]{0.1f, 0.3f, 0.2f};
            final CatBoostPredictions expected = new CatBoostPredictions(1, 1, new double[]{0.029172098906116373});
            final CatBoostPredictions prediction = model.predict(numericFeatuers, (String[]) null);
            assertEqual(expected, prediction);
            assertEqual(expected, model.predict(numericFeatuers, (String[]) null));
        }
    }

    @Test
    public void testFailPredictSingleNumericOnlyWithNullInNumeric() throws CatBoostError {
        try (final CatBoostModel model = loadNumericOnlyTestModel()) {
            try {
                model.predict((float[]) null, (String[]) null);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictSingleNumericOnlywihtInsufficientNumericFeatures() throws CatBoostError {
        try (final CatBoostModel model = loadNumericOnlyTestModel()) {
            try {
                final float[] numericFeatures = new float[]{0.f, 0.f};
                model.predict(numericFeatures, (String[]) null);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictWithInsufficientPredictionSize() throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            try {
                final float[] numericFeatuers = new float[]{0.1f, 0.3f, 0.2f};
                final CatBoostPredictions prediction = new CatBoostPredictions(1, 0);
                model.predict(numericFeatuers, (String[]) null, prediction);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testSuccessfulPredictMultipleNumericOnly() throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            final float[][] numericFeatures = new float[][]{
                    {0.5f, 1.5f, -2.5f},
                    {0.7f, 6.4f, 2.4f},
                    {-2.0f, -1.0f, +6.0f}};
            final CatBoostPredictions expected = new CatBoostPredictions(3, 1, new double[]{
                    0.03547209874741901,
                    0.008157865240661602,
                    0.009992472030400074});
            final CatBoostPredictions prediction = model.predict(numericFeatures, (String[][]) null);
            assertEqual(expected, prediction);
            assertEqual(expected, model.predict(numericFeatures, (String[][]) null));
        }
    }

    @Test
    public void testFailPredictMultipleNumericOnlyNullInNumeric() throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            try {
                model.predict((float[][]) null, (String[][]) null);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictMultipleNumericOnlyInsufficientNumberOfFeatures() throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            try {
                final float[][] numericFeatures = new float[][]{
                        {0.f, 0.f, 0.f},
                        {0.f, 0.f}};
                model.predict(numericFeatures, (String[][]) null);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictMultipleInsufficientPredictionSize() throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            try {
                final float[][] numericFeatures = new float[][]{
                        {0.f, 0.f, 0.f},
                        {0.f, 0.f, 0.f}};
                final CatBoostPredictions prediction = new CatBoostPredictions(1, 1);
                model.predict(numericFeatures, (String[][]) null, prediction);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    // TODO(yazevnul): add test for models that have only categorical features
    // TODO(yazevnul): add test for models that have both numeric and categorical features
}
