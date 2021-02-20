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

    static void assertEqualArrays(@NotNull String[] expected, @NotNull String[] actual) {
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

    static CatBoostModel loadCategoricOnlyTestModel() throws CatBoostError {
        try {
            return CatBoostModel.loadModel(ClassLoader.getSystemResourceAsStream("models/categoric_only_model.cbm"));
        } catch (IOException ioe) {
        }

        fail("failed to load categoric only model from resource, can't run tests without it");
        return null;
    }

    static CatBoostModel loadTestModel() throws CatBoostError {
        try {
            return CatBoostModel.loadModel(ClassLoader.getSystemResourceAsStream("models/model.cbm"));
        } catch (IOException ioe) {
        }

        fail("failed to load categoric only model from resource, can't run tests without it");
        return null;
    }

    static CatBoostModel loadIrisModel() throws CatBoostError {
        try {
            return CatBoostModel.loadModel(ClassLoader.getSystemResourceAsStream("models/iris_model.cbm"));
        } catch (IOException ioe) {
        }

        fail("failed to load categoric only model from resource, can't run tests without it");
        return null;
    }

    @Test
    public void testHashCategoricalFeature() throws CatBoostError {
        final int hash = CatBoostModel.hashCategoricalFeature("foo");
        TestCase.assertEquals(-553946371, hash);
        final int hashUtf8 = CatBoostModel.hashCategoricalFeature("😡");
        TestCase.assertEquals(11426516, hashUtf8);
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
    public void testSuccessfulLoadModelFromFileJsonFormat() throws IOException, CatBoostError {
        final File file = File.createTempFile("numeric_only_model", "json");
        file.deleteOnExit();

        try(OutputStream out = new BufferedOutputStream(new FileOutputStream(file.getAbsoluteFile()))) {
            copyStream(
                    ClassLoader.getSystemResourceAsStream("models/numeric_only_model.json"),
                    out);
        }

        final CatBoostModel model = CatBoostModel.loadModel(file.getAbsolutePath(), "json");
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
            TestCase.assertEquals(3, model.getUsedNumericFeatureCount());
            TestCase.assertEquals(0, model.getUsedCategoricFeatureCount());

            final String[] expected = new String[]{"0", "1", "2"};
            String[] actual = model.getFeatureNames();
            assertEqualArrays(expected, actual);
        }
    }

    public void testCatModelAttributes() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            TestCase.assertEquals(4, model.getFeatures().size());
            TestCase.assertEquals("0", model.getFeatures().get(0).getName());
            TestCase.assertEquals(0, model.getFeatures().get(0).getFlatFeatureIndex());
            TestCase.assertEquals(0, model.getFeatures().get(0).getFeatureIndex());
            TestCase.assertTrue(model.getFeatures().get(0) instanceof CatBoostModel.CatFeature);
            TestCase.assertEquals("3", model.getFeatures().get(3).getName());
            TestCase.assertTrue(model.getFeatures().get(3).isUsedInModel());
            TestCase.assertEquals(false, ((CatBoostModel.FloatFeature) model.getFeatures().get(3)).hasNans());
            TestCase.assertEquals(CatBoostModel.FloatFeature.NanValueTreatment.AsIs, ((CatBoostModel.FloatFeature) model.getFeatures().get(3)).getNanValueTreatment());
        }
    }

    @Test
    public void testModelMetaAttributes() throws CatBoostError {
        try(final CatBoostModel model = loadIrisModel()) {
            TestCase.assertNotNull(model.getMetadata().get("params"));
            // This model has utf-8 in the metadata - make sure it's encoded correctly.
            TestCase.assertTrue(model.getMetadata().get("params").endsWith("}"));
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
    public void testFailPredictSingleNumericOnlywithInsufficientNumericFeatures() throws CatBoostError {
        try (final CatBoostModel model = loadNumericOnlyTestModel()) {
            try {
                final float[] features = new float[]{0.f, 0.f};
                model.predict(features, (String[]) null);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictNumericOnlyWithInsufficientPredictionSize() throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            try {
                final float[] featuers = new float[]{0.1f, 0.3f, 0.2f};
                final CatBoostPredictions prediction = new CatBoostPredictions(1, 0);
                model.predict(featuers, (String[]) null, prediction);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testSuccessfulPredictMultipleNumericOnly() throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            final float[][] features = new float[][]{
                    {0.5f, 1.5f, -2.5f},
                    {0.7f, 6.4f, 2.4f},
                    {-2.0f, -1.0f, +6.0f}};
            final CatBoostPredictions expected = new CatBoostPredictions(3, 1, new double[]{
                    0.03547209874741901,
                    0.008157865240661602,
                    0.009992472030400074});
            final CatBoostPredictions prediction = model.predict(features, (String[][]) null);
            assertEqual(expected, prediction);
            assertEqual(expected, model.predict(features, (String[][]) null));
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
                final float[][] features = new float[][]{
                        {0.f, 0.f, 0.f},
                        {0.f, 0.f}};
                model.predict(features, (String[][]) null);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictMultipleInsufficientPredictionSize() throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            try {
                final float[][] features = new float[][]{
                        {0.f, 0.f, 0.f},
                        {0.f, 0.f, 0.f}};
                final CatBoostPredictions prediction = new CatBoostPredictions(1, 1);
                model.predict(features, (String[][]) null, prediction);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testSuccessfulPredictSingleCategoricOnly() throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            final String[] features = new String[]{"a", "d", "g"};
            final CatBoostPredictions expected = new CatBoostPredictions(1, 1, new double[]{0.04146251510837989});
            final CatBoostPredictions prediction = model.predict((float[])null, features);
            assertEqual(expected, prediction);
            assertEqual(expected, model.predict((float[])null, features));
        }
    }

    @Test
    public void testFailPredictSingleCategoricOnlyWithNullInNumeric() throws CatBoostError {
        try (final CatBoostModel model = loadCategoricOnlyTestModel()) {
            try {
                model.predict((float[]) null, (String[]) null);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictSingleCategoricOnlyWithNullCategoricalFeatures() throws CatBoostError {
        try (final CatBoostModel model = loadCategoricOnlyTestModel()) {
            try {
                final float[] features = null;
                final String[] catFeatures = new String[]{null, null, null};
                model.predict(features, catFeatures);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictSingleCategoricOnlywihtInsufficientCategoricFeatures() throws CatBoostError {
        try (final CatBoostModel model = loadCategoricOnlyTestModel()) {
            try {
                final String[] features = new String[]{"a", "d"};
                model.predict((float[])null, features);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testSuccessfulPredictMultipleCategoricOnly() throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            final String[][] features = new String[][]{
                {"a", "d", "g"},
                {"b", "e", "h"},
                {"c", "f", "k"}};
            final CatBoostPredictions expected = new CatBoostPredictions(3, 1, new double[]{
                0.04146251510837989,
                0.015486266021159064,
                0.04146251510837989});
            final CatBoostPredictions prediction = model.predict((float[][])null, features);
            assertEqual(expected, prediction);
            assertEqual(expected, model.predict((float[][])null, features));
        }
    }

    @Test
    public void testFailPredictMultipleCategoricOnlyNullInCategoric() throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            try {
                model.predict((float[][]) null, (String[][]) null);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictMultipleCategoricOnlyInsufficientNumberOfFeatures() throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            try {
                final String[][] features = new String[][]{
                    {"a", "d", "g"},
                    {"b", "e"}};
                model.predict((float[][])null, features);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testSuccessfulPredictSingle() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            final float[] numericFeatuers = new float[]{0.5f, 1.5f};
            final String[] catFeatures = new String[]{"a", "d", "g"};
            final CatBoostPredictions expected = new CatBoostPredictions(1, 1, new double[]{0.04666924366060905});
            final CatBoostPredictions prediction = model.predict(numericFeatuers, catFeatures);
            assertEqual(expected, prediction);
            assertEqual(expected, model.predict(numericFeatuers, catFeatures));
        }
    }

    @Test
    public void testFailPredictSingleWithNullInNumeric() throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            try {
                final String[] catFeatures = new String[]{"a", "d", "g"};
                model.predict((float[]) null, catFeatures);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictSingleWithNullInCategoric() throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            try {
                final float[] numericFeatuers = new float[]{0.5f, 1.5f};
                model.predict(numericFeatuers, (String[])null);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictSinglewihtInsufficientNumericFeatures() throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            try {
                final float[] numericFeatures = new float[]{};
                final String[] catFeatures = new String[]{"a", "d", "g"};
                model.predict(numericFeatures, catFeatures);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictSinglewihtInsufficientCategoricFeatures() throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            try {
                final float[] numericFeatures = new float[]{0.f, 0.f};
                final String[] catFeatures = new String[]{"a", "d"};
                model.predict(numericFeatures, catFeatures);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictWithInsufficientPredictionSize() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            try {
                final float[] numericFeatuers = new float[]{0.1f, 0.3f};
                final String[] catFeatures = new String[]{"a", "d", "g"};
                final CatBoostPredictions prediction = new CatBoostPredictions(1, 0);
                model.predict(numericFeatuers, catFeatures, prediction);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testSuccessfulPredictMultiple() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            final float[][] numericFeatures = new float[][]{
                {0.5f, 1.5f},
                {0.7f, 6.4f},
                {-2.0f, -1.0f}};
            final String[][] catFeatures = new String[][]{
                {"a", "d", "g"},
                {"b", "e", "h"},
                {"c", "f", "k"}};
            final CatBoostPredictions expected = new CatBoostPredictions(3, 1, new double[]{
                0.04666924366060905,
                0.026244613740247648,
                0.03094452158737013});
            final CatBoostPredictions prediction = model.predict(numericFeatures, catFeatures);
            assertEqual(expected, prediction);
            assertEqual(expected, model.predict(numericFeatures, catFeatures));
        }
    }

    @Test
    public void testFailPredictMultipleNullInNumeric() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            try {
                final String[][] catFeatures = new String[][]{
                    {"a", "d", "g"},
                    {"b", "e", "h"},
                    {"c", "f", "k"}};
                model.predict((float[][]) null, catFeatures);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictMultipleNullInCategoric() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            try {
                final float[][] numericFeatures = new float[][]{
                        {0.5f, 1.5f},
                        {0.7f, 6.4f},
                        {-2.0f, -1.0f}};
                model.predict(numericFeatures, (String[][])null);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictMultipleInsufficientNumberOfNumericFeatures() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            try {
                final float[][] numericFeatures = new float[][]{
                        {0.f, 0.f},
                        {0.f, 0.f},
                        {0.f}};
                final String[][] catFeatures = new String[][]{
                        {"a", "d", "g"},
                        {"b", "e", "h"},
                        {"c", "f", "k"}};
                model.predict(numericFeatures, catFeatures);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictMultipleInsufficientNumberOfCategoricFeatures() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            try {
                final float[][] numericFeatures = new float[][]{
                        {0.f, 0.f},
                        {0.f, 0.f},
                        {0.f, 0.f}};
                final String[][] catFeatures = new String[][]{
                        {"a", "d", "g"},
                        {"b", "e", "h"},
                        {"c", "f"}};
                model.predict(numericFeatures, catFeatures);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictMultipleInsufficientNumberOfNumericRows() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            try {
                final float[][] numericFeatures = new float[][]{
                        {0.f, 0.f},
                        {0.f, 0.f}};
                final String[][] catFeatures = new String[][]{
                        {"a", "d", "g"},
                        {"b", "e", "h"},
                        {"c", "f", "k"}};
                model.predict(numericFeatures, catFeatures);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictMultipleInsufficientNumberOfCategoricRows() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            try {
                final float[][] numericFeatures = new float[][]{
                        {0.f, 0.f},
                        {0.f, 0.f},
                        {0.f, 0.f}};
                final String[][] catFeatures = new String[][]{
                        {"a", "d", "g"},
                        {"b", "e", "h"}};
                model.predict(numericFeatures, catFeatures);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testSuccessfulPredictSingleHashesOnly() throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            final int[] features = new int[]{-805065478, 2136526169, 785836961};
            final CatBoostPredictions expected = new CatBoostPredictions(1, 1, new double[]{0.04146251510837989});
            final CatBoostPredictions prediction = model.predict((float[])null, features);
            assertEqual(expected, prediction);
            assertEqual(expected, model.predict((float[])null, features));
        }
    }

    @Test
    public void testFailPredictSingleHashesOnlyWithNullInNumeric() throws CatBoostError {
        try (final CatBoostModel model = loadCategoricOnlyTestModel()) {
            try {
                model.predict((float[]) null, (int[]) null);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictSingleHashesOnlywihtInsufficientCategoricFeatures() throws CatBoostError {
        try (final CatBoostModel model = loadCategoricOnlyTestModel()) {
            try {
                final int[] features = new int[]{-805065478, 2136526169};
                model.predict((float[])null, features);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testSuccessfulPredictMultipleHashesOnly() throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            final int[][] features = new int[][]{
                    {-805065478, 2136526169, 785836961},
                    {1982436109, 1400211492, 1076941191},
                    {-1883343840, -1452597217, 2122455585}};
            final CatBoostPredictions expected = new CatBoostPredictions(3, 1, new double[]{
                    0.04146251510837989,
                    0.015486266021159064,
                    0.04146251510837989});
            final CatBoostPredictions prediction = model.predict((float[][])null, features);
            assertEqual(expected, prediction);
            assertEqual(expected, model.predict((float[][])null, features));
        }
    }

    @Test
    public void testFailPredictMultipleHashesOnlyNullInCategoric() throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            try {
                model.predict((float[][]) null, (int[][]) null);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictMultipleHashesOnlyInsufficientNumberOfFeatures() throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            try {
                final int[][] features = new int[][]{
                        {-805065478, 2136526169, 785836961},
                        {1982436109, 1400211492}};
                model.predict((float[][])null, features);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testSuccessfulPredictSingleHashes() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            final float[] numericFeatuers = new float[]{0.5f, 1.5f};
            final int[] catFeatures = new int[]{-805065478, 2136526169, 785836961};
            final CatBoostPredictions expected = new CatBoostPredictions(1, 1, new double[]{0.04666924366060905});
            final CatBoostPredictions prediction = model.predict(numericFeatuers, catFeatures);
            assertEqual(expected, prediction);
            assertEqual(expected, model.predict(numericFeatuers, catFeatures));
        }
    }

    @Test
    public void testFailPredictSingleWithNullInNumericHashes() throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            try {
                final int[] catFeatures = new int[]{-805065478, 2136526169, 785836961};
                model.predict((float[]) null, catFeatures);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictSingleWithNullInCategoricHashes() throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            try {
                final float[] numericFeatuers = new float[]{0.5f, 1.5f};
                model.predict(numericFeatuers, (int[])null);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictSinglewihtInsufficientNumericFeaturesHashes() throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            try {
                final float[] numericFeatures = new float[]{};
                final int[] catFeatures = new int[]{-805065478, 2136526169, 785836961};
                model.predict(numericFeatures, catFeatures);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictSinglewihtInsufficientCategoricFeaturesHashes() throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            try {
                final float[] numericFeatures = new float[]{0.f, 0.f};
                final int[] catFeatures = new int[]{-805065478, 2136526169};
                model.predict(numericFeatures, catFeatures);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictWithInsufficientPredictionSizeHashes() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            try {
                final float[] numericFeatuers = new float[]{0.1f, 0.3f};
                final int[] catFeatures = new int[]{-805065478, 2136526169, 785836961};
                final CatBoostPredictions prediction = new CatBoostPredictions(1, 0);
                model.predict(numericFeatuers, catFeatures, prediction);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testSuccessfulPredictMultipleHashes() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            final float[][] numericFeatures = new float[][]{
                    {0.5f, 1.5f},
                    {0.7f, 6.4f},
                    {-2.0f, -1.0f}};
            final int[][] catFeatures = new int[][]{
                    {-805065478, 2136526169, 785836961},
                    {1982436109, 1400211492, 1076941191},
                    {-1883343840, -1452597217, 2122455585}};
            final CatBoostPredictions expected = new CatBoostPredictions(3, 1, new double[]{
                    0.04666924366060905,
                    0.026244613740247648,
                    0.03094452158737013});
            final CatBoostPredictions prediction = model.predict(numericFeatures, catFeatures);
            assertEqual(expected, prediction);
            assertEqual(expected, model.predict(numericFeatures, catFeatures));
        }
    }

    @Test
    public void testFailPredictMultipleNullInNumericHashes() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            try {
                final int[][] catFeatures = new int[][]{
                        {-805065478, 2136526169, 785836961},
                        {1982436109, 1400211492, 1076941191},
                        {-1883343840, -1452597217, 2122455585}};
                model.predict((float[][]) null, catFeatures);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictMultipleNullInCategoricHashes() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            try {
                final float[][] numericFeatures = new float[][]{
                        {0.5f, 1.5f},
                        {0.7f, 6.4f},
                        {-2.0f, -1.0f}};
                model.predict(numericFeatures, (int[][])null);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictMultipleInsufficientNumberOfNumericFeaturesHashes() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            try {
                final float[][] numericFeatures = new float[][]{
                        {0.f, 0.f},
                        {0.f, 0.f},
                        {0.f}};
                final int[][] catFeatures = new int[][]{
                        {-805065478, 2136526169, 785836961},
                        {1982436109, 1400211492, 1076941191},
                        {-1883343840, -1452597217, 2122455585}};
                model.predict(numericFeatures, catFeatures);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictMultipleInsufficientNumberOfCategoricFeaturesHashes() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            try {
                final float[][] numericFeatures = new float[][]{
                        {0.f, 0.f},
                        {0.f, 0.f},
                        {0.f, 0.f}};
                final int[][] catFeatures = new int[][]{
                        {-805065478, 2136526169, 785836961},
                        {1982436109, 1400211492, 1076941191},
                        {-1883343840, -1452597217}};
                model.predict(numericFeatures, catFeatures);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictMultipleInsufficientNumberOfNumericRowsHashes() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            try {
                final float[][] numericFeatures = new float[][]{
                        {0.f, 0.f},
                        {0.f, 0.f}};
                final int[][] catFeatures = new int[][]{
                        {-805065478, 2136526169, 785836961},
                        {1982436109, 1400211492, 1076941191},
                        {-1883343840, -1452597217, 2122455585}};
                model.predict(numericFeatures, catFeatures);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testFailPredictMultipleInsufficientNumberOfCategoricRowsHashes() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            try {
                final float[][] numericFeatures = new float[][]{
                        {0.f, 0.f},
                        {0.f, 0.f},
                        {0.f, 0.f}};
                final int[][] catFeatures = new int[][]{
                        {-805065478, 2136526169, 785836961},
                        {1982436109, 1400211492, 1076941191}};
                model.predict(numericFeatures, catFeatures);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @Test
    public void testEmptyFeaturesArray() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            final float[][] numericFeatures = new float[0][];
            final int[][] catFeatures = new int[0][];
            final CatBoostPredictions expected = new CatBoostPredictions(0, 1, new double[0]);
            final CatBoostPredictions prediction = model.predict(numericFeatures, catFeatures);
            assertEqual(expected, prediction);
            assertEqual(expected, model.predict(numericFeatures, catFeatures));
        }
    }
}
