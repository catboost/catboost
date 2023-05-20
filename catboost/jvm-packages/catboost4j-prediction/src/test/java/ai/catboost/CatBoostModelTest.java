package ai.catboost;

import junit.framework.TestCase;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import ai.catboost.CatBoostModel.FormulaEvaluatorType;

import javax.validation.constraints.NotNull;
import java.io.*;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.fail;

public class CatBoostModelTest {
    private static boolean testOnGPU = false;

    @BeforeAll
    public static void Init() {
        System.setProperty("java.util.logging.config.file", ClassLoader.getSystemResource("logging.properties").getPath());
        final String testOnGPUProperty = System.getProperty("testOnGPU");
        final String[] trueValues = {"y", "yes", "true", "1"};
        testOnGPU = Arrays.asList(trueValues).contains(testOnGPUProperty);
    }

    private static FormulaEvaluatorType[] getFormulaEvaluatorTypes() {
        if (testOnGPU) {
            return new FormulaEvaluatorType[]{FormulaEvaluatorType.CPU, FormulaEvaluatorType.GPU};
        } else {
            return new FormulaEvaluatorType[]{FormulaEvaluatorType.CPU};
        }
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
                        actual.get(objectIndex, predictionIndex),
                        1.e-5);
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

    static CatBoostModel loadTestModelWithNumCatAndTextFeatures() throws CatBoostError {
        try {
            return CatBoostModel.loadModel(ClassLoader.getSystemResourceAsStream("models/model_with_num_cat_and_text_features.cbm"));
        } catch (IOException ioe) {
        }

        fail("failed to load model with numerical, categorical and text features from resource, can't run tests without it");
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
    public void testSuccessfulLoadModelFromJsonStream() throws CatBoostError, IOException {
        final CatBoostModel model = CatBoostModel.loadModel(ClassLoader.getSystemResourceAsStream("models/numeric_only_model.json"), "json");
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
    public void testGetSupportedEvaluatorTypes() throws CatBoostError {
        final FormulaEvaluatorType[] expectedFormulaEvaluatorTypes = getFormulaEvaluatorTypes();

        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            final FormulaEvaluatorType[] formulaEvaluatorTypes = model.getSupportedEvaluatorTypes();
            Set<FormulaEvaluatorType> formulaEvaluatorTypesSet
                = new HashSet<FormulaEvaluatorType>(Arrays.asList(formulaEvaluatorTypes));

            for (FormulaEvaluatorType formulaEvaluatorType : expectedFormulaEvaluatorTypes) {
                TestCase.assertTrue(formulaEvaluatorTypesSet.contains(formulaEvaluatorType));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testSuccessfulPredictSingleNumericOnly(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            model.setEvaluatorType(evaluatorType);

            final float[] numericFeatuers = new float[]{0.1f, 0.3f, 0.2f};
            final CatBoostPredictions expected = new CatBoostPredictions(1, 1, new double[]{0.029172098906116373});
            final CatBoostPredictions prediction = model.predict(numericFeatuers, (String[]) null);
            assertEqual(expected, prediction);
            assertEqual(expected, model.predict(numericFeatuers, (String[]) null));
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictSingleNumericOnlyWithNullInNumeric(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadNumericOnlyTestModel()) {
            model.setEvaluatorType(evaluatorType);
            try {
                model.predict((float[]) null, (String[]) null);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictSingleNumericOnlywithInsufficientNumericFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadNumericOnlyTestModel()) {
            model.setEvaluatorType(evaluatorType);
            try {
                final float[] features = new float[]{0.f, 0.f};
                model.predict(features, (String[]) null);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictNumericOnlyWithInsufficientPredictionSize(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            model.setEvaluatorType(evaluatorType);
            try {
                final float[] featuers = new float[]{0.1f, 0.3f, 0.2f};
                final CatBoostPredictions prediction = new CatBoostPredictions(1, 0);
                model.predict(featuers, (String[]) null, prediction);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testSuccessfulPredictMultipleNumericOnly(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            model.setEvaluatorType(evaluatorType);
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

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictMultipleNumericOnlyNullInNumeric(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            model.setEvaluatorType(evaluatorType);
            try {
                model.predict((float[][]) null, (String[][]) null);
                fail();
            } catch (CatBoostError e) {
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictMultipleNumericOnlyInsufficientNumberOfFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            model.setEvaluatorType(evaluatorType);
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

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictMultipleInsufficientPredictionSize(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            model.setEvaluatorType(evaluatorType);
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

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testSuccessfulPredictSingleCategoricOnly(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);
                final String[] features = new String[]{"a", "d", "g"};
                final CatBoostPredictions expected = new CatBoostPredictions(1, 1, new double[]{0.04146251510837989});
                final CatBoostPredictions prediction = model.predict((float[])null, features);
                assertEqual(expected, prediction);
                assertEqual(expected, model.predict((float[])null, features));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictSingleCategoricOnlyWithNullInNumeric(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);
                try {
                    model.predict((float[]) null, (String[]) null);
                    fail();
                } catch (CatBoostError e) {
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictSingleCategoricOnlyWithNullCategoricalFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);
                try {
                    final float[] features = null;
                    final String[] catFeatures = new String[]{null, null, null};
                    model.predict(features, catFeatures);
                    fail();
                } catch (CatBoostError e) {
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictSingleCategoricOnlywihtInsufficientCategoricFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);
                try {
                    final String[] features = new String[]{"a", "d"};
                    model.predict((float[])null, features);
                    fail();
                } catch (CatBoostError e) {
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testSuccessfulPredictMultipleCategoricOnly(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);
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
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictMultipleCategoricOnlyNullInCategoric(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);
                try {
                    model.predict((float[][]) null, (String[][]) null);
                    fail();
                } catch (CatBoostError e) {
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictMultipleCategoricOnlyInsufficientNumberOfFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);
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
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testSuccessfulPredictSingle(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[] numericFeatuers = new float[]{0.5f, 1.5f};
                final String[] catFeatures = new String[]{"a", "d", "g"};
                final CatBoostPredictions expected = new CatBoostPredictions(1, 1, new double[]{0.04666924366060905});
                final CatBoostPredictions prediction = model.predict(numericFeatuers, catFeatures);
                assertEqual(expected, prediction);
                assertEqual(expected, model.predict(numericFeatuers, catFeatures));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictSingleWithNullInNumeric(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);
                try {
                    final String[] catFeatures = new String[]{"a", "d", "g"};
                    model.predict((float[]) null, catFeatures);
                    fail();
                } catch (CatBoostError e) {
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictSingleWithNullInCategoric(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);
                try {
                    final float[] numericFeatuers = new float[]{0.5f, 1.5f};
                    model.predict(numericFeatuers, (String[])null);
                    fail();
                } catch (CatBoostError e) {
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictSinglewihtInsufficientNumericFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);
                try {
                    final float[] numericFeatures = new float[]{};
                    final String[] catFeatures = new String[]{"a", "d", "g"};
                    model.predict(numericFeatures, catFeatures);
                    fail();
                } catch (CatBoostError e) {
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictSinglewihtInsufficientCategoricFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);
                try {
                    final float[] numericFeatures = new float[]{0.f, 0.f};
                    final String[] catFeatures = new String[]{"a", "d"};
                    model.predict(numericFeatures, catFeatures);
                    fail();
                } catch (CatBoostError e) {
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictWithInsufficientPredictionSize(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);
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
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testSuccessfulPredictSingleWithNumCatAndTextFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModelWithNumCatAndTextFeatures()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);
                final float[] numericFeatuers = new float[]{0.1f, 0.13f};
                final String[] catFeatures = new String[]{"Male"};
                final String[] textFeatures = new String[]{"question 1", "simple answer"};
                final CatBoostPredictions expected = new CatBoostPredictions(1, 3, new double[]{0.37830508558041, -0.11873512511004, -0.25956996047037});
                final CatBoostPredictions prediction = model.predict(numericFeatuers, catFeatures, textFeatures, null);
                assertEqual(expected, prediction);
                assertEqual(expected, model.predict(numericFeatuers, catFeatures, textFeatures, null));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testSuccessfulPredictMultiple(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

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
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testSuccessfulPredictMultipleWithNumCatAndTextFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModelWithNumCatAndTextFeatures()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[][] numericFeatures = new float[][]{
                    {0.1f, 0.13f},
                    {0.0f, 0.2f},
                    {0.33f, 0.65f},
                    {0.2f, 0.1f}};
                final String[][] catFeatures = new String[][]{
                    {"Male"},
                    {"Female"},
                    {"Female"},
                    {"Male"}};
                final String[][] textFeatures = new String[][]{
                    {"question 1", "simple answer"},
                    {"question 2", "strong answer"},
                    {"question 3", "weak answer"},
                    {"question 1", "complicated answer"}};
                final CatBoostPredictions expected = new CatBoostPredictions(4, 3, new double[]{
                    0.37830508558041, -0.11873512511004, -0.25956996047037,
                    -0.12726299984411, 0.13483590199441, -0.00757290215030,
                    -0.12726299984411, -0.00757290215030, 0.13483590199441,
                    0.41077099521589, -0.20538549760794, -0.20538549760794});
                final CatBoostPredictions prediction = model.predict(numericFeatures, catFeatures, textFeatures, null);
                assertEqual(expected, prediction);
                assertEqual(expected, model.predict(numericFeatures, catFeatures, textFeatures, null));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictMultipleNullInNumeric(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

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
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictMultipleNullInCategoric(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

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
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictMultipleInsufficientNumberOfNumericFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

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
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictMultipleInsufficientNumberOfCategoricFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

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
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictMultipleInsufficientNumberOfNumericRows(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

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
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictMultipleInsufficientNumberOfCategoricRows(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

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
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testSuccessfulPredictSingleHashesOnly(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

                final int[] features = new int[]{-805065478, 2136526169, 785836961};
                final CatBoostPredictions expected = new CatBoostPredictions(1, 1, new double[]{0.04146251510837989});
                final CatBoostPredictions prediction = model.predict((float[])null, features);
                assertEqual(expected, prediction);
                assertEqual(expected, model.predict((float[])null, features));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictSingleHashesOnlyWithNullInNumeric(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

                try {
                    model.predict((float[]) null, (int[]) null);
                    fail();
                } catch (CatBoostError e) {
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictSingleHashesOnlywihtInsufficientCategoricFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

                try {
                    final int[] features = new int[]{-805065478, 2136526169};
                    model.predict((float[])null, features);
                    fail();
                } catch (CatBoostError e) {
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testSuccessfulPredictSingleWithNumCatHashesAndTextFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModelWithNumCatAndTextFeatures()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[] numericFeatuers = new float[]{0.1f, 0.13f};
                final int[] catFeatures = new int[]{-1291328762};
                final String[] textFeatures = new String[]{"question 1", "simple answer"};
                final CatBoostPredictions expected = new CatBoostPredictions(1, 3, new double[]{0.37830508558041, -0.11873512511004, -0.25956996047037});
                final CatBoostPredictions prediction = model.predict(numericFeatuers, catFeatures, textFeatures, null);
                assertEqual(expected, prediction);
                assertEqual(expected, model.predict(numericFeatuers, catFeatures, textFeatures, null));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testSuccessfulPredictMultipleHashesOnly(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

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
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictMultipleHashesOnlyNullInCategoric(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

                try {
                    model.predict((float[][]) null, (int[][]) null);
                    fail();
                } catch (CatBoostError e) {
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictMultipleHashesOnlyInsufficientNumberOfFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

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
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testSuccessfulPredictSingleHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[] numericFeatuers = new float[]{0.5f, 1.5f};
                final int[] catFeatures = new int[]{-805065478, 2136526169, 785836961};
                final CatBoostPredictions expected = new CatBoostPredictions(1, 1, new double[]{0.04666924366060905});
                final CatBoostPredictions prediction = model.predict(numericFeatuers, catFeatures);
                assertEqual(expected, prediction);
                assertEqual(expected, model.predict(numericFeatuers, catFeatures));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictSingleWithNullInNumericHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

                try {
                    final int[] catFeatures = new int[]{-805065478, 2136526169, 785836961};
                    model.predict((float[]) null, catFeatures);
                    fail();
                } catch (CatBoostError e) {
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictSingleWithNullInCategoricHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

                try {
                    final float[] numericFeatuers = new float[]{0.5f, 1.5f};
                    model.predict(numericFeatuers, (int[])null);
                    fail();
                } catch (CatBoostError e) {
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictSinglewihtInsufficientNumericFeaturesHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

                try {
                    final float[] numericFeatures = new float[]{};
                    final int[] catFeatures = new int[]{-805065478, 2136526169, 785836961};
                    model.predict(numericFeatures, catFeatures);
                    fail();
                } catch (CatBoostError e) {
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictSinglewihtInsufficientCategoricFeaturesHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

                try {
                    final float[] numericFeatures = new float[]{0.f, 0.f};
                    final int[] catFeatures = new int[]{-805065478, 2136526169};
                    model.predict(numericFeatures, catFeatures);
                    fail();
                } catch (CatBoostError e) {
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictWithInsufficientPredictionSizeHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

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
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testSuccessfulPredictMultipleHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

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
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testSuccessfulPredictMultipleWithNumCatHashedAndTextFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModelWithNumCatAndTextFeatures()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[][] numericFeatures = new float[][]{
                    {0.1f, 0.13f},
                    {0.0f, 0.2f},
                    {0.33f, 0.65f},
                    {0.2f, 0.1f}};
                final int[][] catFeatures = new int[][]{
                    {-1291328762},
                    {-2114564283},
                    {-2114564283},
                    {-1291328762}};
                final String[][] textFeatures = new String[][]{
                    {"question 1", "simple answer"},
                    {"question 2", "strong answer"},
                    {"question 3", "weak answer"},
                    {"question 1", "complicated answer"}};
                final CatBoostPredictions expected = new CatBoostPredictions(4, 3, new double[]{
                    0.37830508558041, -0.11873512511004, -0.25956996047037,
                    -0.12726299984411, 0.13483590199441, -0.00757290215030,
                    -0.12726299984411, -0.00757290215030, 0.13483590199441,
                    0.41077099521589, -0.20538549760794, -0.20538549760794});
                final CatBoostPredictions prediction = model.predict(numericFeatures, catFeatures, textFeatures, null);
                assertEqual(expected, prediction);
                assertEqual(expected, model.predict(numericFeatures, catFeatures, textFeatures, null));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictMultipleNullInNumericHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

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
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictMultipleNullInCategoricHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

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
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictMultipleInsufficientNumberOfNumericFeaturesHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

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
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictMultipleInsufficientNumberOfCategoricFeaturesHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

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
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictMultipleInsufficientNumberOfNumericRowsHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

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
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testFailPredictMultipleInsufficientNumberOfCategoricRowsHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

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
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testEmptyFeaturesArray(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[][] numericFeatures = new float[0][];
                final int[][] catFeatures = new int[0][];
                final CatBoostPredictions expected = new CatBoostPredictions(0, 1, new double[0]);
                final CatBoostPredictions prediction = model.predict(numericFeatures, catFeatures);
                assertEqual(expected, prediction);
                assertEqual(expected, model.predict(numericFeatures, catFeatures));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    public void testEmptyFeaturesArrayWithNumCatAndTextFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModelWithNumCatAndTextFeatures()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                Assertions.assertThrows(CatBoostError.class, () -> { model.setEvaluatorType(evaluatorType); } );
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[][] numericFeatures = new float[0][];
                final int[][] catFeatures = new int[0][];
                final String[][] textFeatures = new String[0][];
                final CatBoostPredictions expected = new CatBoostPredictions(0, 3, new double[0]);
                final CatBoostPredictions prediction = model.predict(numericFeatures, catFeatures, textFeatures, null);
                assertEqual(expected, prediction);
                assertEqual(expected, model.predict(numericFeatures, catFeatures, textFeatures, null));
            }
        }
    }
}
