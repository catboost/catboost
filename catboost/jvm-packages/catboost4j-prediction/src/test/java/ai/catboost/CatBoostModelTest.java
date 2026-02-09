package ai.catboost;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import ai.catboost.CatBoostModel.FormulaEvaluatorType;

import java.io.*;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

class CatBoostModelTest {
    private static boolean testOnGPU = false;

    @BeforeAll
    static void Init() {
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

    static void assertEqualArrays(int[] expected, int[] actual) {
        assertEquals(expected.length, actual.length);

        for (int i = 0; i < expected.length; ++i) {
            assertEquals(expected[i], actual[i], "at " + i);
        }
    }

    static void assertEqualArrays(String[] expected, String[] actual) {
        assertEquals(expected.length, actual.length);

        for (int i = 0; i < expected.length; ++i) {
            assertEquals(expected[i], actual[i], "at " + i);
        }
    }

    static void assertEqual(CatBoostPredictions expected, CatBoostPredictions actual) {
        assertEquals(expected.getObjectCount(), actual.getObjectCount());
        assertEquals(expected.getPredictionDimension(), actual.getPredictionDimension());

        for (int objectIndex = 0; objectIndex < expected.getObjectCount(); ++objectIndex) {
            for (int predictionIndex = 0; predictionIndex < expected.getPredictionDimension(); ++predictionIndex)  {
                assertEquals(expected.get(objectIndex, predictionIndex), actual.get(objectIndex, predictionIndex), 1.e-5,
                    "at objectIndex=" + objectIndex + " predictionIndex=" + predictionIndex);
            }
        }
    }

    static CatBoostModel loadNumericOnlyTestModel() throws CatBoostError {
        try {
            return CatBoostModel.loadModel(loadResource("models/numeric_only_model.cbm"));
        } catch (IOException ioe) {
            return Assertions.fail("failed to load numeric only model from resource, can't run tests without it");
        }
    }

    static CatBoostModel loadCategoricOnlyTestModel() throws CatBoostError {
        try {
            return CatBoostModel.loadModel(loadResource("models/categoric_only_model.cbm"));
        } catch (IOException ioe) {
            return Assertions.fail("failed to load categoric only model from resource, can't run tests without it");
        }
    }

    static CatBoostModel loadTestModel() throws CatBoostError {
        try {
            return CatBoostModel.loadModel(loadResource("models/model.cbm"));
        } catch (IOException ioe) {
            return Assertions.fail("failed to load categoric only model from resource, can't run tests without it");
        }
    }

    static CatBoostModel loadIrisModel() throws CatBoostError {
        try {
            return CatBoostModel.loadModel(loadResource("models/iris_model.cbm"));
        } catch (IOException ioe) {
            return Assertions.fail("failed to load categoric only model from resource, can't run tests without it");
        }
    }

    static CatBoostModel loadTestModelWithNumCatAndTextFeatures() throws CatBoostError {
        try {
            return CatBoostModel.loadModel(loadResource("models/model_with_num_cat_and_text_features.cbm"));
        } catch (IOException ioe) {
            return Assertions.fail("failed to load model with numerical, categorical and text features from resource, can't run tests without it");
        }
    }

    @Test
    void testHashCategoricalFeature() throws CatBoostError {
        final int hash = CatBoostModel.hashCategoricalFeature("foo");
        assertEquals(-553946371, hash);
        final int hashUtf8 = CatBoostModel.hashCategoricalFeature("😡");
        assertEquals(11426516, hashUtf8);
    }

    @Test
    void testHashCategoricalFeatures() throws CatBoostError {
        final String[] catFeatures = new String[]{"foo", "bar", "baz"};
        final int[] expectedHashes = new int[]{-553946371, 50123586, 825262476};

        final int[] hashes1 = CatBoostModel.hashCategoricalFeatures(catFeatures);
        assertEqualArrays(expectedHashes, hashes1);

        final int[] hashes2 = new int[3];
        CatBoostModel.hashCategoricalFeatures(catFeatures, hashes2);
        assertEqualArrays(expectedHashes, hashes2);

        // test insufficient `hashes` size
        assertThrows(CatBoostError.class, () -> {
            final int[] hashes = new int[2];
            CatBoostModel.hashCategoricalFeatures(catFeatures, hashes);
        });
    }

    @Test
    void testSuccessfulLoadModelFromStream() throws CatBoostError, IOException {
        try (InputStream input = loadResource("models/numeric_only_model.cbm")) {
            final CatBoostModel model = CatBoostModel.loadModel(input);
            model.close();
        }
    }

    @Test
    void testSuccessfulLoadModelFromFile(@TempDir final Path directory) throws IOException, CatBoostError {
        final Path file = directory.resolve("numeric_only_model.cbm");

        try(InputStream input = loadResource("models/numeric_only_model.cbm")) {
            Files.copy(input, file);
        }

        final CatBoostModel model = CatBoostModel.loadModel(file.toAbsolutePath().toString());
        model.close();
    }

    @Test
    void testSuccessfulLoadModelFromJsonStream() throws CatBoostError, IOException {
        final CatBoostModel model = CatBoostModel.loadModel(loadResource("models/numeric_only_model.json"), "json");
        model.close();
    }

    @Test
    void testSuccessfulLoadModelFromFileJsonFormat(@TempDir final Path directory) throws IOException, CatBoostError {
        final Path file = directory.resolve("numeric_only_model.json");
        try(InputStream input = loadResource("models/numeric_only_model.json")) {
            Files.copy(input, file);
        }

        final CatBoostModel model = CatBoostModel.loadModel(file.toAbsolutePath().toString(), "json");
        model.close();
    }

    @Test
    void testFailLoadModelFromStream() {
        assertThrows(CatBoostError.class, () -> {
            final CatBoostModel model = CatBoostModel.loadModel(loadResource("models/not_a_model.cbm"));
            model.close();
        });
    }

    @Test
    void testFailLoadModelFromFile(@TempDir final Path directory) {
        final Path file = directory.resolve("not_a_model.cbm");
        assertThrows(CatBoostError.class, () -> {
            try (InputStream input = loadResource("models/not_a_model.cbm")) {
                Files.copy(input, file);
            }
            final CatBoostModel model = CatBoostModel.loadModel(file.toAbsolutePath().toString());
            model.close();
        });
    }

    @Test
    void testModelAttributes() throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            assertEquals(1, model.getPredictionDimension());
            assertEquals(5, model.getTreeCount());
            assertEquals(3, model.getUsedNumericFeatureCount());
            assertEquals(0, model.getUsedCategoricFeatureCount());

            final String[] expected = new String[]{"0", "1", "2"};
            String[] actual = model.getFeatureNames();
            assertEqualArrays(expected, actual);
        }
    }

    @Test
    void testCatModelAttributes() throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            assertEquals(4, model.getFeatures().size());
            assertEquals("0", model.getFeatures().get(0).getName());
            assertEquals(0, model.getFeatures().get(0).getFlatFeatureIndex());
            assertEquals(0, model.getFeatures().get(0).getFeatureIndex());
            assertInstanceOf(CatBoostModel.CatFeature.class, model.getFeatures().get(0));
            assertEquals("3", model.getFeatures().get(3).getName());
            assertTrue(model.getFeatures().get(3).isUsedInModel());
            assertFalse(((CatBoostModel.FloatFeature) model.getFeatures().get(3)).hasNans());
            assertEquals(CatBoostModel.FloatFeature.NanValueTreatment.AsIs,
                ((CatBoostModel.FloatFeature) model.getFeatures().get(3)).getNanValueTreatment());
        }
    }

    @Test
    void testModelMetaAttributes() throws CatBoostError {
        try(final CatBoostModel model = loadIrisModel()) {
            assertNotNull(model.getMetadata().get("params"));
            // This model has utf-8 in the metadata - make sure it's encoded correctly.
            assertTrue(model.getMetadata().get("params").endsWith("}"));
        }
    }

    @Test
    void testGetSupportedEvaluatorTypes() throws CatBoostError {
        final FormulaEvaluatorType[] expectedFormulaEvaluatorTypes = getFormulaEvaluatorTypes();

        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            final FormulaEvaluatorType[] formulaEvaluatorTypes = model.getSupportedEvaluatorTypes();
            Set<FormulaEvaluatorType> formulaEvaluatorTypesSet
                = new HashSet<FormulaEvaluatorType>(Arrays.asList(formulaEvaluatorTypes));

            for (FormulaEvaluatorType formulaEvaluatorType : expectedFormulaEvaluatorTypes) {
                assertTrue(formulaEvaluatorTypesSet.contains(formulaEvaluatorType));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testSuccessfulPredictSingleNumericOnly(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
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
    void testFailPredictSingleNumericOnlyWithNullInNumeric(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadNumericOnlyTestModel()) {
            model.setEvaluatorType(evaluatorType);
            assertThrows(CatBoostError.class, () -> model.predict(null, (String[]) null));
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictSingleNumericOnlyWithInsufficientNumericFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadNumericOnlyTestModel()) {
            model.setEvaluatorType(evaluatorType);
            final float[] features = new float[]{0.f, 0.f};
            assertThrows(CatBoostError.class, () -> model.predict(features, (String[]) null));
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictNumericOnlyWithInsufficientPredictionSize(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            model.setEvaluatorType(evaluatorType);
            final float[] featuers = new float[]{0.1f, 0.3f, 0.2f};
            assertThrows(CatBoostError.class, () -> {
                final CatBoostPredictions prediction = new CatBoostPredictions(1, 0);
                model.predict(featuers, (String[]) null, prediction);
            });
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testSuccessfulPredictMultipleNumericOnly(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
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
    void testSuccessfulPredictMultipleNumericOnlyTransposed(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            model.setEvaluatorType(evaluatorType);
            final float[][] features = new float[][]{
                    {0.5f, 0.7f, -2.0f},
                    {1.5f, 6.4f, -1.0f},
                    {-2.5f, 2.4f, +6.0f}};
            final CatBoostPredictions expected = new CatBoostPredictions(3, 1, new double[]{
                    0.03547209874741901,
                    0.008157865240661602,
                    0.009992472030400074});
            final CatBoostPredictions prediction = model.predictTransposed(features);
            assertEqual(expected, prediction);
            assertEqual(expected, model.predictTransposed(features));
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictMultipleNumericOnlyTransposedIncorrectNumberOfObjects(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            model.setEvaluatorType(evaluatorType);
            final float[][] features = new float[][]{
                    {0.f, 0.f, 0.f},
                    {0.f, 1.f, 2.f},
                    {0.f, 3.f}};
            assertThrows(CatBoostError.class, () -> model.predictTransposed(features));
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictMultipleNumericOnlyTransposedInsifficientNumberOfFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            model.setEvaluatorType(evaluatorType);
            final float[][] features = new float[][]{
                    {0.f, 0.f, 0.f},
                    {0.f, 1.f, 2.f}};
            assertThrows(CatBoostError.class, () -> model.predictTransposed(features));
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictMultipleNumericOnlyNullInNumeric(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            model.setEvaluatorType(evaluatorType);
            assertThrows(CatBoostError.class, () -> model.predict(null, (String[][]) null));
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictMultipleNumericOnlyInsufficientNumberOfFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            model.setEvaluatorType(evaluatorType);
            final float[][] features = new float[][]{
                    {0.f, 0.f, 0.f},
                    {0.f, 0.f}};
            assertThrows(CatBoostError.class, () -> model.predict(features, (String[][]) null));
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictMultipleInsufficientPredictionSize(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            model.setEvaluatorType(evaluatorType);
            final float[][] features = new float[][]{
                    {0.f, 0.f, 0.f},
                    {0.f, 0.f, 0.f}};
            assertThrows(CatBoostError.class, () -> {
                final CatBoostPredictions prediction = new CatBoostPredictions(1, 1);
                model.predict(features, (String[][]) null, prediction);
            });
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testSuccessfulPredictSingleCategoricOnly(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);
                final String[] features = new String[]{"a", "d", "g"};
                final CatBoostPredictions expected = new CatBoostPredictions(1, 1, new double[]{0.04146251510837989});
                final CatBoostPredictions prediction = model.predict(null, features);
                assertEqual(expected, prediction);
                assertEqual(expected, model.predict((float[])null, features));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictSingleCategoricOnlyWithNullInNumeric(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);
                assertThrows(CatBoostError.class, () -> model.predict(null, (String[]) null));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictSingleCategoricOnlyWithNullCategoricalFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);
                assertThrows(CatBoostError.class, () -> {
                    final String[] catFeatures = new String[]{null, null, null};
                    model.predict(null, catFeatures);
                });
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictSingleCategoricOnlywihtInsufficientCategoricFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final String[] features = new String[]{"a", "d"};
                assertThrows(CatBoostError.class, () -> model.predict(null, features));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testSuccessfulPredictMultipleCategoricOnly(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
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
                final CatBoostPredictions prediction = model.predict(null, features);
                assertEqual(expected, prediction);
                assertEqual(expected, model.predict((float[][])null, features));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictMultipleCategoricOnlyNullInCategoric(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);
                assertThrows(CatBoostError.class, () -> model.predict(null, (String[][]) null));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictMultipleCategoricOnlyInsufficientNumberOfFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);
                final String[][] features = new String[][]{
                    {"a", "d", "g"},
                    {"b", "e"}};
                assertThrows(CatBoostError.class, () -> model.predict(null, features));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testSuccessfulPredictSingle(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
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
    void testFailPredictSingleWithNullInNumeric(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                final String[] catFeatures = new String[]{"a", "d", "g"};
                model.setEvaluatorType(evaluatorType);
                assertThrows(CatBoostError.class, () -> model.predict(null, catFeatures));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictSingleWithNullInCategoric(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[] numericFeatuers = new float[]{0.5f, 1.5f};
                assertThrows(CatBoostError.class, () -> model.predict(numericFeatuers, (String[])null));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictSingleWithInsufficientNumericFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[] numericFeatures = new float[]{};
                final String[] catFeatures = new String[]{"a", "d", "g"};
                assertThrows(CatBoostError.class, () -> model.predict(numericFeatures, catFeatures));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictSingleWithInsufficientCategoricFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[] numericFeatures = new float[]{0.f, 0.f};
                final String[] catFeatures = new String[]{"a", "d"};
                assertThrows(CatBoostError.class, () -> model.predict(numericFeatures, catFeatures));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictWithInsufficientPredictionSize(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[] numericFeatuers = new float[]{0.1f, 0.3f};
                final String[] catFeatures = new String[]{"a", "d", "g"};
                assertThrows(CatBoostError.class, () -> {
                    final CatBoostPredictions prediction = new CatBoostPredictions(1, 0);
                    model.predict(numericFeatuers, catFeatures, prediction);
                });
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testSuccessfulPredictSingleWithNumCatAndTextFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModelWithNumCatAndTextFeatures()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
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
    void testSuccessfulPredictMultiple(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
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
    void testSuccessfulPredictMultipleWithNumCatAndTextFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModelWithNumCatAndTextFeatures()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
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
    void testFailPredictMultipleNullInNumeric(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final String[][] catFeatures = new String[][]{
                    {"a", "d", "g"},
                    {"b", "e", "h"},
                    {"c", "f", "k"}};
                assertThrows(CatBoostError.class, () -> model.predict(null, catFeatures));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictMultipleNullInCategoric(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[][] numericFeatures = new float[][]{
                    {0.5f, 1.5f},
                    {0.7f, 6.4f},
                    {-2.0f, -1.0f}};
                assertThrows(CatBoostError.class, () -> model.predict(numericFeatures, (String[][])null));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictMultipleInsufficientNumberOfNumericFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[][] numericFeatures = new float[][]{
                    {0.f, 0.f},
                    {0.f, 0.f},
                    {0.f}};
                final String[][] catFeatures = new String[][]{
                    {"a", "d", "g"},
                    {"b", "e", "h"},
                    {"c", "f", "k"}};
                assertThrows(CatBoostError.class, () -> model.predict(numericFeatures, catFeatures));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictMultipleInsufficientNumberOfCategoricFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[][] numericFeatures = new float[][]{
                    {0.f, 0.f},
                    {0.f, 0.f},
                    {0.f, 0.f}};
                final String[][] catFeatures = new String[][]{
                    {"a", "d", "g"},
                    {"b", "e", "h"},
                    {"c", "f"}};
                assertThrows(CatBoostError.class, () -> model.predict(numericFeatures, catFeatures));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictMultipleInsufficientNumberOfNumericRows(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[][] numericFeatures = new float[][]{
                    {0.f, 0.f},
                    {0.f, 0.f}};
                final String[][] catFeatures = new String[][]{
                    {"a", "d", "g"},
                    {"b", "e", "h"},
                    {"c", "f", "k"}};
                assertThrows(CatBoostError.class, () -> model.predict(numericFeatures, catFeatures));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictMultipleInsufficientNumberOfCategoricRows(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[][] numericFeatures = new float[][]{
                    {0.f, 0.f},
                    {0.f, 0.f},
                    {0.f, 0.f}};
                final String[][] catFeatures = new String[][]{
                    {"a", "d", "g"},
                    {"b", "e", "h"}};
                assertThrows(CatBoostError.class, () -> model.predict(numericFeatures, catFeatures));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testSuccessfulPredictSingleHashesOnly(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final int[] features = new int[]{-805065478, 2136526169, 785836961};
                final CatBoostPredictions expected = new CatBoostPredictions(1, 1, new double[]{0.04146251510837989});
                final CatBoostPredictions prediction = model.predict(null, features);
                assertEqual(expected, prediction);
                assertEqual(expected, model.predict((float[])null, features));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictSingleHashesOnlyWithNullInNumeric(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                assertThrows(CatBoostError.class, () -> model.predict(null, (int[]) null));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictSingleHashesOnlyWithInsufficientCategoricFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final int[] features = new int[]{-805065478, 2136526169};
                assertThrows(CatBoostError.class, () -> model.predict(null, features));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testSuccessfulPredictSingleWithNumCatHashesAndTextFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModelWithNumCatAndTextFeatures()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
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
    void testSuccessfulPredictMultipleHashesOnly(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
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
                final CatBoostPredictions prediction = model.predict(null, features);
                assertEqual(expected, prediction);
                assertEqual(expected, model.predict((float[][])null, features));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictMultipleHashesOnlyNullInCategoric(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                assertThrows(CatBoostError.class, () -> model.predict(null, (int[][]) null));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictMultipleHashesOnlyInsufficientNumberOfFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadCategoricOnlyTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final int[][] features = new int[][]{
                    {-805065478, 2136526169, 785836961},
                    {1982436109, 1400211492}};
                assertThrows(CatBoostError.class, () -> model.predict(null, features));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testSuccessfulPredictSingleHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
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
    void testFailPredictSingleWithNullInNumericHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final int[] catFeatures = new int[]{-805065478, 2136526169, 785836961};
                assertThrows(CatBoostError.class, () -> model.predict(null, catFeatures));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictSingleWithNullInCategoricHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[] numericFeatures = new float[]{0.5f, 1.5f};
                assertThrows(CatBoostError.class, () -> model.predict(numericFeatures, (int[])null));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictSingleWithInsufficientNumericFeaturesHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[] numericFeatures = new float[]{};
                final int[] catFeatures = new int[]{-805065478, 2136526169, 785836961};
                assertThrows(CatBoostError.class, () -> model.predict(numericFeatures, catFeatures));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictSingleWithInsufficientCategoricFeaturesHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try (final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[] numericFeatures = new float[]{0.f, 0.f};
                final int[] catFeatures = new int[]{-805065478, 2136526169};
                assertThrows(CatBoostError.class, () -> model.predict(numericFeatures, catFeatures));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictWithInsufficientPredictionSizeHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[] numericFeatuers = new float[]{0.1f, 0.3f};
                final int[] catFeatures = new int[]{-805065478, 2136526169, 785836961};
                assertThrows(CatBoostError.class, () -> {
                    final CatBoostPredictions prediction = new CatBoostPredictions(1, 0);
                    model.predict(numericFeatuers, catFeatures, prediction);
                });
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testSuccessfulPredictMultipleHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
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
    void testSuccessfulPredictMultipleWithNumCatHashedAndTextFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModelWithNumCatAndTextFeatures()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
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
    void testFailPredictMultipleNullInNumericHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final int[][] catFeatures = new int[][]{
                    {-805065478, 2136526169, 785836961},
                    {1982436109, 1400211492, 1076941191},
                    {-1883343840, -1452597217, 2122455585}};
                assertThrows(CatBoostError.class, () -> model.predict(null, catFeatures));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictMultipleNullInCategoricHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[][] numericFeatures = new float[][]{
                    {0.5f, 1.5f},
                    {0.7f, 6.4f},
                    {-2.0f, -1.0f}};
                assertThrows(CatBoostError.class, () -> model.predict(numericFeatures, (int[][])null));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictMultipleInsufficientNumberOfNumericFeaturesHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[][] numericFeatures = new float[][]{
                    {0.f, 0.f},
                    {0.f, 0.f},
                    {0.f}};
                final int[][] catFeatures = new int[][]{
                    {-805065478, 2136526169, 785836961},
                    {1982436109, 1400211492, 1076941191},
                    {-1883343840, -1452597217, 2122455585}};
                assertThrows(CatBoostError.class, () -> model.predict(numericFeatures, catFeatures));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictMultipleInsufficientNumberOfCategoricFeaturesHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[][] numericFeatures = new float[][]{
                    {0.f, 0.f},
                    {0.f, 0.f},
                    {0.f, 0.f}};
                final int[][] catFeatures = new int[][]{
                    {-805065478, 2136526169, 785836961},
                    {1982436109, 1400211492, 1076941191},
                    {-1883343840, -1452597217}};
                assertThrows(CatBoostError.class, () -> model.predict(numericFeatures, catFeatures));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictMultipleInsufficientNumberOfNumericRowsHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[][] numericFeatures = new float[][]{
                    {0.f, 0.f},
                    {0.f, 0.f}};
                final int[][] catFeatures = new int[][]{
                    {-805065478, 2136526169, 785836961},
                    {1982436109, 1400211492, 1076941191},
                    {-1883343840, -1452597217, 2122455585}};
                assertThrows(CatBoostError.class, () -> model.predict(numericFeatures, catFeatures));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testFailPredictMultipleInsufficientNumberOfCategoricRowsHashes(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
            } else {
                model.setEvaluatorType(evaluatorType);

                final float[][] numericFeatures = new float[][]{
                    {0.f, 0.f},
                    {0.f, 0.f},
                    {0.f, 0.f}};
                final int[][] catFeatures = new int[][]{
                    {-805065478, 2136526169, 785836961},
                    {1982436109, 1400211492, 1076941191}};
                assertThrows(CatBoostError.class, () -> model.predict(numericFeatures, catFeatures));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("getFormulaEvaluatorTypes")
    void testEmptyFeaturesArray(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModel()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
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
    void testEmptyFeaturesArrayWithNumCatAndTextFeatures(CatBoostModel.FormulaEvaluatorType evaluatorType) throws CatBoostError {
        try(final CatBoostModel model = loadTestModelWithNumCatAndTextFeatures()) {
            if (evaluatorType == FormulaEvaluatorType.GPU) {
                assertThrows(CatBoostError.class, () -> model.setEvaluatorType(evaluatorType));
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

    private static InputStream loadResource(final String path) {
        InputStream resource = ClassLoader.getSystemResourceAsStream(path);
        if (resource == null) {
            throw new IllegalStateException("There is no model at path: " + path);
        }
        return resource;
    }
}
