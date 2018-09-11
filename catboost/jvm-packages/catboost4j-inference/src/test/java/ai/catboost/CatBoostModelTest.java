package ai.catboost;

import junit.framework.TestCase;
import org.junit.Test;

import javax.validation.constraints.NotNull;

import static org.junit.Assert.fail;

public class CatBoostModelTest {
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

    static CatBoostModel loadNumericOnlyTestModel() throws CatBoostException {
        return CatBoostModel.loadModel("test_data/numeric_only_model.cbm");
    }

    @Test
    public void testHashCategoricalFeature() throws CatBoostException {
        final int hash = CatBoostModel.hashCategoricalFeature("foo");
        TestCase.assertEquals(-553946371, hash);
    }

    @Test
    public void testHashCategoricalFeatures() throws CatBoostException {
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
        } catch (CatBoostException e) {
        }
    }

    @Test
    public void testLoadModel() throws CatBoostException {
        {
            final CatBoostModel model = loadNumericOnlyTestModel();
            model.close();
        }

        try {
            final CatBoostModel model = CatBoostModel.loadModel("test_data/not_a_model.cbm");
            model.close();
            fail();
        } catch (CatBoostException e) {
        }
    }

    @Test
    public void testModelAttributes() throws CatBoostException {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            TestCase.assertEquals(1, model.getPredictionDimension());
            TestCase.assertEquals(5, model.getTreeCount());
            TestCase.assertEquals(3, model.getNumericFeatureCount());
            TestCase.assertEquals(0, model.getCategoricFeatureCount());
        }
    }

    @Test
    public void testPredictSingle() throws CatBoostException {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            final float[] numericFeatuers = new float[]{0.1f, 0.3f, 0.2f};

            // test valid case
            {
                final CatBoostPredictions expected = new CatBoostPredictions(1, 1, new double[]{0.029172098906116373});
                final CatBoostPredictions prediction = model.predict(numericFeatuers, (String[]) null);
                assertEqual(expected, prediction);
                assertEqual(expected, model.predict(numericFeatuers, (String[]) null));
            }

            // test absence of numeric features
            try {
                model.predict((float[])null, (String[]) null);
                fail();
            } catch (CatBoostException e) {
            }

            // test insufficient number of numeric features
            try {
                final float[] numericFeatures = new float[]{0.f, 0.f};
                model.predict(numericFeatures, (String[]) null);
                fail();
            } catch (CatBoostException e) {
            }

            // test insufficient `prediction` size
            try {
                final CatBoostPredictions prediction = new CatBoostPredictions(1, 0);
                model.predict(numericFeatuers, (String[]) null, prediction);
                fail();
            } catch (CatBoostException e) {
            }
        }
    }

    @Test
    public void testPredictMultiple() throws CatBoostException {
        try(final CatBoostModel model = loadNumericOnlyTestModel()) {
            // test valid case
            {
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

            // test absence of numeric features
            try {
                model.predict((float[][])null, (String[][]) null);
                fail();
            } catch (CatBoostException e) {
            }

            // test insufficient number of numeric features
            try {
                final float[][] numericFeatures = new float[][]{
                        {0.f, 0.f, 0.f},
                        {0.f, 0.f}};
                model.predict(numericFeatures, (String[][]) null);
                fail();
            } catch (CatBoostException e) {
            }

            // insufficient size of prediction (by number of objects)
            try {
                final float[][] numericFeatures = new float[][]{
                        {0.f, 0.f, 0.f},
                        {0.f, 0.f, 0.f}};
                final CatBoostPredictions prediction = new CatBoostPredictions(1, 1);
                model.predict(numericFeatures, (String[][]) null, prediction);
                fail();
            } catch (CatBoostException e) {
            }
        }
    }

    // TODO(yazevnul): add test for models that have only categorical features
    // TODO(yazevnul): add test for models that have both numeric and categorical features
}
