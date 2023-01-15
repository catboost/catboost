import java.util.Arrays;

import org.apache.commons.math3.util.Precision;

import junit.framework.TestCase;
import org.junit.Test;

import ai.catboost.common.IteratorUtils;

public class IteratorUtilsTest {
    @Test
    public void testEmpty() {
        TestCase.assertTrue(
            IteratorUtils.elementsEqual(
                Arrays.stream(new int[0]).iterator(),
                Arrays.stream(new int[0]).iterator(),
                (l, r) -> l == r
            )
        );
    }

    @Test
    public void testEmptyAndNonEmpty() {
        TestCase.assertFalse(
            IteratorUtils.elementsEqual(
                Arrays.stream(new int[] {4, 5, 6, 1}).iterator(),
                Arrays.stream(new int[0]).iterator(),
                (l, r) -> l == r
            )
        );
    }

    @Test
    public void testSameSizeAndEqual() {
        TestCase.assertTrue(
            IteratorUtils.elementsEqual(
                Arrays.stream(new String[] {"a", "b", "c"}).iterator(),
                Arrays.stream(new String[] {"a", "b", "c"}).iterator(),
                (l, r) -> l.equals(r)
            )
        );
    }

    @Test
    public void testSameSizeAndNonEqual() {
        TestCase.assertFalse(
            IteratorUtils.elementsEqual(
                Arrays.stream(new int[] {4, 5, 6, 1}).iterator(),
                Arrays.stream(new int[] {4, 5, 3, 1}).iterator(),
                (l, r) -> l == r
            )
        );
    }

    @Test
    public void testDifferentSizeSamePrefix() {
        TestCase.assertFalse(
            IteratorUtils.elementsEqual(
                Arrays.stream(new int[] {1, 2, 3, 4, 5}).iterator(),
                Arrays.stream(new int[] {1, 2, 3}).iterator(),
                (l, r) -> l == r
            )
        );
    }

    @Test
    public void testDifferentSizeDifferentPrefix() {
        TestCase.assertFalse(
            IteratorUtils.elementsEqual(
                Arrays.stream(new String[] {"a", "z", "c", "d"}).iterator(),
                Arrays.stream(new String[] {"a", "b", "c"}).iterator(),
                (l, r) -> l.equals(r)
            )
        );
    }

    @Test
    public void testEqualWithPrecision() {
        TestCase.assertTrue(
            IteratorUtils.elementsEqual(
                Arrays.stream(new double[] {1.e-4, 1.0}).iterator(),
                Arrays.stream(new double[] {1.e-5, 1.0}).iterator(),
                (l, r) -> Precision.equals(l, r, 1e-3)
            )
        );
    }

    @Test
    public void testNonEqualWithPrecision() {
        TestCase.assertFalse(
            IteratorUtils.elementsEqual(
                Arrays.stream(new double[] {1.e-4, 1.0}).iterator(),
                Arrays.stream(new double[] {1.e-5, 1.0}).iterator(),
                (l, r) -> Precision.equals(l, r, 1e-6)
            )
        );
    }
}
