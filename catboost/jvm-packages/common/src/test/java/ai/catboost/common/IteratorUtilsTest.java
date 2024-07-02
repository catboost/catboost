package ai.catboost.common;

import java.util.Arrays;
import java.util.Objects;

import org.apache.commons.math3.util.Precision;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class IteratorUtilsTest {
    @Test
    public void testEmpty() {
        assertTrue(
            IteratorUtils.elementsEqual(
                Arrays.stream(new int[0]).iterator(),
                Arrays.stream(new int[0]).iterator(),
                    Objects::equals
            )
        );
    }

    @Test
    public void testEmptyAndNonEmpty() {
        assertFalse(
            IteratorUtils.elementsEqual(
                Arrays.stream(new int[] {4, 5, 6, 1}).iterator(),
                Arrays.stream(new int[0]).iterator(),
                    Objects::equals
            )
        );
    }

    @Test
    public void testSameSizeAndEqual() {
        assertTrue(
            IteratorUtils.elementsEqual(
                Arrays.stream(new String[] {"a", "b", "c"}).iterator(),
                Arrays.stream(new String[] {"a", "b", "c"}).iterator(),
                    String::equals
            )
        );
    }

    @Test
    public void testSameSizeAndNonEqual() {
        assertFalse(
            IteratorUtils.elementsEqual(
                Arrays.stream(new int[] {4, 5, 6, 1}).iterator(),
                Arrays.stream(new int[] {4, 5, 3, 1}).iterator(),
                    Objects::equals
            )
        );
    }

    @Test
    public void testDifferentSizeSamePrefix() {
        assertFalse(
            IteratorUtils.elementsEqual(
                Arrays.stream(new int[] {1, 2, 3, 4, 5}).iterator(),
                Arrays.stream(new int[] {1, 2, 3}).iterator(),
                    Objects::equals
            )
        );
    }

    @Test
    public void testDifferentSizeDifferentPrefix() {
        assertFalse(
            IteratorUtils.elementsEqual(
                Arrays.stream(new String[] {"a", "z", "c", "d"}).iterator(),
                Arrays.stream(new String[] {"a", "b", "c"}).iterator(),
                    String::equals
            )
        );
    }

    @Test
    public void testEqualWithPrecision() {
        assertTrue(
            IteratorUtils.elementsEqual(
                Arrays.stream(new double[] {1.e-4, 1.0}).iterator(),
                Arrays.stream(new double[] {1.e-5, 1.0}).iterator(),
                (l, r) -> Precision.equals(l, r, 1e-3)
            )
        );
    }

    @Test
    public void testNonEqualWithPrecision() {
        assertFalse(
            IteratorUtils.elementsEqual(
                Arrays.stream(new double[] {1.e-4, 1.0}).iterator(),
                Arrays.stream(new double[] {1.e-5, 1.0}).iterator(),
                (l, r) -> Precision.equals(l, r, 1e-6)
            )
        );
    }
}
