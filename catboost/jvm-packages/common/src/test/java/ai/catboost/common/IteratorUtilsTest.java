package ai.catboost.common;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

class IteratorUtilsTest {
    @Test
    void testEmpty() {
        assertTrue(
            IteratorUtils.elementsEqual(
                Arrays.stream(new int[0]).iterator(),
                Arrays.stream(new int[0]).iterator(),
                Integer::equals
            )
        );
    }

    @Test
    void testEmptyAndNonEmpty() {
        assertFalse(
            IteratorUtils.elementsEqual(
                Arrays.stream(new int[] {4, 5, 6, 1}).iterator(),
                Arrays.stream(new int[0]).iterator(),
                (l, r) -> l == r
            )
        );
    }

    @Test
    void testSameSizeAndEqual() {
        assertTrue(
            IteratorUtils.elementsEqual(
                Arrays.stream(new String[] {"a", "b", "c"}).iterator(),
                Arrays.stream(new String[] {"a", "b", "c"}).iterator(),
                String::equals
            )
        );
    }

    @Test
    void testSameSizeAndNonEqual() {
        assertFalse(
            IteratorUtils.elementsEqual(
                Arrays.stream(new int[] {4, 5, 6, 1}).iterator(),
                Arrays.stream(new int[] {4, 5, 3, 1}).iterator(),
                Integer::equals
            )
        );
    }

    @Test
    void testDifferentSizeSamePrefix() {
        assertFalse(
            IteratorUtils.elementsEqual(
                Arrays.stream(new int[] {1, 2, 3, 4, 5}).iterator(),
                Arrays.stream(new int[] {1, 2, 3}).iterator(),
                Integer::equals
            )
        );
    }

    @Test
    void testDifferentSizeDifferentPrefix() {
        assertFalse(
            IteratorUtils.elementsEqual(
                Arrays.stream(new String[] {"a", "z", "c", "d"}).iterator(),
                Arrays.stream(new String[] {"a", "b", "c"}).iterator(),
                String::equals
            )
        );
    }

    @Test
    void testEqualWithPrecision() {
        assertTrue(
            IteratorUtils.elementsEqual(
                Arrays.stream(new double[] {1.e-4, 1.0}).iterator(),
                Arrays.stream(new double[] {1.e-5, 1.0}).iterator(),
                (l, r) -> Math.abs(l - r) <= 1e-3
            )
        );
    }

    @Test
    void testNonEqualWithPrecision() {
        assertFalse(
            IteratorUtils.elementsEqual(
                Arrays.stream(new double[] {1.e-4, 1.0}).iterator(),
                Arrays.stream(new double[] {1.e-5, 1.0}).iterator(),
                (l, r) -> Math.abs(l - r) <= 1e-6
            )
        );
    }
}
