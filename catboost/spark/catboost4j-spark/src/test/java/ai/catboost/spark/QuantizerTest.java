package ai.catboost.spark;

import java.util.Iterator;
import java.util.stream.IntStream; 

import org.apache.commons.math3.util.Precision;

import org.junit.Assert;
import org.junit.Test;

import ai.catboost.CatBoostError;
import ai.catboost.common.IteratorUtils;

import ru.yandex.catboost.spark.catboost4j_spark.src.native_impl.*;


public class QuantizerTest {

    private void testBestSplit(
        TVector_float values,
        TVector_float initialBorders,
        TVector_float expectedBorders
    ) throws Exception {
        Quantizer quantizer = new Quantizer();

        TVector_float borders = new TVector_float();

        boolp hasDefaultQuantizedBin = new boolp();
        TDefaultQuantizedBin defaultQuantizedBin = new TDefaultQuantizedBin();

        native_impl.BestSplit(
            values,
            /*valuesSorted*/ false,
            /*defaultValue*/ null,
            /*featureValuesMayContainNans*/ false,
            /*maxBordersCount*/ 255,
            EBorderSelectionType.GreedyLogSum,
            /*quantizedDefaultBinFraction*/ null,
            /*initialBorders*/ initialBorders,
            borders,
            hasDefaultQuantizedBin.cast(),
            defaultQuantizedBin
        );

        Assert.assertTrue(
            IteratorUtils.elementsEqual( 
                borders.iterator(),
                expectedBorders.iterator(),
                (l, r) -> Precision.equals(l, r, 1e-5)
            )
        );
    }

    @Test
    public void testEmpty() throws Exception {
        Quantizer quantizer = new Quantizer();
        
        TVector_float values = new TVector_float();
        TVector_float initialBorders = null;
        TVector_float borders = new TVector_float();

        testBestSplit(values, initialBorders, borders);
    }
    
    @Test
    public void testTwoElements() throws Exception {
        Quantizer quantizer = new Quantizer();

        TVector_float values = new TVector_float(new float[] {0.12f, 0.33f});
        TVector_float initialBorders = null;
        TVector_float borders = new TVector_float(new float[] {0.225f});

        testBestSplit(values, initialBorders, borders);
    }

    @Test
    public void testManyElements() throws Exception {
        Quantizer quantizer = new Quantizer();

        TVector_float values = new TVector_float(new float[] {0.12f, 0.33f, 0.33f, 0.75f, 1.1f, 1.2f});
        TVector_float initialBorders = null;
        TVector_float borders = new TVector_float(new float[] {0.225f, 0.54f, 0.925f, 1.15f});

        testBestSplit(values, initialBorders, borders);
    }

    @Test
    public void testInitialBorders() throws Exception {
        Quantizer quantizer = new Quantizer();

        TVector_float values = new TVector_float(new float[] {0.12f, 0.33f, 0.33f, 0.75f, 1.1f, 1.2f});
        TVector_float initialBorders = new TVector_float(new float[] {0.5f, 0.6f});
        TVector_float borders = new TVector_float(new float[] {0.225f, 0.5f, 0.925f, 1.15f});

        testBestSplit(values, initialBorders, borders);
    }
}
