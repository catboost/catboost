package ai.catboost.spark;

import ai.catboost.common.NativeLib;
//import org.apache.spark.

import ru.yandex.catboost.spark.catboost4j_spark.src.native_impl.*;

public class Quantizer {
    static {
        try {
            NativeLib.smartLoad("catboost4j-spark-impl");
        } catch (Exception ex) {
            throw new RuntimeException("Failed to load catboost4j-spark-impl native library", ex);
        }
    }
    
    public void Do() throws Exception {
        
    }
}
