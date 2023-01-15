package ai.catboost

import ai.catboost.common.NativeLib;
import java.lang.System

package object spark {
  var nativeLibLoaded = false
  
  def ensureNativeLibLoaded = {
    this.synchronized {
      if (!nativeLibLoaded) {
        try {
          NativeLib.smartLoad("catboost4j-spark-impl");
        } catch {
          case e: Exception => throw new RuntimeException("Failed to load catboost4j-spark-impl native library", e);
        }
        nativeLibLoaded = true
      }
    }
  }
}