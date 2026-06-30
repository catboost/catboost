package ai.catboost.spark;

import org.junit.rules.Verifier
import org.junit.Assert


object ProcessingFinishedVerifier extends Verifier {
override def verify(): Unit = {
    val runningNonDaemonThreads = Thread.getAllStackTraces.keySet.stream
      .filter(t => !t.isDaemon && t != Thread.currentThread())
      .map[String](_.getName)
      .toArray
    if (runningNonDaemonThreads.nonEmpty) {
      Assert.fail(s"Some non-daemon threads are still running: ${runningNonDaemonThreads.mkString(", ")}")
    }
  }
}

