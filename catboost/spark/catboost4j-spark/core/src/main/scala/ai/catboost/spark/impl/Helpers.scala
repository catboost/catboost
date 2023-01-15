package ai.catboost.spark.impl

import java.nio.file.{Files,Path}
import java.nio.charset.StandardCharsets
import java.util.concurrent.Future


private[spark] object Helpers {
  def writeToTempFile(
    content: String,
    tmpFilePrefix: String,
    tmpFileSuffix: Option[String] = None
  ) : Path = {
    val tempFile = Files.createTempFile(tmpFilePrefix, tmpFileSuffix.getOrElse(".tmp"))
    tempFile.toFile.deleteOnExit
    Files.write(tempFile, content.getBytes(StandardCharsets.UTF_8))
    tempFile
  }

  def checkOneFutureAndWaitForOther(
    future1: Future[Unit],
    future2: Future[Unit],
    future1Description: String
  ) = {
    try {
      future1.get
    } catch {
      case t: Throwable => {
        future2.cancel(true)
        throw new RuntimeException("Error while executing " + future1Description, t)
      }
    }
    future2.get
  }
}
