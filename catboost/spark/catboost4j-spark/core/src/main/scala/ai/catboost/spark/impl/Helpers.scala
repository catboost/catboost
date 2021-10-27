package ai.catboost.spark.impl

import java.nio.file.{Files,Path}
import java.nio.charset.StandardCharsets
import java.util.concurrent.{ExecutorCompletionService,Future}


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
    future1Description: String,
    future2: Future[Unit],
    future2Description: String
  ) = {
    try {
      future1.get
    } catch {
      case e : java.util.concurrent.ExecutionException => {
        future2.cancel(true)
        throw new java.util.concurrent.ExecutionException(
          "Error while executing " + future1Description,
          e.getCause
        )
      }
    }
    try {
      future2.get
    } catch {
      case e : java.util.concurrent.ExecutionException => {
        throw new java.util.concurrent.ExecutionException(
          "Error while executing " + future2Description,
          e.getCause
        )
      }
    }
  }

  def waitForTwoFutures(
    ecs: ExecutorCompletionService[Unit],
    future1: Future[Unit],
    future1Description: String,
    future2: Future[Unit],
    future2Description: String
  ) = {
    val firstCompletedFuture = ecs.take()

    if (firstCompletedFuture == future1) {
      checkOneFutureAndWaitForOther(future1, future1Description, future2, future2Description)
    } else {
      checkOneFutureAndWaitForOther(future2, future2Description, future1, future1Description)
    }
  }
}
