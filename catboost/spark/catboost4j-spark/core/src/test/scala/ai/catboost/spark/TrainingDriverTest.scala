package ai.catboost.spark

import collection.mutable

import java.io._
import java.net._
import java.util.concurrent.{CancellationException,Executors,FutureTask}

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.{Await,Future}
import scala.concurrent.duration._

import org.junit.{Assert,Test};


class TrainingDriverTest {
  @Test
  def simpleTest = {
    val workerCount = 5

    val expectedWorkersInfo = Array[WorkerInfo](
        new WorkerInfo(0, 10, "host0", 8082),
        new WorkerInfo(1, 20, "host1", 8083),
        new WorkerInfo(2, 30, "host2", 8084),
        new WorkerInfo(3, 40, "host3", 8085),
        new WorkerInfo(4, 50, "host4", 8086)
    )

    val callback = {
      workersInfo : Array[WorkerInfo] => {
        Assert.assertArrayEquals(
          workersInfo.asInstanceOf[Array[Object]],
          expectedWorkersInfo.asInstanceOf[Array[Object]]
        )
      }
    }

    val trainingDriver : TrainingDriver = new TrainingDriver(
      listeningPort = 0,
      workerCount = workerCount,
      startMasterCallback = callback,
      workerInitializationTimeout = java.time.Duration.ofSeconds(10)
    )
    val listeningPort = trainingDriver.getListeningPort

    val trainingDriverFuture = Executors.newSingleThreadExecutor.submit(trainingDriver)

    val futures = new mutable.ArrayBuffer[Future[Unit]]()
    for (i <- 0 until workerCount) {
      val port = 8082 + i
      val info = new WorkerInfo(i, i*10 + 10, s"host$i", port)

      futures += Future {
        Thread.sleep(Duration(2, SECONDS).toMillis)

        val socket = new Socket("localhost", listeningPort);
        try {
          val outputStream = socket.getOutputStream
          val objectOutputStream = new ObjectOutputStream(outputStream)
          objectOutputStream.writeUnshared(info)
          objectOutputStream.close
          outputStream.close
        } finally {
          socket.close
        }
      }
    }
    for (future <- futures) {
      Await.result(future, Duration(1, MINUTES))
    }

    val dummy = trainingDriverFuture.get
  }
}
