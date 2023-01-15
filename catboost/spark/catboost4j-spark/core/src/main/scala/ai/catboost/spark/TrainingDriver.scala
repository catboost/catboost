package ai.catboost.spark

import java.io._
import java.net._
import java.util.Arrays
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.{CancellationException,Executors,FutureTask}

import scala.concurrent.duration._

import ai.catboost.CatBoostError


// use in startMasterCallback
class CatBoostWorkersConnectionLostException(message: String) extends IOException(message) {
  def this(message: String, cause: Throwable) {
    this(message)
    initCause(cause)
  }

  def this(cause: Throwable) {
    this(Option(cause).map(_.toString).orNull, cause)
  }

  def this() {
    this(null: String)
  }
}


class WorkerInfo (
  val partitionId : Int,
  val host: String,
  val port: Int
) extends Serializable {
  override def equals(rhs: Any) : Boolean = {
    System.out.println("WorkerInfo.equals")
    val rhsWorkerInfo = rhs.asInstanceOf[WorkerInfo]
    if (rhsWorkerInfo == null) {
      false
    } else {
      partitionId.equals(rhsWorkerInfo.partitionId) &&
      host.equals(rhsWorkerInfo.host) &&
      port.equals(rhsWorkerInfo.port)
    }
  }
}


class UpdatableWorkersInfo (
  private val workersInfo: Array[WorkerInfo],
  var workerRegistrationUpdatedSinceLastMasterStart : AtomicBoolean,
  val serverSocket: ServerSocket,
  private val fullTimeout: Duration
) extends Runnable with Closeable {
  def this(
    listeningPort: Int,
    workerCount: Int,
    fullTimeout: Duration
  ) = this(
    workersInfo = new Array[WorkerInfo](workerCount),
    workerRegistrationUpdatedSinceLastMasterStart = new AtomicBoolean(false),
    serverSocket = new ServerSocket(listeningPort),
    fullTimeout = fullTimeout
  )

  private def acceptAndProcessWorkerInfo(callback : WorkerInfo => Unit) = {
    val socket = serverSocket.accept
    try {
      val inputStream = socket.getInputStream
      try {
        val objectInputStream = new ObjectInputStream(inputStream)
        try {
          callback(objectInputStream.readUnshared().asInstanceOf[WorkerInfo])
        } finally {
          objectInputStream.close
        }
      } finally {
        inputStream.close
      }
    } finally {
      socket.close
    }
  }

  def initWorkers(workerInitializationTimeout: Duration) = {
    serverSocket.setSoTimeout(workerInitializationTimeout.toMillis.toInt)
    try {
      var registeredWorkerCount: Int = 0
      while (registeredWorkerCount < workersInfo.length) {
        acceptAndProcessWorkerInfo(
          workerInfo => {
            if (workersInfo(workerInfo.partitionId) == null) {
              registeredWorkerCount = registeredWorkerCount + 1
            }
            workersInfo(workerInfo.partitionId) = workerInfo
          }
        )
      }
    } catch {
      case _: SocketTimeoutException => throw new CatBoostError(
        s"Initial worker wait timeout of $workerInitializationTimeout expired"
      )
    }
  }

  def run() = {
    try {
      serverSocket.setSoTimeout(fullTimeout.toMillis.toInt)

      while (true) {
        acceptAndProcessWorkerInfo(
          workerInfo => {
            this.synchronized {
              workersInfo(workerInfo.partitionId) = workerInfo
              workerRegistrationUpdatedSinceLastMasterStart.set(true)
            }
          }
        )
      }

    } catch {
      case _: InterruptedException => Unit
    }
  }

  def close = {
    serverSocket.close
  }

  def getWorkersInfo : Array[WorkerInfo] = {
    this.synchronized {
      workerRegistrationUpdatedSinceLastMasterStart.set(false)
      Arrays.copyOf(workersInfo, workersInfo.length)
    }
  }
}


/**
 * Responsible for high-level coordination between master and workers.
 * Should be run in a separate thread using Future (to capture possible exceptions).
 *  1) waits until all workers announced their readiness by
 *     sending their host:port coordinates
 *  2) start a separate thread to track workers' coordinates updates
 *  3) start CatBoost master via callback with workers' coordinates
 *  4) if CatBoost master fails see if we can attempt to restart it with updated workers
 *     (wait for workerInitializationTimeout before attempting that).
 *     Go to step 3 if there are updated workers.
 *  5) returns if master callback finished without exceptions
 */
class TrainingDriver (
  val updatableWorkersInfo: UpdatableWorkersInfo,
  val workerInitializationTimeout: Duration,
  val startMasterCallback: Array[WorkerInfo] => Unit
) extends Runnable {

  /**
   * @param listeningPort Port to listen for connections from workers. Pass 0 to autoassign (see ServerSocket constructor documentation)
   *  @param fullTimeout Timeout for maximum wait for the whole training process
   *  @param workerCount How many workers == partitions participate in training
   *  @param startMasterCallback pass a routine to start CatBoost master process given WorkerInfos.
   * 		Throw CatBoostWorkersConnectionLostException if master failed due to loss of workers
   *  @param workerInitializationTimeout Timeout to wait until CatBoost workers on Spark executors are initalized and sent their WorkerInfo
   */
  def this(
    listeningPort: Int,
    workerCount: Int,
    fullTimeout: Duration,
    startMasterCallback: Array[WorkerInfo] => Unit,
    workerInitializationTimeout: Duration = Duration(10, MINUTES)
  ) = this(
    updatableWorkersInfo = new UpdatableWorkersInfo(listeningPort, workerCount, fullTimeout),
    workerInitializationTimeout = workerInitializationTimeout,
    startMasterCallback = startMasterCallback
  )

  def getListeningPort : Int = this.updatableWorkersInfo.serverSocket.getLocalPort

  def run = {
    try {
      updatableWorkersInfo.initWorkers(workerInitializationTimeout)

      val workersUpdateFuture = Executors.newSingleThreadExecutor.submit(updatableWorkersInfo)

      try {
        var success = false
        do {
          val workersInfo = updatableWorkersInfo.getWorkersInfo
          try {
            startMasterCallback(workersInfo)
            success = true
          } catch {
            // try to relaunch if
            case e: CatBoostWorkersConnectionLostException => {
              // wait for some relaunched workers
              Thread.sleep(workerInitializationTimeout.toMillis)
              if (!updatableWorkersInfo.workerRegistrationUpdatedSinceLastMasterStart.get()) {
                throw new CatBoostError(
                  "Master won't be restarted - no relaunched workers after timeout " +
                  s"$workerInitializationTimeout expired"
                )
              }
            }
          }
        } while (!success)
      } finally {
        workersUpdateFuture.cancel(true)
        try {
          workersUpdateFuture.get // to throw exceptions if there were any
        } catch {
          case _: CancellationException => Unit
        }
      }
    } finally {
      updatableWorkersInfo.close
    }
  }
}
