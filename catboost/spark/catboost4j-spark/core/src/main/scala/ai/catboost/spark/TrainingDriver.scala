package ai.catboost.spark

import java.io._
import java.net._
import java.util.Arrays
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.{CancellationException,Executors,FutureTask}

import org.apache.spark.internal.Logging

import ai.catboost.CatBoostError


// use in startMasterCallback
private[spark] class CatBoostWorkersConnectionLostException(message: String) extends IOException(message) {
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


private[spark] class WorkerInfo (
  val partitionId : Int,
  val partitionSize: Int,
  val host: String,
  val port: Int
) extends Serializable {
  override def equals(rhs: Any) : Boolean = {
    val rhsWorkerInfo = rhs.asInstanceOf[WorkerInfo]
    if (rhsWorkerInfo == null) {
      false
    } else {
      partitionId.equals(rhsWorkerInfo.partitionId) &&
      partitionSize.equals(rhsWorkerInfo.partitionSize) &&
      host.equals(rhsWorkerInfo.host) &&
      port.equals(rhsWorkerInfo.port)
    }
  }
  
  override def toString() : String = {
    s"partitionId=${partitionId}, partitionSize=${partitionSize}, host=${host}, port=${port}"
  }
}


private[spark] class UpdatableWorkersInfo (
  private val workersInfo: Array[WorkerInfo],
  var workerRegistrationUpdatedSinceLastMasterStart : AtomicBoolean,
  val serverSocket: ServerSocket
) extends Runnable with Closeable with Logging {
  def this(
    listeningPort: Int,
    workerCount: Int
  ) = this(
    workersInfo = new Array[WorkerInfo](workerCount),
    workerRegistrationUpdatedSinceLastMasterStart = new AtomicBoolean(false),
    serverSocket = new ServerSocket(listeningPort)
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

  def initWorkers(workerInitializationTimeout: java.time.Duration) = {
    serverSocket.setSoTimeout(workerInitializationTimeout.toMillis.toInt)
    try {
      var registeredWorkerCount: Int = 0
      while (registeredWorkerCount < workersInfo.length) {
        acceptAndProcessWorkerInfo(
          workerInfo => {
            log.info(s"received workerInfo=${workerInfo}")
            if (workersInfo(workerInfo.partitionId) == null) {
              registeredWorkerCount = registeredWorkerCount + 1
            }
            workersInfo(workerInfo.partitionId) = workerInfo
          }
        )
      }
    } catch {
      case _: SocketTimeoutException => throw new CatBoostError(
        s"Initial worker wait timeout of ${impl.TimeHelpers.format(workerInitializationTimeout)} expired"
      )
    }
  }

  def run() = {
    try {
      serverSocket.setSoTimeout(0)

      while (true) {
        acceptAndProcessWorkerInfo(
          workerInfo => {
            this.synchronized {
              log.info(s"received workerInfo=${workerInfo}")
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
private[spark] class TrainingDriver (
  val updatableWorkersInfo: UpdatableWorkersInfo,
  val workerInitializationTimeout: java.time.Duration,
  val startMasterCallback: Array[WorkerInfo] => Unit
) extends Runnable with Logging {

  /**
   * @param listeningPort Port to listen for connections from workers. Pass 0 to autoassign (see ServerSocket constructor documentation)
   *  @param workerCount How many workers == partitions participate in training
   *  @param startMasterCallback pass a routine to start CatBoost master process given WorkerInfos.
   * 		Throw CatBoostWorkersConnectionLostException if master failed due to loss of workers
   *  @param workerInitializationTimeout Timeout to wait until CatBoost workers on Spark executors are initalized and sent their WorkerInfo
   */
  def this(
    listeningPort: Int,
    workerCount: Int,
    startMasterCallback: Array[WorkerInfo] => Unit,
    workerInitializationTimeout: java.time.Duration = java.time.Duration.ofMinutes(10)
  ) = this(
    updatableWorkersInfo = new UpdatableWorkersInfo(listeningPort, workerCount),
    workerInitializationTimeout = workerInitializationTimeout,
    startMasterCallback = startMasterCallback
  )

  def getListeningPort : Int = this.updatableWorkersInfo.serverSocket.getLocalPort

  def run = {
    try {
      log.info("started")
      log.debug(s"workerInitializationTimeout=${workerInitializationTimeout}")
      updatableWorkersInfo.initWorkers(workerInitializationTimeout)

      val workersUpdateFuture = Executors.newSingleThreadExecutor.submit(updatableWorkersInfo)

      try {
        var success = false
        do {
          log.info("wait for workers info")
          val workersInfo = updatableWorkersInfo.getWorkersInfo
          try {
            log.info("CatBoost master: starting")
            startMasterCallback(workersInfo)
            log.info("CatBoost master: finished successfully")
            success = true
          } catch {
            // try to relaunch if
            case e: CatBoostWorkersConnectionLostException => {
              log.info(s"wait for missing workers to relaunch for ${workerInitializationTimeout}")
              Thread.sleep(workerInitializationTimeout.toMillis)
              if (!updatableWorkersInfo.workerRegistrationUpdatedSinceLastMasterStart.get()) {
                throw new CatBoostError(
                  "CatBoost master won't be restarted - no relaunched workers after timeout " +
                  s"${impl.TimeHelpers.format(workerInitializationTimeout)} expired"
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
      log.info("finished")
    }
  }
}

// use on workers
private[spark] object TrainingDriver extends Logging {

  /**
   * @returns CatBoost worker port. There's a possibility that this port will be reuse before binding
   *  in CatBoost code, in this case worker will fail and will be restarted
   */
  def getWorkerPort() : Int = {
    val serverSocket = new ServerSocket(0)
    try {
      val localPort = serverSocket.getLocalPort
      log.info(s"Reserved port ${localPort} for CatBoost worker")
      localPort
    } finally {
      serverSocket.close
    }
  }

  private def isWorkerListening(port: Int) : Boolean = {
    var socket : Socket = null
    try {
      socket = new Socket("localhost", port)
      true
    } catch {
      case _ : Throwable => false
    } finally {
      if (socket != null) {
        socket.close
      }
    }
  }


  def waitForListeningPortAndSendWorkerInfo(
    trainingDriverListeningAddress: InetSocketAddress,
    partitionId: Int,
    partitionSize: Int,
    workerPort: Int
  ) = {
    if (partitionSize > 0) {
      log.info(s"wait for CatBoost worker to start listening at port ${workerPort}")
      do {
        Thread.sleep(10)
      } while (!isWorkerListening(workerPort))
      log.info(s"CatBoost worker started listening at port ${workerPort}")
    }

    log.info(s"send WorkerInfo to CatBoost training driver at ${trainingDriverListeningAddress}")
    val socket = new Socket(
      trainingDriverListeningAddress.getAddress,
      trainingDriverListeningAddress.getPort
    )

    val workerInfo = new WorkerInfo(
      partitionId,
      partitionSize,
      socket.getLocalAddress.getHostAddress,
      workerPort
    )
    log.info(s"WorkerInfo=${workerInfo}")
    try {
      val outputStream = socket.getOutputStream
      try {
        val objectOutputStream = new ObjectOutputStream(outputStream)
        try {
          objectOutputStream.writeUnshared(workerInfo)
        } finally {
          objectOutputStream.close
        }
      } finally {
        outputStream.close
      }
    } finally {
      socket.close
    }
    log.info("WorkerInfo has been successfully sent to CatBoost training driver")
  }
}
