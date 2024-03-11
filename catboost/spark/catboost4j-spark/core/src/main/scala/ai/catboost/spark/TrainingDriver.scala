package ai.catboost.spark

import util.control.Breaks._

import java.io._
import java.net._
import java.nio.file.{Files,Path}
import java.util.Arrays
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.{CancellationException,Executors,FutureTask}

import sun.net.util.IPAddressUtil

import org.apache.spark.internal.Logging

import ru.yandex.catboost.spark.catboost4j_spark.core.src.native_impl

import ai.catboost.CatBoostError

import ai.catboost.spark.impl.{RunClassInNewProcess,ShutdownWorkersApp}

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

// use in waitForListeningPortAndSendWorkerInfo
private[spark] class CatBoostTrainingDriverConnectException(message: String) extends IOException(message) {
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
  val serverSocket: ServerSocket
) extends Runnable with Closeable with Logging {
  def this(
    listeningPort: Int,
    workerCount: Int
  ) = this(
    workersInfo = new Array[WorkerInfo](workerCount),
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
            }
          }
        )
      }

    } catch {
      case _: InterruptedException => ()
    }
  }

  def close = {
    serverSocket.close
  }

  def getWorkersInfo : Array[WorkerInfo] = {
    this.synchronized {
      Arrays.copyOf(workersInfo, workersInfo.length)
    }
  }

  def shutdownRemainingWorkers(
    connectTimeout: java.time.Duration,
    workerShutdownOptimisticTimeout: java.time.Duration,
    workerShutdownPessimisticTimeout: java.time.Duration
  ) = {
    log.info("Shutdown remaining workers: start")

    val remainingWorkersInfo = workersInfo.filter(
      workerInfo => {
        val isListening
          = (workerInfo != null) &&
            TrainingDriver.isWorkerListening(workerInfo.host, workerInfo.port, connectTimeout)
        if (isListening) {
          log.info(s"remaining listening workerInfo=${workerInfo}")
        }
        isListening
      }
    )

    if (remainingWorkersInfo.isEmpty) {
      log.info("Shutdown remaining workers: no remaining workers")
    } else {
      val tmpDirPath = Files.createTempDirectory("catboost_train")

      val hostsFilePath = tmpDirPath.resolve("worker_hosts.txt")
      TrainingDriver.saveHostsListToFile(hostsFilePath, remainingWorkersInfo)

      val shutdownWorkersAppProcess = RunClassInNewProcess(
        ShutdownWorkersApp.getClass,
        args = Some(Array(hostsFilePath.toString, workerShutdownPessimisticTimeout.getSeconds.toString))
      )

      val returnValue = shutdownWorkersAppProcess.waitFor
      if (returnValue != 0) {
        throw new CatBoostError(s"Shutdown workers process failed: exited with code $returnValue")
      }

      // wait to ensure that workers have exited
      Thread.sleep(workerShutdownOptimisticTimeout.toMillis)
    }

    log.info("Shutdown remaining workers: finish")
  }
}


/**
 * Responsible for high-level coordination between master and workers.
 * Should be run in a separate thread using Future (to capture possible exceptions).
 *  1) waits until all workers announced their readiness by
 *     sending their host:port coordinates
 *  2) start CatBoost master via callback with workers' coordinates
 *  3) if master finished unsuccessfully shutdown remaining workers to avoid their infinite waiting
 */
private[spark] class TrainingDriver (
  val updatableWorkersInfo: UpdatableWorkersInfo,
  val connectTimeout: java.time.Duration,
  val workerInitializationTimeout: java.time.Duration,
  val workerShutdownOptimisticTimeout: java.time.Duration,
  val workerShutdownPessimisticTimeout: java.time.Duration,
  val startMasterCallback: Array[WorkerInfo] => Unit,
  var closed: Boolean
) extends Runnable with Logging {

  /**
   * @param listeningPort Port to listen for connections from workers. Pass 0 to autoassign (see ServerSocket constructor documentation)
   *  @param workerCount How many workers == partitions participate in training
   *  @param startMasterCallback pass a routine to start CatBoost master process given WorkerInfos.
   * 		Throw CatBoostWorkersConnectionLostException if master failed due to loss of workers
   *  @param connectTimeout Timeout to wait while establishing socket connections between TrainingDriver and workers
   *  @param workerInitializationTimeout Timeout to wait until CatBoost workers on Spark executors are initalized and sent their WorkerInfo
   *  @param workerShutdownOptimisticTimeout Timeout to wait for CatBoost workers to shut down after CatBoost master failed
   *   (assuming CatBoost master has been able to issue 'stop' command to workers)
   *  @param workerShutdownPessimisticTimeout Timeout to wait for CatBoost workers to shut down in
   *   explicit shutdownRemainingWorkers procedure.
   */
  def this(
    listeningPort: Int,
    workerCount: Int,
    startMasterCallback: Array[WorkerInfo] => Unit,
    connectTimeout: java.time.Duration = java.time.Duration.ofMinutes(1),
    workerInitializationTimeout: java.time.Duration = java.time.Duration.ofMinutes(10),
    workerShutdownOptimisticTimeout : java.time.Duration = java.time.Duration.ofSeconds(40),
    workerShutdownPessimisticTimeout : java.time.Duration = java.time.Duration.ofMinutes(5)
  ) = this(
    updatableWorkersInfo = new UpdatableWorkersInfo(listeningPort, workerCount),
    connectTimeout = connectTimeout,
    workerInitializationTimeout = workerInitializationTimeout,
    workerShutdownOptimisticTimeout = workerShutdownOptimisticTimeout,
    workerShutdownPessimisticTimeout = workerShutdownPessimisticTimeout,
    startMasterCallback = startMasterCallback,
    closed = false
  )

  def getListeningPort : Int = this.updatableWorkersInfo.serverSocket.getLocalPort

  def run = {
    var success = false
    try {
      log.info("started")
      log.info("wait for workers info")
      log.debug(s"workerInitializationTimeout=${workerInitializationTimeout}")
      updatableWorkersInfo.initWorkers(workerInitializationTimeout)

      val workersUpdateFuture = Executors.newSingleThreadExecutor.submit(updatableWorkersInfo)

      try {
        val workersInfo = updatableWorkersInfo.getWorkersInfo
        log.info("CatBoost master: starting")
        startMasterCallback(workersInfo)
        log.info("CatBoost master: finished successfully")
        success = true
      } finally {
        workersUpdateFuture.cancel(true)
        try {
          workersUpdateFuture.get // to throw exceptions if there were any
        } catch {
          case _: CancellationException => ()
        }
      }
    } finally {
      /* wait for CatBoost workers to shut down after CatBoost master failed
         (assuming CatBoost master has been able to issue 'stop' command to workers)
      */
      this.close(tryToShutdownWorkers = !success, waitToShutdownWorkers = true)
      log.info("finished")
    }
  }

  def close(tryToShutdownWorkers:Boolean, waitToShutdownWorkers:Boolean) = {
    this.synchronized {
      if (!closed) {
        log.info("close updatableWorkersInfo")
        updatableWorkersInfo.close
        if (tryToShutdownWorkers) {
          if (waitToShutdownWorkers) {
            log.info(s"wait for workers to finish by themselves for $workerShutdownOptimisticTimeout")
            Thread.sleep(workerShutdownOptimisticTimeout.toMillis)
          }
          updatableWorkersInfo.shutdownRemainingWorkers(
            connectTimeout,
            workerShutdownOptimisticTimeout,
            workerShutdownPessimisticTimeout
          )
        }
        closed = true
        log.info("closed")
      }
    }
  }
}


private[spark] object TrainingDriver extends Logging {
  def saveHostsListToFile(hostsFilePath: Path, workersInfo: Array[WorkerInfo]) = {
    val pw = new PrintWriter(hostsFilePath.toFile)
    try {
      for (workerInfo <- workersInfo) {
        if (workerInfo.partitionSize > 0) {
          if (IPAddressUtil.isIPv6LiteralAddress(workerInfo.host)) {
            pw.println(s"[${workerInfo.host}]:${workerInfo.port}")
          } else {
            pw.println(s"${workerInfo.host}:${workerInfo.port}")
          }
        }
      }
    } finally {
      pw.close
    }
  }

  /**
   * Use on workers
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

  def isWorkerListening(host: String, port: Int, connectTimeout: java.time.Duration) : Boolean = {
    val socket = new Socket
    val socketAddress = new InetSocketAddress(host, port)
    val connected = try {
      socket.connect(socketAddress, connectTimeout.toMillis.toInt)
      true
    } catch {
      case _ : Throwable => false
    }
    if (connected) {
      socket.close
    }
    connected
  }

  // use on workers
  def waitForListeningPortAndSendWorkerInfo(
    trainingDriverListeningAddress: InetSocketAddress,
    partitionId: Int,
    partitionSize: Int,
    workerPort: Int,
    connectTimeout:  java.time.Duration,
    workerInitializationTimeout: java.time.Duration
  ) = {
    if (workerInitializationTimeout.toMillis < 10) {
      throw new RuntimeException("workerInitializationTimeout must be >= 10 ms")
    }

    if (partitionSize > 0) {
      log.info(s"wait for CatBoost worker to start listening at port ${workerPort}")
      val initializationDeadline = java.time.Instant.now().plus(workerInitializationTimeout)

      breakable {
        while (true) {
          Thread.sleep(10)
          if (isWorkerListening("localhost", workerPort, workerInitializationTimeout)) {
            log.info(s"CatBoost worker started listening at port ${workerPort}")
            break
          }
          if (java.time.Instant.now().compareTo(initializationDeadline) > 0) {
            throw new CatBoostError(
              s"Initial worker wait timeout of ${impl.TimeHelpers.format(workerInitializationTimeout)} expired"
            )
          }
        }
      }
    }

    log.info(s"send WorkerInfo to CatBoost training driver at ${trainingDriverListeningAddress}")
    val socket = new Socket
    try {
      socket.connect(trainingDriverListeningAddress, connectTimeout.toMillis.toInt)
    } catch {
      case e : Throwable => {
        throw new CatBoostTrainingDriverConnectException(e)
      }
    }

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
