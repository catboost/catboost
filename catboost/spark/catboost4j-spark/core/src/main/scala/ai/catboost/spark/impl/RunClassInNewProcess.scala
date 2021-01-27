package ai.catboost.spark.impl

import collection.JavaConverters._
import collection.mutable

import java.nio.file.Paths


private[spark] object RunClassInNewProcess {
  private def getClassPathString() : String = {
    val classPathURIs = (new io.github.classgraph.ClassGraph()).getClasspathURIs().asScala
    classPathURIs.flatMap(
        uri => {
          uri.getScheme match {
            case "file" | "local" => Seq(uri.getPath)
            case _ => Seq()
          }
        }
    ).mkString(System.getProperty("path.separator"))
  }
  
  // Strips trailing $ from Scala's object class name to get companion class with static "main" method.
  private def getAppMainClassName(className: String) : String = {
    if (className.endsWith("$")) { className.substring(0, className.length - 1) } else { className }
  }


  def apply(
    clazz: Class[_],
    jvmArgs: Option[Array[String]] = None,
    args: Option[Array[String]] = None,
    inheritIO: Boolean = true,
    redirectInput: Option[ProcessBuilder.Redirect] = None,
    redirectOutput: Option[ProcessBuilder.Redirect] = None,
    redirectError: Option[ProcessBuilder.Redirect] = None
  ) : java.lang.Process = {
    val javaBin = Paths.get(System.getProperty("java.home"), "bin", "java").toString
    val classpath = getClassPathString()
    
    val cmd = new mutable.ArrayBuffer[String]
    cmd += javaBin
    if (jvmArgs.isDefined) {
      cmd ++= jvmArgs.get
    }
    cmd ++= Seq("-cp", classpath, getAppMainClassName(clazz.getName()))
    if (args.isDefined) {
      cmd ++= args.get
    }

    val processBuilder = new ProcessBuilder(cmd: _*)
    if (inheritIO) {
      if (redirectInput.isDefined || redirectOutput.isDefined || redirectError.isDefined) {
        throw new java.lang.IllegalArgumentException(
          "inheritIO specified together with one of redirect* arguments"
        )
      }
      processBuilder.inheritIO
    } else {
      if (redirectInput.isDefined) {
        processBuilder.redirectInput(redirectInput.get)
      }
      if (redirectOutput.isDefined) {
        processBuilder.redirectOutput(redirectOutput.get)
      }
      if (redirectError.isDefined) {
        processBuilder.redirectError(redirectError.get)
      }
    }

    processBuilder.start()
  }
}
