package ai.catboost.spark.impl

import collection.JavaConverters._
import collection.mutable

import java.io.{File,FileOutputStream, IOException}
import java.nio.file.{Files,Paths}
import java.util.jar.{JarOutputStream,Manifest}


private[spark] object RunClassInNewProcess {
  // Pathing Jar code is slightly modified version of code from Gradle's JavaExecHandleBuilder.java
  // Licensed under the Apache License, Version 2.0 (the "License"): http://www.apache.org/licenses/LICENSE-2.0

  @throws[IOException]
  private def writePathingJarFile(classPath: Seq[String]) : File = {
    val pathingJarFile = Files.createTempFile("classpath", ".tmp").toFile
    pathingJarFile.deleteOnExit
    val fileOutputStream = new FileOutputStream(pathingJarFile)
    try {
      val jarOutputStream = new JarOutputStream(fileOutputStream, toManifest(classPath))
      try {
        jarOutputStream.putNextEntry(new java.util.zip.ZipEntry("META-INF/"))
      } finally {
        jarOutputStream.close()
      }
    } finally {
      fileOutputStream.close()
    }
    pathingJarFile
  }

  private def toManifest(classPath: Seq[String]): Manifest = {
    val manifest = new Manifest()
    val attributes = manifest.getMainAttributes()
    attributes.put(java.util.jar.Attributes.Name.MANIFEST_VERSION, "1.0")
    attributes.putValue("Class-Path", classPath.mkString(" "))
    manifest
  }

  private def getClassPath() : Seq[String] = {
    val classPathURIs = (new io.github.classgraph.ClassGraph()).getClasspathURIs().asScala
    classPathURIs.iterator.flatMap(
        uri => {
          uri.getScheme match {
            case "file" | "local" => Seq(uri.getPath)
            case _ => Seq()
          }
        }
    ).toSeq
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
    val classPath = getClassPath()
    val classpathJarFile = writePathingJarFile(classPath)

    val cmd = new mutable.ArrayBuffer[String]
    cmd += javaBin
    if (jvmArgs.isDefined) {
      cmd ++= jvmArgs.get
    }
    cmd ++= Seq("-cp", classpathJarFile.getAbsolutePath(), getAppMainClassName(clazz.getName()))
    if (args.isDefined) {
      cmd ++= args.get
    }

    val processBuilder = new ProcessBuilder(cmd.toSeq: _*)
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
