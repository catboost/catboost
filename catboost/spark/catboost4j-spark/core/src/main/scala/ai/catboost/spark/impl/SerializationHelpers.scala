package ai.catboost.spark.impl

import java.io.{ObjectInputStream,ObjectOutputStream}

import ai.catboost.CatBoostError


object SerializationHelpers {
  def readObject[T >: Null <: AnyRef](
    fileSystem: org.apache.hadoop.fs.FileSystem,
    fsPath: org.apache.hadoop.fs.Path,
    optional: Boolean = false
  ) : T = {
    if (!optional || fileSystem.exists(fsPath)) {
      val inputStream = fileSystem.open(fsPath)
      try {
        val objectStream = new ObjectInputStream(inputStream)
        try {
          objectStream.readObject().asInstanceOf[T]
        } finally {
          objectStream.close()
        }
      } finally {
        inputStream.close()
      }
    } else {
      null
    }
  }

  def writeObject[T](
    fileSystem: org.apache.hadoop.fs.FileSystem,
    fsPath: org.apache.hadoop.fs.Path,
    data: T // don't write if null
  ) = {
    if (data != null) {
      val outputStream = fileSystem.create(fsPath, true)
      try {
        val objectStream = new ObjectOutputStream(outputStream)
        try {
          objectStream.writeObject(data)
        } finally {
          objectStream.close()
        }
      } finally {
        outputStream.close()
      }
    }
  }
}

