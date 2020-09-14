package ai.catboost.spark;

import collection.mutable

import org.apache.spark.ml.attribute._
import org.apache.spark.sql.types.Metadata

import ai.catboost.CatBoostError


object DataHelpers {
  def makeFeaturesMetadata(initialFeatureNames: Array[String]) : Metadata = {
    val featureNames = new Array[String](initialFeatureNames.length)

    val featureNamesSet = new mutable.HashSet[String]()

    for (i <- 0 until featureNames.size) {
      val name = initialFeatureNames(i)
      if (name.isEmpty) {
        val generatedName = s"_f$i"
        if (featureNamesSet.contains(generatedName)) {
          throw new CatBoostError(
            s"""Unable to use generated name "$generatedName" for feature with unspecified name because"""
            + " it has been already used for another feature"
          )
        }
        featureNames(i) = generatedName
      } else {
        featureNames(i) = name
      }
      featureNamesSet.add(featureNames(i))
    }

    val defaultAttr = NumericAttribute.defaultAttr
    val attrs = featureNames.map {
      name => defaultAttr.withName(name).asInstanceOf[Attribute]
    }.toArray
    val attrGroup = new AttributeGroup("userFeatures", attrs)
    attrGroup.toMetadata
  }
}
