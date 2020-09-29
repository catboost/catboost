package ai.catboost.spark.params.macros;

import scala.annotation.{StaticAnnotation, compileTimeOnly}
import scala.language.experimental.macros
import scala.reflect.macros.whitebox

@compileTimeOnly("enable macro paradise to expand macro annotations")
private[spark] class ParamGetterSetter extends StaticAnnotation {
  def macroTransform(annottees: Any*): Any = macro ParamGetterSetterMacro.impl
}

private[spark] object ParamGetterSetterMacro {
  def impl(c: whitebox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._

    c.Expr[Any](
      annottees.head.tree match {
        case q"final val $id: $paramType = new $paramType2(this, $name, $doc)" =>
          val capitalizedId = TermName(id.decodedName.toString.capitalize)
          val getId = TermName("get" + capitalizedId)
          val setId = TermName("set" + capitalizedId)
          
          val paramValueType = paramType match {
            case tq"Param[..$typeList]" => {
              if (typeList.size != 1) {
                c.abort(
                  c.enclosingPosition,
                  s"Param must have one type parameter"
                )
              }
              typeList(0)
            }
            case tq"EnumParam[..$typeList]" => {
              if (typeList.size != 1) {
                c.abort(
                  c.enclosingPosition,
                  s"Param must have one type parameter"
                )
              }
              typeList(0)
            }
            case tq"OrderedStringMapParam[..$typeList]" => {
              if (typeList.size != 1) {
                c.abort(
                  c.enclosingPosition,
                  s"OrderedStringMapParam must have one type parameter"
                )
              }
              tq"java.util.LinkedHashMap[String, ${typeList(0)}]"
            }
            case tq"BooleanParam" => tq"Boolean"
            case tq"DoubleArrayParam" => tq"Array[Double]"
            case tq"DoubleParam" => tq"Double"
            case tq"FloatParam" => tq"Float"
            case tq"IntArrayParam" => tq"Array[Int]"
            case tq"IntParam" => tq"Int"
            case tq"LongParam" => tq"Long"
            case tq"StringArrayParam" => tq"Array[String]"
            case tq"DurationParam" => tq"java.time.Duration"
            case _ => c.abort(
              c.enclosingPosition,
              s"Bad paramType: $paramType"
            )
          }

          q"""
            final val $id: $paramType = new $paramType2(this, $name, $doc); 
            final def $getId: $paramValueType = $$($id);
            final def $setId(value: $paramValueType): this.type = set($id, value)
          """
        case _ => c.abort(
          c.enclosingPosition,
          s"Annotation @ParamGetterSetter can be used only with Param declarations"
        )
      }
    )
  }
}
