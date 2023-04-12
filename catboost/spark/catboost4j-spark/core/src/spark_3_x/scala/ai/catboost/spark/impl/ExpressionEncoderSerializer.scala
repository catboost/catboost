package ai.catboost.spark.impl

import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.encoders.{ExpressionEncoder,RowEncoder}
import org.apache.spark.sql.types.StructType


object ExpressionEncoderSerializer {
  def apply(schema: StructType) : ExpressionEncoderSerializer = {
    new ExpressionEncoderSerializer(RowEncoder(schema).createSerializer())
  }
}


class ExpressionEncoderSerializer(val serializer: ExpressionEncoder.Serializer[Row]) {
  def toInternalRow(row : Row) : InternalRow = {
    serializer(row)
  }
}
