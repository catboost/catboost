package ai.catboost.spark.impl

import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType

object RowEncoderConstructor {
    def construct(schema: StructType) : ExpressionEncoder[Row] = {
        ExpressionEncoder(schema)
    }
}
