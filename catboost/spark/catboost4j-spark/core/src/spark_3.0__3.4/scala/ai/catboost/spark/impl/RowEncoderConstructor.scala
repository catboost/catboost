package ai.catboost.spark.impl

import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.types.StructType

object RowEncoderConstructor {
    def construct(schema: StructType) : RowEncoder = {
        RowEncoder(schema)
    }
}
