package ai.catboost.spark.impl

import org.apache.commons.lang3.time.DurationFormatUtils


private[spark] object TimeHelpers {
  def format(duration: java.time.Duration) : String = {
    val formatString = if (duration.toDays > 0) {
      "dd'days' hh'hours' mm'min' ss'sec'"
    } else if (duration.toHours > 0) {
      "hh'hours' mm'min' ss'sec'"
    } else if (duration.toMinutes > 0) {
      "mm'min' ss'sec'"
    } else {
      "ss'sec'"
    }
    DurationFormatUtils.formatDuration(duration.toMillis, formatString)
  }
}
