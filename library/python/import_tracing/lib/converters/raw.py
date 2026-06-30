import library.python.import_tracing.lib.converters.base as base_converter
import library.python.import_tracing.lib.constants as constants


class RawTextTraceConverter(base_converter.BaseTraceConverter):
    @staticmethod
    def _get_columns_length(events):
        max_filename = 0
        max_cumtime = 0
        max_end_time = 0

        for event in events:
            max_filename = max(max_filename, len(event.filename))
            max_cumtime = max(max_cumtime, event.end_time - event.start_time)
            max_end_time = max(max_end_time, event.end_time)

        return len(str(max_cumtime)), max_filename, max_end_time

    @staticmethod
    def _get_sorted_events(events):
        return sorted(events, key=lambda event: event.end_time - event.start_time, reverse=True)

    @staticmethod
    def _format_line(cumtime, filename, max_cumtime, max_filename):
        return "{0:<{max_cumtime}}\t{1:<{max_filename}}\n".format(
            cumtime,
            filename,
            max_cumtime=max_cumtime,
            max_filename=max_filename,
        )

    def dump(self, events, filepath):
        max_cumtime, max_filename, max_end_time = self._get_columns_length(events)
        max_line_length = max_cumtime + max_filename

        with open(filepath, "w") as file:
            # total time taken
            file.write("total time taken (seconds): {0:.4f}\n".format(max_end_time / constants.MCS_IN_SEC))
            file.write("-" * max_line_length + "\n")

            # header
            file.write(self._format_line("cumtime", "filename", max_cumtime, max_filename))
            file.write("-" * max_line_length + "\n")

            # trace info
            for event in self._get_sorted_events(events):
                time_taken = format(((event.end_time - event.start_time) / constants.MCS_IN_SEC), ".6f")

                file.write(
                    self._format_line(
                        time_taken,
                        event.filename,
                        max_cumtime,
                        max_filename,
                    )
                )
