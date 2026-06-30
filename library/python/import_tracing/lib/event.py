class Event:
    __slots__ = ("modname", "filename", "tid", "start_time", "end_time")

    def __init__(self, modname, filename, tid=None, start_time=None, end_time=None):
        self.modname = modname
        self.filename = filename
        self.tid = tid
        self.start_time = start_time
        self.end_time = end_time
