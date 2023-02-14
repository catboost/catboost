def copy_stream(read, *writers, **kwargs):
    """
    function that reads data by its first argument 'read' function
    and writes that data by all 'writers' functions specified as positional args
    optional kwarg 'size' controls chunk size for read-write operations
    """
    chunk_size = kwargs.get('size', 1024 * 1024)
    while True:
        data = read(chunk_size)
        if not data:
            break
        for write in writers:
            write(data)
