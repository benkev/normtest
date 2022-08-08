def read_chunk(fileobj, chunksize=None):
    """
     A generator to read a file piece by piece.
    """
    while True:
        data = fileobj.read(chunksize)
        if not data:
            break
        yield data
