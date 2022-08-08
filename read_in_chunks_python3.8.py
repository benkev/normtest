def rows(f, chunksize=1024, sep='|'):
    """
    Read a file where the row separator is '|' lazily.

    Usage:

    >>> with open('big.csv') as f:
    >>>     for r in rows(f):
    >>>         process(r)
    """
    row = ''
    while (chunk := f.read(chunksize)) != '':   # End of file
        while (i := chunk.find(sep)) != -1:     # No separator found
            yield row + chunk[:i]
            chunk = chunk[i+1:]
            row = ''
        row += chunk
    yield row

    
