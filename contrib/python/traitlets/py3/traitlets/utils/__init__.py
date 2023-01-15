
# vestigal things from IPython_genutils.
def cast_unicode(s, encoding='utf-8'):
    if isinstance(s, bytes):
        return s.decode(encoding, 'replace')
    return s
