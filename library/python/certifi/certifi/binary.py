import ssl


def builtin_ca():
    return None, None, ssl.builtin_cadata()


# Normally certifi.where() returns a path to a certificate file;
# here it returns a callable.
def where():
    return builtin_ca


# Patch ssl module to accept a callable cafile.
load_verify_locations = ssl.SSLContext.load_verify_locations


def load_verify_locations__callable(self, cafile=None, capath=None, cadata=None):
    if callable(cafile):
        cafile, capath, cadata = cafile()

    return load_verify_locations(self, cafile, capath, cadata)


ssl.SSLContext.load_verify_locations = load_verify_locations__callable
