import ssl
import os


def builtin_ca():
    cafile, capath = os.getenv("SSL_CERT_FILE"), os.getenv("SSL_CERT_DIR")
    if cafile is None and capath is None:
        return None, None, ssl.builtin_cadata()
    return cafile, capath, None


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
