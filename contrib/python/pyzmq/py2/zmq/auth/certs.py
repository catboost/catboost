"""0MQ authentication related functions and classes."""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.


import datetime
import glob
import io
import os
import zmq
from zmq.utils.strtypes import bytes, unicode, b, u


_cert_secret_banner = u("""#   ****  Generated on {0} by pyzmq  ****
#   ZeroMQ CURVE **Secret** Certificate
#   DO NOT PROVIDE THIS FILE TO OTHER USERS nor change its permissions.

""")

_cert_public_banner = u("""#   ****  Generated on {0} by pyzmq  ****
#   ZeroMQ CURVE Public Certificate
#   Exchange securely, or use a secure mechanism to verify the contents
#   of this file after exchange. Store public certificates in your home
#   directory, in the .curve subdirectory.

""")

def _write_key_file(key_filename, banner, public_key, secret_key=None, metadata=None, encoding='utf-8'):
    """Create a certificate file"""
    if isinstance(public_key, bytes):
        public_key = public_key.decode(encoding)
    if isinstance(secret_key, bytes):
        secret_key = secret_key.decode(encoding)
    with io.open(key_filename, 'w', encoding='utf8') as f:
        f.write(banner.format(datetime.datetime.now()))

        f.write(u('metadata\n'))
        if metadata:
            for k, v in metadata.items():
                if isinstance(k, bytes):
                    k = k.decode(encoding)
                if isinstance(v, bytes):
                    v = v.decode(encoding)
                f.write(u("    {0} = {1}\n").format(k, v))

        f.write(u('curve\n'))
        f.write(u("    public-key = \"{0}\"\n").format(public_key))

        if secret_key:
            f.write(u("    secret-key = \"{0}\"\n").format(secret_key))


def create_certificates(key_dir, name, metadata=None):
    """Create zmq certificates.
    
    Returns the file paths to the public and secret certificate files.
    """
    public_key, secret_key = zmq.curve_keypair()
    base_filename = os.path.join(key_dir, name)
    secret_key_file = "{0}.key_secret".format(base_filename)
    public_key_file = "{0}.key".format(base_filename)
    now = datetime.datetime.now()

    _write_key_file(public_key_file,
                    _cert_public_banner.format(now),
                    public_key)

    _write_key_file(secret_key_file,
                    _cert_secret_banner.format(now),
                    public_key,
                    secret_key=secret_key,
                    metadata=metadata)

    return public_key_file, secret_key_file


def load_certificate(filename):
    """Load public and secret key from a zmq certificate.
    
    Returns (public_key, secret_key)
    
    If the certificate file only contains the public key,
    secret_key will be None.
    
    If there is no public key found in the file, ValueError will be raised.
    """
    public_key = None
    secret_key = None
    if not os.path.exists(filename):
        raise IOError("Invalid certificate file: {0}".format(filename))

    with open(filename, 'rb') as f:
        for line in f:
            line = line.strip()
            if line.startswith(b'#'):
                continue
            if line.startswith(b'public-key'):
                public_key = line.split(b"=", 1)[1].strip(b' \t\'"')
            if line.startswith(b'secret-key'):
                secret_key = line.split(b"=", 1)[1].strip(b' \t\'"')
            if public_key and secret_key:
                break
    
    if public_key is None:
        raise ValueError("No public key found in %s" % filename)
    
    return public_key, secret_key


def load_certificates(directory='.'):
    """Load public keys from all certificates in a directory"""
    certs = {}
    if not os.path.isdir(directory):
        raise IOError("Invalid certificate directory: {0}".format(directory))
    # Follow czmq pattern of public keys stored in *.key files.
    glob_string = os.path.join(directory, "*.key")
    
    cert_files = glob.glob(glob_string)
    for cert_file in cert_files:
        public_key, _ = load_certificate(cert_file)
        if public_key:
            certs[public_key] = True
    return certs

__all__ = ['create_certificates', 'load_certificate', 'load_certificates']
