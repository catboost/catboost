import os.path

pem = os.path.abspath(__file__)
pem = os.path.dirname(pem)
pem = os.path.dirname(pem)
pem = os.path.dirname(pem)
pem = os.path.dirname(pem)
pem = os.path.dirname(pem)
pem = os.path.join(pem, "certs", "cacert.pem")


def where():
    return pem
