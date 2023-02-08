#!/usr/bin/env python

import codecs
import sys
import unittest

import idna.codec

class IDNACodecTests(unittest.TestCase):
    
    def testCodec(self):
        pass

    def testIncrementalDecoder(self):

        # Tests derived from Python standard library test/test_codecs.py

        incremental_tests = (
            (u"python.org", b"python.org"),
            (u"python.org.", b"python.org."),
            (u"pyth\xf6n.org", b"xn--pythn-mua.org"),
            (u"pyth\xf6n.org.", b"xn--pythn-mua.org."),
        )

        for decoded, encoded in incremental_tests:
            if sys.version_info[0] == 2:
                self.assertEqual("".join(codecs.iterdecode(encoded, "idna")),
                                decoded)
            else:
                self.assertEqual("".join(codecs.iterdecode((bytes([c]) for c in encoded), "idna")),
                                decoded)

        decoder = codecs.getincrementaldecoder("idna")()
        self.assertEqual(decoder.decode(b"xn--xam", ), u"")
        self.assertEqual(decoder.decode(b"ple-9ta.o", ), u"\xe4xample.")
        self.assertEqual(decoder.decode(b"rg"), u"")
        self.assertEqual(decoder.decode(b"", True), u"org")

        decoder.reset()
        self.assertEqual(decoder.decode(b"xn--xam", ), u"")
        self.assertEqual(decoder.decode(b"ple-9ta.o", ), u"\xe4xample.")
        self.assertEqual(decoder.decode(b"rg."), u"org.")
        self.assertEqual(decoder.decode(b"", True), u"")


    def testIncrementalEncoder(self):

        # Tests derived from Python standard library test/test_codecs.py

        incremental_tests = (
            (u"python.org", b"python.org"),
            (u"python.org.", b"python.org."),
            (u"pyth\xf6n.org", b"xn--pythn-mua.org"),
            (u"pyth\xf6n.org.", b"xn--pythn-mua.org."),
        )
        for decoded, encoded in incremental_tests:
            self.assertEqual(b"".join(codecs.iterencode(decoded, "idna")),
                             encoded)

        encoder = codecs.getincrementalencoder("idna")()
        self.assertEqual(encoder.encode(u"\xe4x"), b"")
        self.assertEqual(encoder.encode(u"ample.org"), b"xn--xample-9ta.")
        self.assertEqual(encoder.encode(u"", True), b"org")

        encoder.reset()
        self.assertEqual(encoder.encode(u"\xe4x"), b"")
        self.assertEqual(encoder.encode(u"ample.org."), b"xn--xample-9ta.org.")
        self.assertEqual(encoder.encode(u"", True), b"")

if __name__ == '__main__':
    unittest.main()
