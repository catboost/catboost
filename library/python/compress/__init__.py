from io import open

import struct
import json
import os
import logging

import library.python.par_apply as lpp
import library.python.codecs as lpc


logger = logging.getLogger('compress')


def list_all_codecs():
    return sorted(frozenset(lpc.list_all_codecs()))


def find_codec(ext):
    def ext_compress(x):
        return lpc.dumps(ext, x)

    def ext_decompress(x):
        return lpc.loads(ext, x)

    ext_decompress(ext_compress(b''))

    return {'c': ext_compress, 'd': ext_decompress, 'n': ext}


def codec_for(path):
    for ext in reversed(path.split('.')):
        try:
            return find_codec(ext)
        except Exception as e:
            logger.debug('in codec_for(): %s', e)

    raise Exception('unsupported file %s' % path)


def compress(fr, to, codec=None, fopen=open, threads=1):
    if codec:
        codec = find_codec(codec)
    else:
        codec = codec_for(to)

    func = codec['c']

    def iter_blocks():
        with fopen(fr, 'rb') as f:
            while True:
                chunk = f.read(16 * 1024 * 1024)

                if chunk:
                    yield chunk
                else:
                    yield b''

                    return

    def iter_results():
        info = {
            'codec': codec['n'],
        }

        if fr:
            info['size'] = os.path.getsize(fr)

        yield json.dumps(info, sort_keys=True) + '\n'

        for c in lpp.par_apply(iter_blocks(), func, threads):
            yield c

    with fopen(to, 'wb') as f:
        for c in iter_results():
            logger.debug('complete %s', len(c))
            f.write(struct.pack('<I', len(c)))

            try:
                f.write(c)
            except TypeError:
                f.write(c.encode('utf-8'))


def decompress(fr, to, codec=None, fopen=open, threads=1):
    def iter_chunks():
        with fopen(fr, 'rb') as f:
            cnt = 0

            while True:
                ll = f.read(4)

                if ll:
                    ll = struct.unpack('<I', ll)[0]

                if ll:
                    if ll > 100000000:
                        raise Exception('broken stream')

                    yield f.read(ll)

                    cnt += ll
                else:
                    if not cnt:
                        raise Exception('empty stream')

                    return

    it = iter_chunks()
    extra = []

    for chunk in it:
        hdr = {}

        try:
            hdr = json.loads(chunk)
        except Exception as e:
            logger.info('can not parse header, suspect old format: %s', e)
            extra.append(chunk)

        break

    def resolve_codec():
        if 'codec' in hdr:
            return find_codec(hdr['codec'])

        if codec:
            return find_codec(codec)

        return codec_for(fr)

    dc = resolve_codec()['d']

    def iter_all_chunks():
        for x in extra:
            yield x

        for x in it:
            yield x

    with fopen(to, 'wb') as f:
        for c in lpp.par_apply(iter_all_chunks(), dc, threads):
            if c:
                logger.debug('complete %s', len(c))
                f.write(c)
            else:
                break
