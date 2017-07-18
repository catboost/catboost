#!/usr/bin/env python

import sys
import os


def BuildPlnHeader():
    if len(sys.argv) < 2:
        print >>sys.stderr, "Usage: build_pln_header.py <absolute/path/to/OutFile>"
        sys.exit(1)

    print >>sys.stdout, "Build Pln Header..."
    outPath = sys.argv[1]
    tmpPath = outPath + '.tmp'
    tmpFile = open(tmpPath, 'w')

    tmpFile.write('#include <library/sse2neon/sse_adhoc.h>\n')
    tmpFile.write('#include <kernel/relevfml/relev_fml.h>\n')
    for path in sys.argv[2:]:
        name = os.path.basename(path).split(".")[0]  # name without extensions
        tmpFile.write('\nextern SRelevanceFormula fml{0};\n'.format(name))
        tmpFile.write('float {0}(const float* f);\n'.format(name))
        tmpFile.write('void {0}SSE(const float* const* factors, float* result);\n'.format(name))
    tmpFile.close()
    try:
        os.remove(outPath)
    except:
        pass
    try:
        os.rename(tmpPath, outPath)
    except:
        print >>sys.stdout, 'Error: Failed to rename ' + tmpPath + ' to ' + outPath

if __name__ == '__main__':
    BuildPlnHeader()
