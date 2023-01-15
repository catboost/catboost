PY23_LIBRARY()

LICENSE(BSD-3-Clause)



NO_COMPILER_WARNINGS()

PEERDIR(
    contrib/python/numpy
)

SRCS(
    bspline_util.c
    C_bspline_util.c
    D_bspline_util.c
    firfilter.c
    medianfilter.c
    S_bspline_util.c
    sigtoolsmodule.c
    splinemodule.c
    Z_bspline_util.c
    correlate_nd.c
    lfilter.c
)

PY_REGISTER(scipy.signal.sigtools)
PY_REGISTER(scipy.signal.spline)

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.signal

    __init__.py
    _arraytools.py
    bsplines.py
    filter_design.py
    fir_filter_design.py
    lti_conversion.py
    ltisys.py
    _max_len_seq.py
    _peak_finding.py
    _savitzky_golay.py
    signaltools.py
    spectral.py
    _upfirdn.py
    waveforms.py
    wavelets.py
    windows.py

    CYTHON_C
    _max_len_seq_inner.pyx
    _spectral.pyx
    _upfirdn_apply.pyx
)

END()
